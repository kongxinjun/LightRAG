# hunt/ — LLM-free bug-hunting harness for LightRAG

Not for upstream. Lives on the fork's `hunt/*` branches only (`hunt/harness`
round 1, `hunt/round2` rebased onto a newer `main`).

Everything is driven by deterministic fakes (`fakes.py`):

* **FakeLLM** answers every prompt kind LightRAG sends: extraction (one
  `entity` row per vocabulary name in the chunk, one `relation` row per
  adjacent pair in a sentence), gleaning (`<|COMPLETE|>`), keyword extraction
  (JSON), description summary, query answer. `fail_on=` injects LLM outages.
* **fake embedding**: character-trigram hashing into 64 dims, L2-normalised.
* Vocabulary mixes ASCII, multi-word, CJK, Greek, emoji, accented names, a
  prefix pair (`Neo`/`Neon`) and a concatenation collision (`AB`+`C` ==
  `A`+`BC`, the known relation-VDB-id collision).
* Real tiktoken (`gpt-4o-mini`), `chunk_token_size=64`, so chunk boundaries
  fall inside multi-byte text.

## Setup

```bash
uv sync --python 3.12 --extra test --extra offline
uv pip install --python .venv/bin/python hypothesis      # not in uv.lock
docker compose -f hunt/docker-compose.hunt.yml up -d     # only for DB profiles
# ... run things ...
docker compose -f hunt/docker-compose.hunt.yml down -v   # always clean up
```

Never point these at a `.env` with real keys; nothing here needs one.

## Running against another code tree

`HUNT_CODE=/path/to/checkout` makes every tool, probe and pytest run import
`lightrag` from that tree (a candidate fix, a closed PR) and fails loudly if it
could not. Do not rely on `PYTHONPATH`: pytest puts the `hunt/` package's
parent first on `sys.path`, so a `PYTHONPATH` override is silently ignored.

```bash
cd /tmp/somewhere   # keep the hypothesis database apart per code tree
HUNT_CODE=/tmp/proto PYTHONPATH=$REPO $REPO/.venv/bin/python -m pytest $REPO/hunt/test_stateful.py -p no:cacheprovider
```

**A candidate fix is only ready to propose after** the state machine and the
property tests pass against it (`HUNT_CODE`), including the hand-made-object
checks below, and the upstream offline suite still passes.

## Tools

| Command | What it does |
|---|---|
| `python -m hunt.run_scenario --profile P --out X.json` | 12-step script (insert, dup insert, delete, rename, merge, relation edit, manual create, entity delete, re-insert, delete with cache). Snapshot + invariant check + 5 queries after each step. |
| `python -m hunt.compare A.json B.json ...` | Step-by-step diff of reports (first one is the reference). |
| `python -m hunt.run_nasty --profile P --out X.json` | Hostile entity names (quotes, backslash, `%_`, backtick, 300 chars, combining accents, `<SEP>`, emoji, …) through every graph read API, label search, subgraph, edit, rename, delete. |
| `pytest hunt/test_props.py` | Hypothesis properties: delete-all leaves nothing, insert+delete restores, order independence, idempotent re-insert, rename/merge consistency. `HUNT_PROFILE`, `HUNT_EXAMPLES`. |
| `pytest hunt/test_stateful.py` | Hypothesis state machine over insert / delete / rename / merge / relation edit / create / entity delete, invariants after every step. |
| `python -m hunt.fault_sweep --op OP [--exc runtime\|cancel]` | Fail the k-th storage write of OP for every k, retry, compare with a clean run. OPs: delete, delete_cache, merge, rename, edit_relation, delete_entity, create_relation. |
| `python -m hunt.insert_sweep` | Same for ingestion: fault while a doc ingests, then delete the FAILED doc and check the graph is back to before. |
| `python -m hunt.race --seeds N` | Concurrent mixes of insert/delete/admin edits with a slow LLM; invariants after the pipeline settles. |
| `python hunt/probes/*.py` | Minimal reproductions of each reported candidate. |
| `python hunt/probes/probe_parallel_merges.py P N merge\|rename` | Two concurrent admin edits of adjacent entities, N trials (round 2, R1). |

Profiles (`backends.py`): `file` (Json/NetworkX/Nano), `file-faiss`, `pg`
(AGE graph), `pg-table`, `neo4j-redis-qdrant`, `mongo` (Atlas local, vector
search), `milvus`, `memgraph-milvus`.

## Hand-made objects (`manual.py`, round 2)

`ManualLedger` follows every `acreate_entity` / `acreate_relation` /
`ainsert_custom_kg` object through renames and merges (the regression that got
#4688 closed deleted a source-less manual relation, and nothing here noticed):

* M1 a manual entity survives unless explicitly deleted; the one documented
  exception (`acreate_*` docstrings, #3856) is a manual object that later gained
  document evidence, purged with its last real source -- filtered as
  `documented` (`HUNT_FILTER_C1=0` shows it);
* M2 a manual relation survives whenever both endpoints survive;
* M3 a manual object keeps a description.

The state machine checks them after every step and also has `custom_kg` and
`delete_relation` rules; `test_props.py::test_manual_objects_survive_document_deletion`
checks them across delete / rename / merge / delete-and-recreate.
Q1 (state machine, `HUNT_QUERY=0` to skip): every query mode returns only
live entities, relations and chunks.

The documented rename/merge residue (#4716) is filtered precisely: a dangling
chunk reference is accepted only when every dead id was on an object a
rename/merge moved, at move time.

## Invariants (`state.check_invariants`)

1. every graph node/edge has its vector row; vector `get_exact_count` equals what the graph implies (no orphans);
2. node/edge `source_id` names only live chunks, never empty; edge weight >= distinct evidence count; both endpoints exist;
3. `entity_chunks` / `relation_chunks` rows exist for every object and only for live objects; tracking covers `source_id`;
4. every chunk belongs to a live `full_docs` row (custom-KG chunks are the documented exception);
5. PROCESSED doc_status `chunks_list` / `chunks_count` match `text_chunks`;
6. after `delete_llm_cache=True`, the document's extract-cache rows are gone.

Violations that come from the known relation-VDB-id collision (`A~BC` vs
`AB~C`) are expected and are already covered upstream by open PRs.
