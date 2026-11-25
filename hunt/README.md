# hunt/ — LLM-free bug-hunting harness for LightRAG

Not for upstream. Lives on the fork's `hunt/harness` branch only.

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

Profiles (`backends.py`): `file` (Json/NetworkX/Nano), `file-faiss`, `pg`
(AGE graph), `pg-table`, `neo4j-redis-qdrant`, `mongo` (Atlas local, vector
search), `milvus`, `memgraph-milvus`.

## Invariants (`state.check_invariants`)

1. every graph node/edge has its vector row; vector `get_exact_count` equals what the graph implies (no orphans);
2. node/edge `source_id` names only live chunks, never empty; edge weight >= distinct evidence count; both endpoints exist;
3. `entity_chunks` / `relation_chunks` rows exist for every object and only for live objects; tracking covers `source_id`;
4. every chunk belongs to a live `full_docs` row (custom-KG chunks are the documented exception);
5. PROCESSED doc_status `chunks_list` / `chunks_count` match `text_chunks`;
6. after `delete_llm_cache=True`, the document's extract-cache rows are gone.

Violations that come from the known relation-VDB-id collision (`A~BC` vs
`AB~C`) are expected and are already covered upstream by open PRs.
