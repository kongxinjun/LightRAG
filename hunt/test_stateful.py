"""Stateful fuzzing: random interleavings of ingest, delete and admin edits.

    HUNT_EXAMPLES=50 .venv/bin/python -m pytest hunt/test_stateful.py -p no:cacheprovider -q

After every step the cross-store invariants in ``hunt.state`` must hold.
Violations that are documented accepted residues are filtered so they do
not mask new ones:

* rename/merge products keep the deleted document's chunk refs after it is
  deleted (*Merge and rename failure model*, first accepted residue, #4716).

Every hand-made object (``acreate_entity`` / ``acreate_relation`` /
``ainsert_custom_kg``) is followed by ``hunt.manual.ManualLedger``; its
M1-M3 checks run after every step (the lesson of the closed #4688).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from hunt.manual import ManualLedger
from hunt.state import check_invariants, snapshot
from hunt.test_props import FILLER, PROFILE, SAFE_VOCAB, document

LOOP = asyncio.new_event_loop()
# Documented decision (acreate_entity / acreate_relation docstrings, #3856):
# a manual object that later gained document evidence is purged with its last
# real source. Filtered by default; HUNT_FILTER_C1=0 shows it
# (probes/probe_manual_entity_purged.py).
FILTER_C1 = os.environ.get("HUNT_FILTER_C1", "1") != "0"


def run(coro):
    return LOOP.run_until_complete(coro)


class LightRAGMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        from hunt.run_scenario import make_rag

        self.dir = tempfile.mkdtemp(prefix="hunt-sm-")
        self.rag, self.llm = run(make_rag(PROFILE, self.dir, f"sm_{uuid.uuid4().hex[:10]}"))
        self.docs: dict[str, str] = {}
        self.n = 0
        self.derived: set[str] = set()  # names produced by rename/merge/create
        self.log: list[str] = []
        self.manual = ManualLedger()
        # chunk ids carried by objects a rename/merge moved (accepted residue)
        self.moved_chunks: set[str] = set()
        self.n_ckg = 0

    def teardown(self):
        import collections, json
        c = collections.Counter(x.split()[0] for x in self.log if not x.startswith("  "))
        with open("/tmp/hunt/sm_ops.jsonl", "a") as f:
            f.write(json.dumps(c) + "\n")
        run(self.rag.finalize_storages())
        shutil.rmtree(self.dir, ignore_errors=True)

    @initialize(seed=st.lists(document, min_size=1, max_size=3, unique=True))
    def seed_corpus(self, seed):
        for text in seed:
            self.insert(text)

    def _record_moved(self, names):
        snap = run(snapshot(self.rag))
        for n in names:
            self.moved_chunks.update(snap["nodes"].get(n, {}).get("source_id", []))
            for k, e in snap["edges"].items():
                if n in k.split("<SEP>"):
                    self.moved_chunks.update(e["source_id"])

    async def _nodes(self):
        return sorted(await self.rag.chunk_entity_relation_graph.get_all_labels())

    @rule(text=document)
    def insert(self, text):
        if text in self.docs.values():
            return
        did = f"d{self.n}"
        self.n += 1
        self.log.append(f"insert {did} {text[:60]!r}")
        run(self.rag.ainsert(text, ids=[did], file_paths=[f"{did}.txt"]))
        self.docs[did] = text

    @rule(data=st.data())
    def delete_doc(self, data):
        if not self.docs:
            return
        did = data.draw(st.sampled_from(sorted(self.docs)))
        cache = data.draw(st.booleans())
        self.log.append(f"delete {did} cache={cache}")
        pre = run(snapshot(self.rag)) if FILTER_C1 else None
        r = run(self.rag.adelete_by_doc_id(did, delete_llm_cache=cache))
        assert r.status == "success", (r.status, r.message)
        del self.docs[did]
        if FILTER_C1:
            post = run(snapshot(self.rag))
            for name in self.manual.c1_victims(pre, post, f"{did}-chunk-"):
                self.log.append(f"  documented: manual {name} purged with {did}")

    @rule(data=st.data())
    def rename(self, data):
        nodes = run(self._nodes())
        if not nodes:
            return
        a = data.draw(st.sampled_from(nodes))
        new = data.draw(st.sampled_from([f"{a}2", a.upper(), a.lower(), f" {a} ", f"{a}的"]))
        self.log.append(f"rename {a!r} -> {new!r}")
        self._record_moved([a])
        try:
            r = run(self.rag.aedit_entity(a, {"entity_name": new}))
        except ValueError as e:  # refusals (exists, normalizes to same...) are fine
            self.log.append(f"  refused: {e}")
            return
        stored = r.get("entity_name", new)
        self.derived.add(stored)
        self.manual.renamed(a, stored)

    @rule(data=st.data())
    def merge(self, data):
        nodes = run(self._nodes())
        if len(nodes) < 2:
            return
        srcs = data.draw(st.lists(st.sampled_from(nodes), min_size=1, max_size=3, unique=True))
        tgt = data.draw(st.one_of(st.sampled_from(nodes), st.sampled_from(SAFE_VOCAB)))
        self.log.append(f"merge {srcs} -> {tgt!r}")
        self._record_moved(list(srcs) + [tgt])
        try:
            r = run(self.rag.amerge_entities(srcs, tgt))
        except ValueError as e:
            self.log.append(f"  refused: {e}")
            return
        stored = (r or {}).get("entity_name", tgt) if isinstance(r, dict) else tgt
        self.derived.add(stored)
        self.manual.merged(srcs, stored)

    @rule(data=st.data())
    def delete_entity(self, data):
        nodes = run(self._nodes())
        if not nodes:
            return
        a = data.draw(st.sampled_from(nodes))
        self.log.append(f"delete_entity {a!r}")
        r = run(self.rag.adelete_by_entity(a))
        assert r.status == "success", (r.status, r.message)
        self.manual.entity_deleted(a)

    @rule(data=st.data())
    def delete_relation(self, data):
        edges = run(self.rag.chunk_entity_relation_graph.get_all_edges())
        if not edges:
            return
        s, t = data.draw(st.sampled_from(sorted((x["source"], x["target"]) for x in edges)))
        self.log.append(f"delete_relation {s!r}~{t!r}")
        r = run(self.rag.adelete_by_relation(s, t))
        assert r.status == "success", (r.status, r.message)
        self.manual.relation_deleted(s, t)

    @rule(
        names=st.lists(st.sampled_from(SAFE_VOCAB + ["Custom K", "定制"]), min_size=1, max_size=3, unique=True),
        with_src=st.booleans(),
    )
    def custom_kg(self, names, with_src):
        cid = f"ckg{self.n_ckg}"
        self.n_ckg += 1
        chunk = f"{cid}-chunk"
        kg = {
            "chunks": [{"content": " ".join(names) + " custom.", "source_id": chunk, "file_path": f"{cid}.txt"}],
            "entities": [
                {"entity_name": n, "entity_type": "custom", "description": f"custom {n}", "source_id": chunk}
                for n in names
            ],
            "relationships": [],
        }
        for a, b in zip(names, names[1:]):
            rel = {"src_id": a, "tgt_id": b, "description": f"custom {a}-{b}", "keywords": "ck", "weight": 1.0}
            if with_src:
                rel["source_id"] = chunk
            else:
                rel["weight"] = 0.5
            kg["relationships"].append(rel)
        self.log.append(f"custom_kg {cid} {names} src={with_src}")
        run(self.rag.ainsert_custom_kg(kg, full_doc_id=cid))
        for n in names:
            self.manual.add_entity(n)
        for a, b in zip(names, names[1:]):
            self.manual.add_relation(a, b)

    @rule(data=st.data())
    def edit_relation(self, data):
        edges = run(self.rag.chunk_entity_relation_graph.get_all_edges())
        if not edges:
            return
        e = data.draw(st.sampled_from(sorted((x["source"], x["target"]) for x in edges)))
        flip = data.draw(st.booleans())
        s, t = (e[1], e[0]) if flip else e
        self.log.append(f"edit_relation {s!r}~{t!r}")
        run(self.rag.aedit_relation(s, t, {"description": "edited " + s}))

    @rule(name=st.sampled_from(SAFE_VOCAB + ["Manual X", "手动"]), other=st.sampled_from(SAFE_VOCAB))
    def create(self, name, other):
        self.log.append(f"create {name!r} + rel to {other!r}")
        try:
            run(self.rag.acreate_entity(name, {"description": "manual", "entity_type": "t"}))
            self.derived.add(name)
            self.manual.add_entity(name)
        except ValueError as e:
            self.log.append(f"  refused: {e}")
        if run(self.rag.chunk_entity_relation_graph.has_node(other)) and other != name:
            try:
                run(self.rag.acreate_relation(name, other, {"description": "m", "keywords": "k", "weight": 0.5}))
                self.manual.add_relation(name, other)
            except ValueError as e:
                self.log.append(f"  refused rel: {e}")

    @invariant()
    def consistent(self):
        s = run(snapshot(self.rag))
        errs = check_invariants(s)
        # Accepted residue (Merge and rename failure model): objects a
        # rename/merge moved keep a later-deleted document's chunk ids, and a
        # later ingestion can carry them on to a neighbour. Known only when
        # every dead id was on a moved object at move time.
        live = set(s["kv"].get("text_chunks", {}))
        dead_ok = set()
        for kind in ("nodes", "edges"):
            for k, d in s[kind].items():
                dead = {c for c in d["source_id"] if c not in live and c not in ("", "UNKNOWN", "manual_creation")}
                if dead and dead <= self.moved_chunks:
                    dead_ok.add(k)
        known = [
            e for e in errs
            if "references deleted chunks" in e
            and any(e.startswith(f"node {k!r} ") or e.startswith(f"edge {k!r} ") for k in dead_ok)
        ]
        new = [e for e in errs if e not in known]
        new += self.manual.check(s)
        self.manual.prune_missing_endpoints(s)
        assert new == [], "\n".join(self.log[-15:] + ["--"] + new)


LightRAGMachine.TestCase.settings = settings(
    max_examples=int(os.environ.get("HUNT_EXAMPLES", "30")),
    stateful_step_count=int(os.environ.get("HUNT_STEPS", "12")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestLightRAG = LightRAGMachine.TestCase
