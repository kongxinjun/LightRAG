"""Stateful fuzzing: random interleavings of ingest, delete and admin edits.

    HUNT_EXAMPLES=50 .venv/bin/python -m pytest hunt/test_stateful.py -p no:cacheprovider -q

After every step the cross-store invariants in ``hunt.state`` must hold.
Violations already attributed to a known defect are filtered by ``KNOWN``
so they do not mask new ones:

* rename/merge products keep dangling chunk refs after the source document
  is deleted (anchors are not rewritten on rename/merge) -- harness finding #1.
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

from hunt.state import check_invariants, snapshot
from hunt.test_props import FILLER, PROFILE, SAFE_VOCAB, document

LOOP = asyncio.new_event_loop()


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

    def teardown(self):
        import collections, json
        c = collections.Counter(x.split()[0] for x in self.log if not x.startswith("  "))
        with open("/tmp/hunt/sm_ops.jsonl", "a") as f:
            f.write(json.dumps(c) + "\n")
        run(self.rag.finalize_storages())
        shutil.rmtree(self.dir, ignore_errors=True)

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
        r = run(self.rag.adelete_by_doc_id(did, delete_llm_cache=cache))
        assert r.status == "success", (r.status, r.message)
        del self.docs[did]

    @rule(data=st.data())
    def rename(self, data):
        nodes = run(self._nodes())
        if not nodes:
            return
        a = data.draw(st.sampled_from(nodes))
        new = data.draw(st.sampled_from([f"{a}2", a.upper(), a.lower(), f" {a} ", f"{a}的"]))
        self.log.append(f"rename {a!r} -> {new!r}")
        try:
            r = run(self.rag.aedit_entity(a, {"entity_name": new}))
        except ValueError as e:  # refusals (exists, normalizes to same...) are fine
            self.log.append(f"  refused: {e}")
            return
        self.derived.add(r.get("entity_name", new))

    @rule(data=st.data())
    def merge(self, data):
        nodes = run(self._nodes())
        if len(nodes) < 2:
            return
        srcs = data.draw(st.lists(st.sampled_from(nodes), min_size=1, max_size=3, unique=True))
        tgt = data.draw(st.one_of(st.sampled_from(nodes), st.sampled_from(SAFE_VOCAB)))
        self.log.append(f"merge {srcs} -> {tgt!r}")
        try:
            run(self.rag.amerge_entities(srcs, tgt))
        except ValueError as e:
            self.log.append(f"  refused: {e}")
            return
        self.derived.add(tgt)

    @rule(data=st.data())
    def delete_entity(self, data):
        nodes = run(self._nodes())
        if not nodes:
            return
        a = data.draw(st.sampled_from(nodes))
        self.log.append(f"delete_entity {a!r}")
        r = run(self.rag.adelete_by_entity(a))
        assert r.status == "success", (r.status, r.message)

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
        except ValueError as e:
            self.log.append(f"  refused: {e}")
        if run(self.rag.chunk_entity_relation_graph.has_node(other)) and other != name:
            try:
                run(self.rag.acreate_relation(name, other, {"description": "m", "keywords": "k"}))
            except ValueError as e:
                self.log.append(f"  refused rel: {e}")

    @invariant()
    def consistent(self):
        s = run(snapshot(self.rag))
        errs = check_invariants(s)
        known = []
        for e in errs:
            if "references deleted chunks" in e and any(repr(d) in e or f"{d}<SEP>" in e or f"<SEP>{d}" in e for d in self.derived):
                known.append(e)
        new = [e for e in errs if e not in known]
        assert new == [], "\n".join(self.log[-15:] + ["--"] + new)


LightRAGMachine.TestCase.settings = settings(
    max_examples=int(os.environ.get("HUNT_EXAMPLES", "30")),
    stateful_step_count=int(os.environ.get("HUNT_STEPS", "12")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestLightRAG = LightRAGMachine.TestCase
