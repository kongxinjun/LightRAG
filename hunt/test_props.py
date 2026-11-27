"""Property tests over random documents and operation sequences.

    .venv/bin/python -m pytest hunt/test_props.py -p no:cacheprovider -q \
        --hypothesis-show-statistics

Runs on the file-backed profile (``HUNT_PROFILE`` selects another one). Each
example builds a fresh workspace, so examples never share state.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hunt import fakes
from hunt.state import GRAPH_FIELD_SEP, check_invariants, snapshot

PROFILE = os.environ.get("HUNT_PROFILE", "file")
if PROFILE not in ("file", "file-faiss"):
    from hunt.backends import DOCKER_ENV

    for k, v in DOCKER_ENV.items():
        os.environ.setdefault(k, v)

# Names whose concatenations collide (#4213) are left out so the known VDB-id
# collision does not mask everything else.
SAFE_VOCAB = [n for n in fakes.VOCAB if n not in ("AB", "C", "A", "BC")]
FILLER = ["the", "met", "and", "visited", "数据", "は", "über", "🙂", "x" * 30, "lorem ipsum dolor"]

sentence = st.lists(
    st.one_of(st.sampled_from(SAFE_VOCAB), st.sampled_from(FILLER)), min_size=1, max_size=8
).map(lambda ws: " ".join(ws) + ".")
document = st.lists(sentence, min_size=1, max_size=12).map(" ".join)
corpus = st.lists(document, min_size=1, max_size=4, unique=True)

settings.register_profile(
    "hunt",
    max_examples=int(os.environ.get("HUNT_EXAMPLES", "40")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
    derandomize=bool(os.environ.get("HUNT_DERANDOMIZE")),
)
settings.load_profile("hunt")


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _rag(**kw):
    from hunt.run_scenario import make_rag

    d = tempfile.mkdtemp(prefix="hunt-prop-")
    rag, llm = await make_rag(PROFILE, d, f"p_{uuid.uuid4().hex[:10]}", **kw)
    rag._hunt_dir = d
    return rag, llm


async def _close(rag):
    if PROFILE not in ("file", "file-faiss"):
        for name in ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                     "relation_chunks", "llm_response_cache", "chunk_entity_relation_graph",
                     "entities_vdb", "relationships_vdb", "chunks_vdb", "doc_status"):
            try:
                await getattr(rag, name).drop()
            except Exception:
                pass
    await rag.finalize_storages()
    shutil.rmtree(rag._hunt_dir, ignore_errors=True)


async def _insert(rag, docs: dict[str, str]):
    await rag.ainsert(list(docs.values()), ids=list(docs), file_paths=[f"{k}.txt" for k in docs])


def _logical(s):
    """Graph shape that must not depend on history: names, endpoints, evidence."""
    return (
        {n: tuple(d["source_id"]) for n, d in s["nodes"].items()},
        {k: tuple(d["source_id"]) for k, d in s["edges"].items()},
    )


@given(corpus)
def test_insert_then_delete_all_leaves_nothing(docs):
    async def go():
        rag, _ = await _rag()
        try:
            ids = {f"d{i}": t for i, t in enumerate(docs)}
            await _insert(rag, ids)
            s = await snapshot(rag)
            assert check_invariants(s) == []
            for d in ids:
                r = await rag.adelete_by_doc_id(d, delete_llm_cache=True)
                assert r.status == "success", r
                assert check_invariants(await snapshot(rag)) == []
            s = await snapshot(rag)
            assert s["nodes"] == {} and s["edges"] == {}
            leftovers = {k: list(v) for k, v in s["kv"].items() if v and k != "llm_response_cache"}
            assert leftovers == {}
            assert [k for k in s["kv"].get("llm_response_cache", {}) if ":extract:" in k] == []
            assert {k: v["exact_count"] for k, v in s["vectors"].items()} == {
                "entities": 0, "relationships": 0, "chunks": 0}
        finally:
            await _close(rag)

    _run(go())


@given(corpus, corpus)
def test_delete_restores_previous_graph(base, extra):
    """insert(base); insert(extra); delete(extra) == insert(base)."""

    async def go():
        rag, _ = await _rag()
        try:
            b = {f"b{i}": t for i, t in enumerate(base)}
            e = {f"e{i}": t for i, t in enumerate(extra) if t not in base}
            await _insert(rag, b)
            before = _logical(await snapshot(rag))
            if e:
                await _insert(rag, e)
                for d in e:
                    r = await rag.adelete_by_doc_id(d)
                    assert r.status == "success", r
            s = await snapshot(rag)
            assert check_invariants(s) == []
            assert _logical(s) == before
        finally:
            await _close(rag)

    _run(go())


@given(corpus, st.randoms(use_true_random=False))
def test_insert_order_does_not_matter(docs, rnd):
    async def go():
        ids = {f"d{i}": t for i, t in enumerate(docs)}
        order = list(ids)
        rnd.shuffle(order)
        out = []
        for seq in (list(ids), order):
            rag, _ = await _rag()
            try:
                for d in seq:
                    await _insert(rag, {d: ids[d]})
                out.append(_logical(await snapshot(rag)))
            finally:
                await _close(rag)
        assert out[0] == out[1]

    _run(go())


@given(corpus, st.sampled_from(SAFE_VOCAB), st.sampled_from(SAFE_VOCAB), st.booleans())
def test_rename_or_merge_keeps_stores_consistent(docs, a, b, merge):
    async def go():
        rag, _ = await _rag()
        try:
            ids = {f"d{i}": t for i, t in enumerate(docs)}
            await _insert(rag, ids)
            s0 = await snapshot(rag)
            if a not in s0["nodes"]:
                return
            if merge:
                if b not in s0["nodes"] or a == b:
                    return
                await rag.amerge_entities([a], b)
                gone, kept = a, b
            else:
                new = f"{a} renamed"
                r = await rag.aedit_entity(a, {"entity_name": new})
                gone, kept = a, r["entity_name"]  # stored (normalized) name
            s = await snapshot(rag)
            assert check_invariants(s) == []
            assert gone not in s["nodes"] and kept in s["nodes"]
            # evidence of the old name moved to the kept name
            want = set(s0["nodes"][a]["source_id"])
            assert want <= set(s["nodes"][kept]["source_id"])
            # every doc can still be deleted afterwards. What the rename/merge
            # moved may stay with dead chunk ids (accepted residue, #4716,
            # *Merge and rename failure model*); nothing else may.
            for d in ids:
                r = await rag.adelete_by_doc_id(d, delete_llm_cache=True)
                assert r.status == "success", r
            s = await snapshot(rag)
            assert set(s["nodes"]) <= {kept}, s["nodes"].keys()
            assert all(kept in k.split(GRAPH_FIELD_SEP) for k in s["edges"]), s["edges"].keys()
            residue = [e for e in check_invariants(s) if "references deleted chunks" in e and repr(kept) in e]
            assert [e for e in check_invariants(s) if e not in residue] == []
        finally:
            await _close(rag)

    _run(go())


@given(corpus)
def test_reinsert_is_idempotent(docs):
    async def go():
        rag, llm = await _rag()
        try:
            ids = {f"d{i}": t for i, t in enumerate(docs)}
            await _insert(rag, ids)
            s1 = await snapshot(rag)
            n_calls = len(llm.calls)
            await _insert(rag, ids)
            s2 = await snapshot(rag)
            assert _logical(s1) == _logical(s2)
            assert s1["kv"]["text_chunks"].keys() == s2["kv"]["text_chunks"].keys()
            assert len(llm.calls) == n_calls
        finally:
            await _close(rag)

    _run(go())


@pytest.mark.parametrize("dup_with_new_id", [True])
@given(corpus)
def test_same_content_new_id(dup_with_new_id, docs):
    """Same text under a second id is a duplicate, not a second copy."""

    async def go():
        rag, _ = await _rag()
        try:
            await _insert(rag, {"x0": docs[0]})
            s1 = await snapshot(rag)
            await _insert(rag, {"x1": docs[0]})
            s2 = await snapshot(rag)
            assert _logical(s1) == _logical(s2)
            assert check_invariants(s2) == []
        finally:
            await _close(rag)

    _run(go())


MANUAL_NAMES = ["Manual Zed", "手工实体", "Ωmega Manual"]  # never produced by a document


@given(corpus, st.lists(st.tuples(st.sampled_from(MANUAL_NAMES), st.sampled_from(SAFE_VOCAB)), min_size=1, max_size=4),
       st.sampled_from(["none", "rename", "merge", "delete_relation", "delete_entity", "rename_recreate",
                        "lower_weight"]),
       st.sampled_from(SAFE_VOCAB),
       st.sampled_from(SAFE_VOCAB))
def test_manual_objects_survive_document_deletion(docs, links, op, a, b):
    """Hand-made entities, and hand-made relations whose endpoints survive,
    are never removed by deleting documents, before or after a rename/merge
    (the regression that closed #4688). Manual names never occur in a
    document, so the documented "last real source" rule never applies."""
    from hunt.manual import ManualLedger

    async def go():
        rag, _ = await _rag()
        led = ManualLedger()
        try:
            ids = {f"d{i}": t for i, t in enumerate(docs)}
            await _insert(rag, ids)
            nodes = set((await snapshot(rag))["nodes"])
            for m in {m for m, _ in links}:
                await rag.acreate_entity(m, {"description": f"manual {m}", "entity_type": "t"})
                led.add_entity(m)
            for m, other in links:
                if other in nodes and not await rag.chunk_entity_relation_graph.has_edge(m, other):
                    await rag.acreate_relation(m, other, {"description": "m", "keywords": "k", "weight": 0.5})
                    led.add_relation(m, other)
            if op == "delete_relation" and a in nodes and b in nodes and a != b \
                    and await rag.chunk_entity_relation_graph.has_edge(a, b):
                # explicit deletion, then a hand-made relation under the same key
                await rag.adelete_by_relation(a, b)
                await rag.acreate_relation(a, b, {"description": "curated", "keywords": "k", "weight": 0.5})
                led.add_relation(a, b)
            elif op == "delete_entity" and a in nodes and b in nodes and a != b:
                # the anchors keep naming a and a~b; recreate both by hand
                await rag.adelete_by_entity(a)
                led.entity_deleted(a)  # its relations go with it
                await rag.acreate_entity(a, {"description": "curated", "entity_type": "t", "source_id": ""})
                await rag.acreate_relation(a, b, {"description": "curated", "keywords": "k", "weight": 0.5})
                led.add_entity(a)
                led.add_relation(a, b)
            elif op == "rename_recreate" and a in nodes and b in nodes and a != b:
                r = await rag.aedit_entity(a, {"entity_name": f"{a} renamed"})
                led.renamed(a, r["entity_name"])
                await rag.acreate_entity(a, {"description": "curated", "entity_type": "t"})
                await rag.acreate_relation(a, b, {"description": "curated", "keywords": "k", "weight": 0.5})
                led.add_entity(a)
                led.add_relation(a, b)
            elif op == "lower_weight" and a in nodes and b in nodes and a != b \
                    and await rag.chunk_entity_relation_graph.has_edge(a, b):
                await rag.aedit_relation(a, b, {"description": "lowered", "source_id": "", "weight": 0.25})
                led.add_relation(a, b)
            elif op == "rename" and a in nodes:
                r = await rag.aedit_entity(a, {"entity_name": f"{a} renamed"})
                led.renamed(a, r["entity_name"])
            elif op == "merge" and a in nodes and b in nodes and a != b:
                await rag.amerge_entities([a], b)
                led.merged([a], b)
            assert led.check(await snapshot(rag)) == []
            for d in ids:
                r = await rag.adelete_by_doc_id(d)
                assert r.status == "success", r
                s = await snapshot(rag)
                assert led.check(s) == [], (op, d)
                led.prune_missing_endpoints(s)
        finally:
            await _close(rag)

    _run(go())
