"""A stale anchor makes document deletion delete a later hand-made relation.

    .venv/bin/python hunt/probes/probe_stale_anchor_manual.py [profile]

d0 = "Alice met Bob."      (anchors: Alice, Bob, Alice~Bob)
d1 = "Alice is here. Bob is here."   (keeps both endpoints alive)
adelete_by_relation(Alice, Bob)          -- admin removes the extracted edge
acreate_relation(Alice, Bob, weight 0.5) -- admin curates a source-less one
   (variant: ainsert_custom_kg relation without source_id)
adelete_by_doc_id(d0)                    -- the curated edge is deleted
"""
import asyncio, os, sys, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants
from hunt.probes.probe_manual_entity_purged import drop  # noqa: E402

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "file"
if PROFILE not in ("file", "file-faiss"):
    for k, v in DOCKER_ENV.items():
        os.environ.setdefault(k, v)


async def scenario(kind):
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Bob.", "Alice is here. Bob is here."], ids=["d0", "d1"], file_paths=["d0.txt", "d1.txt"])
        print(f"[{PROFILE}] {kind}: d0 anchors:", (await rag.full_relations.get_by_id("d0") or {}).get("relation_pairs"))
        r = await rag.adelete_by_relation("Alice", "Bob")
        print("    delete relation:", r.status)
        if kind == "acreate_relation":
            await rag.acreate_relation("Alice", "Bob", {"description": "curated A-B", "keywords": "k", "weight": 0.5})
        else:
            await rag.ainsert_custom_kg({"chunks": [], "entities": [], "relationships": [
                {"src_id": "Alice", "tgt_id": "Bob", "description": "curated A-B", "keywords": "k", "weight": 0.5}]})
        s = await snapshot(rag)
        print("    curated edge:", s["edges"].get("Alice<SEP>Bob"))
        r = await rag.adelete_by_doc_id("d0")
        s = await snapshot(rag)
        print("    delete d0:", r.status, "| Alice, Bob alive:", "Alice" in s["nodes"], "Bob" in s["nodes"],
              "| curated edge after:", s["edges"].get("Alice<SEP>Bob"))
        print("    invariants:", check_invariants(s))
    finally:
        await drop(rag)


async def main():
    for kind in ("acreate_relation", "custom_kg"):
        await scenario(kind)

if __name__ == "__main__":
    asyncio.run(main())


async def edit_to_sourceless():
    """Documented weight lowering (ProgramingWithCore, Relation Weight Contract):
    set source_id to "" in the same edit. Both endpoints survive via d1."""
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Bob.", "Alice is here. Bob is here."], ids=["d0", "d1"], file_paths=["d0.txt", "d1.txt"])
        await rag.aedit_relation("Alice", "Bob", {"description": "curated A-B", "source_id": "", "weight": 0.3})
        s = await snapshot(rag)
        print(f"[{PROFILE}] edit_to_sourceless: edge:", s["edges"].get("Alice<SEP>Bob"),
              "| tracking:", s["kv"]["relation_chunks"].get("Alice<SEP>Bob"))
        r = await rag.adelete_by_doc_id("d0")
        s = await snapshot(rag)
        print("    delete d0:", r.status, "| Alice, Bob alive:", "Alice" in s["nodes"], "Bob" in s["nodes"],
              "| edge after:", s["edges"].get("Alice<SEP>Bob"))
    finally:
        await drop(rag)
