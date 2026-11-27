"""Baseline: what main does to manually created objects across delete / rename / merge.

    .venv/bin/python hunt/probes/probe_manual_baseline.py [profile]

Each scenario: d0 = "Alice met Bob.", d1 = "Bob met Dave."; manual entity Zed,
manual relations Alice~Zed and Zed~Dave; then an admin op, then delete d0.
"""
import asyncio, json, os, sys, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "file"
if PROFILE not in ("file", "file-faiss"):
    for k, v in DOCKER_ENV.items():
        os.environ.setdefault(k, v)


async def scenario(op):
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Bob.", "Bob met Dave."], ids=["d0", "d1"], file_paths=["d0.txt", "d1.txt"])
        await rag.acreate_entity("Zed", {"description": "manual Zed", "entity_type": "person"})
        await rag.acreate_relation("Alice", "Zed", {"description": "manual A-Z", "keywords": "k"})
        await rag.acreate_relation("Zed", "Dave", {"description": "manual Z-D", "keywords": "k"})
        if op == "rename":
            await rag.aedit_entity("Alice", {"entity_name": "Alicia"})
        elif op == "merge":
            await rag.amerge_entities(["Alice"], "Bob")
        elif op == "edit-merge":
            await rag.aedit_entity("Alice", {"entity_name": "Bob"}, allow_merge=True)
        r = await rag.adelete_by_doc_id("d0")
        s = await snapshot(rag)
        print(f"[{PROFILE}] {op:10s} delete d0: {r.status}")
        print("   nodes:", json.dumps({k: v["source_id"] for k, v in s["nodes"].items()}, ensure_ascii=False))
        print("   edges:", json.dumps({k: v["source_id"] for k, v in s["edges"].items()}, ensure_ascii=False))
    finally:
        for name in ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                     "relation_chunks", "llm_response_cache", "chunk_entity_relation_graph",
                     "entities_vdb", "relationships_vdb", "chunks_vdb", "doc_status"):
            try:
                await getattr(rag, name).drop()
            except Exception:
                pass
        await rag.finalize_storages()


async def main():
    for op in ("none", "rename", "merge", "edit-merge"):
        await scenario(op)

if __name__ == "__main__":
    asyncio.run(main())
