"""A hand-made entity is deleted with the first document that mentions it.

    .venv/bin/python hunt/probes/probe_manual_entity_purged.py [profile]

acreate_entity("Alice") -> ainsert("Alice met Bob.", d1) -> adelete_by_doc_id(d1).
The entity existed before the document; deleting the document should leave it.
Variants: manual relation instead of entity, ainsert_custom_kg instead of acreate.
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


async def drop(rag):
    for name in ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                 "relation_chunks", "llm_response_cache", "chunk_entity_relation_graph",
                 "entities_vdb", "relationships_vdb", "chunks_vdb", "doc_status"):
        try:
            await getattr(rag, name).drop()
        except Exception:
            pass
    await rag.finalize_storages()


async def scenario(kind):
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Bob met Dave."], ids=["d0"], file_paths=["d0.txt"])
        if kind == "acreate":
            await rag.acreate_entity("Alice", {"description": "curated Alice", "entity_type": "person"})
            await rag.acreate_relation("Alice", "Dave", {"description": "curated A-D", "keywords": "k", "weight": 0.5})
        elif kind == "custom_kg":
            await rag.ainsert_custom_kg({
                "chunks": [{"content": "Alice is curated.", "source_id": "ck-1", "file_path": "ck.txt"}],
                "entities": [{"entity_name": "Alice", "entity_type": "person", "description": "curated Alice", "source_id": "ck-1"}],
                "relationships": [{"src_id": "Alice", "tgt_id": "Dave", "description": "curated A-D", "keywords": "k", "weight": 1.0, "source_id": "ck-1"}],
            }, full_doc_id="ck")
        before = await snapshot(rag)
        print(f"[{PROFILE}] {kind:9s} before:", {k: v["source_id"] for k, v in before["nodes"].items() if k == "Alice"},
              "| tracking:", before["kv"]["entity_chunks"].get("Alice", {}).get("chunk_ids"))
        await rag.ainsert(["Alice met Bob."], ids=["d1"], file_paths=["d1.txt"])
        mid = await snapshot(rag)
        print("    after d1 insert:", mid["nodes"].get("Alice", {}).get("source_id"),
              "| tracking:", mid["kv"]["entity_chunks"].get("Alice", {}).get("chunk_ids"),
              "| desc:", (mid["nodes"].get("Alice", {}).get("description") or "")[:60].replace("\n", " / "))
        r = await rag.adelete_by_doc_id("d1")
        s = await snapshot(rag)
        print("    delete d1:", r.status, "| Alice in graph:", "Alice" in s["nodes"],
              "| Alice~Dave edge:", "Alice<SEP>Dave" in s["edges"],
              "| tracking:", s["kv"]["entity_chunks"].get("Alice", {}).get("chunk_ids"))
    finally:
        await drop(rag)


async def relation_only():
    """Both endpoints come from d0; only the relation is hand-made."""
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Bob is here. Dave is here."], ids=["d0"], file_paths=["d0.txt"])
        await rag.acreate_relation("Bob", "Dave", {"description": "curated B-D", "keywords": "k", "weight": 0.5})
        await rag.ainsert(["Bob met Dave."], ids=["d1"], file_paths=["d1.txt"])
        mid = await snapshot(rag)
        print(f"[{PROFILE}] relation  after d1 insert:", mid["edges"].get("Bob<SEP>Dave"))
        r = await rag.adelete_by_doc_id("d1")
        s = await snapshot(rag)
        print("    delete d1:", r.status, "| Bob, Dave in graph:", "Bob" in s["nodes"], "Dave" in s["nodes"],
              "| Bob~Dave edge:", s["edges"].get("Bob<SEP>Dave"))
    finally:
        await drop(rag)


async def main():
    for kind in ("acreate", "custom_kg"):
        await scenario(kind)
    await relation_only()

if __name__ == "__main__":
    asyncio.run(main())
