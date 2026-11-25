import os, asyncio, sys, tempfile, uuid, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants

async def main(kind):
    rag, llm = await make_rag("file", tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    await rag.ainsert(["Alice met Bob.", "Bob met Dave."], ids=["d0", "d1"], file_paths=["d0.txt", "d1.txt"])
    if kind == "rename":
        await rag.aedit_entity("Alice", {"entity_name": "Alicia"})
    else:
        await rag.amerge_entities(["Alice"], "Dave")
    print("anchors d0:", await rag.full_entities.get_by_id("d0"), await rag.full_relations.get_by_id("d0"))
    r = await rag.adelete_by_doc_id("d0")
    print("delete:", r.status, r.message)
    s = await snapshot(rag)
    print("nodes:", json.dumps({k: v["source_id"] for k, v in s["nodes"].items()}, ensure_ascii=False))
    print("edges:", json.dumps({k: v["source_id"] for k, v in s["edges"].items()}, ensure_ascii=False))
    print("entity_chunks:", {k: v.get("chunk_ids") for k, v in s["kv"]["entity_chunks"].items()})
    print("violations:", check_invariants(s))
    from lightrag import QueryParam
    d = await rag.aquery_data("Alicia Dave", param=QueryParam(mode="local", top_k=5))
    print("query local chunks:", [c.get("chunk_id") for c in d["data"]["chunks"]], "entities:", [e["entity_name"] for e in d["data"]["entities"]])
    await rag.finalize_storages()
asyncio.run(main(sys.argv[1]))
