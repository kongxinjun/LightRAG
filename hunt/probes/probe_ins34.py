import os, asyncio, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.fault_sweep import Injector, WRITE_METHODS
from hunt.insert_sweep import base, NEW
from hunt.state import snapshot, check_invariants
async def main():
    inj = Injector(WRITE_METHODS); rag = await base("file", inj)
    inj.k, inj.armed = 34, True
    await rag.ainsert(NEW, ids=["d3"], file_paths=["d3.txt"]); inj.armed = False
    print("fault:", inj.fired, "| d3:", (await rag.doc_status.get_by_id("d3"))["status"])
    r = await rag.adelete_by_doc_id("d3"); print("delete failed d3:", r.status, r.status_code)
    print("orphan tracking Neo:", (await rag.entity_chunks.get_by_id("Neo") or {}).get("chunk_ids"), "| graph has Neo:", await rag.chunk_entity_relation_graph.has_node("Neo"))
    await rag.ainsert("Neo visits Bob.", ids=["d9"], file_paths=["d9.txt"])
    print("new doc d9 -> Neo source_id:", (await rag.chunk_entity_relation_graph.get_node("Neo"))["source_id"])
    r = await rag.adelete_by_doc_id("d9"); print("delete d9:", r.status)
    n = await rag.chunk_entity_relation_graph.get_node("Neo")
    print("after deleting every doc that mentions Neo:", n and {"source_id": n["source_id"], "description": n["description"]})
    print(check_invariants(await snapshot(rag)))
    await rag.finalize_storages()
asyncio.run(main())
