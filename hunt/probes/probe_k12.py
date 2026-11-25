import os, asyncio, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.fault_sweep import Injector, build, WRITE_METHODS
from hunt.state import snapshot, check_invariants
async def main(k):
    inj = Injector(WRITE_METHODS); rag = await build("file", inj)
    inj.k, inj.armed = k, True
    r = await rag.adelete_by_doc_id("d1"); inj.armed=False
    print("first:", r.status, r.status_code, inj.fired)
    r = await rag.adelete_by_doc_id("d1"); print("retry:", r.status, r.status_code)
    print("entity_chunks Acme Corp:", await rag.entity_chunks.get_by_id("Acme Corp"))
    await rag.ainsert("Acme Corp hires Neo.", ids=["d9"], file_paths=["d9.txt"])
    n = await rag.chunk_entity_relation_graph.get_node("Acme Corp")
    print("after new doc, Acme Corp source_id:", n["source_id"], "| tracking:", (await rag.entity_chunks.get_by_id("Acme Corp"))["chunk_ids"])
    r = await rag.adelete_by_doc_id("d9"); print("delete d9:", r.status)
    n = await rag.chunk_entity_relation_graph.get_node("Acme Corp")
    print("after deleting d9, Acme Corp:", n and (n["source_id"], n["description"][:80]))
    print(check_invariants(await snapshot(rag)))
    await rag.finalize_storages()
asyncio.run(main(int(sys.argv[1])))
