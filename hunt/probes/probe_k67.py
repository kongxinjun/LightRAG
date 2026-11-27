import os, asyncio, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.fault_sweep import Injector, build, DOCS, WRITE_METHODS
async def main():
    inj = Injector(WRITE_METHODS); rag = await build("file", inj)
    inj.k, inj.armed = 67, True
    r = await rag.adelete_by_doc_id("d1"); inj.armed=False
    print("first:", r.status, r.status_code, inj.fired)
    print("full_docs d1:", bool(await rag.full_docs.get_by_id("d1")), "doc_status d1:", await rag.doc_status.get_by_id("d1"))
    await rag.ainsert(DOCS["d1"], ids=["d1"], file_paths=["d1.txt"])
    st = await rag.doc_status.get_by_id("d1")
    print("re-insert d1 ->", st and st.get("status"), "chunks:", st and st.get("chunks_list"))
    print("graph has 北京:", await rag.chunk_entity_relation_graph.has_node("北京"))
    ds = await rag.doc_status.get_docs_by_statuses(["failed"])
    print("failed docs:", {k:(v.get('error_msg') if isinstance(v,dict) else getattr(v,'error_msg',None)) for k,v in ds.items()})
    await rag.finalize_storages()
asyncio.run(main())
