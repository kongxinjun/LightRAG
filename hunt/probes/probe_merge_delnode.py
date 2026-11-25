import os, asyncio, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.fault_sweep import Injector, build, WRITE_METHODS
async def main():
    inj = Injector({"delete_node"}); rag = await build("file", inj)
    inj.k, inj.armed = 1, True
    try:
        r = await rag.aedit_entity("Alice", {"entity_name": "Bob"}, allow_merge=True)
        print("aedit_entity returned:", json.dumps({k: r.get(k) for k in ("entity_name",)}, ensure_ascii=False), json.dumps(r.get("operation_summary"), ensure_ascii=False))
    except Exception as e:
        print("raised", type(e).__name__, str(e)[:200])
    inj.armed=False
    g = rag.chunk_entity_relation_graph
    print("fired:", inj.fired, "| Alice exists:", await g.has_node("Alice"), "| Bob source_id:", (await g.get_node("Bob"))["source_id"])
    print("Bob edges:", sorted(map(tuple, await g.get_node_edges("Bob"))))
    await rag.finalize_storages()
asyncio.run(main())
