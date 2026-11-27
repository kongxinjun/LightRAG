import asyncio, sys, os, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
for k,v in DOCKER_ENV.items(): os.environ.setdefault(k,v)
from hunt.run_scenario import make_rag
async def main(profile):
    rag,_=await make_rag(profile, tempfile.mkdtemp(), f"e_{uuid.uuid4().hex[:8]}")
    g=rag.chunk_entity_relation_graph
    for n in ("Src","Dst"): await g.upsert_node(n, {"entity_id":n,"description":"d","entity_type":"t","source_id":"c1","file_path":"f"})
    await g.upsert_edge("Src","Dst",{"description":"e","keywords":"k","weight":1.0,"source_id":"c1","file_path":"f"})
    await g.index_done_callback()
    print(f"{profile:20} get_node_edges('Dst') = {await g.get_node_edges('Dst')}   get_node_edges('Src') = {await g.get_node_edges('Src')}")
    await g.drop(); await rag.finalize_storages()
asyncio.run(main(sys.argv[1]))
