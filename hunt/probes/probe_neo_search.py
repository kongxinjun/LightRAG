import asyncio, sys, os, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
for k,v in DOCKER_ENV.items(): os.environ.setdefault(k,v)
from hunt.run_scenario import make_rag
NAMES=["hash#name","colon:name","a`b","emoji🚀name","angle<b>name","plain name","dot.name","C++ Builder","node.js","AT&T","R&D部门"]
async def main():
    rag,_=await make_rag("neo4j-redis-qdrant", tempfile.mkdtemp(), f"s_{uuid.uuid4().hex[:8]}")
    g=rag.chunk_entity_relation_graph
    for n in NAMES: await rag.acreate_entity(n, {"description":"d","entity_type":"t"})
    await asyncio.sleep(3)
    for n in NAMES:
        full=await g.search_labels(n, limit=50); pre=await g.search_labels(n[:4], limit=50)
        print(f"{n!r:18} full-query hit={n in full} ({len(full)})  prefix {n[:4]!r} hit={n in pre}")
    await g.drop(); await rag.finalize_storages()
asyncio.run(main())
