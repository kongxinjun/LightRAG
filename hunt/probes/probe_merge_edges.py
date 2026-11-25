import asyncio, sys, os, tempfile, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
for k,v in DOCKER_ENV.items(): os.environ.setdefault(k,v)
from hunt.run_scenario import make_rag
async def main(profile):
    rag,_=await make_rag(profile, tempfile.mkdtemp(), f"m_{uuid.uuid4().hex[:8]}")
    g=rag.chunk_entity_relation_graph
    await rag.ainsert("Alice met Zoë. Zoë met Dave. Bob is here.", ids=["d1"], file_paths=["d1.txt"])
    print(f"{profile:20} Zoë edges before: {sorted(map(tuple, await g.get_node_edges('Zoë')))}")
    await rag.amerge_entities(["Zoë"], "Bob")
    be = sorted(tuple(sorted(e)) for e in await g.get_node_edges("Bob"))
    rc = [k for k in ["Alice<SEP>Zoë", "Dave<SEP>Zoë", "Alice<SEP>Bob", "Bob<SEP>Dave"] if await rag.relation_chunks.get_by_id(k)]
    print(f"{profile:20} Bob edges after merge: {be} | relation_chunks rows: {rc} | rel vdb count: {await rag.relationships_vdb.get_exact_count()}")
    for st in (rag.full_docs, rag.text_chunks, rag.full_entities, rag.full_relations, rag.entity_chunks, rag.relation_chunks, rag.llm_response_cache, g, rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb, rag.doc_status):
        try: await st.drop()
        except Exception: pass
    await rag.finalize_storages()
asyncio.run(main(sys.argv[1]))
