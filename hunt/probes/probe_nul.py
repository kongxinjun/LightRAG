import asyncio, sys, os, tempfile, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
for k,v in DOCKER_ENV.items(): os.environ.setdefault(k,v)
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants
async def main(profile):
    rag, _ = await make_rag(profile, tempfile.mkdtemp(), f"z_{uuid.uuid4().hex[:8]}")
    kg = {"chunks":[{"content":"nul chunk","source_id":"src-z","file_path":"z.txt"}],
          "entities":[{"entity_name":"Good","entity_type":"t","description":"ok","source_id":"src-z"},
                      {"entity_name":"Bad\x00Name","entity_type":"t","description":"has \x00 nul","source_id":"src-z"}],
          "relationships":[{"src_id":"Good","tgt_id":"Bad\x00Name","description":"r","keywords":"k","weight":1.0,"source_id":"src-z"}]}
    try:
        await rag.ainsert_custom_kg(kg); print(profile, "custom_kg OK")
    except Exception as e:
        print(profile, "custom_kg raised", type(e).__name__, str(e)[:120].replace("\n"," "))
    s = await snapshot(rag)
    print("   nodes:", list(s["nodes"]), "| chunks:", list(s["kv"]["text_chunks"]), "| vdb:", {k: v["exact_count"] for k, v in s["vectors"].items()}, "| full_docs:", list(s["kv"]["full_docs"]))
    print("   violations:", check_invariants(s)[:4])
    for st in (rag.full_docs, rag.text_chunks, rag.full_entities, rag.full_relations, rag.entity_chunks, rag.relation_chunks, rag.llm_response_cache, rag.chunk_entity_relation_graph, rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb, rag.doc_status):
        try: await st.drop()
        except Exception: pass
    await rag.finalize_storages()
asyncio.run(main(sys.argv[1]))
