"""Stale anchors after explicit entity deletion; custom KG re-creates the pair.

    .venv/bin/python hunt/probes/probe_stale_anchor_entities.py [profile]

d0 = "Alice Bob Carol Ann."; adelete_by_entity(Alice), adelete_by_entity(Bob);
ainsert_custom_kg(entities Alice, Bob with a custom chunk; relation Alice~Bob
without source_id); adelete_by_doc_id(d0).
"""
import asyncio, os, sys, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot
from hunt.probes.probe_manual_entity_purged import drop

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "file"
if PROFILE not in ("file", "file-faiss"):
    for k, v in DOCKER_ENV.items():
        os.environ.setdefault(k, v)


async def main():
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice Bob Carol Ann."], ids=["d0"], file_paths=["d0.txt"])
        await rag.adelete_by_entity("Alice")
        await rag.adelete_by_entity("Bob")
        await rag.ainsert_custom_kg({
            "chunks": [{"content": "Alice Bob custom.", "source_id": "ck", "file_path": "ck.txt"}],
            "entities": [{"entity_name": n, "entity_type": "custom", "description": f"custom {n}", "source_id": "ck"} for n in ("Alice", "Bob")],
            "relationships": [{"src_id": "Alice", "tgt_id": "Bob", "description": "custom A-B", "keywords": "ck", "weight": 0.5}],
        }, full_doc_id="ckg0")
        s = await snapshot(rag)
        print(f"[{PROFILE}] before: nodes", {k: v["source_id"] for k, v in s["nodes"].items()})
        print("    edge:", s["edges"].get("Alice<SEP>Bob"), "| tracking:", s["kv"]["relation_chunks"].get("Alice<SEP>Bob"))
        r = await rag.adelete_by_doc_id("d0")
        s = await snapshot(rag)
        print("    delete d0:", r.status, "| nodes", {k: v["source_id"] for k, v in s["nodes"].items()},
              "| edge after:", s["edges"].get("Alice<SEP>Bob"))
    finally:
        await drop(rag)

if __name__ == "__main__":
    asyncio.run(main())
