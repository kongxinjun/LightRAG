"""amerge_entities crashes when a neighbour is renamed concurrently.

    .venv/bin/python hunt/probes/probe_merge_neighbor_race.py [profile]   # pg, neo4j-redis-qdrant, mongo, memgraph

The merge locks only its source and target names. Between
``get_node_edges(source)`` and ``get_edge(src, tgt)`` a concurrent admin write
on a *neighbour* (here: renaming Dave, which moves the edge Alice~Dave to
Alice~David) removes the edge, ``get_edge`` returns None, and the merge dies
with AttributeError after it has already upserted the target node.
Found by ``hunt.race --profile pg`` (seeds 3, 4); this pins the interleaving
deterministically by running the rename from inside the merge's first
``get_edge``. Not reachable on NetworkX: the admin write gate serialises
admin writes there.
"""
import asyncio, os, sys, tempfile, traceback, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants
from hunt.probes.probe_manual_entity_purged import drop

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "pg"
for k, v in DOCKER_ENV.items():
    os.environ.setdefault(k, v)


async def main():
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Dave.", "Bob is here."], ids=["d0", "d1"], file_paths=["d0.txt", "d1.txt"])
        g = rag.chunk_entity_relation_graph
        real_get_edge = g.get_edge
        fired = []

        async def get_edge(src, tgt):
            if not fired and {src, tgt} == {"Alice", "Dave"}:
                fired.append(1)
                r = await rag.aedit_entity("Dave", {"entity_name": "David"})  # concurrent neighbour edit
                print(f"[{PROFILE}] concurrent rename Dave->David:", r.get("entity_name"))
            return await real_get_edge(src, tgt)

        g.get_edge = get_edge
        try:
            r = await rag.amerge_entities(["Alice"], "Bob")
            print("    merge returned:", r.get("entity_name") if isinstance(r, dict) else r)
        except Exception as e:
            print("    merge raised:", type(e).__name__, e)
            tb = traceback.extract_tb(e.__traceback__)
            print("    at:", [f"{os.path.basename(f.filename)}:{f.lineno}" for f in tb][-2:])
        finally:
            g.get_edge = real_get_edge
        s = await snapshot(rag)
        print("    nodes:", sorted(s["nodes"]), "| edges:", sorted(s["edges"]))
        print("    invariants:", check_invariants(s))
    finally:
        await drop(rag)

if __name__ == "__main__":
    asyncio.run(main())
