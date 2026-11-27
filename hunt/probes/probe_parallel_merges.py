"""Two concurrent admin edits that share an edge (no hooks).

    .venv/bin/python hunt/probes/probe_parallel_merges.py [profile] [trials] [merge|rename]

rename: aedit_entity(Alice -> Alicia) || aedit_entity(Carol Ann -> Carol);
the relation must end up as Alicia~Carol.

Doc: "Alice met Carol Ann." plus "Bob is here." / "Dave is here." docs.
merge(Alice -> Bob) || merge(Carol Ann -> Dave): each locks only its own
source+target, but both rewrite the shared edge Alice~Carol Ann. Afterwards
the relation must exist as Bob~Dave (both merges redirect their side).
Counts trials that raised, and trials that silently lost the relation.
"""
import asyncio, collections, os, sys, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants
from hunt.probes.probe_manual_entity_purged import drop

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "pg"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
KIND = sys.argv[3] if len(sys.argv) > 3 else "merge"   # merge | rename
for k, v in DOCKER_ENV.items():
    os.environ.setdefault(k, v)


async def trial():
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Carol Ann.", "Bob is here.", "Dave is here."], ids=["d0", "d1", "d2"],
                          file_paths=["d0.txt", "d1.txt", "d2.txt"])
        if KIND == "merge":
            ops = [rag.amerge_entities(["Alice"], "Bob"), rag.amerge_entities(["Carol Ann"], "Dave")]
            want = "Bob<SEP>Dave"
        else:
            ops = [rag.aedit_entity("Alice", {"entity_name": "Alicia"}), rag.aedit_entity("Carol Ann", {"entity_name": "Carol"})]
            want = "Alicia<SEP>Carol"
        res = await asyncio.gather(*ops, return_exceptions=True)
        errs = [f"{type(r).__name__}: {str(r)[-160:].replace(chr(10)," ")}" for r in res if isinstance(r, BaseException)]
        s = await snapshot(rag)
        return errs, sorted(s["edges"]), sorted(s["nodes"]), check_invariants(s), want
    finally:
        await drop(rag)


async def main():
    outcomes = collections.Counter()
    for i in range(TRIALS):
        errs, edges, nodes, inv, want = await trial()
        key = ("raised " + "; ".join(errs)) if errs else ("ok" if want in edges else f"LOST relation: edges={edges} nodes={nodes}")
        outcomes[key + (f" | invariants={inv}" if inv else "")] += 1
    for k, n in outcomes.most_common():
        print(f"[{PROFILE} {KIND}] {n}/{TRIALS}  {k}")

if __name__ == "__main__":
    asyncio.run(main())
