"""An admin edit racing a document deletion (no hooks).

    .venv/bin/python hunt/probes/probe_edit_vs_delete.py [profile] [trials] [rename|merge]

d0 = "Alice met Carol Ann." (only source of Alice and of the edge),
d1 = "Carol Ann is here. Dave is here."
rename: adelete_by_doc_id(d0) || aedit_entity(Carol Ann -> Carol)
merge:  adelete_by_doc_id(d0) || amerge_entities([Carol Ann], Dave)
Afterwards Alice and every edge to it must be gone, and the stores agree.
The purge takes no per-entity keyed lock, so the admin lock alone cannot
order the two; only the busy reservation (the full gate) does.
"""
import asyncio, collections, os, sys, tempfile, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import hunt  # noqa: F401  (honours HUNT_CODE)
from hunt.backends import DOCKER_ENV
from hunt.run_scenario import make_rag
from hunt.state import snapshot, check_invariants
from hunt.probes.probe_manual_entity_purged import drop

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "pg"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
KIND = sys.argv[3] if len(sys.argv) > 3 else "rename"
for k, v in DOCKER_ENV.items():
    os.environ.setdefault(k, v)


async def trial():
    rag, _ = await make_rag(PROFILE, tempfile.mkdtemp(), f"w{uuid.uuid4().hex[:6]}")
    try:
        await rag.ainsert(["Alice met Carol Ann.", "Carol Ann is here. Dave is here."], ids=["d0", "d1"],
                          file_paths=["d0.txt", "d1.txt"])
        edit = (rag.aedit_entity("Carol Ann", {"entity_name": "Carol"}) if KIND == "rename"
                else rag.amerge_entities(["Carol Ann"], "Dave"))
        res = await asyncio.gather(rag.adelete_by_doc_id("d0"), edit, return_exceptions=True)
        errs = [f"{type(r).__name__}: {str(r)[-120:]}" for r in res if isinstance(r, BaseException)]
        s = await snapshot(rag)
        bad = check_invariants(s)
        if "Alice" in s["nodes"] or any("Alice" in k.split("<SEP>") for k in s["edges"]):
            bad.append("deleted document's entity Alice or an edge to it survived")
        return errs, bad
    finally:
        await drop(rag)


async def main():
    out = collections.Counter()
    for _ in range(TRIALS):
        errs, bad = await trial()
        out[("raised " + "; ".join(errs) + " | " if errs else "") + ("ok" if not bad else "BAD " + "; ".join(sorted(set(bad))[:3]))] += 1
    for k, n in out.most_common():
        print(f"[{PROFILE} {KIND} vs delete] {n}/{TRIALS}  {k}")

if __name__ == "__main__":
    asyncio.run(main())
