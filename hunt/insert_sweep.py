"""Fault sweep over ingestion: fail the k-th storage call while d3 ingests.

    python -m hunt.insert_sweep [--profile file] [--exc runtime|cancel]

After the faulted insert the document must be FAILED (never PROCESSED with a
fault swallowed), deleting it must succeed and leave the graph exactly as
before the insert, and a clean re-insert must reproduce the clean run.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from hunt.backends import DOCKER_ENV
from hunt.fault_sweep import DOCS, WRITE_METHODS, Injector, drop, logical
from hunt.state import check_invariants, diff, snapshot

NEW = "Alice met Neo. Neo founded Acme Corp in 東京タワー. Zoë and Dave joined."


async def base(profile, inj):
    import tempfile, uuid
    from hunt.run_scenario import make_rag

    rag, _ = await make_rag(profile, tempfile.mkdtemp(), f"i_{uuid.uuid4().hex[:8]}")
    await rag.ainsert([DOCS["d1"], DOCS["d2"]], ids=["d1", "d2"], file_paths=["d1.txt", "d2.txt"])
    inj.wrap(rag)
    return rag


async def one(profile, k, exc):
    inj = Injector(WRITE_METHODS)
    rag = await base(profile, inj)
    rec = {"k": k}
    try:
        before = logical(await snapshot(rag))
        inj.k, inj.armed, inj.exc = k, True, exc
        try:
            await rag.ainsert(NEW, ids=["d3"], file_paths=["d3.txt"])
            rec["insert"] = "returned"
        except BaseException as e:  # noqa: BLE001
            rec["insert"] = f"raised {type(e).__name__}: {str(e)[:150]}"
        inj.armed = False
        rec["fired"], rec["calls"] = inj.fired, inj.n
        st = await rag.doc_status.get_by_id("d3")
        rec["status"] = st and str(st.get("status"))
        s_after = await snapshot(rag)
        rec["after_fault"] = logical(s_after)
        if inj.fired and rec["status"] and rec["status"].endswith("processed"):
            rec.setdefault("problems", []).append("doc PROCESSED although a fault fired")
        if rec["status"] and not rec["status"].endswith("processed"):
            r = await rag.adelete_by_doc_id("d3")
            rec["delete"] = f"{r.status}:{r.status_code}:{(r.message or '')[:120]}"
            s = await snapshot(rag)
            if r.status != "success":
                rec.setdefault("problems", []).append(f"delete of failed doc: {rec['delete']}")
            elif logical(s) != before:
                rec.setdefault("problems", []).append(
                    "after deleting the failed doc the graph differs from before the insert: "
                    + "; ".join(diff(before, logical(s))[:3])
                )
            v = check_invariants(s)
            if v:
                rec.setdefault("problems", []).append(f"violations after delete: {v[:4]}")
    finally:
        await drop(rag, profile)
    return rec


async def main_async(profile, exc, kmax):
    ref = await one(profile, 10**9, exc)
    n = ref["calls"]
    print(f"reference: {n} write calls, status={ref['status']}")
    bad = 0
    for k in range(1, min(n, kmax) + 1):
        rec = await one(profile, k, exc)
        if rec.get("problems"):
            bad += 1
            print(f"  k={k} fired={rec['fired']} insert={rec['insert'][:60]!r} status={rec['status']} delete={rec.get('delete','')[:80]!r}")
            for p in rec["problems"]:
                print("     -", p[:350])
    print(f"{bad}/{min(n, kmax)} fault points left a problem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="file")
    ap.add_argument("--exc", default="runtime", choices=["runtime", "cancel"])
    ap.add_argument("--kmax", type=int, default=300)
    a = ap.parse_args()
    if a.profile not in ("file", "file-faiss"):
        for k, v in DOCKER_ENV.items():
            os.environ.setdefault(k, v)
    asyncio.run(main_async(a.profile, a.exc, a.kmax))


if __name__ == "__main__":
    main()
