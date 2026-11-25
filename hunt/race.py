"""Concurrent operations on one workspace, then invariant check.

    python -m hunt.race --seeds 30 [--profile file]

Each seed: a base corpus is ingested, then a random mix of inserts,
deletions and admin edits runs concurrently (asyncio.gather) with a
slow fake LLM so the pipeline interleaves with the admin writes. Every
operation must either succeed or fail loudly; afterwards the stores must
agree (``hunt.state.check_invariants``) and no deleted document may still
attribute anything.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import tempfile
import uuid

from hunt.backends import DOCKER_ENV
from hunt.fakes import FakeLLM
from hunt.state import check_invariants, snapshot

NAMES = ["Alice", "Bob", "Carol Ann", "Dave", "Acme Corp", "北京", "Zoë", "Neo", "Orion🚀"]


class SlowLLM(FakeLLM):
    def __init__(self, rnd):
        super().__init__()
        self.rnd = rnd

    async def __call__(self, *a, **kw):
        await asyncio.sleep(self.rnd.random() * 0.02)
        return await super().__call__(*a, **kw)


def doc(rnd):
    return " ".join(
        f"{rnd.choice(NAMES)} met {rnd.choice(NAMES)} and {rnd.choice(NAMES)}." for _ in range(rnd.randint(1, 4))
    )


async def one(profile, seed):
    from hunt.run_scenario import make_rag

    rnd = random.Random(seed)
    llm = SlowLLM(rnd)
    rag, _ = await make_rag(profile, tempfile.mkdtemp(), f"r_{uuid.uuid4().hex[:8]}", llm=llm)
    base = {f"b{i}": doc(rnd) for i in range(4)}
    await rag.ainsert(list(base.values()), ids=list(base), file_paths=[f"{k}.txt" for k in base])
    log, deleted = [], set()

    async def op(i):
        await asyncio.sleep(rnd.random() * 0.05)
        kind = rnd.choice(["insert", "insert", "delete", "rename", "merge", "edit", "delete_entity"])
        try:
            if kind == "insert":
                d = f"n{seed}_{i}"
                await rag.ainsert(doc(rnd), ids=[d], file_paths=[f"{d}.txt"])
                r = "ok"
            elif kind == "delete":
                d = rnd.choice(sorted(base))
                res = await rag.adelete_by_doc_id(d)
                r = f"{res.status}:{res.status_code}"
                if res.status == "success":
                    deleted.add(d)
            elif kind == "rename":
                a = rnd.choice(NAMES)
                res = await rag.aedit_entity(a, {"entity_name": a + " X"})
                r = (res.get("operation_summary") or {}).get("operation_status")
            elif kind == "merge":
                a, b = rnd.sample(NAMES, 2)
                await rag.amerge_entities([a], b)
                r = "ok"
            elif kind == "edit":
                a, b = rnd.sample(NAMES, 2)
                await rag.aedit_relation(a, b, {"description": "edited"})
                r = "ok"
            else:
                res = await rag.adelete_by_entity(rnd.choice(NAMES))
                r = f"{res.status}:{res.status_code}"
        except Exception as e:
            r = f"EXC {type(e).__name__}: {str(e)[:150]}"
        log.append((i, kind, r))

    await asyncio.gather(*(op(i) for i in range(10)))
    # let any deferred pipeline run finish
    await rag.apipeline_process_enqueue_documents()
    from lightrag.kg.shared_storage import get_namespace_data

    ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    for _ in range(200):  # background drives (ingress mailbox) may still run
        if not ps.get("busy"):
            break
        await asyncio.sleep(0.05)
    idle_after = ps.get("busy")
    s = await snapshot(rag)
    if idle_after:
        log.append((99, "wait", "pipeline still busy after 10s"))
    errs = check_invariants(s)
    # docs reported deleted must not attribute anything
    for d in deleted:
        if d in s["doc_status"]:
            errs.append(f"deleted doc {d} still has doc_status {s['doc_status'][d]['status']}")
        for n, v in s["nodes"].items():
            if any(c.startswith(f"{d}-") for c in v["source_id"]):
                errs.append(f"node {n!r} still cites deleted doc {d}")
    stuck = {d: v["status"] for d, v in s["doc_status"].items() if v["status"] not in ("processed", "failed")}
    if stuck:
        errs.append(f"docs not settled: {stuck}")
    await rag.finalize_storages()
    return log, errs


async def main_async(profile, seeds):
    bad = 0
    for seed in range(seeds):
        log, errs = await one(profile, seed)
        weird = [x for x in log if (x[2].startswith("EXC") or x[1] == "wait") and "does not exist" not in x[2] and "AdminWriteGateRefused" not in x[2]
                 and "already exists" not in x[2] and "not found" not in x[2].lower()]
        if errs or weird:
            bad += 1
            print(f"seed={seed}")
            for x in sorted(log):
                print("   ", x)
            for e in errs[:8]:
                print("   !!", e)
    print(f"{bad}/{seeds} seeds with problems")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="file")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    if a.profile not in ("file", "file-faiss"):
        for k, v in DOCKER_ENV.items():
            os.environ.setdefault(k, v)
    asyncio.run(main_async(a.profile, a.seeds))


if __name__ == "__main__":
    main()
