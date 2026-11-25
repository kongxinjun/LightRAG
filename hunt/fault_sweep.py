"""Fault-injection sweep: fail the k-th storage call of an operation, retry.

    python -m hunt.fault_sweep --op delete [--profile file] [--exc runtime|cancel]

For each k in 1..N (N = storage calls the operation makes when nothing
fails) a fresh workspace is built, the k-th call raises, the operation is
retried with faults off, and the end state is compared with a clean run:

* the retry must succeed (or be refused for a documented reason);
* the logical graph must equal the clean run's;
* the cross-store invariants of ``hunt.state`` must hold;
* the first attempt must not report success while its fault fired
  (``AGENTS.md``: a failure must never be swallowed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import uuid

from hunt.backends import DOCKER_ENV
from hunt.state import check_invariants, snapshot

STORES = (
    "full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
    "relation_chunks", "llm_response_cache", "chunk_entity_relation_graph",
    "entities_vdb", "relationships_vdb", "chunks_vdb", "doc_status",
)
WRITE_METHODS = {
    "upsert", "delete", "index_done_callback", "upsert_node", "upsert_edge",
    "upsert_nodes_batch", "upsert_edges_batch", "delete_node", "remove_nodes",
    "remove_edges", "delete_entity", "delete_entity_relation", "update_doc_status_fields",
}
READ_METHODS = {"get_by_id", "get_by_ids", "get_node", "get_edge", "get_nodes_batch",
                "get_edges_batch", "get_node_edges", "has_node", "has_edge", "get_by_id_strict",
                "get_by_ids_strict", "filter_keys", "get_docs_by_statuses", "get_vectors_by_ids"}

DOCS = {
    "d1": "Alice met Bob in 北京. Bob works at Acme Corp. Zoë visited Dave.",
    "d2": "Bob met Dave. Dave studies at 上海交通大学. Alice knows Zoë.",
    "d3": "Carol Ann met Alice. Neo met Orion🚀 in 東京タワー.",
}


class Injector:
    def __init__(self, methods):
        self.methods = methods
        self.armed = False
        self.k = None
        self.n = 0
        self.fired = None
        self.exc = "runtime"
        self.trace: list[str] = []

    def wrap(self, rag):
        for name in STORES:
            st = getattr(rag, name)
            for m in self.methods:
                fn = getattr(st, m, None)
                if fn is None or not asyncio.iscoroutinefunction(fn):
                    continue
                setattr(st, m, self._mk(name, m, fn))

    def _mk(self, store, m, fn):
        async def w(*a, **kw):
            if self.armed:
                self.n += 1
                self.trace.append(f"{store}.{m}")
                if self.n == self.k:
                    self.fired = f"{store}.{m}#{self.n}"
                    if self.exc == "cancel":
                        raise asyncio.CancelledError()
                    raise RuntimeError(f"injected fault at {self.fired}")
            return await fn(*a, **kw)

        return w


def logical(s):
    return (
        {n: sorted(d["source_id"]) for n, d in s["nodes"].items()},
        {k: sorted(d["source_id"]) for k, d in s["edges"].items()},
        sorted(s["kv"]["text_chunks"]),
        sorted(s["kv"]["full_docs"]),
        {d: v["status"] for d, v in s["doc_status"].items() if not d.startswith("dup-")},
    )


async def build(profile, inj):
    from hunt.run_scenario import make_rag

    rag, llm = await make_rag(profile, tempfile.mkdtemp(), f"f_{uuid.uuid4().hex[:8]}")
    await rag.ainsert(list(DOCS.values()), ids=list(DOCS), file_paths=[f"{d}.txt" for d in DOCS])
    inj.wrap(rag)
    return rag


async def drop(rag, profile):
    if profile not in ("file", "file-faiss"):
        for n in STORES:
            try:
                await getattr(rag, n).drop()
            except Exception:
                pass
    try:
        await rag.finalize_storages()
    except Exception:
        pass


OPS = {
    "delete": lambda rag: rag.adelete_by_doc_id("d1"),
    "delete_cache": lambda rag: rag.adelete_by_doc_id("d1", delete_llm_cache=True),
    "merge": lambda rag: rag.amerge_entities(["Alice", "Zoë"], "Bob"),
    "rename": lambda rag: rag.aedit_entity("Alice", {"entity_name": "Alicia"}),
    "edit_relation": lambda rag: rag.aedit_relation("Alice", "Bob", {"description": "x", "weight": 9.0}),
    "delete_entity": lambda rag: rag.adelete_by_entity("Bob"),
    "create_relation": lambda rag: rag.acreate_relation("Neo", "Bob", {"description": "x", "keywords": "k"}),
}


def _ok(op, res):
    if op.startswith("delete") and hasattr(res, "status"):
        return res.status == "success", f"{res.status}:{res.status_code}:{(res.message or '')[:160]}"
    return True, "returned"


async def one(profile, op, k, methods, exc):
    inj = Injector(methods)
    rag = await build(profile, inj)
    rec = {"k": k}
    try:
        inj.k, inj.armed, inj.exc = k, True, exc
        try:
            res = await OPS[op](rag)
            ok, msg = _ok(op, res)
            rec["first"] = ("ok " if ok else "fail ") + msg
        except BaseException as e:  # noqa: BLE001 - CancelledError too
            rec["first"] = f"raised {type(e).__name__}: {str(e)[:200]}"
        inj.armed = False
        rec["fired"] = inj.fired
        rec["calls"] = inj.n
        if inj.fired and rec["first"].startswith("ok"):
            rec["SWALLOWED"] = True
        # retry once with faults off
        try:
            res = await OPS[op](rag)
            ok, msg = _ok(op, res)
            rec["retry"] = ("ok " if ok else "fail ") + msg
        except Exception as e:
            rec["retry"] = f"raised {type(e).__name__}: {str(e)[:200]}"
        s = await snapshot(rag)
        rec["violations"] = check_invariants(s)
        rec["logical"] = logical(s)
    finally:
        await drop(rag, profile)
    return rec, inj.trace


async def main_async(profile, op, kinds, exc, kmax):
    methods = set()
    if "write" in kinds:
        methods |= WRITE_METHODS
    if "read" in kinds:
        methods |= READ_METHODS
    # reference: count calls, no fault
    ref, trace = await one(profile, op, 10**9, methods, exc)
    n = ref["calls"]
    print(f"[{op}] reference: {n} storage calls; first={ref['first']} violations={ref['violations']}")
    out = {"op": op, "profile": profile, "ref": ref, "trace": trace, "runs": []}
    bad = 0
    for k in range(1, min(n, kmax) + 1):
        rec, _ = await one(profile, op, k, methods, exc)
        problems = []
        if rec.get("SWALLOWED"):
            problems.append("first attempt reported success although a fault fired")
        if not rec["retry"].startswith("ok") and not rec["first"].startswith("ok"):
            problems.append(f"retry did not succeed: {rec['retry']}")
        new_v = [v for v in rec["violations"] if v not in ref["violations"]]
        if new_v:
            problems.append(f"violations: {new_v[:4]}")
        if rec["logical"] != ref["logical"]:
            from hunt.state import diff

            problems.append(f"state differs from clean run: {diff(ref['logical'], rec['logical'])[:4]}")
        rec["problems"] = problems
        out["runs"].append(rec)
        if problems:
            bad += 1
            print(f"  k={k} fired={rec['fired']} first={rec['first'][:120]!r} retry={rec['retry'][:80]!r}")
            for p in problems:
                print(f"      - {p[:400]}")
    print(f"[{op}] {bad}/{min(n, kmax)} fault points left a problem")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--profile", default="file")
    ap.add_argument("--kinds", default="write")
    ap.add_argument("--exc", default="runtime", choices=["runtime", "cancel"])
    ap.add_argument("--kmax", type=int, default=400)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.profile not in ("file", "file-faiss"):
        for k, v in DOCKER_ENV.items():
            os.environ.setdefault(k, v)
    out = asyncio.run(main_async(a.profile, a.op, a.kinds.split(","), a.exc, a.kmax))
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
