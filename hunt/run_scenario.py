"""Run one deterministic operation script against one backend profile.

    python -m hunt.run_scenario --profile file --out /tmp/hunt/file.json

Writes a JSON report: per-step snapshot, invariant violations and query
results. ``hunt.compare`` diffs two reports. One process per run so the
process-global shared storage starts empty.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
import uuid

from hunt.backends import DOCKER_ENV, PROFILES

DOCS = {
    "doc-a": (
        "Alice founded Acme Corp in 北京. Bob joined Acme Corp later. "
        "Alice and Bob met Carol Ann at 上海交通大学. Zoë visited Ünïcode Lab with Dave. "
    )
    * 3,
    "doc-b": (
        "Bob works with Dave. Dave studies at 上海交通大学 with Zoë. "
        "Neon is not Neo. Neo met Orion🚀 in 東京タワー. AB knows C. "
    )
    * 3,
    "doc-c": (
        "A visits BC. Δέλτα lab hosts Orion🚀 and Neon. Carol Ann writes to Alice. "
        "北京和上海交通大学合作。東京タワー位于东京。"
    )
    * 4,
    "doc-d": "Zoë alone. Δέλτα alone. Acme Corp hires Zoë. ",
}

QUERIES = [
    ("Who is Alice and how is Acme Corp related to Bob?", "local"),
    ("What does Dave study at 上海交通大学?", "global"),
    ("Tell me about Neo and Orion🚀", "hybrid"),
    ("Δέλτα and Neon", "mix"),
    ("Alice 北京", "naive"),
]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


async def make_rag(profile: str, workdir: str, workspace: str, llm=None, **extra):
    from lightrag import LightRAG
    from lightrag.utils import Tokenizer
    import tiktoken

    from hunt.fakes import FakeLLM, make_embedding_func

    class _Tik:
        def __init__(self):
            self.e = tiktoken.encoding_for_model("gpt-4o-mini")

        def encode(self, s):
            return self.e.encode(s)

        def decode(self, t):
            return self.e.decode(t)

    llm = llm or FakeLLM()
    rag = LightRAG(
        working_dir=workdir,
        workspace=workspace,
        llm_model_func=llm,
        embedding_func=make_embedding_func(64),
        tokenizer=Tokenizer("gpt-4o-mini", _Tik()),
        chunk_token_size=64,
        chunk_overlap_token_size=8,
        entity_extract_max_gleaning=1,
        max_parallel_insert=2,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.0},
        **PROFILES[profile],
        **extra,
    )
    await rag.initialize_storages()
    return rag, llm


async def query_digest(rag):
    from lightrag import QueryParam

    out = {}
    for q, mode in QUERIES:
        try:
            r = await rag.aquery_data(q, param=QueryParam(mode=mode, top_k=10, chunk_top_k=5))
            data = r.get("data", {}) if isinstance(r, dict) else {}
            out[f"{mode}:{q}"] = {
                "status": r.get("status") if isinstance(r, dict) else None,
                "entities": sorted(e.get("entity_name") for e in data.get("entities", [])),
                "relations": sorted(
                    "~".join(sorted((x.get("src_id"), x.get("tgt_id"))))
                    for x in data.get("relationships", [])
                ),
                "chunks": sorted(c.get("chunk_id") or c.get("content", "")[:20] for c in data.get("chunks", [])),
            }
        except Exception as e:
            out[f"{mode}:{q}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


async def run(profile: str, out_path: str, keep: bool = False):
    from hunt.state import check_invariants, snapshot

    report: dict = {"profile": profile, "steps": []}
    workdir = tempfile.mkdtemp(prefix=f"hunt-{profile}-")
    workspace = f"hunt_{uuid.uuid4().hex[:8]}"
    rag, llm = await make_rag(profile, workdir, workspace)

    async def step(name, coro_fn, query=False):
        t0 = time.time()
        rec = {"step": name}
        try:
            res = await coro_fn()
            rec["result"] = _jsonable(res)[:2000] if isinstance(res, str) else _jsonable(res)
        except Exception as e:
            rec["exception"] = f"{type(e).__name__}: {e}"[:600]
            rec["traceback"] = traceback.format_exc()[-3000:]
        try:
            s = await snapshot(rag)
            rec["snapshot"] = _jsonable(s)
            rec["violations"] = check_invariants(s)
        except Exception as e:
            rec["snapshot_error"] = f"{type(e).__name__}: {e}"
            rec["traceback_snap"] = traceback.format_exc()[-3000:]
        if query:
            rec["queries"] = await query_digest(rag)
        rec["secs"] = round(time.time() - t0, 2)
        report["steps"].append(rec)
        print(f"[{profile}] {name}: {rec.get('exception', 'ok')} violations={len(rec.get('violations', []))} {rec['secs']}s", flush=True)

    async def insert(ids):
        await rag.ainsert([DOCS[i] for i in ids], ids=list(ids), file_paths=[f"{i}.txt" for i in ids])
        return [ (await rag.doc_status.get_by_id(i) or {}).get("status") for i in ids]

    try:
        await step("insert a,b,c", lambda: insert(["doc-a", "doc-b", "doc-c"]), query=True)
        await step("re-insert a (dup)", lambda: insert(["doc-a"]))
        await step("insert d", lambda: insert(["doc-d"]))
        await step("delete b", lambda: _deletion(rag.adelete_by_doc_id("doc-b")), query=True)
        await step("rename Alice->Alicia", lambda: rag.aedit_entity("Alice", {"entity_name": "Alicia"}))
        await step("merge [Neon,Δέλτα]->Neon", lambda: rag.amerge_entities(["Neon", "Δέλτα"], "Neon"))
        await step("edit relation Alicia~Acme Corp", lambda: rag.aedit_relation("Alicia", "Acme Corp", {"description": "edited"}))
        await step("create entity+relation", lambda: _create(rag))
        await step("delete entity Dave", lambda: _deletion(rag.adelete_by_entity("Dave")))
        await step("delete a", lambda: _deletion(rag.adelete_by_doc_id("doc-a")), query=True)
        await step("re-insert b", lambda: insert(["doc-b"]), query=True)
        async def delete_c_with_cache():
            # rows doc-c's chunks reference, minus rows other live chunks share
            st = await rag.doc_status.get_by_id("doc-c")
            own = await rag.text_chunks.get_by_ids(list(st["chunks_list"]))
            mine = {c for r in own if r for c in (r.get("llm_cache_list") or [])}
            res = await _deletion(rag.adelete_by_doc_id("doc-c", delete_llm_cache=True))
            others = await rag.text_chunks.get_by_ids(
                [c for d in ("doc-b", "doc-d") for c in ((await rag.doc_status.get_by_id(d)) or {}).get("chunks_list", [])]
            )
            shared = {c for r in others if r for c in (r.get("llm_cache_list") or [])}
            left = [c for c in sorted(mine - shared) if await rag.llm_response_cache.get_by_id(c)]
            res["own_cache_rows"] = len(mine)
            res["own_cache_rows_left"] = left
            return res

        await step("delete c (cache too)", delete_c_with_cache)
        report["llm_calls"] = {k: llm.calls.count(k) for k in set(llm.calls)}
    finally:
        try:
            if not keep:
                for st in (rag.full_docs, rag.text_chunks, rag.full_entities, rag.full_relations,
                           rag.entity_chunks, rag.relation_chunks, rag.llm_response_cache,
                           rag.chunk_entity_relation_graph, rag.entities_vdb, rag.relationships_vdb,
                           rag.chunks_vdb, rag.doc_status):
                    try:
                        await st.drop()
                    except Exception:
                        pass
            await rag.finalize_storages()
        except Exception as e:
            report["finalize_error"] = repr(e)
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)


async def _deletion(coro):
    r = await coro
    return {"status": getattr(r, "status", None), "message": getattr(r, "message", None), "status_code": getattr(r, "status_code", None)}


async def _create(rag):
    await rag.acreate_entity("Hunt Probe", {"description": "probe entity", "entity_type": "test"})
    await rag.acreate_relation("Hunt Probe", "Bob", {"description": "probe rel", "keywords": "probe", "weight": 0.5})
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.profile not in ("file", "file-faiss"):
        for k, v in DOCKER_ENV.items():
            os.environ.setdefault(k, v)
    asyncio.run(run(a.profile, a.out, a.keep))


if __name__ == "__main__":
    sys.exit(main())
