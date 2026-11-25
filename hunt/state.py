"""Backend-neutral snapshot of a LightRAG instance, plus invariant checks.

``snapshot`` reads every store through the public storage interfaces only, so
the same code runs on any backend combination. Values that legitimately vary
between backends (timestamps, vectors, row order) are dropped or sorted.
"""

from __future__ import annotations

from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key, make_relation_vdb_ids

VOLATILE = {
    "created_at",
    "updated_at",
    "timestamp",
    "create_time",
    "update_time",
    "_id",
    "id",
    "vector",
    "__vector__",
    "track_id",
    "workspace",
    "metadata",
    "error_msg",
    "__created_at__",
}


def _split(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return sorted(str(x) for x in v)
    return sorted(x for x in str(v).split(GRAPH_FIELD_SEP) if x)


async def _kv_rows(storage) -> dict[str, dict] | None:
    try:
        rows = {}
        async for row in storage.iter_rows(page_size=500):
            rid = row.get("_id") or row.get("id")
            rows[rid] = row
        return rows
    except Exception as e:  # backend without enumeration
        return {"__unsupported__": {"error": type(e).__name__}}


def _clean(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in VOLATILE:
            continue
        if isinstance(v, list):
            try:
                v = sorted(v)
            except TypeError:
                pass
        out[k] = v
    return out


async def snapshot(rag) -> dict[str, Any]:
    g = rag.chunk_entity_relation_graph
    nodes = await g.get_all_nodes()
    edges = await g.get_all_edges()
    node_map = {}
    for n in nodes:
        nid = n.get("entity_id") or n.get("id")
        node_map[nid] = {
            "source_id": _split(n.get("source_id")),
            "file_path": _split(n.get("file_path")),
            "entity_type": n.get("entity_type"),
            "description": n.get("description"),
        }
    edge_map = {}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        key = GRAPH_FIELD_SEP.join(sorted((s, t)))
        edge_map[key] = {
            "source_id": _split(e.get("source_id")),
            "weight": float(e.get("weight") or 0.0),
            "keywords": e.get("keywords"),
            "description": e.get("description"),
        }

    kv = {}
    for name in (
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
    ):
        st = getattr(rag, name, None)
        if st is None:
            continue
        rows = await _kv_rows(st)
        kv[name] = {k: _clean(v) for k, v in sorted(rows.items())}

    from lightrag.base import DocStatus

    ds = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    doc_status = {}
    for k, v in sorted(ds.items()):
        g = (lambda f: v.get(f)) if isinstance(v, dict) else (lambda f: getattr(v, f, None))
        st = g("status")
        doc_status[k] = {
            "status": getattr(st, "value", str(st)),
            "chunks_count": g("chunks_count"),
            "chunks_list": sorted(g("chunks_list") or []),
        }

    # Vector stores: no enumeration on the base class, so probe by expected
    # ids and compare the exact count with what the graph implies.
    ent_ids = {compute_mdhash_id(n, prefix="ent-"): n for n in node_map}
    rel_ids = {}
    for key in edge_map:
        s, t = key.split(GRAPH_FIELD_SEP)
        rel_ids[make_relation_vdb_ids(s, t)[0]] = key
    chunk_ids = set(kv.get("text_chunks", {}).keys())
    vec = {}
    for label, st, ids in (
        ("entities", rag.entities_vdb, ent_ids),
        ("relationships", rag.relationships_vdb, rel_ids),
        ("chunks", rag.chunks_vdb, {c: c for c in chunk_ids}),
    ):
        found = await st.get_by_ids(sorted(ids)) if ids else []
        present = sorted({r.get("id") or r.get("_id") for r in found if r})
        try:
            count = await st.get_exact_count()
        except Exception as e:
            count = f"n/a:{type(e).__name__}"
        vec[label] = {
            "expected": len(ids),
            "present": len(present),
            "missing": sorted(ids[i] if isinstance(ids, dict) else i for i in set(ids) - set(present)),
            "exact_count": count,
            "content": {
                (ids[r.get("id") or r.get("_id")]): _vec_payload(r)
                for r in found
                if r
            },
        }
    return {
        "nodes": dict(sorted(node_map.items())),
        "edges": dict(sorted(edge_map.items())),
        "kv": kv,
        "doc_status": doc_status,
        "vectors": vec,
    }


def _vec_payload(r: dict) -> dict:
    keep = {}
    for k in ("entity_name", "src_id", "tgt_id", "source_id", "file_path", "full_doc_id"):
        if k in r:
            keep[k] = _split(r[k]) if k in ("source_id", "file_path") else r[k]
    return keep


PLACEHOLDERS = {"", "UNKNOWN", "manual_creation"}


def check_invariants(s: dict[str, Any], *, cache_purged: bool = False) -> list[str]:
    """Cross-store invariants that must hold in any quiescent state."""
    errs: list[str] = []
    chunks = set(s["kv"].get("text_chunks", {}))
    docs_done = {k for k, v in s["doc_status"].items() if v["status"] == "processed"}

    # 1. every graph node / edge has a vector row, and nothing extra exists
    for label in ("entities", "relationships", "chunks"):
        v = s["vectors"][label]
        if v["missing"]:
            errs.append(f"vdb[{label}] missing rows for {v['missing'][:5]}")
        if isinstance(v["exact_count"], int) and v["exact_count"] != v["expected"]:
            errs.append(
                f"vdb[{label}] exact_count={v['exact_count']} but graph/kv implies {v['expected']} (orphans or dupes)"
            )
    # 2. source_id of every node/edge references live chunks only
    for n, d in s["nodes"].items():
        dead = [c for c in d["source_id"] if c not in PLACEHOLDERS and c not in chunks]
        if dead:
            errs.append(f"node {n!r} references deleted chunks {dead[:3]}")
        if not d["source_id"]:
            errs.append(f"node {n!r} has empty source_id")
    for k, d in s["edges"].items():
        dead = [c for c in d["source_id"] if c not in PLACEHOLDERS and c not in chunks]
        if dead:
            errs.append(f"edge {k!r} references deleted chunks {dead[:3]}")
        n_real = len([c for c in d["source_id"] if c not in ("", "UNKNOWN", "manual_creation")])
        if d["weight"] + 1e-9 < n_real:
            errs.append(f"edge {k!r} weight {d['weight']} < evidence count {n_real}")
        a, b = k.split(GRAPH_FIELD_SEP)
        if a not in s["nodes"] or b not in s["nodes"]:
            errs.append(f"edge {k!r} has a missing endpoint")
    # 3. chunk tracking agrees with graph source_id
    ec = s["kv"].get("entity_chunks", {})
    for n, d in s["nodes"].items():
        row = ec.get(n)
        if row is None:
            errs.append(f"entity_chunks has no row for node {n!r}")
            continue
        tracked = sorted(row.get("chunk_ids") or [])
        real = {c for c in d["source_id"] if c not in PLACEHOLDERS}
        if real - set(tracked):
            errs.append(f"node {n!r} source_id not in entity_chunks: {sorted(real - set(tracked))[:3]}")
    for n in ec:
        if n not in s["nodes"]:
            errs.append(f"entity_chunks row {n!r} has no graph node")
    rc = s["kv"].get("relation_chunks", {})
    for k in rc:
        if k not in s["edges"]:
            errs.append(f"relation_chunks row {k!r} has no graph edge")
    # 4. every chunk belongs to a doc that still exists
    full_docs = set(s["kv"].get("full_docs", {}))
    for cid, row in s["kv"].get("text_chunks", {}).items():
        if row.get("full_doc_id") not in full_docs:
            errs.append(f"chunk {cid} belongs to missing doc {row.get('full_doc_id')}")
    # 5. doc_status chunk list matches text_chunks
    for d, row in s["doc_status"].items():
        if d not in docs_done:
            continue
        lst = set(row["chunks_list"])
        if lst - chunks:
            errs.append(f"doc {d} lists missing chunks {sorted(lst - chunks)[:3]}")
        if row["chunks_count"] is not None and row["chunks_count"] != len(lst):
            errs.append(f"doc {d} chunks_count {row['chunks_count']} != len(chunks_list) {len(lst)}")
    # 6. llm cache rows are reachable from some chunk's llm_cache_list
    reachable = set()
    for row in s["kv"].get("text_chunks", {}).values():
        reachable.update(row.get("llm_cache_list") or [])
    # Only meaningful right after a delete_llm_cache=True purge, and only for
    # rows the caller knows were attributed to the purged document.
    for cid in s["kv"].get("llm_response_cache", {}) if cache_purged else ():
        if cid.startswith("default:extract:") and cid not in reachable:
            errs.append(f"extract cache row {cid} unreachable from any chunk")
    return errs


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Readable structural diff, capped."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append(f"{path}/{k}: only in B -> {str(b[k])[:160]}")
            elif k not in b:
                out.append(f"{path}/{k}: only in A -> {str(a[k])[:160]}")
            else:
                out.extend(diff(a[k], b[k], f"{path}/{k}"))
    elif a != b:
        out.append(f"{path}: A={str(a)[:160]} | B={str(b)[:160]}")
    return out
