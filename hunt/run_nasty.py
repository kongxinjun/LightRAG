"""Cross-backend probe with hostile entity names.

    python -m hunt.run_nasty --profile pg --out /tmp/hunt/nasty-pg.json

For every name: custom-KG insert, read back through every graph read API,
label search, subgraph, admin edit, rename and delete. The report records
what each call returned so ``hunt.compare_nasty`` can diff backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import traceback
import uuid

from hunt.backends import DOCKER_ENV, PROFILES

NAMES = [
    "O'Brien",
    'Say "Hi"',
    "back\\slash",
    "trailing\\",
    "100% _pure_",
    "a`b",
    "$dollar {brace}",
    "semi;colon -- comment",
    "x" * 300,
    "汉字'引号",
    "école",  # combining accent
    "école",
    "Tab\tName",
    "multi  space",
    "UPPER lower",
    "upper LOWER",
    "dot.name",
    "slash/name",
    "colon:name",
    "hash#name",
    "pipe|name",
    "angle<SEP>name",
    "emoji🚀name",
    # "null\x00byte" -- probed separately (hunt/probe_nul.py)
    "  padded  ",
]


def _short(v, n=300):
    s = json.dumps(v, ensure_ascii=False, default=str, sort_keys=True)
    return s if len(s) <= n else s[:n] + "..."


async def run(profile, out):
    from hunt.run_scenario import make_rag

    rag, _ = await make_rag(profile, tempfile.mkdtemp(), f"n_{uuid.uuid4().hex[:8]}")
    g = rag.chunk_entity_relation_graph
    rep = {"profile": profile, "names": {}}

    async def call(rec, label, fn):
        try:
            rec[label] = await fn()
        except Exception as e:
            rec[label] = f"EXC {type(e).__name__}: {str(e)[:200]}"

    # one custom KG: Hub + every name, each linked to Hub
    kg = {
        "chunks": [{"content": "nasty names chunk", "source_id": "src-nasty", "file_path": "nasty.txt"}],
        "entities": [{"entity_name": "Hub", "entity_type": "t", "description": "hub", "source_id": "src-nasty"}]
        + [{"entity_name": n, "entity_type": "t", "description": f"desc of {n}", "source_id": "src-nasty"} for n in NAMES],
        "relationships": [
            {"src_id": "Hub", "tgt_id": n, "description": f"hub-{n}", "keywords": "k", "weight": 1.0, "source_id": "src-nasty"}
            for n in NAMES
        ],
    }
    try:
        await rag.ainsert_custom_kg(kg)
        rep["custom_kg"] = "ok"
    except Exception as e:
        rep["custom_kg"] = f"EXC {type(e).__name__}: {e}"[:400]
        rep["custom_kg_tb"] = traceback.format_exc()[-2000:]

    labels = await g.get_all_labels()
    rep["labels"] = sorted(labels)
    for n in NAMES:
        rec: dict = {}
        stored = n if n in labels else next((l for l in labels if l.strip() == n.strip() or l.replace("\t", " ") == n.replace("\t", " ")), None)
        rec["stored_as"] = stored
        name = stored or n
        await call(rec, "has_node", lambda: g.has_node(name))
        await call(rec, "get_node.desc", lambda: _desc(g.get_node(name)))
        await call(rec, "node_degree", lambda: g.node_degree(name))
        await call(rec, "get_node_edges", lambda: _edges(g.get_node_edges(name)))
        await call(rec, "has_edge", lambda: g.has_edge("Hub", name))
        await call(rec, "get_edge.desc", lambda: _desc(g.get_edge(name, "Hub")))
        await call(rec, "batch_nodes", lambda: _keys(g.get_nodes_batch([name, "Hub"])))
        await call(rec, "search_labels", lambda: _contains(g.search_labels(name[:12], limit=500), name))
        await call(rec, "kg_subgraph", lambda: _kg(g.get_knowledge_graph(name, max_depth=1, max_nodes=50)))
        await call(rec, "edit_desc", lambda: _ok(rag.aedit_entity(name, {"description": "edited"})))
        await call(rec, "after_edit", lambda: _desc(g.get_node(name)))
        await call(rec, "rename", lambda: _ok(rag.aedit_entity(name, {"entity_name": name + " R"})))
        await call(rec, "renamed_exists", lambda: g.has_node(name + " R"))
        await call(rec, "old_exists", lambda: g.has_node(name))
        await call(rec, "renamed_edge", lambda: g.has_edge("Hub", name + " R"))
        await call(rec, "delete", lambda: _del(rag.adelete_by_entity(name + " R")))
        await call(rec, "deleted_gone", lambda: _not(g.has_node(name + " R")))
        rep["names"][n] = rec
    rep["final_labels"] = sorted(await g.get_all_labels())
    rep["final_vdb_counts"] = {}
    for label, st in (("entities", rag.entities_vdb), ("relationships", rag.relationships_vdb)):
        try:
            rep["final_vdb_counts"][label] = await st.get_exact_count()
        except Exception as e:
            rep["final_vdb_counts"][label] = f"EXC {type(e).__name__}"
    for st in (rag.full_docs, rag.text_chunks, rag.full_entities, rag.full_relations, rag.entity_chunks,
               rag.relation_chunks, rag.llm_response_cache, g, rag.entities_vdb, rag.relationships_vdb,
               rag.chunks_vdb, rag.doc_status):
        try:
            await st.drop()
        except Exception:
            pass
    await rag.finalize_storages()
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1, default=str)


async def _desc(c):
    r = await c
    return None if r is None else r.get("description")


async def _edges(c):
    r = await c
    return None if r is None else sorted(tuple(sorted(e)) for e in r)


async def _keys(c):
    return sorted((await c).keys())


async def _contains(c, name):
    r = await c
    return {"n": len(r), "hit": name in r}


async def _kg(c):
    r = await c
    return {"nodes": sorted(n.id for n in r.nodes), "edges": len(r.edges), "truncated": r.is_truncated}


async def _ok(c):
    r = await c
    return {"entity_name": r.get("entity_name"), "status": (r.get("operation_summary") or {}).get("operation_status")}


async def _del(c):
    r = await c
    return r.status


async def _not(c):
    return not await c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.profile not in ("file", "file-faiss"):
        for k, v in DOCKER_ENV.items():
            os.environ.setdefault(k, v)
    asyncio.run(run(a.profile, a.out))


if __name__ == "__main__":
    main()
