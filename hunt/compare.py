"""Diff scenario reports step by step.

    python -m hunt.compare /tmp/hunt/file.json /tmp/hunt/pg.json [...]

The first report is the reference. Prints, per step, the violations of each
profile and every snapshot / query difference against the reference.
"""

from __future__ import annotations

import json
import sys

from hunt.state import diff

# Fields that legitimately differ between backends.
IGNORE_PREFIXES = (
    "/kv/llm_response_cache",  # row payload shape differs per backend
    "/vectors/chunks/content",
    "/vectors/entities/content",  # PG returns raw table columns (documented shape gap)
    "/vectors/relationships/content",
)


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items() if x not in (None, {}, [], "")}
    if isinstance(v, str) and "<SEP>" in v:
        return "<SEP>".join(sorted(v.split("<SEP>")))
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def _strip(snap: dict) -> dict:
    s = json.loads(json.dumps(snap))
    # duplicate-placeholder doc ids embed a random suffix
    ds = s.get("doc_status", {})
    for k in [k for k in ds if k.startswith("dup-")]:
        ds.setdefault("dup-*", []).append(ds.pop(k))
    for name, rows in s.get("kv", {}).items():
        for k, row in list(rows.items()):
            if isinstance(row, dict):
                for f in ("count", "llm_cache_list_count", "original_prompt", "return", "cache_type", "chunk_id", "mode", "queryparam"):
                    row.pop(f, None)
    return _norm(s)


def main(paths: list[str]) -> int:
    reps = [json.load(open(p)) for p in paths]
    ref = reps[0]
    bad = 0
    for i, st in enumerate(ref["steps"]):
        print(f"\n### step {i}: {st['step']}")
        for r in reps:
            s = r["steps"][i]
            ex = s.get("exception") or s.get("snapshot_error")
            print(f"  [{r['profile']}] exc={ex!r} violations={s.get('violations')}")
        for r in reps[1:]:
            a, b = ref["steps"][i], r["steps"][i]
            d = [
                x
                for x in diff(_strip(a.get("snapshot", {})), _strip(b.get("snapshot", {})))
                if not x.startswith(IGNORE_PREFIXES)
            ]
            dq = diff(a.get("queries", {}), b.get("queries", {}), "/queries")
            dr = diff(a.get("result"), b.get("result"), "/result")
            for x in (d + dq + dr)[:25]:
                print(f"  DIFF {ref['profile']} vs {r['profile']}: {x}")
            bad += len(d + dq + dr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
