"""Ledger of hand-made graph objects and the invariant they must satisfy.

Lesson of the closed #4688: a fix that cleaned up rename/merge residue also
deleted a source-less manual relation on the next document deletion, and no
harness check noticed. ``ManualLedger`` records every object created by
``acreate_entity`` / ``acreate_relation`` / ``ainsert_custom_kg``, follows it
through renames and merges, and ``check`` asserts:

M1  a manual entity is in the graph unless it was explicitly deleted
    (``adelete_by_entity``) -- deleting an unrelated document, renaming or
    merging never removes it (a merge moves it to the target name). Exception,
    documented in the ``acreate_*`` docstrings (#3856): once documents mention
    it, purging its last real source may delete it (``c1_victims``);
M2  a manual relation is in the graph whenever both of its (mapped) endpoints
    are; only an explicit relation/entity deletion, or the loss of an endpoint
    that a document alone supported, may remove it. A merge collapsing it into
    a self-loop drops it from the ledger (documented merge behaviour);
M3  a manual object keeps a non-empty description.

What main does when a manual relation's *document-derived* endpoint loses its
last document (the endpoint is purged and the relation with it) is recorded by
``probes/probe_manual_baseline.py`` and is allowed by M2.
"""

from __future__ import annotations

from lightrag.constants import GRAPH_FIELD_SEP


def _key(a: str, b: str) -> str:
    return GRAPH_FIELD_SEP.join(sorted((a, b)))


class ManualLedger:
    def __init__(self):
        self.entities: set[str] = set()
        self.relations: set[tuple[str, str]] = set()

    # --- recording -------------------------------------------------------
    def add_entity(self, name: str):
        self.entities.add(name)

    def add_relation(self, a: str, b: str):
        if a != b:
            self.relations.add(tuple(sorted((a, b))))

    def renamed(self, old: str, new: str):
        self._map({old: new})

    def merged(self, sources: list[str], target: str):
        self._map({s: target for s in sources if s != target})

    def entity_deleted(self, name: str):
        self.entities.discard(name)
        self.relations = {r for r in self.relations if name not in r}

    def relation_deleted(self, a: str, b: str):
        self.relations.discard(tuple(sorted((a, b))))

    def _map(self, m: dict[str, str]):
        self.entities = {m.get(e, e) for e in self.entities}
        rels = set()
        for a, b in self.relations:
            a2, b2 = m.get(a, a), m.get(b, b)
            if a2 != b2:
                rels.add(tuple(sorted((a2, b2))))
        self.relations = rels

    # --- checking --------------------------------------------------------
    def check(self, snap: dict) -> list[str]:
        errs = []
        nodes, edges = snap["nodes"], snap["edges"]
        for e in sorted(self.entities):
            if e not in nodes:
                errs.append(f"MANUAL entity {e!r} vanished")
            elif not (nodes[e].get("description") or "").strip():
                errs.append(f"MANUAL entity {e!r} lost its description")
        for a, b in sorted(self.relations):
            if a in nodes and b in nodes and _key(a, b) not in edges:
                errs.append(f"MANUAL relation {a!r}~{b!r} vanished while both endpoints exist")
        return errs

    def c1_victims(self, pre: dict, post: dict, chunk_prefix: str) -> list[str]:
        """Forget manual objects lost by the documented rule: every tracked
        chunk belonged to the deleted document, so the purge saw no source."""
        out = []

        def only_deleted(row):
            ids = (row or {}).get("chunk_ids") or []
            return bool(ids) and all(c.startswith(chunk_prefix) for c in ids)

        for e in sorted(self.entities):
            if e in pre["nodes"] and e not in post["nodes"] and only_deleted(pre["kv"]["entity_chunks"].get(e)):
                self.entity_deleted(e)
                out.append(e)
        for a, b in sorted(self.relations):
            k = _key(a, b)
            if k in pre["edges"] and k not in post["edges"] and only_deleted(pre["kv"]["relation_chunks"].get(k)):
                self.relation_deleted(a, b)
                out.append(k)
        return out

    def prune_missing_endpoints(self, snap: dict):
        """Forget relations whose endpoint is legitimately gone (M2)."""
        nodes = snap["nodes"]
        self.relations = {r for r in self.relations if r[0] in nodes and r[1] in nodes}
