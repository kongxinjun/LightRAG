"""Deterministic stand-ins for the LLM and the embedding model.

The fake LLM answers every prompt kind LightRAG sends without a network:

* entity extraction  -> one ``entity`` row per vocabulary name found in the
  chunk, one ``relation`` row per adjacent pair of names inside a sentence;
* gleaning           -> ``<|COMPLETE|>`` (nothing missed);
* keyword extraction -> JSON with the vocabulary names found in the query;
* description summary-> a sorted, de-duplicated join of the descriptions;
* query answer       -> a digest of the entity names the context mentions.

Everything is a pure function of the prompt, so two backends fed the same
documents must end up with the same logical state.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

TD = "<|#|>"
DONE = "<|COMPLETE|>"

# Vocabulary of entity names. Mixes ASCII, multi-word, CJK, accented and
# emoji names, plus names whose concatenations collide ("AB"+"C" == "A"+"BC").
VOCAB: list[str] = [
    "Alice",
    "Bob",
    "Carol Ann",
    "Dave",
    "Acme Corp",
    "北京",
    "上海交通大学",
    "Zoë",
    "Ünïcode Lab",
    "AB",
    "C",
    "A",
    "BC",
    "Neo",
    "Neon",
    "Orion🚀",
    "東京タワー",
    "Δέλτα",
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s*")
INPUT_RE = re.compile(r"---Input Text---\n```\n(.*?)\n```", re.S)
QUERY_RE = re.compile(r"User Query:\s*(.*?)\n\n---Output---", re.S)
DESC_NAME_RE = re.compile(r"^(Entity|Relation) Name: (.*)$", re.M)
DESC_LIST_RE = re.compile(r"Description List:\n\n```\n(.*?)\n```", re.S)


def _find_names(text: str) -> list[str]:
    """Vocabulary names in ``text`` in order of first appearance.

    A name only counts when it is not part of a longer vocabulary name at the
    same spot ("Neo" inside "Neon" does not count), and when its neighbours
    are not word characters for ASCII names.
    """
    hits: list[tuple[int, int, str]] = []
    for name in VOCAB:
        if name.isascii():
            pat = r"(?<![\w])" + re.escape(name) + r"(?![\w])"
        else:
            pat = re.escape(name)
        for m in re.finditer(pat, text):
            hits.append((m.start(), m.end(), name))
    # drop hits covered by a longer hit
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, n in hits:
        if s < last_end:
            continue
        kept.append((s, e, n))
        last_end = e
    seen: list[str] = []
    for _, _, n in kept:
        if n not in seen:
            seen.append(n)
    return seen


def extraction_rows(chunk: str) -> list[str]:
    rows: list[str] = []
    ents: list[str] = []
    rels: list[tuple[str, str, str]] = []
    for sent in SENTENCE_SPLIT.split(chunk):
        sent = sent.strip()
        if not sent:
            continue
        names = _find_names(sent)
        for n in names:
            if n not in ents:
                ents.append(n)
        for a, b in zip(names, names[1:]):
            if a != b:
                rels.append((a, b, sent))
    digest = hashlib.md5(chunk.encode()).hexdigest()[:6]
    for n in ents:
        rows.append(f"entity{TD}{n}{TD}concept{TD}{n} mentioned in chunk {digest}.")
    seen_pairs = set()
    for a, b, sent in rels:
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        rows.append(
            f"relation{TD}{a}{TD}{b}{TD}co-occurs{TD}{a} appears with {b} in chunk {digest}."
        )
    return rows


@dataclass
class FakeLLM:
    """Callable with the ``llm_model_func`` signature.

    ``fail_on`` is an optional predicate ``(kind, prompt) -> bool``; when it
    returns True the call raises, which lets a test inject LLM outages into a
    single phase (extraction, summary, keywords, answer).
    """

    calls: list[str] = field(default_factory=list)
    fail_on: object = None

    async def __call__(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        kind, answer = self.answer(prompt, system_prompt, history_messages, kwargs)
        self.calls.append(kind)
        if self.fail_on is not None and self.fail_on(kind, prompt):
            raise RuntimeError(f"injected LLM failure on {kind}")
        if kwargs.get("stream"):

            async def gen():
                yield answer

            return gen()
        return answer

    def answer(self, prompt, system_prompt, history_messages, kwargs):
        prompt = prompt or ""
        sp = system_prompt or ""
        if history_messages and "Based on the last extraction task" in prompt:
            return "glean", DONE
        m = INPUT_RE.search(prompt)
        if m and "Extract entities and relationships" in prompt:
            return "extract", "\n".join(extraction_rows(m.group(1)) + [DONE])
        if "high_level_keywords" in prompt and "User Query:" in prompt:
            q = QUERY_RE.search(prompt)
            query = q.group(1) if q else prompt
            names = _find_names(query)
            return "keywords", json.dumps(
                {"high_level_keywords": ["topic"], "low_level_keywords": names},
                ensure_ascii=False,
            )
        if "Description List:" in prompt:
            dl = DESC_LIST_RE.search(prompt)
            descs = []
            for line in (dl.group(1) if dl else "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    descs.append(json.loads(line).get("description", line))
                except Exception:
                    descs.append(line)
            return "summary", " | ".join(sorted(set(descs)))
        # Query answer: digest of the context.
        ctx = sp + "\n" + prompt
        names = sorted(_find_names(ctx))
        return "answer", "ANSWER: " + ", ".join(names)


def fake_embed_vector(text: str, dim: int) -> np.ndarray:
    """Bag of character trigrams hashed into ``dim`` buckets, L2-normalized.

    Texts that share words get a high cosine, so vector search ranks them the
    way a real model would on toy data, and the result is reproducible.
    """
    v = np.zeros(dim, dtype=np.float64)
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = int.from_bytes(hashlib.md5(t[i : i + 3].encode()).digest()[:4], "little")
        v[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    n = np.linalg.norm(v)
    if n == 0:
        v[0] = 1.0
        n = 1.0
    return (v / n).astype(np.float32)


def make_embedding_func(dim: int = 64):
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts, **_):
        return np.stack([fake_embed_vector(t, dim) for t in texts])

    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=_embed, model_name="fake-trigram")
