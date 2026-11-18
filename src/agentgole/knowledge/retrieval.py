"""Keyword and vector ranking for local retrieval."""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter, defaultdict

from ..errors import ModelError
from ..protocols.types import Chunk

_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af"
_WORDS = re.compile(rf"([{_CJK}]+)|((?:(?![{_CJK}])[^\W_])+)")


def _tokens(text: str) -> list[str]:
    terms = []
    for cjk, word in _WORDS.findall(text):
        if word:
            terms.append(word.lower())
        elif len(cjk) == 1:
            terms.append(cjk)
        else:
            terms.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
    return terms


class BM25:
    """Index a corpus using BM25 with word tokens and CJK bigrams."""

    def __init__(self, texts: list[str]) -> None:
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            tokens = _tokens(text)
            self._lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings[term].append((index, count))
        self._average = sum(self._lengths) / len(texts) if texts else 0

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k matching document positions and positive scores."""
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(_tokens(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log1p((len(self._lengths) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, freq in postings:
                scale = 1.5 * (0.25 + 0.75 * self._lengths[index] / self._average)
                scores[index] += idf * freq * 2.5 / (freq + scale)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(0, k)]


def unit_vector(values: list[float]) -> array[float]:
    """Encode a finite vector as normalized native float32 values."""
    length = math.hypot(*values)
    if not values or not math.isfinite(length):
        raise ModelError("The embedding model returned an empty or non-finite vector.")
    return array("f", (value / (length or 1.0) for value in values))


class SearchIndex:
    """Rank a consistent snapshot of passages and their normalized embeddings."""

    def __init__(self, chunks: list[Chunk], vectors: list[array[float]]) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self._keywords: BM25 | None = None

    def _vector_rank(self, query: array[float], limit: int) -> list[tuple[int, float]]:
        try:
            import numpy as np  # pyright: ignore[reportMissingImports]
        except ImportError:
            scores = [sum(x * y for x, y in zip(vec, query, strict=True)) for vec in self.vectors]
        else:
            scores = (np.asarray(self.vectors) @ np.asarray(query)).tolist()
        return sorted(enumerate(scores), key=lambda item: -item[1])[:limit]

    def search(self, query: str, k: int, mode: str, vector: array[float] | None) -> list[Chunk]:
        """Return scored chunks using BM25, cosine similarity or reciprocal rank fusion."""
        rankings: list[list[tuple[int, float]]] = []
        limit = k * 4 if mode == "hybrid" else k
        if mode != "vector":
            if self._keywords is None:
                self._keywords = BM25([chunk.text for chunk in self.chunks])
            rankings.append(self._keywords.rank(query, limit))
        if vector is not None:
            rankings.append(self._vector_rank(vector, limit))
        if mode == "hybrid":
            fused: dict[int, float] = defaultdict(float)
            for ranking in rankings:
                for rank, (index, _) in enumerate(ranking, start=1):
                    fused[index] += 1 / (60 + rank)
            best = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        else:
            best = rankings[0]
        return [self.chunks[i].model_copy(update={"score": score}) for i, score in best]
