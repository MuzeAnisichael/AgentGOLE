"""Keyword and vector ranking for local retrieval."""

import math
import re
from collections import Counter, defaultdict

_CJK = (
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    "\uf900-\ufaff\U00020000-\U000323af"
)
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
