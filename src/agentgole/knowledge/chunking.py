"""Split source text on document and sentence boundaries."""

import hashlib
import re
from bisect import bisect_right

from ..protocols.types import Chunk, Document

_PARAGRAPH = re.compile(r"\n[ \t]*\n+|(?m:^[ \t]{0,3}(?=#{1,6}[ \t]))")
_SENTENCE = re.compile(r"[。！？.!?]+|\n")


def split_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Pack paragraphs, then sentences, retaining exact character overlap."""
    text = document.text
    boundaries = [
        [match.end() for match in pattern.finditer(text)] for pattern in (_PARAGRAPH, _SENTENCE)
    ]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for positions in boundaries:
                last = bisect_right(positions, end) - 1
                if last >= 0 and positions[last] > start + overlap:
                    end = positions[last]
                    break
        passage = text[start:end]
        if passage.strip():
            identity = f"{document.source}\0{document.page}\0{start}\0{passage}"
            chunks.append(
                Chunk(
                    id=hashlib.sha256(identity.encode()).hexdigest(),
                    text=passage,
                    source=document.source,
                    page=document.page,
                )
            )
        if end == len(text):
            break
        start = end - overlap
    return chunks
