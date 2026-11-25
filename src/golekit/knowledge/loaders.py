"""Discover knowledge sources and load their text into documents."""

import asyncio
import glob
import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from golekit._html import visible_text
from golekit.errors import ConfigError, MissingDependencyError
from golekit.protocols.types import Document

_logger = logging.getLogger(__name__)
_TEXT = {".txt", ".md", ".markdown"}
_SUPPORTED = _TEXT | {".pdf", ".docx"}
_INSTALL = 'pip install "golekit[rag]"'


def _is_url(source: str) -> bool:
    return source.lower().startswith(("https://", "http://"))


def _discover(sources: Sequence[str | Path]) -> list[str]:
    found: dict[str, None] = {}
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                # Hidden entries such as .git or the index store are only read when named.
                if child.name.startswith(".") or (child.is_symlink() and child.is_dir()):
                    continue
                visit(child)
        elif path.is_file():
            if path.suffix.lower() in _SUPPORTED:
                found[str(path)] = None
            else:
                _logger.warning("Skipping unsupported knowledge source: %s", path)

    for source in sources:
        if _is_url(str(source)):
            found[str(source)] = None
            continue
        path = Path(source).expanduser()
        if path.exists():
            visit(path)
        elif glob.has_magic(str(path)):
            matches = sorted(glob.iglob(str(path), recursive=True))
            if not matches:
                _logger.warning("Knowledge pattern matched no files: %s", path)
            for match in matches:
                visit(Path(match))
        else:
            raise ConfigError(f"Knowledge source not found: {path}. Check the path.")
    return list(found)


async def discover_sources(sources: Sequence[str | Path]) -> list[str]:
    """Expand directories and globs into unique absolute paths and preserve URL sources."""
    return await asyncio.to_thread(_discover, sources)


def _load_file(path: Path) -> list[Document]:
    source = str(path.resolve())
    suffix = path.suffix.lower()
    if suffix in _TEXT:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(text=text, source=source)]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise MissingDependencyError(f"Loading PDF requires pypdf. Run {_INSTALL}.") from exc
        with path.open("rb") as stream:
            return [
                Document(text=page.extract_text() or "", source=source, page=number)
                for number, page in enumerate(PdfReader(stream).pages, 1)
            ]
    if suffix == ".docx":
        try:
            from docx import Document as WordDocument  # pyright: ignore[reportMissingImports]
            from docx.table import Table  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise MissingDependencyError(
                f"Loading DOCX requires python-docx. Run {_INSTALL}."
            ) from exc
        blocks: list[str] = []
        for block in WordDocument(str(path)).iter_inner_content():
            if not isinstance(block, Table):
                blocks.append(block.text)
                continue
            lines: list[str] = []
            for row in block.rows:
                # A merged cell is reported once per spanned column.
                cells = [
                    c
                    for i, c in enumerate(row.cells)
                    if i == 0 or c._tc is not row.cells[i - 1]._tc
                ]
                lines.append("\t".join(cell.text.strip() for cell in cells))
            blocks.append("\n".join(lines))
        return [Document(text="\n\n".join(blocks), source=source)]
    _logger.warning("Skipping unsupported knowledge source: %s", path)
    return []


async def load_source(
    source: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> list[Document]:
    """Read one discovered source, keeping one-based page numbers for PDFs."""
    if not _is_url(source):
        return await asyncio.to_thread(_load_file, Path(source).expanduser())
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        try:
            response = await client.get(source)
            response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            raise ConfigError(
                f"Knowledge source unavailable ({reason}). Check the URL and connection."
            ) from None
        text = response.text
        if (
            response.headers.get("content-type", "").partition(";")[0].strip().lower()
            == "text/html"
        ):
            text = visible_text(text)
        return [Document(text=text, source=source)]
