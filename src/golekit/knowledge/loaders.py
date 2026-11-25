"""Discover knowledge sources and load their text into documents."""

import asyncio
import glob
import logging
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path

import httpx

from golekit.errors import ConfigError, MissingDependencyError
from golekit.protocols.types import Document

_logger = logging.getLogger(__name__)
_TEXT = {".txt", ".md", ".markdown"}
_SUPPORTED = _TEXT | {".pdf", ".docx"}
_INSTALL = 'pip install "golekit[rag]"'
_HIDDEN_TAGS = {"head", "script", "style", "template", "title"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


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
                if not (child.is_symlink() and child.is_dir()):
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


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.hidden_stack:
            if tag == self.hidden_stack[-1]:
                self.hidden_stack.append(tag)
            return
        if tag in _HIDDEN_TAGS or any(key == "hidden" for key, _ in attrs):
            if tag not in _VOID_TAGS:
                self.hidden_stack.append(tag)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden_stack:
            if tag == self.hidden_stack[-1]:
                self.hidden_stack.pop()
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_stack:
            self.parts.append(data)

    def text(self) -> str:
        lines = "".join(self.parts).splitlines()
        return "\n".join(" ".join(line.split()) for line in lines if line.strip())


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
        except ImportError as exc:
            raise MissingDependencyError(
                f"Loading DOCX requires python-docx. Run {_INSTALL}."
            ) from exc
        text = "\n\n".join(paragraph.text for paragraph in WordDocument(str(path)).paragraphs)
        return [Document(text=text, source=source)]
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
            parser = _VisibleHTML()
            parser.feed(text)
            parser.close()
            text = parser.text()
        return [Document(text=text, source=source)]
