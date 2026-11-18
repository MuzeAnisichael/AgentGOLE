import logging
import sys
from pathlib import Path

import httpx
import pytest

from agentgole.errors import MissingDependencyError
from agentgole.knowledge.loaders import discover_sources, load_source
from agentgole.protocols.types import Document

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


async def test_discover_files_directories_globs_and_urls(tmp_path, caplog):
    notes = tmp_path / "notes"
    notes.mkdir()
    first = notes / "a.MD"
    first.touch()
    nested = notes / "nested"
    nested.mkdir()
    second = nested / "policy.markdown"
    second.touch()
    unsupported = notes / "image.png"
    unsupported.touch()
    extra = tmp_path / "more.txt"
    extra.touch()
    url = "https://example.test/faq?q=[policy]"

    with caplog.at_level(logging.WARNING):
        sources = await discover_sources([notes, str(tmp_path / "*.txt"), url, first, url])

    assert sources == [str(first), str(second), str(extra), url]
    assert str(unsupported) in caplog.text


async def test_text_and_markdown_decode_utf8_with_replacement(tmp_path):
    for suffix in (".txt", ".md", ".markdown"):
        source = tmp_path / f"notes{suffix}"
        source.write_bytes("报销".encode() + b"\xff\n")
        assert await load_source(str(source)) == [Document(text="报销�\n", source=str(source))]


async def test_html_url_extracts_visible_text_and_keeps_source():
    url = "https://example.test/policy"
    markup = """
    <head><title>Metadata</title></head>
    <h1>Travel &amp; expenses</h1><script>var ignored = true;</script>
    <style>p {color: red}</style><div hidden><div>Internal</div>Draft</div>
    <p>Re<strong>ceipts</strong> are required.</p>
    """

    def serve(request):
        assert request.method == "GET" and str(request.url) == url
        return httpx.Response(
            200, headers={"Content-Type": "text/html; charset=utf-8"}, text=markup
        )

    docs = await load_source(url, transport=httpx.MockTransport(serve))
    assert docs == [Document(source=url, text="Travel & expenses\nReceipts are required.")]


async def test_plain_text_url_preserves_literal_markup():
    url = "http://example.test/manual.txt"
    text = "Print <help> to begin.\n"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=text))
    assert await load_source(url, transport=transport) == [Document(text=text, source=url)]


async def test_pdf_preserves_page_numbers():
    pytest.importorskip("pypdf")
    source = str(FIXTURES / "two-pages.pdf")
    docs = await load_source(source)
    assert [(doc.page, doc.text.strip(), doc.source) for doc in docs] == [
        (1, "First page", source),
        (2, "Second page", source),
    ]


async def test_docx_preserves_paragraph_order():
    pytest.importorskip("docx")
    source = str(FIXTURES / "paragraphs.docx")
    docs = await load_source(source)
    assert docs == [Document(text="First paragraph\n\nSecond paragraph", source=source)]


@pytest.mark.parametrize(
    ("module", "name"), [("pypdf", "two-pages.pdf"), ("docx", "paragraphs.docx")]
)
async def test_missing_format_dependency_provides_install_command(module, name, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MissingDependencyError) as exc:
        await load_source(str(FIXTURES / name))
    assert 'pip install "agentgole[rag]"' in str(exc.value)


async def test_unavailable_url_reports_actionable_error_without_query_credentials():
    from agentgole.errors import ConfigError

    with pytest.raises(ConfigError, match="HTTP 404") as caught:
        await load_source(
            "https://example.test/missing?key=secret",
            transport=httpx.MockTransport(lambda _: httpx.Response(404)),
        )
    assert "secret" not in str(caught.value) and "URL" in str(caught.value)
