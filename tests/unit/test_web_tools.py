"""Built-in web_search and web_fetch tools against scripted HTTP responses."""

import httpx
import pytest

import golekit as gole
from golekit.errors import ConfigError
from golekit.testing import FakeModel
from golekit.tools import web

DUCKDUCKGO = """
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&amp;rut=abc">Welcome to <b>Python</b>.org</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F">
      The official home of the <b>Python</b> Programming Language</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title"><a class="result__a" href="https://docs.python.org/3/">3 Documentation</a></h2>
  <a class="result__snippet" href="https://docs.python.org/3/">Tutorials and the library reference</a>
</div>
"""


async def test_duckduckgo_results_are_parsed_and_numbered():
    def serve(request):
        assert request.url.host == "html.duckduckgo.com" and request.url.params["q"] == "python"
        return httpx.Response(200, text=DUCKDUCKGO)

    search = web._search_tool("duckduckgo", 1, httpx.MockTransport(serve))
    assert await search.call(query="python") == (
        "[1] Welcome to Python.org\n    https://www.python.org/\n"
        "    The official home of the Python Programming Language"
    )
    both = await web._search_tool(None, 5, httpx.MockTransport(serve)).call(query="python")
    assert "[2] 3 Documentation\n    https://docs.python.org/3/" in both


async def test_tavily_uses_its_key_and_reports_failures(monkeypatch):
    monkeypatch.setenv("GOLE_SEARCH_PROVIDER", "tavily")
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        gole.Agent(FakeModel([]), tools=["web_search"])
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")

    def serve(request):
        assert request.headers["Authorization"] == "Bearer tvly-secret"
        if b"broken" in request.content:
            return httpx.Response(500, text="tvly-secret")
        items = [{"title": "GoleKit", "url": "https://example.com/g", "content": "Agent kernel"}]
        return httpx.Response(200, json={"results": items})

    search = web._search_tool(None, 5, httpx.MockTransport(serve))
    assert await search.call(query="golekit") == (
        "[1] GoleKit\n    https://example.com/g\n    Agent kernel"
    )
    with pytest.raises(Exception, match=r"Search failed \(HTTP 500\)") as failure:
        await search.call(query="broken")
    assert "tvly-secret" not in str(failure.value)


def test_search_provider_comes_from_config(tmp_path, monkeypatch):
    (tmp_path / "gole.yaml").write_text("tools:\n  web_search:\n    provider: bing\n")
    monkeypatch.setenv("GOLE_CONFIG", str(tmp_path / "gole.yaml"))
    with pytest.raises(ConfigError, match="duckduckgo, tavily"):
        gole.tools.web_search()
    assert gole.tools.web_search(provider="duckduckgo").spec.name == "web_search"


async def test_fetch_extracts_page_text_and_rejects_other_content():
    def serve(request):
        if request.url.path == "/image":
            return httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"\x89PNG")
        if request.url.path == "/plain":
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="<b>raw</b>")
        markup = "<html><head><title>t</title></head><body><h1>News</h1><script>x()</script><p>Body text</p></body></html>"
        return httpx.Response(
            200, headers={"Content-Type": "text/html; charset=utf-8"}, text=markup
        )

    fetch = web._fetch_tool(False, httpx.MockTransport(serve))
    assert await fetch.call(url="https://93.184.216.34/page") == "News\nBody text"
    assert await fetch.call(url="https://93.184.216.34/plain") == "<b>raw</b>"
    with pytest.raises(Exception, match="image/png"):
        await fetch.call(url="https://93.184.216.34/image")
    with pytest.raises(Exception, match="http or https"):
        await fetch.call(url="file:///etc/passwd")


async def test_fetch_refuses_private_addresses_even_after_a_redirect():
    requested: list[str] = []

    def serve(request):
        requested.append(str(request.url))
        if request.url.path == "/hop":
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest"})
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="internal")

    fetch = web._fetch_tool(False, httpx.MockTransport(serve))
    for url in ("http://127.0.0.1:8080/", "http://10.0.0.5/", "http://localhost/", "http://[::1]/"):
        with pytest.raises(Exception, match="blocked"):
            await fetch.call(url=url)
    with pytest.raises(Exception, match="blocked"):
        await fetch.call(url="http://93.184.216.34/hop")
    assert requested == ["http://93.184.216.34/hop"]
    allowed = web._fetch_tool(True, httpx.MockTransport(serve))
    assert await allowed.call(url="http://127.0.0.1:8080/") == "internal"
