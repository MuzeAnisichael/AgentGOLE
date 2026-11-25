"""Built-in tools that search the web and fetch pages."""

import asyncio
import ipaddress
import os
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from .._html import visible_text
from ..config import load_config
from ..errors import ConfigError, ToolError
from ..protocols import Tool
from .function import tool

_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; GoleKit)"}
_PROVIDERS = ("duckduckgo", "tavily")
_FETCH_TIMEOUT = 20
_MAX_REDIRECTS = 5
_MAX_BYTES = 2 * 1024 * 1024


class _DuckDuckGo(HTMLParser):
    """Collect result titles, links and snippets from the HTML results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if "result__a" in classes:
            self.results.append({"title": "", "url": _target(values.get("href") or ""), "text": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "text"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def _target(href: str) -> str:
    # Result links pass through a redirect that carries the real address in "uddg".
    link = urlsplit(href)
    wrapped = parse_qs(link.query).get("uddg")
    if wrapped:
        return wrapped[0]
    return "https:" + href if href.startswith("//") else href


async def _duckduckgo(http: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
    response = await http.get("https://html.duckduckgo.com/html/", params={"q": query})
    response.raise_for_status()
    parser = _DuckDuckGo()
    parser.feed(response.text)
    parser.close()
    return parser.results


async def _tavily(
    http: httpx.AsyncClient, query: str, key: str, limit: int
) -> list[dict[str, str]]:
    response = await http.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": limit},
        headers={"Authorization": f"Bearer {key}"},
    )
    response.raise_for_status()
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "text": str(item.get("content") or ""),
        }
        for item in response.json()["results"]
    ]


def _search_tool(
    provider: str | None, max_results: int, transport: httpx.AsyncBaseTransport | None
) -> Tool:
    settings = load_config().get("tools", {}).get("web_search", {})
    name = provider or os.environ.get("GOLE_SEARCH_PROVIDER") or settings.get("provider")
    name = name or "duckduckgo"
    if name not in _PROVIDERS:
        raise ConfigError(f"Choose a web_search provider from: {', '.join(_PROVIDERS)}.")
    key = os.environ.get("TAVILY_API_KEY", "")
    if name == "tavily" and not key:
        raise ConfigError("Set TAVILY_API_KEY to use the tavily search provider.")
    if max_results < 1:
        raise ConfigError("Choose max_results of one or more.")

    async def web_search(query: str) -> str:
        """Search the web and return numbered results with title, link and snippet.

        Args:
            query: What to search for.
        """
        try:
            async with httpx.AsyncClient(
                transport=transport, headers=_AGENT, timeout=_FETCH_TIMEOUT, follow_redirects=True
            ) as http:
                if name == "tavily":
                    found = await _tavily(http, query, key, max_results)
                else:
                    found = await _duckduckgo(http, query)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            raise ToolError(f"Search failed ({reason}). Try again or rephrase.") from None
        found = [item for item in found if item["url"]][:max_results]
        if not found:
            raise ToolError("No search results were found. Try different keywords.")
        return "\n".join(
            f"[{index}] {' '.join(item['title'].split())}\n    {item['url']}\n"
            f"    {' '.join(item['text'].split())}"
            for index, item in enumerate(found, 1)
        )

    return tool(web_search, timeout=_FETCH_TIMEOUT + 10)


def web_search(provider: str | None = None, max_results: int = 5) -> Tool:
    """Create the ``web_search`` tool.

    ``provider`` is ``"duckduckgo"`` (no key needed) or ``"tavily"`` (reads TAVILY_API_KEY).
    When omitted it comes from GOLE_SEARCH_PROVIDER, then ``tools.web_search.provider`` in
    gole.yaml, then defaults to duckduckgo.
    """
    return _search_tool(provider, max_results, None)


async def _check_public(url: httpx.URL) -> None:
    host = url.host
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, None)
        except OSError:
            raise ToolError(f"Cannot resolve host {host!r}.") from None
        addresses = [ipaddress.ip_address(str(info[4][0]).partition("%")[0]) for info in infos]
    for address in addresses:
        mapped = getattr(address, "ipv4_mapped", None) or address
        if not mapped.is_global or mapped.is_multicast:
            raise ToolError(
                f"Refusing to fetch {host!r}: loopback, private and link-local addresses "
                "are blocked."
            )


def _fetch_tool(allow_private: bool, transport: httpx.AsyncBaseTransport | None) -> Tool:
    async def web_fetch(url: str) -> str:
        """Fetch a web page and return its readable text.

        Args:
            url: An http or https address.
        """
        try:
            target = httpx.URL(url)
        except (httpx.InvalidURL, TypeError):
            raise ToolError("Give a valid http or https URL.") from None
        try:
            async with httpx.AsyncClient(
                transport=transport, headers=_AGENT, timeout=_FETCH_TIMEOUT
            ) as http:
                for _ in range(_MAX_REDIRECTS + 1):
                    if target.scheme not in ("http", "https") or not target.host:
                        raise ToolError("Give a valid http or https URL.")
                    if not allow_private:
                        await _check_public(target)
                    async with http.stream("GET", target) as response:
                        if response.is_redirect and "location" in response.headers:
                            target = httpx.URL(urljoin(str(target), response.headers["location"]))
                            continue
                        response.raise_for_status()
                        kind = response.headers.get("content-type", "").partition(";")[0]
                        kind = kind.strip().lower()
                        if not kind.startswith("text/") and kind != "application/xhtml+xml":
                            raise ToolError(
                                f"Cannot read content of type {kind or 'unknown'}; only text."
                            )
                        body = bytearray()
                        async for part in response.aiter_bytes():
                            body += part
                            if len(body) >= _MAX_BYTES:
                                break
                        text = bytes(body[:_MAX_BYTES]).decode(
                            response.charset_encoding or "utf-8", errors="replace"
                        )
                        if kind in ("text/html", "application/xhtml+xml"):
                            text = visible_text(text)
                        return text
                raise ToolError(f"Stopped after {_MAX_REDIRECTS} redirects.")
        except (httpx.HTTPError, httpx.InvalidURL, LookupError) as exc:
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            raise ToolError(f"Fetch failed ({reason}).") from None

    return tool(web_fetch, timeout=_FETCH_TIMEOUT + 10)


def web_fetch(allow_private: bool = False) -> Tool:
    """Create the ``web_fetch`` tool, which reads the text of an http(s) page.

    Loopback, private and link-local addresses are refused, including after redirects,
    unless ``allow_private=True``.
    """
    return _fetch_tool(allow_private, None)
