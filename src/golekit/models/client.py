"""Async HTTP client shared by OpenAI-compatible model providers."""

import asyncio
import math
import re
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx

from ..errors import ModelError, UnsupportedToolsError
from ..protocols.types import Capabilities, Message, ModelChunk, ModelResponse, ToolSpec
from .wire import completed_response, stream_response, wire_message

_T = TypeVar("_T")

_TOOL_PARAM = r"(?:tools|tool[_ -]calling|function[_ -]calling|functions)"
_UNSUPPORTED_TOOLS = re.compile(
    rf"(?:does(?:n't| not) support|cannot use|unsupported (?:parameter:? )?|"
    rf"unrecognized request argument supplied:|unknown parameter:?)\s*(?:the\s+)?['`\"]?{_TOOL_PARAM}\b"
    rf"|\b{_TOOL_PARAM}['`\"]?\s+(?:(?:is|are)\s+)?(?:not supported|unsupported|not available)",
    re.IGNORECASE,
)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = -1
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return 0.5 * 2**attempt


class CompatibleClient:
    """OpenAI-compatible chat transport configured for one model."""

    def __init__(
        self,
        provider: str,
        model: str,
        name: str,
        base_url: str,
        api_key: str | None,
        defaults: dict[str, Any],
        timeout: float,
        retries: int,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.name = name
        self._model = model
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._defaults = defaults
        self._timeout = timeout
        self._retries = retries
        self._transport = transport

    def _http(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return httpx.AsyncClient(headers=headers, timeout=self._timeout, transport=self._transport)

    async def probe(self) -> Capabilities:
        """Resolve tool support, asking Ollama for the installed model's metadata."""
        if self._provider != "ollama":
            return Capabilities(tools=self._provider in {"openai", "deepseek", "dashscope"})
        url = self._base_url.removesuffix("/v1") + "/api/show"
        try:
            async with self._http() as http:
                response = await http.post(url, json={"model": self._model})
            response.raise_for_status()
            data = response.json()
            capabilities = data.get("capabilities")
            if not isinstance(capabilities, list):
                return Capabilities()
            return Capabilities(tools="tools" in capabilities, vision="vision" in capabilities)
        except (httpx.HTTPError, ValueError, AttributeError):
            return Capabilities()

    async def check_connection(self) -> None:
        """Check that the configured service can answer for this specific model."""
        await self.generate([Message(role="user", content="Hi")], max_tokens=1)

    def _status_error(self, response: httpx.Response, used_tools: bool) -> ModelError:
        try:
            error = response.json()["error"]
            detail = error["message"] if isinstance(error, dict) else error
        except (ValueError, TypeError, KeyError):
            detail = response.text
        detail = str(detail or response.reason_phrase)
        if self._api_key:
            detail = detail.replace(self._api_key, "[redacted]")
        detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", detail)
        error_type = ModelError
        if used_tools and response.status_code == 400 and _UNSUPPORTED_TOOLS.search(detail):
            error_type = UnsupportedToolsError
        summary = " ".join(detail.split())[:300]
        return error_type(f"{self._provider} HTTP {response.status_code}: {summary}")

    async def _request(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        params: dict[str, Any],
        stream: bool,
    ) -> AsyncIterator[ModelChunk]:
        payload = {
            **self._defaults,
            **params,
            "model": self._model,
            "messages": [wire_message(msg) for msg in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.model_dump()} for tool in tools
            ]
        if stream:
            payload.setdefault("stream_options", {"include_usage": True})

        async def decode(response: httpx.Response) -> AsyncIterator[ModelChunk]:
            if stream:
                async for chunk in stream_response(response, self.name):
                    yield chunk
            else:
                await response.aread()
                yield ModelChunk(response=completed_response(response.json(), self.name))

        async for chunk in self._post(
            "/chat/completions", payload, decode, "chat completion", bool(tools)
        ):
            yield chunk

    async def embedding_data(self, texts: list[str]) -> Any:
        """Request a batch of embeddings with the same retry policy as chat."""

        async def decode(response: httpx.Response) -> AsyncIterator[Any]:
            await response.aread()
            yield response.json()["data"]

        payload = {"model": self._model, "input": texts}
        result = None
        async for data in self._post("/embeddings", payload, decode, "embedding"):
            result = data
        return result

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        decode: Callable[[httpx.Response], AsyncIterator[_T]],
        operation: str,
        used_tools: bool = False,
    ) -> AsyncIterator[_T]:
        url = self._base_url + path
        async with self._http() as http:
            for attempt in range(self._retries + 1):
                delay = _retry_delay(attempt, None)
                emitted = False
                status: int | None = None
                try:
                    async with http.stream("POST", url, json=payload) as response:
                        status = response.status_code
                        if response.is_success:
                            async for item in decode(response):
                                emitted = True
                                yield item
                            return
                        await response.aread()
                        if attempt == self._retries or not (
                            response.status_code == 429 or 500 <= response.status_code < 600
                        ):
                            raise self._status_error(response, used_tools)
                        delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                except httpx.TransportError as exc:
                    # Replaying a partially delivered stream would duplicate visible tokens.
                    if emitted or attempt == self._retries:
                        raise ModelError(
                            f"{self._provider} connection failed ({type(exc).__name__}). Check the endpoint."
                        ) from None
                except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                    if status is None:
                        raise ModelError(
                            f"{self._provider}: invalid {operation} request."
                        ) from None
                    raise ModelError(
                        f"{self._provider} HTTP {status}: invalid {operation} response."
                    ) from None
                await asyncio.sleep(delay)

    async def generate(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> ModelResponse:
        """Return a complete assistant reply and usage."""
        result: ModelResponse | None = None
        async for chunk in self._request(messages, tools, params, False):
            result = chunk.response
        assert result is not None
        return result

    def stream(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> AsyncIterator[ModelChunk]:
        """Stream visible text followed by a complete assistant response."""
        return self._request(messages, tools, params, True)
