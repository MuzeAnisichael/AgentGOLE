"""Public model facade."""

import asyncio
from concurrent.futures import Future
from threading import Lock
from typing import Any

from ._sync import run_sync
from .config import load_config
from .errors import ConfigError
from .models import create_client
from .protocols import ModelClient
from .protocols.types import Capabilities, Message, ModelResponse, ResolvedMode, ToolMode


def _check_mode(mode: ToolMode | None) -> None:
    if mode not in (None, "auto", "native", "prompt"):
        raise ConfigError("Choose tool_mode 'auto', 'native' or 'prompt'.")


class _Binding:
    """Share capability discovery and server downgrades across all users of a Model."""

    def __init__(self, client: ModelClient, mode: ToolMode):
        self.client = client
        self.mode: ToolMode = mode
        self.rejected_tools = False
        self.rejected_schema = False
        self._probe: Future[Capabilities | None] | None = None
        self._lock = Lock()

    async def capabilities(self) -> Capabilities:
        while True:
            with self._lock:
                pending = self._probe
                owner = pending is None
                if pending is None:
                    pending = self._probe = Future()
            if owner:
                try:
                    pending.set_result(await self.client.probe())
                except asyncio.CancelledError:
                    with self._lock:
                        self._probe = None
                    # Another run can take ownership with its own time budget.
                    pending.set_result(None)
                    raise
                except BaseException as error:
                    with self._lock:
                        self._probe = None
                    pending.set_exception(error)
                    raise
            caps = await asyncio.shield(asyncio.wrap_future(pending))
            if caps is not None:
                rejected: dict[str, bool] = {}
                if self.rejected_tools:
                    rejected["tools"] = False
                if self.rejected_schema:
                    rejected["json_schema"] = False
                return caps.model_copy(update=rejected)

    def downgrade(self, feature: str = "tools") -> None:
        """Remember that the server rejected native tools or JSON Schema responses."""
        if feature == "json_schema":
            self.rejected_schema = True
        else:
            self.rejected_tools = True

    async def resolve_mode(self, override: ToolMode | None) -> ResolvedMode:
        if self.rejected_tools:
            return "prompt"
        mode = override or self.mode
        if mode == "auto":
            return "native" if (await self.capabilities()).tools else "prompt"
        return mode


class Model:
    """Configure an OpenAI-compatible provider or a model alias.

    ``constrained`` controls JSON Schema constrained decoding: ``None`` follows the
    provider default, ``True`` or ``False`` forces it on or off. ``context`` is the context
    length in tokens that the server really applies to this model; GoleKit only uses it to
    warn before a prompt is truncated and never changes the server's setting.
    """

    def __init__(
        self,
        spec: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        tool_mode: ToolMode = "auto",
        constrained: bool | None = None,
        context: int | None = None,
    ) -> None:
        _check_mode(tool_mode)
        client = create_client(
            spec,
            config=load_config(),
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            constrained=constrained,
            context=context,
        )
        self._binding = _Binding(client, tool_mode)

    def generate(self, prompt_or_messages: str | list[Message], **params: Any) -> ModelResponse:
        """Generate a reply from a prompt or conversation."""
        return run_sync(self.agenerate(prompt_or_messages, **params))

    async def agenerate(
        self, prompt_or_messages: str | list[Message], **params: Any
    ) -> ModelResponse:
        """Generate a reply asynchronously."""
        messages = (
            [Message(role="user", content=prompt_or_messages)]
            if isinstance(prompt_or_messages, str)
            else [Message.model_validate(message) for message in prompt_or_messages]
        )
        return await self._binding.client.generate(messages, **params)

    @property
    def capabilities(self) -> Capabilities:
        """Probe the provider once and cache its capabilities."""
        return run_sync(self._binding.capabilities())
