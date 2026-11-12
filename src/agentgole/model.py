"""Public model facade."""

from typing import Any

from .protocols.types import Capabilities, Message, ModelResponse, ToolMode


class Model:
    """Configure an OpenAI-compatible provider or a model alias."""

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
    ) -> None:
        raise NotImplementedError

    def generate(self, prompt_or_messages: str | list[Message], **params: Any) -> ModelResponse:
        """Generate a reply from a prompt or conversation."""
        raise NotImplementedError

    async def agenerate(
        self, prompt_or_messages: str | list[Message], **params: Any
    ) -> ModelResponse:
        """Generate a reply asynchronously."""
        raise NotImplementedError

    @property
    def capabilities(self) -> Capabilities:
        """Probe the provider once and cache its capabilities."""
        raise NotImplementedError
