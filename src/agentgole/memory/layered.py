"""Compose recent conversational context with durable facts."""

import asyncio
from typing import Any

from ..protocols import EmbeddingClient, ModelClient, Tool
from ..protocols.types import Message
from .longterm import LongTerm
from .summary import Summary
from .window import Window


class Layered:
    """Load durable facts before recent history and save each turn to both layers."""

    def __init__(self, short: Window | Summary, long: LongTerm) -> None:
        self.short = short
        self.long = long

    def bind(self, model: ModelClient, embedding: EmbeddingClient | None) -> None:
        """Share the agent's configured models with both memory layers."""
        if bind := getattr(self.short, "bind", None):
            bind(model, embedding)
        self.long.bind(model, embedding)

    async def load(self, session: str, query: str | None = None) -> list[Message]:
        """Retrieve facts before the short-term messages."""
        facts = await self.long.load(session, query)
        return facts + await self.short.load(session, query)

    async def save(self, session: str, messages: list[Message]) -> None:
        """Write the same complete turn to both memories before returning."""
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.short.save(session, messages))
            tasks.create_task(self.long.save(session, messages))

    def tools(self, session: str = "default") -> list[Tool]:
        """Expose the long-term layer's session-bound memory tools."""
        return self.long.tools(session)

    def add(self, text: str, *, session: str = "default") -> None:
        """Add a durable fact for this session."""
        self.long.add(text, session=session)

    def search(self, query: str, *, session: str = "default", k: int = 5) -> list[str]:
        """Search the durable facts in the current session."""
        return self.long.search(query, session=session, k=k)

    def export(self, *, session: str = "default") -> dict[str, Any]:
        """Export both layers as independent JSON-compatible data."""
        return {
            "short": self.short.export(session=session),
            "long": self.long.export(session=session),
        }

    def clear(self, *, session: str = "default") -> None:
        """Clear both layers for a single session."""
        self.long.clear(session=session)
        self.short.clear(session=session)
