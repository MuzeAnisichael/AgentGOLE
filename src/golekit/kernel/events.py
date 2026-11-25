"""Event recording and delivery for one agent run."""

import logging
from typing import Any

from ..protocols import EventSink
from ..protocols.types import Chunk, Citation, Event, Trace

_logger = logging.getLogger("golekit.kernel.events")


class RunEvents:
    """Record each event before notifying the run's consumer."""

    def __init__(self, run_id: str, sink: EventSink) -> None:
        self.run_id = run_id
        self.sink = sink
        self.step: int | None = None
        self.trace = Trace()
        self.citations: list[Citation] = []
        self._cited: set[str] = set()

    async def retrieved(self, query: str, chunks: list[Chunk]) -> None:
        """Record all retrieval events and keep each passage's first citation."""
        for chunk in chunks:
            if chunk.id not in self._cited:
                self._cited.add(chunk.id)
                self.citations.append(
                    Citation(source=chunk.source, page=chunk.page, text=chunk.text)
                )
        await self.send("retrieve", query=query, chunks=chunks)

    async def send(self, kind: str, **fields: Any) -> None:
        """Deliver a timestamped event with the current action number."""
        event = Event(type=kind, run_id=self.run_id, step=self.step, **fields)
        self.trace.events.append(event)
        try:
            await self.sink(event)
        except Exception:
            _logger.exception("Event callback failed for %s", kind)

    async def recover(self, exc: Exception) -> None:
        """Record a recoverable error and continue the run."""
        await self.send("error", exception=exc, recoverable=True)
