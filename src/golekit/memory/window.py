"""Bounded, process-local conversation memory."""

from collections import defaultdict, deque
from threading import Lock
from typing import Any

from ..protocols.types import Message


def copy_turn(messages: list[Message]) -> tuple[Message, Message]:
    """Validate and snapshot a complete conversation turn."""
    if len(messages) != 2 or messages[0].role != "user" or messages[1].role != "assistant":
        raise ValueError("A memory turn must contain a user message and its final assistant reply")
    return messages[0].model_copy(deep=True), messages[1].model_copy(deep=True)


class Window:
    """Keep the most recent complete user/assistant turns for each session."""

    def __init__(self, turns: int = 20) -> None:
        if turns < 0:
            raise ValueError("Window turns must be nonnegative")
        self.turns = turns
        self._sessions: defaultdict[str, deque[tuple[Message, Message]]] = defaultdict(
            lambda: deque(maxlen=turns)
        )
        self._lock = Lock()

    async def load(self, session: str, query: str | None = None) -> list[Message]:
        """Return independent copies of the retained conversation."""
        with self._lock:
            turns = tuple(self._sessions.get(session, ()))
        return [message.model_copy(deep=True) for turn in turns for message in turn]

    async def save(self, session: str, messages: list[Message]) -> None:
        """Append a complete turn to this session's window."""
        pair = copy_turn(messages)
        if self.turns:
            with self._lock:
                self._sessions[session].append(pair)

    def export(self, *, session: str = "default") -> dict[str, Any]:
        """Export this session's retained messages as JSON-compatible data."""
        with self._lock:
            turns = tuple(self._sessions.get(session, ()))
        return {"messages": [m.model_dump(mode="json") for turn in turns for m in turn]}

    def clear(self, *, session: str = "default") -> None:
        """Remove the retained turns for one session."""
        with self._lock:
            self._sessions.pop(session, None)
