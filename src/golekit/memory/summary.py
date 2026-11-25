"""Rolling summaries of session-local conversation history."""

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..errors import ConfigError
from ..kernel.context import estimate_tokens
from ..model import Model
from ..protocols import EmbeddingClient, ModelClient
from ..protocols.types import Message
from .window import copy_turn

_log = logging.getLogger("golekit.memory.summary")


@dataclass
class _Session:
    summary: str = ""
    turns: list[tuple[Message, Message]] = field(default_factory=list)
    summarized: int = 0
    failed_through: int = 0


class Summary:
    """Compress older turns when a session exceeds its estimated token budget."""

    def __init__(
        self, max_tokens: int = 2000, keep_turns: int = 4, model: str | ModelClient | None = None
    ) -> None:
        if max_tokens < 0 or keep_turns < 0:
            raise ConfigError("Choose nonnegative max_tokens and keep_turns for summary memory.")
        self.max_tokens = max_tokens
        self.keep_turns = keep_turns
        self._explicit = Model(model)._binding.client if isinstance(model, str) else model
        self._model = self._explicit
        self._sessions: dict[str, _Session] = {}
        self._lock = Lock()

    def bind(self, model: ModelClient, embedding: EmbeddingClient | None) -> None:
        """Use the agent's model unless a summary-specific model was supplied."""
        self._model = self._explicit if self._explicit is not None else model

    async def load(self, session: str, query: str | None = None) -> list[Message]:
        """Return the summary and recent turns, retaining all originals on failure."""
        with self._lock:
            state = self._sessions.get(session)
            if state is None:
                return []
            start = max(0, len(state.turns) - self.keep_turns)
            turns = state.turns[0 if state.failed_through > state.summarized else start :]
            messages = [Message(role="system", content=state.summary)] if state.summary else []
        messages.extend(m.model_copy(deep=True) for turn in turns for m in turn)
        return messages

    async def save(self, session: str, messages: list[Message]) -> None:
        """Append one turn and roll its older history into the summary if needed."""
        turn = copy_turn(messages)
        with self._lock:
            state = self._sessions.setdefault(session, _Session())
            state.turns.append(turn)
            count = len(state.turns) - self.keep_turns
            if count <= 0:
                return
            size = estimate_tokens(state.summary) + sum(
                estimate_tokens(m.content or "") for pair in state.turns for m in pair
            )
            if size <= self.max_tokens:
                return
            summary, older = state.summary, state.turns[:count]
            through = state.summarized + count
        history = "\n".join(f"{m.role}: {m.content or ''}" for pair in older for m in pair)
        try:
            if self._model is None:
                raise ConfigError("Pass a model to Summary or bind it to an Agent.")
            response = await self._model.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "Update the conversation summary using the older turns below. Retain facts, "
                            "decisions and open questions. Use the conversation's language. "
                            "Return only the concise new summary."
                        ),
                    ),
                    Message(
                        role="user",
                        content=f"Previous summary:\n{summary}\nOlder turns:\n{history}",
                    ),
                ]
            )
            updated = (response.message.content or "").strip()
            if not updated:
                raise ValueError("The model returned an empty summary.")
        except Exception as error:  # noqa: BLE001 - custom model failures retain the original turns.
            with self._lock:
                state.failed_through = max(state.failed_through, through)
            _log.warning(
                "Summary compaction failed (%s). Retaining original turns.", type(error).__name__
            )
            return
        with self._lock:
            # A concurrent save may have already summarized a longer prefix.
            if through > state.summarized:
                del state.turns[: through - state.summarized]
                state.summarized = through
                state.summary = updated

    def export(self, *, session: str = "default") -> dict[str, Any]:
        """Export the summary and every turn still held in its original form."""
        with self._lock:
            state = self._sessions.get(session)
            if state is None:
                return {"summary": "", "messages": []}
            return {
                "summary": state.summary,
                "messages": [m.model_dump(mode="json") for pair in state.turns for m in pair],
            }

    def clear(self, *, session: str = "default") -> None:
        """Forget the session, including any pending compaction result."""
        with self._lock:
            self._sessions.pop(session, None)
