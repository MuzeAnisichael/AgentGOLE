"""Helpers shared by paradigms that orchestrate several model calls."""

from typing import Any

from ..protocols import ModelClient
from ..protocols.types import Action, CallModel, Message, RunState

# A model spec or alias, a Model facade, or a ModelClient.
ModelRef = Any


def model_source(model: ModelRef) -> Any:
    """Turn a spec or alias into a Model facade; other values pass through."""
    if isinstance(model, str):
        from ..model import Model

        return Model(model)
    return model


def client_of(source: Any) -> ModelClient | None:
    """Return the client behind a Model facade, or the client itself."""
    binding = getattr(source, "_binding", None)
    return source if binding is None else binding.client


def leading_system(messages: list[Message]) -> list[Message]:
    """Return the system messages that open a conversation."""
    lead: list[Message] = []
    for message in messages:
        if message.role != "system":
            break
        lead.append(message)
    return lead


def with_request(base: list[Message], request: str) -> list[Message]:
    """Copy a conversation, replacing its final user message with a new request."""
    return [*base[:-1], Message(role="user", content=request)]


def restart(state: RunState, messages: list[Message]) -> None:
    """Replace the working conversation and forget the reply that preceded it."""
    state.messages[:] = messages
    state.last_response = None
    state.scratch["react_retries"] = 0


def hidden(action: Action) -> Action:
    """Mark a delegated model call as intermediate."""
    return action.model_copy(update={"visible": False}) if isinstance(action, CallModel) else action


def numbered(items: list[str]) -> str:
    """Render steps as a numbered list."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
