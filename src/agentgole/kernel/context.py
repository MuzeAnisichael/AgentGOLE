"""Conversation assembly for native and prompt-based tools."""

from ..protocols.types import Message, RunState, ToolResult


def initial_state(input: str, instructions: str | None, memory: list[Message]) -> RunState:
    """Build a fresh conversation from instructions, memory, and user input."""
    messages = [Message(role="system", content=instructions)] if instructions else []
    messages.extend(memory)
    messages.append(Message(role="user", content=input))
    return RunState(input=input, messages=messages)


def append_results(state: RunState, results: list[ToolResult], as_observation: bool) -> None:
    """Feed a batch of tool results back into the conversation."""
    if as_observation:
        observations = "\n".join(f"{result.name}: {result.output}" for result in results)
        state.messages.append(Message(role="user", content=f"Observation: {observations}"))
    else:
        state.messages.extend(
            Message(
                role="tool", name=result.name, tool_call_id=result.call_id, content=result.output
            )
            for result in results
        )
