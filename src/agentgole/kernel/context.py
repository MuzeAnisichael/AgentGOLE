"""Conversation assembly for native and prompt-based tools."""

import re

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


_CJK = re.compile(
    r"[\u1100-\u11ff\u3040-\u30ff\u3100-\u312f\u31a0-\u31bf"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
    r"\U00020000-\U000323af]"
)


def estimate_tokens(text: str) -> int:
    """Count CJK characters individually and other characters in groups of four."""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4
