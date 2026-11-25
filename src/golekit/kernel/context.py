"""Conversation assembly for native and prompt-based tools."""

import re
from collections.abc import Sequence

from ..errors import ConfigError
from ..protocols import Retriever, StepContext, Tool
from ..protocols.types import Message, RunState, ToolResult
from ..tools import tool
from .events import RunEvents


def initial_state(input: str, instructions: str | None, memory: list[Message]) -> RunState:
    """Build a fresh conversation from instructions, memory, and user input."""
    messages = [Message(role="system", content=instructions)] if instructions else []
    messages.extend(memory)
    messages.append(Message(role="user", content=input))
    # Paradigms that extend the instructions need to tell them apart from memory messages.
    return RunState(input=input, messages=messages, scratch={"instructions": bool(instructions)})


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


_NO_MATCH = "未找到相关内容。"


async def _search(retriever: Retriever, query: str, events: RunEvents, start: int = 1) -> str:
    chunks = await retriever.search(query, k=5)
    await events.retrieved(query, chunks)
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start):
        location = chunk.source if chunk.page is None else f"{chunk.source} p.{chunk.page}"
        lines.append(f"[{index}] ({location}) {chunk.text}")
    return "\n".join(lines)


def _search_tool(retriever: Retriever, events: RunEvents) -> Tool:
    @tool(name=f"search_{retriever.name}", description=retriever.description)
    async def search(query: str) -> str:
        return await _search(retriever, query, events) or _NO_MATCH

    return search


async def prepare_knowledge(
    state: RunState,
    ctx: StepContext,
    events: RunEvents,
    retrievers: Sequence[Retriever],
    mode: str,
    *,
    after_downgrade: bool = False,
) -> None:
    """Configure retrieval for this run and its resolved tool mode."""
    if not retrievers or (after_downgrade and mode != "auto"):
        return
    if mode == "tool" or (mode == "auto" and ctx.tool_mode == "native"):
        for retriever in retrievers:
            item = _search_tool(retriever, events)
            if item.spec.name in ctx.tools:
                raise ConfigError(
                    f"Tool {item.spec.name!r} already exists. Set a distinct knowledge name."
                )
            ctx.tools[item.spec.name] = item
        return
    sections: list[str] = []
    start = 1
    for retriever in retrievers:
        if after_downgrade:
            ctx.tools.pop(f"search_{retriever.name}", None)
        section = await _search(retriever, state.input, events, start)
        if section:
            sections.append(section)
            start += len(events.trace.events[-1].chunks)
    if not sections:
        return
    references = "参考资料：\n" + "\n".join(sections)
    # Keep the instructions and memory preamble first; some chat templates require that.
    index = next(
        (i for i, m in enumerate(state.messages) if m.role != "system"), len(state.messages)
    )
    state.messages.insert(index, Message(role="system", content=references))
