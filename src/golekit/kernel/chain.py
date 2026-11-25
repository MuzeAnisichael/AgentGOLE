"""Ordered interception of tool calls.

Each link receives the call and a function that hands it to the next link. A link may
pass the call on, change the returned value, or answer with an error.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from ..errors import ToolError
from ..protocols import StepContext
from ..protocols.types import ToolCall
from .events import RunEvents


class Refusal(Exception):
    """A link declined the call; the message is returned to the model verbatim."""


Proceed = Callable[[], Awaitable[Any]]
Link = Callable[[ToolCall, StepContext, RunEvents, Proceed], Awaitable[Any]]


async def _validate(call: ToolCall, ctx: StepContext, events: RunEvents, proceed: Proceed) -> Any:
    if call.invalid_arguments:
        raise Refusal("invalid arguments JSON")
    tool = ctx.tools.get(call.name)
    if tool is None:
        raise ToolError(f"Unknown tool '{call.name}'. Choose a registered tool.")
    check = getattr(tool, "check_arguments", None)
    if callable(check):
        check(call.arguments)
    return await proceed()


async def _approve(call: ToolCall, ctx: StepContext, events: RunEvents, proceed: Proceed) -> Any:
    if not getattr(ctx.tools[call.name], "confirm", False):
        return await proceed()
    await events.send("approval_required", tool=call.name, args=call.arguments)
    if ctx.approver is None:
        raise Refusal(
            "This tool requires confirmation and the agent has no approval method. "
            "Pass approval= to the Agent."
        )
    try:
        approved = await ctx.approver.approve(call.name, call.arguments)
    except Exception as exc:  # noqa: BLE001 - a failing approver must not run the tool
        await ctx.report_error(exc)
        approved = False
    if not approved:
        raise Refusal("Rejected by the user.")
    return await proceed()


async def _execute(call: ToolCall, ctx: StepContext, events: RunEvents, proceed: Proceed) -> Any:
    return await ctx.tools[call.name].call(**call.arguments)


_CHAIN: tuple[Link, ...] = (_validate, _approve, _execute)


async def run_tool(call: ToolCall, ctx: StepContext, events: RunEvents) -> Any:
    """Send one call through every link and return the tool's value."""

    async def advance(index: int) -> Any:
        return await _CHAIN[index](call, ctx, events, lambda: advance(index + 1))

    return await advance(0)
