"""Select native tool calls or a prompt-based fallback."""

from ..protocols import StepContext
from ..protocols.types import Action, CallModel, CallTool, Finish, RunState
from .react import React


class Auto:
    """Follow the provider's native calls or delegate the run to ReAct."""

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        if ctx.tool_mode == "prompt":
            return await React().step(state, ctx)
        if not state.history or state.history[-1].action != "model" or state.last_response is None:
            return CallModel()
        reply = state.last_response
        return (
            CallTool(calls=reply.tool_calls)
            if reply.tool_calls
            else Finish(output=reply.content or "")
        )
