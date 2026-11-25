"""Answer with a single model call."""

from ..protocols import StepContext
from ..protocols.types import Action, CallModel, Finish, RunState


class Direct:
    """Send the conversation to the model once, without tools, and return its reply.

    Example:
        gole.Agent("ollama/qwen3:8b", paradigm="direct").run("Summarize this note.")
    """

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        if not state.history or state.history[-1].action != "model" or state.last_response is None:
            return CallModel(use_tools=False)
        return Finish(output=state.last_response.content or "")
