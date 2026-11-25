"""Draft an answer, have a critic review it, and revise until it passes."""

from typing import Any

from pydantic import BaseModel, Field

from .._json import OutputSchema
from ..errors import ConfigError
from ..protocols import ModelClient, StepContext
from ..protocols.types import Action, CallModel, Finish, Message, RunState
from ._shared import ModelRef, client_of, hidden, leading_system, model_source, restart
from .auto import Auto

_CRITERIA = "回答准确、完整、切题"
_REVIEW_PROMPT = """用户的问题：{task}

待评审的回答：
{draft}

评审标准：{criteria}

请判断这份回答是否符合评审标准。只输出一个 JSON 对象：
{{"pass": true 或 false, "feedback": "不通过时写出具体的修改意见，通过时留空"}}"""
_REVISE_PROMPT = "评审意见：{feedback}\n\n请根据评审意见修订你的回答，只输出修订后的完整回答。"


class _Verdict(BaseModel):
    passed: bool = Field(alias="pass")
    feedback: str = ""


_VERDICT = OutputSchema(_Verdict)


class Reflect:
    """Review the draft answer with a critic and revise it up to ``rounds`` times.

    ``critic`` accepts a model spec, an alias, a Model or a ModelClient and defaults to
    the Agent's model. ``criteria`` describes what the critic checks.

    Example:
        gole.Agent(paradigm=gole.paradigms.Reflect(critic="smart", rounds=1))
    """

    def __init__(
        self, critic: ModelRef = None, rounds: int = 2, criteria: str | None = None
    ) -> None:
        if rounds < 0:
            raise ConfigError("Choose rounds of zero or more.")
        self._critic = client_of(model_source(critic))
        self.rounds = rounds
        self.criteria = criteria or _CRITERIA
        self._agent_model: ModelClient | None = None
        self._auto = Auto()

    def bind(self, model: ModelClient) -> None:
        """Use the Agent's model for reviews unless a critic was given."""
        self._agent_model = model

    def _review(self, state: RunState, run: dict[str, Any]) -> CallModel:
        run["phase"] = "review"
        request = _REVIEW_PROMPT.format(
            task=state.input, draft=run["draft"], criteria=self.criteria
        )
        return CallModel(
            messages=[*leading_system(run["base"]), Message(role="user", content=request)],
            use_tools=False,
            visible=False,
            model=self._critic or self._agent_model,
            response_schema=_VERDICT.schema,
        )

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        run: dict[str, Any] | None = state.scratch.get("_reflect")
        if run is None:
            base = [message.model_copy(deep=True) for message in state.messages]
            run = state.scratch["_reflect"] = {"phase": "draft", "base": base, "revisions": 0}
        reply = (state.last_response.content or "") if state.last_response else ""

        if run["phase"] == "draft":
            action = await self._auto.step(state, ctx)
            if not isinstance(action, Finish):
                return action if self.rounds == 0 else hidden(action)
            if self.rounds == 0:
                return action
            run["draft"] = action.output
            return self._review(state, run)

        if run["phase"] == "review":
            try:
                verdict = _VERDICT.validate(reply)
            except (ValueError, SyntaxError, TypeError) as exc:
                await ctx.report_error(exc)
                return Finish(output=run["draft"])
            if verdict.passed:
                return Finish(output=run["draft"])
            run["revisions"] += 1
            run["phase"] = "revise"
            # The revision continues the original conversation so output corrections can follow.
            restart(
                state,
                [
                    *run["base"],
                    Message(role="assistant", content=run["draft"]),
                    Message(role="user", content=_REVISE_PROMPT.format(feedback=verdict.feedback)),
                ],
            )
            return CallModel(use_tools=False, visible=run["revisions"] == self.rounds)

        run["draft"] = reply
        if run["revisions"] == self.rounds:
            return Finish(output=reply)
        return self._review(state, run)
