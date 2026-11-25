"""Plan a task, execute each step, check the outcome and write the answer."""

from typing import Any

from pydantic import BaseModel

from .._json import OutputSchema
from ..errors import ConfigError
from ..protocols import ModelClient, StepContext
from ..protocols.types import Action, CallModel, Finish, Message, RunState
from ._shared import (
    ModelRef,
    client_of,
    hidden,
    leading_system,
    model_source,
    numbered,
    restart,
    with_request,
)
from .auto import Auto

_MAX_STEPS = 8
_CARRY = 2000
_PLAN = OutputSchema(list[str])

_PLAN_PROMPT = """任务：{task}
{tools}
请把完成这个任务需要做的事拆成按顺序执行的步骤，最多 8 步，每步一句话，说清要做什么。简单的任务只需要一步。
只输出一个 JSON 字符串数组，例如 ["第一步", "第二步"]，不要输出其他内容。"""
_STEP_PROMPT = """总任务：{task}

已完成的步骤：
{done}

当前步骤：{step}

只完成当前步骤，并给出这一步的结果。"""
_CHECK_PROMPT = """任务：{task}

已执行的步骤和结果：
{done}

判断这些结果是否已经足够回答任务。只输出一个 JSON 对象：
足够时输出 {{"done": true, "plan": []}}；
不够时输出 {{"done": false, "plan": ["还需要做的步骤"]}}，最多 8 步。"""
_ANSWER_PROMPT = """任务：{task}

各步骤的结果：
{done}

请根据以上结果写出给用户的最终回答，使用与任务相同的语言，不要提及步骤本身。"""


class _Check(BaseModel):
    done: bool
    plan: list[str] = []


_CHECK = OutputSchema(_Check)


def _done(run: dict[str, Any]) -> str:
    lines = [
        f"{index}. {step}\n   结果：{result[:_CARRY]}"
        for index, (step, result) in enumerate(run["done"], 1)
    ]
    return "\n".join(lines) or "（无）"


def _steps(value: list[str]) -> list[str]:
    return [step.strip() for step in value if step.strip()][:_MAX_STEPS]


class PlanExecute:
    """Plan with one model, execute each step with tools, then synthesize the answer.

    ``planner`` writes the plan, checks the results and writes the final answer;
    ``executor`` runs the steps and becomes the run's primary model. Each accepts a
    model spec, an alias, a Model or a ModelClient, and defaults to the Agent's model.

    Example:
        gole.Agent(tools=["calculator"], paradigm=gole.paradigms.PlanExecute(planner="smart"))
    """

    def __init__(
        self, planner: ModelRef = None, executor: ModelRef = None, max_replans: int = 2
    ) -> None:
        if max_replans < 0:
            raise ConfigError("Choose max_replans of zero or more.")
        self._planner = client_of(model_source(planner))
        self.primary_model = model_source(executor)
        self.max_replans = max_replans
        self._agent_model: ModelClient | None = None
        self._auto = Auto()

    def bind(self, model: ModelClient) -> None:
        """Use the Agent's model wherever no specific model was given."""
        self._agent_model = model

    def _ask(self, messages: list[Message], schema: OutputSchema) -> CallModel:
        return CallModel(
            messages=messages,
            use_tools=False,
            visible=False,
            model=self._planner or self._agent_model,
            response_schema=schema.schema,
        )

    def _begin(self, state: RunState, run: dict[str, Any]) -> None:
        step = state.scratch["plan"][run["index"]]
        request = _STEP_PROMPT.format(task=state.input, done=_done(run), step=step)
        restart(state, [*leading_system(state.messages), Message(role="user", content=request)])
        run["phase"] = "execute"

    def _adopt(self, state: RunState, run: dict[str, Any], plan: list[str]) -> None:
        state.scratch["plan"] = plan
        state.scratch["thought"] = numbered(plan)
        run["index"] = 0
        self._begin(state, run)

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        run: dict[str, Any] | None = state.scratch.get("_plan")
        if run is None:
            base = [message.model_copy(deep=True) for message in state.messages]
            tools = "".join(
                f"\n- {tool.spec.name}：{tool.spec.description}" for tool in ctx.tools.values()
            )
            request = _PLAN_PROMPT.format(
                task=state.input, tools=f"\n可用工具：{tools}\n" if tools else ""
            )
            run = state.scratch["_plan"] = {
                "phase": "plan",
                "base": base,
                "ask": with_request(base, request),
                "retried": False,
                "done": [],
                "replans": 0,
                "index": 0,
            }
            return self._ask(run["ask"], _PLAN)
        reply = (state.last_response.content or "") if state.last_response else ""

        if run["phase"] == "plan":
            try:
                plan = _steps(_PLAN.validate(reply))
                if not plan:
                    raise ValueError("The plan has no steps")
            except (ValueError, SyntaxError, TypeError) as exc:
                if not run["retried"]:
                    run["retried"] = True
                    retry = [
                        *run["ask"],
                        Message(role="assistant", content=reply),
                        Message(
                            role="user",
                            content=f"无法解析这个计划：{exc}。只输出一个 JSON 字符串数组。",
                        ),
                    ]
                    return self._ask(retry, _PLAN)
                await ctx.report_error(exc)
                plan = [state.input]
            self._adopt(state, run, plan)

        elif run["phase"] == "check":
            try:
                verdict = _CHECK.validate(reply)
                plan = [] if verdict.done else _steps(verdict.plan)
            except (ValueError, SyntaxError, TypeError):
                plan = []
            if plan:
                run["replans"] += 1
                self._adopt(state, run, plan)
            else:
                return self._answer(state, run)

        elif run["phase"] == "answer":
            return Finish(output=reply)

        action = await self._auto.step(state, ctx)
        if not isinstance(action, Finish):
            return hidden(action)
        plan = state.scratch["plan"]
        run["done"].append((plan[run["index"]], action.output))
        run["index"] += 1
        if run["index"] < len(plan):
            self._begin(state, run)
            return hidden(await self._auto.step(state, ctx))
        if run["replans"] < self.max_replans:
            run["phase"] = "check"
            request = _CHECK_PROMPT.format(task=state.input, done=_done(run))
            return self._ask(with_request(run["base"], request), _CHECK)
        return self._answer(state, run)

    def _answer(self, state: RunState, run: dict[str, Any]) -> CallModel:
        run["phase"] = "answer"
        request = _ANSWER_PROMPT.format(task=state.input, done=_done(run))
        # The answer continues the working conversation so output corrections can follow it.
        restart(state, with_request(run["base"], request))
        return CallModel(use_tools=False, model=self._planner or self._agent_model)
