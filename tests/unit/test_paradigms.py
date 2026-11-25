from unittest.mock import AsyncMock

from pydantic import BaseModel, Field

from golekit.paradigms import Auto, React
from golekit.protocols import StepContext
from golekit.protocols.types import (
    CallModel,
    CallTool,
    Finish,
    Message,
    RunState,
    Step,
    ToolCall,
    ToolSpec,
)
from golekit.tools import tool


class ExternalTool:
    spec = ToolSpec(
        name="find",
        description="Find a record.",
        parameters={
            "type": "object",
            "required": ["ids"],
            "additionalProperties": False,
            "properties": {"ids": {"type": "array", "items": {"type": "integer", "minimum": 1}}},
        },
    )

    async def call(self, **arguments):
        raise AssertionError("Validation must never execute the tool")


def context(mode="prompt"):
    return StepContext.model_construct(
        tool_mode=mode, tools={"find": ExternalTool()}, report_error=AsyncMock(), run_id="r"
    )


def record(state, message):
    state.history.append(Step(turn=len(state.history) + 1, action="model"))
    state.messages.append(message)
    state.last_response = message


async def test_native_dispatch_then_finish_ignores_stale_response():
    state = RunState(input="Hi", messages=[])
    ctx = context("native")
    auto = Auto()
    assert await auto.step(state, ctx) == CallModel()
    calls = [ToolCall(id="a", name="find", arguments={"ids": [1]})]
    record(state, Message(role="assistant", tool_calls=calls))
    assert await auto.step(state, ctx) == CallTool(calls=calls)
    state.history.append(Step(turn=2, action="tool"))
    assert await auto.step(state, ctx) == CallModel()
    record(state, Message(role="assistant", content="Done"))
    assert await auto.step(state, ctx) == Finish(output="Done")


async def test_react_downgrade_repairs_and_observes_without_duplicate_prompt():
    state = RunState(input="Read", messages=[Message(role="system", content="Be brief.")])
    ctx = context()
    auto = Auto()
    record(state, Message(role="assistant", content="stale native answer"))
    ask = CallModel(use_tools=False, emit_tokens_after="Final Answer:")
    assert await auto.step(state, ctx) == ask
    prompt = state.messages[0].content or ""
    assert (
        prompt.startswith("Be brief.\n") and "Find a record." in prompt and '"minimum": 1' in prompt
    )
    record(
        state,
        Message(
            role="assistant",
            content="Action:\n```json\n{'tool':'find', 'arguments':{'ids':[1,],},}",
        ),
    )
    selected = await auto.step(state, ctx)
    assert isinstance(selected, CallTool) and selected.as_observation
    assert selected.calls[0].name == "find" and selected.calls[0].arguments == {"ids": [1]}
    state.history.append(Step(turn=3, action="tool"))
    state.messages.append(Message(role="user", content="Observation: Found"))
    assert await auto.step(state, ctx) == ask
    assert state.messages[0].content == prompt
    record(state, Message(role="assistant", content="Thought: {ignored}\nFinal Answer: Found\n"))
    assert await auto.step(state, ctx) == Finish(output="Found")


async def test_react_retries_twice_reports_and_finishes_with_raw_reply():
    state = RunState(input="Hi", messages=[])
    ctx = context()
    react = React()
    await react.step(state, ctx)
    for raw in (
        '{"tool": "missing", "arguments": {}} {"tool":"find","arguments":{"ids":[1]}}',
        '{"tool":"find","arguments":{"ids":[0]}}',
    ):
        record(state, Message(role="assistant", content=raw))
        assert isinstance(await react.step(state, ctx), CallModel)
        assert state.messages[-1].role == "user" and "Error:" in (state.messages[-1].content or "")
    raw = "{still invalid"
    record(state, Message(role="assistant", content=raw))
    assert await react.step(state, ctx) == Finish(output=raw)
    assert isinstance(ctx.report_error, AsyncMock)
    ctx.report_error.assert_awaited_once()


class Search(BaseModel):
    query: str = Field(min_length=3)


async def test_react_function_validation_rejects_invalid_nested_fields():
    @tool
    def search(item: Search):
        raise AssertionError("Validation must never execute the tool")

    state = RunState(input="Read", messages=[])
    ctx = context()
    ctx.tools = {"search": search}
    react = React()
    await react.step(state, ctx)
    record(
        state,
        Message(role="assistant", content='{"tool":"search","arguments":{"item":{"query":"a"}}}'),
    )
    assert isinstance(await react.step(state, ctx), CallModel)
    record(
        state,
        Message(
            role="assistant",
            content="""{"tool":"search","arguments":{"item":{"query":"x,}'y"}}} extra""",
        ),
    )
    selected = await react.step(state, ctx)
    assert (
        isinstance(selected, CallTool) and selected.calls[0].arguments["item"]["query"] == "x,}'y"
    )


async def test_unknown_schema_constraint_is_reported_before_dispatch():
    state = RunState(input="Hi", messages=[])
    ctx = context()
    ctx.tools["find"].spec = ctx.tools["find"].spec.model_copy(
        update={"parameters": {"not": {"required": ["x"]}}}
    )
    react = React()
    await react.step(state, ctx)
    record(state, Message(role="assistant", content='{"tool":"find","arguments":{}}'))
    assert isinstance(await react.step(state, ctx), CallModel)
    assert "Unsupported tool schema" in (state.messages[-1].content or "")


async def test_plain_prompt_reply_finishes_without_corrections():
    for paradigm in (Auto(), React()):
        ctx = context()
        ctx.tools.clear()
        state = RunState(input="Hi", messages=[])
        assert await paradigm.step(state, ctx) == CallModel(use_tools=False)
        record(state, Message(role="assistant", content="Hello"))
        assert await paradigm.step(state, ctx) == Finish(output="Hello")
        assert len(state.messages) == 1

    ctx = context()
    state = RunState(input="Hi", messages=[])
    react = React()
    await react.step(state, ctx)
    record(state, Message(role="assistant", content="Thought: no tool needed\nHello"))
    assert await react.step(state, ctx) == Finish(output="Hello")
