import asyncio
import json
from typing import Any

import pytest

from golekit.errors import MaxStepsError, ModelError, RunTimeoutError, UnsupportedToolsError
from golekit.kernel.engine import execute
from golekit.protocols import StepContext
from golekit.protocols.types import (
    Action,
    CallModel,
    CallTool,
    Event,
    Finish,
    Message,
    ModelResponse,
    RunState,
    ToolSpec,
    Usage,
)
from golekit.testing import FakeModel, fake_tool_call


def field(event: Event, name: str) -> Any:
    return getattr(event, name)


class Script:
    def __init__(self, *actions: Action):
        self.actions = actions

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        return self.actions[state.turn]


class Echo:
    spec = ToolSpec(name="echo", description="Echo a value", parameters={})

    async def call(self, **arguments: Any) -> Any:
        value = arguments["value"]
        if value is None:
            raise ValueError("missing value " + "!" * 9000)
        return value


class Store:
    def __init__(self):
        self.saved: list[Message] = []

    async def load(self, session: str, query: str | None = None) -> list[Message]:
        assert session == "session1"
        return [Message(role="assistant", content="previous")]

    async def save(self, session: str, messages: list[Message]) -> None:
        assert session == "session1"
        self.saved = messages


class CountedModel(FakeModel):
    async def generate(self, messages, *, tools=None, **params) -> ModelResponse:
        reply = await super().generate(messages, tools=tools, **params)
        reply.usage = Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        return reply


class Harness:
    def __init__(self, model: FakeModel):
        self.events: list[Event] = []
        self.downgrades = 0
        self.ctx = StepContext(
            model=model,
            tools={"echo": Echo()},
            run_id="run",
            tool_mode="native",
            report_error=self.unbound,
            downgrade=self.downgrade,
        )

    async def unbound(self, exc: Exception) -> None:
        raise AssertionError("Run errors should be collected by the engine")

    def downgrade(self):
        self.downgrades += 1

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def run(self, paradigm: Script, **kwargs):
        params: dict[str, Any] = {
            "session": "session1",
            "instructions": "system",
            "memory": None,
            "ctx": self.ctx,
            "max_steps": 10,
            "timeout": 10,
            "streaming": False,
            "emit": self.emit,
        }
        params.update(kwargs)
        return await execute("request", paradigm=paradigm, **params)


async def test_action_order_usage_and_memory():
    call = fake_tool_call("echo", {"value": {"text": "中"}})
    model = CountedModel([call, "answer"])
    harness = Harness(model)
    memory = Store()
    result = await harness.run(
        Script(CallModel(), CallTool(calls=[call]), CallModel(), Finish(output="answer")),
        memory=memory,
    )
    assert [e.type for e in harness.events] == [
        "run_start",
        "memory_load",
        "model_start",
        "model_end",
        "step",
        "tool_call",
        "tool_result",
        "step",
        "model_start",
        "model_end",
        "step",
        "step",
        "memory_save",
        "run_end",
    ]
    assert result.trace.events == harness.events
    assert result.model_dump()["trace"]["events"][-1]["result"]["text"] == "answer"
    assert [m.content for m in model.requests[0]["messages"]] == ["system", "previous", "request"]
    observation = model.requests[1]["messages"][-1]
    assert observation.role == "tool" and observation.tool_call_id == call.id
    assert json.loads(observation.content or "") == {"text": "中"}
    assert [(s.turn, s.action) for s in result.steps] == [
        (1, "model"),
        (2, "tool"),
        (3, "model"),
        (4, "finish"),
    ]
    assert result.usage == Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    assert [(m.role, m.content) for m in memory.saved] == [
        ("user", "request"),
        ("assistant", "answer"),
    ]
    assert field(harness.events[2], "model") == "fake"
    assert field(harness.events[6], "tool") == "echo"


async def test_batch_observation_includes_errors_and_truncated_output():
    calls = [
        fake_tool_call("echo", {"value": None}),
        fake_tool_call("absent", {}),
        fake_tool_call("echo", {"value": "x" * 9000}),
    ]
    model = FakeModel(["recovered"])
    harness = Harness(model)
    result = await harness.run(
        Script(CallTool(calls=calls, as_observation=True), CallModel(), Finish(output="recovered"))
    )
    messages = model.requests[0]["messages"]
    assert len(messages) == 3 and messages[-1].role == "user"
    text = messages[-1].content or ""
    assert text.startswith("Observation:") and "Error: ValueError: missing value" in text
    assert "Error: ToolError: Unknown tool 'absent'" in text
    failed, missing, large = result.steps[0].results
    assert failed.error and missing.error and large.error is None
    assert len(failed.output) == 8000 and failed.output.endswith("...[truncated]")
    assert len(large.output) == 8000 and large.output.endswith("...[truncated]")


async def test_hidden_calls_emit_no_tokens_and_finish_supplies_the_answer():
    model = FakeModel(["privateFinal:hidden", "unfinished", "all"])
    harness = Harness(model)
    await harness.run(
        Script(
            CallModel(emit_tokens_after="Final:", visible=False),
            CallModel(visible=False),
            CallModel(messages=[], use_tools=False, visible=False),
            Finish(output="done"),
        ),
        streaming=True,
    )
    tokens = [e for e in harness.events if e.type == "token"]
    assert [field(e, "text") for e in tokens] == ["done"]
    assert tokens[0].step == 4
    assert model.requests[-1] == {"messages": [], "tools": None}


async def test_call_can_name_its_model_and_paradigm_can_annotate_the_step():
    class Planned(Script):
        async def step(self, state, ctx):
            if state.turn == 1:
                state.scratch["thought"] = "plan: answer directly"
            return await super().step(state, ctx)

    primary, helper = CountedModel(["final"]), CountedModel(["draft"], name="helper")
    harness = Harness(primary)
    result = await harness.run(
        Planned(
            CallModel(model=helper, visible=False, use_tools=False),
            CallModel(),
            Finish(output="final"),
        ),
        streaming=True,
    )
    starts = [field(e, "model") for e in harness.events if e.type == "model_start"]
    assert starts == ["helper", "fake"]
    assert len(helper.requests) == len(primary.requests) == 1
    assert result.usage.total_tokens == 6
    steps = [e for e in harness.events if e.type == "step"]
    assert [field(e, "thought") for e in steps] == ["plan: answer directly", None, None]
    assert "".join(field(e, "text") for e in harness.events if e.type == "token") == "final"


async def test_max_steps_includes_finish_and_reports_error():
    harness = Harness(FakeModel(["first"]))
    with pytest.raises(MaxStepsError):
        await harness.run(Script(CallModel(), Finish(output="unused")), max_steps=1)
    terminal = harness.events[-1]
    assert terminal.type == "error" and isinstance(field(terminal, "exception"), MaxStepsError)
    assert not field(terminal, "recoverable")


async def test_deadline_includes_probe_and_cleans_up():
    harness = Harness(FakeModel([]))
    cancelled = asyncio.Event()

    async def probe():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(RunTimeoutError):
        await harness.run(Script(Finish(output="unused")), resolve_mode=probe, timeout=0.01)
    assert cancelled.is_set()
    assert [e.type for e in harness.events] == ["run_start", "error"]
    assert isinstance(field(harness.events[-1], "exception"), RunTimeoutError)


async def test_rejection_retries_without_charging_action_and_records_recovery():
    class RejectTools(FakeModel):
        async def generate(self, messages, *, tools=None, **params):
            if tools:
                raise UnsupportedToolsError("unsupported tools")
            return await super().generate(messages, tools=tools, **params)

    class Recover(Script):
        async def step(self, state, ctx):
            if state.turn == 1:
                await ctx.report_error(ValueError("recoverable parse issue"))
            return await super().step(state, ctx)

    harness = Harness(RejectTools(["answer"]))
    result = await harness.run(Recover(CallModel(), Finish(output="answer")), max_steps=2)
    assert harness.downgrades == 1 and harness.ctx.tool_mode == "prompt"
    errors = [e for e in result.trace.events if e.type == "error"]
    assert [type(field(e, "exception")) for e in errors] == [UnsupportedToolsError, ValueError]
    assert all(field(e, "recoverable") for e in errors)
    assert [s.turn for s in result.steps] == [1, 2]


async def test_fatal_exception_and_callback_isolation(caplog):
    class BrokenModel(FakeModel):
        async def generate(self, messages, *, tools=None, **params):
            raise ModelError("model failed")

    harness = Harness(BrokenModel([]))

    async def hook(event):
        await harness.emit(event)
        if event.type == "model_start":
            raise ValueError("broken handler")

    with pytest.raises(ModelError, match="model failed"):
        await harness.run(Script(CallModel()), emit=hook)
    assert "broken handler" in caplog.text
    assert [e.type for e in harness.events] == ["run_start", "model_start", "error"]
    assert field(harness.events[-1], "recoverable") is False and harness.downgrades == 0


async def test_finish_supplies_token_before_step_if_marker_absent():
    harness = Harness(FakeModel(["private notes"]))
    result = await harness.run(
        Script(CallModel(emit_tokens_after="Final:"), Finish(output="public answer")),
        streaming=True,
    )
    events = harness.events
    finish = next(i for i, e in enumerate(events) if e.type == "step" and e.action == "finish")
    assert [(e.type, e.text) for e in events if e.type == "token"] == [("token", result.text)]
    assert events[finish - 1].type == "token"


async def test_malformed_native_arguments_become_one_observation():
    from golekit.models.wire import completed_response

    calls = completed_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "echo", "arguments": text}}
                            for text in ("", "{'value': 2,}", "broken")
                        ]
                    }
                }
            ]
        },
        "fake",
    ).message.tool_calls
    assert len({call.id for call in calls if call.id}) == 3
    assert [c.arguments for c in calls[:2]] == [{}, {"value": 2}]
    model = FakeModel(["recovered"])
    harness = Harness(model)
    result = await harness.run(
        Script(CallTool(calls=calls), CallModel(), Finish(output="recovered"))
    )
    assert result.steps[0].results[1].output == "2"
    assert model.requests[0]["messages"][-1].content == "Error: invalid arguments JSON"


async def test_memory_receives_current_query():
    class QueryStore(Store):
        async def load(self, session, query=None):
            return [Message(role="system", content=f"{session} / {query}")]

    model = FakeModel(["ok"])
    await Harness(model).run(Script(CallModel(), Finish(output="ok")), memory=QueryStore())
    assert model.requests[0]["messages"][1].content == "session1 / request"


@pytest.mark.parametrize("stall", [False, True])
async def test_failed_memory_save_keeps_the_result(stall, caplog):
    class BrokenStore(Store):
        async def save(self, session, messages):
            if stall:
                await asyncio.Event().wait()
            raise ModelError("fact extraction failed")

    harness = Harness(FakeModel(["answer"]))
    result = await harness.run(
        Script(CallModel(), Finish(output="answer")),
        memory=BrokenStore(),
        timeout=0.05 if stall else 10,
    )
    assert result.text == "answer"
    kinds = [e.type for e in harness.events]
    assert "memory_save" not in kinds and kinds[-2:] == ["error", "run_end"]
    error = harness.events[-2]
    assert field(error, "recoverable")
    assert isinstance(field(error, "exception"), RunTimeoutError if stall else ModelError)
    assert "Memory save failed" in caplog.text
