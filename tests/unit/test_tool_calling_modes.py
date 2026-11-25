"""Run the same tool-assisted conversation through native and prompt calling."""

import pytest

import golekit as gole
from golekit.errors import UnsupportedToolsError
from golekit.testing import FakeModel, fake_tool_call


@gole.tool
def weather(city: str) -> str:
    """Look up a city's weather."""
    return f"{city} 18°C"


@pytest.mark.parametrize("mode", ["native", "prompt"])
async def test_weather_then_answer(mode):
    prompt_call = 'Thought: get weather\nAction:\n```json\n{"tool":"weather","arguments":{"city":"Paris"}}\n```'
    native = mode == "native"
    model = FakeModel(
        [
            fake_tool_call("weather", {"city": "Paris"}) if native else prompt_call,
            "Paris 18°C" if native else "Thought: done\nFinal Answer:Paris 18°C",
        ]
    )
    agent = gole.Agent(model, tools=[weather], tool_mode=mode)
    events = [ev async for ev in agent.astream("Weather?")]
    result = events[-1].result
    assert str(result) == "Paris 18°C"
    assert "".join(ev.text for ev in events if ev.type == "token") == result.text
    assert [ev.output for ev in events if ev.type == "tool_result"] == ["Paris 18°C"]
    assert bool(model.requests[0]["tools"]) is native
    observation = model.requests[1]["messages"][-1]
    assert observation.role == ("tool" if native else "user")
    assert "Paris 18°C" in (observation.content or "")
    assert [step.action for step in result.steps] == ["model", "tool", "model", "finish"]


async def test_server_rejection_persists_on_shared_model(monkeypatch):
    class RejectingModel(FakeModel):
        rejects = 0

        async def generate(self, messages, *, tools=None, **params):
            if tools:
                self.rejects += 1
                raise UnsupportedToolsError("400: tools not supported")
            return await super().generate(messages, tools=tools, **params)

    fake = RejectingModel(["Final Answer:one", "Final Answer:two"])
    monkeypatch.setattr("golekit.model.create_client", lambda *args, **kwargs: fake)
    shared = gole.Model("ollama/demo")
    agent = gole.Agent(shared, tools=[weather])
    result = await agent.arun("one")
    assert result.text == "one"
    assert any(e.type == "error" and e.recoverable for e in result.trace.events)
    assert (await gole.Agent(shared, tools=[weather]).arun("two")).text == "two"
    assert fake.rejects == 1 and not shared.capabilities.tools


@pytest.mark.parametrize("prefix", ["", "begin"])
async def test_direct_stream_completes_missing_deltas(prefix):
    from golekit.protocols.types import ModelChunk

    class Partial(FakeModel):
        async def stream(self, messages, **kwargs):
            if prefix:
                yield ModelChunk(text=prefix)
            yield ModelChunk(response=await self.generate(messages, **kwargs))

    events = [event async for event in gole.Agent(Partial(["begin end"])).astream("hello")]
    assert (
        "".join(event.text for event in events if event.type == "token") == events[-1].result.text
    )
    if prefix:
        assert next(i for i, e in enumerate(events) if e.type == "token") < next(
            i for i, e in enumerate(events) if e.type == "model_end"
        )


async def test_marked_prompt_answer_arrives_before_model_end():
    model = FakeModel(["Thought: done\nFinal Answer:  first  second  "])
    agent = gole.Agent(model, tools=[weather], tool_mode="prompt")
    events = [event async for event in agent.astream("weather")]
    end = next(i for i, e in enumerate(events) if e.type == "model_end")
    assert "".join(e.text for e in events[:end] if e.type == "token") == "first  second"
    assert events[-1].result.text == "first  second"


async def test_native_text_streams_before_stream_end_and_tool_call():
    import asyncio

    from golekit.protocols.types import ModelChunk

    class Slow(FakeModel):
        finished = 0

        async def stream(self, messages, *, tools=None, **params):
            response = await self.generate(messages, tools=tools, **params)
            if response.message.tool_calls:
                response.message.content = "Let me check. "
            content = response.message.content or ""
            for index in range(0, len(content), 4):
                await asyncio.sleep(0.01)
                yield ModelChunk(text=content[index : index + 4])
            self.finished += 1
            yield ModelChunk(response=response)

    model = Slow([fake_tool_call("weather", {"city": "Paris"}), "Paris is mild."])
    agent = gole.Agent(model, tools=[weather], tool_mode="native")
    seen: list[tuple[str, int]] = []
    agent.on("token")(lambda event: seen.append((event.text, model.finished)))
    events = [event async for event in agent.astream("Weather?")]
    result = events[-1].result
    # The first delta of each reply arrives while that reply is still being generated.
    assert seen[0] == ("Let ", 0)
    assert next(text for text, finished in seen if finished == 1) == "Pari"
    kinds = [event.type for event in events]
    last_tool = len(kinds) - 1 - kinds[::-1].index("tool_result")
    assert "token" in kinds[: kinds.index("tool_call")]
    assert "".join(e.text for e in events[last_tool:] if e.type == "token") == result.text
    assert result.text == "Paris is mild."
