import asyncio
import json
from datetime import UTC, datetime
from threading import Event

import pytest
from pydantic import BaseModel, ValidationError

from golekit.protocols.types import ToolSpec
from golekit.tools import builtins, resolve_tools, tool


class Point(BaseModel):
    x: int


async def test_signature_docs_and_nested_parameters():
    @tool
    async def move(point: Point, scale: int = 2):
        """Scale the point.

        Args:
            point (Point): Starting coordinate.
            scale: Scale factor for
                the coordinate.
        """
        return Point(x=point.x * scale)

    assert move.spec.description == "Scale the point."
    schema = move.spec.parameters
    assert schema["required"] == ["point"]
    assert schema["properties"]["scale"]["default"] == 2
    assert schema["properties"]["scale"]["description"] == "Scale factor for the coordinate."
    assert schema["properties"]["point"]["description"] == "Starting coordinate."
    assert json.loads(await move.call(point={"x": 3})) == {"x": 6}
    with pytest.raises(ValidationError):
        await move.call(point={"x": "bad"})


async def test_flat_docs_and_sync_retries_after_validation():
    attempts = []

    @tool(name="lookup", description="Custom description.", retries=1)
    def fetch(query: str):
        """Fetch a value.
        query: Lookup key.
        """
        attempts.append(query)
        if len(attempts) == 1:
            raise OSError("Temporary failure")
        return ["é", query]

    assert fetch.spec.name == "lookup" and fetch.spec.description == "Custom description."
    assert fetch.spec.parameters["properties"]["query"]["description"] == "Lookup key."
    with pytest.raises(ValidationError):
        await fetch.call(q="bad")
    assert attempts == []
    assert json.loads(await fetch.call(query="x")) == ["é", "x"]
    assert attempts == ["x", "x"]


async def test_async_and_sync_tool_timeouts():
    release = Event()

    @tool(timeout=0.05)
    def blocking():
        release.wait(2)

    @tool(timeout=0.05)
    async def waiting():
        await asyncio.Event().wait()

    try:
        for wrapped in (blocking, waiting):
            with pytest.raises(TimeoutError):
                await wrapped.call()
    finally:
        release.set()


async def test_serialization_limit_and_unmodified_strings():
    @tool
    def repeat(count: int) -> str:
        return "é" * count

    assert await repeat.call(count=8000) == "é" * 8000
    output = await repeat.call(count=8100)
    assert len(output) == 8000 and output.endswith("...[truncated]")


async def test_calculator_whitelist_and_exponent_limit():
    calc = resolve_tools(["calculator"])["calculator"]
    assert await calc.call(expression="sqrt(9) + (2 ** 3) // 2 - -1") == "8.0"
    assert await calc.call(expression="math.cos(pi) + log(e)") == "0.0"
    for expression in ("2 ** -1001", "math.__dict__", "open('x')", "True + 1", "[1][0]"):
        with pytest.raises(ValueError):
            await calc.call(expression=expression)


async def test_datetime_iana_offset_and_weekday(monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 7, 10, 1, 2, 3, tzinfo=UTC)

    monkeypatch.setattr(builtins, "datetime", FixedClock)
    date_tool = resolve_tools(["datetime"])["datetime"]
    assert json.loads(await date_tool.call(timezone="America/New_York", offset_days=1)) == {
        "date": "2024-07-10",
        "time": "21:02:03",
        "weekday": "Wednesday",
    }
    assert json.loads(await date_tool.call(timezone="Asia/Shanghai"))["time"] == "09:02:03"


def test_tool_resolution_rejects_unknown_or_ambiguous_names():
    with pytest.raises(ValueError, match="Unknown built-in"):
        resolve_tools(["absent"])
    existing = resolve_tools(["calculator"])["calculator"]
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_tools(iter([existing, "calculator"]))


async def test_external_tool_validates_then_times_out(monkeypatch):
    class Waiting:
        spec = ToolSpec(
            name="wait",
            description="Wait.",
            parameters={
                "type": "object",
                "$comment": "Annotations do not restrict arguments.",
                "required": ["signal"],
            },
        )

        async def call(self, **arguments):
            await asyncio.Event().wait()

    monkeypatch.setattr("golekit.tools._DEFAULT_TIMEOUT", 0.01)
    wrapped = resolve_tools([Waiting()])["wait"]
    with pytest.raises(ValueError, match="missing"):
        await wrapped.call()
    with pytest.raises(TimeoutError):
        await wrapped.call(signal=True)
