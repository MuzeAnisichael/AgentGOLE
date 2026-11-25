"""Exporting a run's events as JSON, by call and through GOLE_TRACE_DIR."""

import json
import logging

import pytest

import golekit as gole
from golekit.errors import MaxStepsError
from golekit.testing import FakeModel, fake_tool_call


@gole.tool(confirm=True)
def wipe(path: str) -> str:
    """Delete a path."""
    return "deleted"


def broken_approval(tool, args):
    raise RuntimeError("approval service down")


def agent(script, **options):
    return gole.Agent(
        FakeModel(script),
        tools=[wipe],
        approval=gole.approval.Callback(broken_approval),
        **options,
    )


async def test_document_round_trips_with_exceptions_and_without_nested_trace(tmp_path):
    result = await agent([fake_tool_call("wipe", {"path": "数据"}), "done"]).arun("clean")
    target = tmp_path / "nested" / "run.json"
    text = result.trace.to_json(target)
    document = json.loads(text)
    assert target.read_text(encoding="utf-8") == text and "数据" in text
    assert document["run_id"] == result.run_id and document["golekit"] == gole.__version__
    events = document["events"]
    assert [e["type"] for e in events] == [e.type for e in result.trace.events]
    error = next(e for e in events if e["type"] == "error")
    assert error["exception"] == {"type": "RuntimeError", "message": "approval service down"}
    end = events[-1]
    assert end["result"]["text"] == "done" and "trace" not in end["result"]
    start = next(e for e in events if e["type"] == "model_start")
    assert start["messages"][-1] == {
        "role": "user",
        "content": "clean",
        "tool_calls": [],
        "tool_call_id": None,
        "name": None,
        "reasoning": None,
    }


async def test_failed_run_carries_its_trace(tmp_path, monkeypatch):
    with pytest.raises(MaxStepsError) as failure:
        await agent(["never used"], max_steps=1).arun("clean")
    trace = failure.value.trace  # pyright: ignore[reportAttributeAccessIssue]
    document = json.loads(trace.to_json())
    assert document["events"][-1]["exception"]["type"] == "MaxStepsError"
    assert document["events"][-1]["recoverable"] is False


async def test_trace_directory_receives_successful_and_failed_runs(tmp_path, monkeypatch, caplog):
    directory = tmp_path / "traces"
    monkeypatch.setenv("GOLE_TRACE_DIR", str(directory))
    result = await agent(["done"]).arun("clean")
    with pytest.raises(MaxStepsError) as failure:
        await agent(["never used"], max_steps=1).arun("clean")
    failed = failure.value.trace.events[0].run_id  # pyright: ignore[reportAttributeAccessIssue]
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [f"{result.run_id}.json", f"{failed}.json"]
    )
    saved = json.loads((directory / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert saved["events"][-1]["type"] == "run_end"
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setenv("GOLE_TRACE_DIR", str(blocked / "sub"))
    with caplog.at_level(logging.WARNING):
        assert (await agent(["done"]).arun("clean")).text == "done"
    assert "Could not write the run trace" in caplog.text
