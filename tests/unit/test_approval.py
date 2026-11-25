"""Confirmation of tool calls before they run."""

import io

import pytest

import golekit as gole
from golekit import approval
from golekit.errors import ConfigError
from golekit.testing import FakeModel, fake_tool_call

calls: list[str] = []


@gole.tool(confirm=True)
def wipe(path: str) -> str:
    """Delete a path."""
    calls.append(path)
    return f"deleted {path}"


def run(approver, script=None):
    calls.clear()
    model = FakeModel(script or [fake_tool_call("wipe", {"path": "tmp"}), "done"])
    return model, gole.Agent(model, tools=[wipe], approval=approver)


@pytest.mark.parametrize(
    "answer,approved", [("YES\n", True), ("y\n", True), ("n\n", False), ("", False)]
)
async def test_console_reads_one_line(answer, approved, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(answer))
    model, agent = run(approval.Console())
    result = await agent.arun("clean up")
    assert calls == (["tmp"] if approved else [])
    observation = model.requests[1]["messages"][-1].content
    assert observation == ("deleted tmp" if approved else "Error: Rejected by the user.")
    assert result.text == "done"
    prompt = capsys.readouterr().err
    assert "wipe" in prompt and '{"path": "tmp"}' in prompt


async def test_callback_decides_and_every_call_announces_itself():
    seen: list[tuple[str, dict]] = []

    async def decide(tool, args):
        seen.append((tool, args))
        return args["path"] == "tmp"

    script = [
        fake_tool_call("wipe", {"path": "tmp"}),
        fake_tool_call("wipe", {"path": "/"}),
        fake_tool_call("wipe", {"path": 3}),
        "done",
    ]
    _, agent = run(approval.Callback(decide), script)
    result = await agent.arun("clean up")
    assert calls == ["tmp"] and seen == [("wipe", {"path": "tmp"}), ("wipe", {"path": "/"})]
    required = [e for e in result.trace.events if e.type == "approval_required"]
    # Invalid arguments are refused before anyone is asked.
    assert [(e.tool, e.args) for e in required] == seen
    outputs = [step.results[0] for step in result.steps if step.action == "tool"]
    assert outputs[1].error == "Rejected by the user." and "ValidationError" in (
        outputs[2].error or ""
    )
    kinds = [e.type for e in result.trace.events]
    assert kinds.index("tool_call") < kinds.index("approval_required") < kinds.index("tool_result")


async def test_failing_callback_rejects_and_is_reported():
    def decide(tool, args):
        raise RuntimeError("approval service down")

    _, agent = run(approval.Callback(decide))
    result = await agent.arun("clean up")
    assert calls == [] and result.text == "done"
    errors = [e for e in result.trace.events if e.type == "error"]
    assert len(errors) == 1 and errors[0].recoverable


async def test_without_an_approver_the_tool_never_runs():
    model, agent = run(None)
    result = await agent.arun("clean up")
    assert calls == []
    assert "approval=" in (model.requests[1]["messages"][-1].content or "")
    assert [e.type for e in result.trace.events].count("approval_required") == 1
    with pytest.raises(ConfigError, match="approval"):
        gole.Agent(model, approval=lambda tool, args: True)  # pyright: ignore[reportArgumentType]
