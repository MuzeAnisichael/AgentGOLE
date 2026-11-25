"""Built-in python, shell and file tools."""

import os
import sys

import pytest

import golekit as gole
from golekit.errors import ConfigError
from golekit.testing import FakeModel, fake_tool_call
from golekit.tools.files import files
from golekit.tools.system import python, shell


async def test_python_returns_output_without_inherited_secrets(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "secret")
    code = "import os, sys\nprint(sorted(k for k in os.environ if k.endswith('_API_KEY')))\nprint(os.getcwd() != sys.argv[1])"
    output = await python().call(code=code.replace("sys.argv[1]", repr(os.getcwd())))
    assert output.splitlines() == ["[]", "True"]
    failed = await python().call(code="import sys\nprint('partial')\nsys.exit(3)")
    assert failed == "partial\n[exit code 3]"


async def test_python_blocks_sockets_and_stops_at_the_time_limit():
    blocked = await python().call(
        code="import urllib.request\nurllib.request.urlopen('http://127.0.0.1:9')"
    )
    assert "Network access is disabled" in blocked and "[exit code 1]" in blocked
    with pytest.raises(Exception, match="Timed out after 0.5s"):
        await python(timeout=0.5).call(code="import time\ntime.sleep(30)")


async def test_shell_reports_output_streams_and_exit_code(tmp_path):
    (tmp_path / "marker.txt").touch()
    listing = await shell(cwd=tmp_path).call(command="dir" if os.name == "nt" else "ls")
    assert "marker.txt" in listing
    command = (
        f'"{sys.executable}" -c "import sys; print(1); print(2, file=sys.stderr); sys.exit(4)"'
    )
    output = await shell().call(command=command)
    assert output == "1\n[stderr]\n2\n[exit code 4]"
    with pytest.raises(Exception, match="Timed out"):
        await shell(timeout=0.5).call(
            command=f'"{sys.executable}" -c "import time; time.sleep(30)"'
        )


async def test_python_and_shell_need_confirmation_by_default():
    model = FakeModel(
        [
            fake_tool_call("python", {"code": "print(1)"}),
            fake_tool_call("shell", {"command": "x"}),
            "ok",
        ]
    )
    result = await gole.Agent(model, tools=["python", "shell"]).arun("go")
    outputs = [step.results[0].output for step in result.steps if step.action == "tool"]
    assert all("requires confirmation" in output for output in outputs)
    assert python(confirm=False).confirm is False  # pyright: ignore[reportAttributeAccessIssue]


async def test_files_stay_inside_the_root(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "note.txt").write_bytes("内容".encode() + b"\xff")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    read_file, list_dir = files(root)
    assert await read_file.call(path="note.txt") == "内容�"
    assert await list_dir.call() == "note.txt\nsub/"
    escapes = ["../secret.txt", "sub/../../secret.txt", str(secret)]
    if hasattr(os, "symlink") and os.name != "nt":
        (root / "link.txt").symlink_to(secret)
        escapes.append("link.txt")
    for path in escapes:
        with pytest.raises(Exception, match="root directory"):
            await read_file.call(path=path)
    with pytest.raises(ConfigError, match="root"):
        files(tmp_path / "missing")


async def test_write_file_is_opt_in_and_creates_parents(tmp_path):
    assert [item.spec.name for item in files(tmp_path)] == ["read_file", "list_dir"]
    write_file = files(tmp_path, write=True)[2]
    assert "new/deep.txt" in await write_file.call(path="new/deep.txt", content="hello")
    assert (tmp_path / "new" / "deep.txt").read_text(encoding="utf-8") == "hello"
    with pytest.raises(Exception, match="root directory"):
        await write_file.call(path="../outside.txt", content="x")
    agent = gole.Agent(FakeModel([]), tools=["calculator", gole.tools.files(tmp_path, write=True)])
    assert list(agent._tools) == ["calculator", "read_file", "list_dir", "write_file"]
    with pytest.raises(ConfigError, match=r"gole\.tools\.files\(root="):
        gole.Agent(FakeModel([]), tools=["files"])
