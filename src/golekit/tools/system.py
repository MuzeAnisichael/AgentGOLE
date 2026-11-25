"""Built-in tools that run code and commands on the local machine."""

import locale
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from ..errors import ToolError
from ..protocols import Tool
from .function import tool

# Disable sockets before the user's script runs, then execute it as __main__.
_BOOTSTRAP = """\
import runpy, socket, sys

class _Blocked(socket.socket):
    def __init__(self, *args, **kwargs):
        raise OSError("Network access is disabled in the python tool.")

def _refuse(*args, **kwargs):
    raise OSError("Network access is disabled in the python tool.")

socket.socket = _Blocked
socket.create_connection = _refuse
socket.getaddrinfo = _refuse
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
# What an interpreter needs to start; credentials such as *_API_KEY are never copied.
_KEPT_ENV = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG")


def _kill_tree(process: "subprocess.Popen[bytes]") -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _decode(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Windows shells answer in the console code page.
        text = data.decode(locale.getpreferredencoding(False), errors="replace")
    return text.replace("\r\n", "\n")


def _run(
    command: str | list[str],
    *,
    cwd: str | Path | None,
    env: dict[str, str] | None,
    timeout: float,
) -> str:
    windows = os.name == "nt"
    process = subprocess.Popen(
        command,
        shell=isinstance(command, str),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # A separate group or session lets a timeout stop the children as well.
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if windows else 0,
        start_new_session=not windows,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise ToolError(f"Timed out after {timeout:g}s; the process was terminated.") from None
    text = _decode(stdout)
    if stderr:
        text += ("" if not text or text.endswith("\n") else "\n") + "[stderr]\n" + _decode(stderr)
    if process.returncode:
        text += "" if not text or text.endswith("\n") else "\n"
        text += f"[exit code {process.returncode}]"
    return text or "(no output)"


def python(timeout: float = 10, confirm: bool = True) -> Tool:
    """Create the ``python`` tool, which runs model-written code in a child process.

    The child is a separate isolated interpreter in an empty temporary directory, with a
    minimal environment and sockets disabled. This limits accidents; it is not a security
    boundary, and the code can still read and write the user's files. Keep ``confirm=True``
    unless the input is trusted.
    """
    if timeout <= 0:
        raise ValueError("Timeout must be positive")

    def run(code: str) -> str:
        """Run Python code in a fresh process and return what it prints.

        The process has no network access and stops after a time limit.

        Args:
            code: Complete Python source. Use print() to produce output.
        """
        # On Windows a process that is still exiting can keep the directory busy for a moment.
        with tempfile.TemporaryDirectory(
            prefix="gole-python-", ignore_cleanup_errors=True
        ) as directory:
            script = Path(directory) / "main.py"
            script.write_text(code, encoding="utf-8")
            env = {name: os.environ[name] for name in _KEPT_ENV if name in os.environ}
            command = [sys.executable, "-I", "-X", "utf8", "-c", _BOOTSTRAP, str(script)]
            return _run(command, cwd=directory, env=env, timeout=timeout)

    # The outer limit only guards against a hung pipe; the process limit fires first.
    return tool(run, name="python", timeout=timeout + 10, confirm=confirm)


def shell(cwd: str | Path | None = None, timeout: float = 30, confirm: bool = True) -> Tool:
    """Create the ``shell`` tool, which runs a command line through the system shell.

    Commands run with the user's permissions in ``cwd`` (the current directory by default).
    """
    if timeout <= 0:
        raise ValueError("Timeout must be positive")

    def run(command: str) -> str:
        """Run a shell command and return its output and exit code.

        Args:
            command: Command line for the system shell.
        """
        return _run(command, cwd=cwd, env=None, timeout=timeout)

    return tool(run, name="shell", timeout=timeout + 10, confirm=confirm)
