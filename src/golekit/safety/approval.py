"""Ways to approve a tool call that requires confirmation."""

import asyncio
import inspect
import json
import sys
from collections.abc import Awaitable, Callable
from threading import Thread
from typing import Any

__all__ = ["Callback", "Console"]


class Console:
    """Ask on the terminal before each call; only ``y`` or ``yes`` approves.

    Example:
        gole.Agent(tools=["shell"], approval=gole.approval.Console())
    """

    async def approve(self, tool: str, args: dict[str, Any]) -> bool:
        """Print the call to stderr and read one line from stdin."""
        text = json.dumps(args, ensure_ascii=False, default=str)
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"工具 {tool} 请求执行，参数：{text}", file=sys.stderr)
        print("允许吗？[y/N] ", end="", file=sys.stderr, flush=True)
        loop = asyncio.get_running_loop()
        answer: asyncio.Future[str] = loop.create_future()

        def read() -> None:
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                line = ""
            try:
                loop.call_soon_threadsafe(lambda: answer.done() or answer.set_result(line))
            except RuntimeError:
                pass  # The run ended, for example on its timeout, before the user answered.

        # A daemon thread cannot keep the process alive while it waits for input.
        Thread(target=read, daemon=True).start()
        return (await answer).strip().lower() in ("y", "yes")


class Callback:
    """Delegate the decision to ``fn(tool, args) -> bool``, sync or async.

    Example:
        gole.Agent(tools=["shell"], approval=gole.approval.Callback(lambda tool, args: True))
    """

    def __init__(self, fn: Callable[[str, dict[str, Any]], bool | Awaitable[bool]]) -> None:
        self._fn = fn

    async def approve(self, tool: str, args: dict[str, Any]) -> bool:
        """Return the callback's decision; an exception counts as a rejection upstream."""
        decision = self._fn(tool, args)
        if inspect.isawaitable(decision):
            decision = await decision
        return bool(decision)
