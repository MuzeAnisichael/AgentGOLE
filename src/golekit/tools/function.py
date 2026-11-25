"""Adapt typed Python functions to the Tool protocol."""

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from concurrent.futures import Future
from contextvars import copy_context
from threading import Thread
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, create_model

from ..protocols import Tool
from ..protocols.types import ToolSpec

_PARAMETER = re.compile(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)")


def _documentation(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    lines = (inspect.getdoc(fn) or "").splitlines()
    names = inspect.signature(fn).parameters
    summary: list[str] = []
    descriptions: dict[str, str] = {}
    parameter: str | None = None
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:"}:
            in_args = True
            parameter = None
        elif (match := _PARAMETER.fullmatch(stripped)) and match[1] in names:
            in_args = True
            parameter = match.group(1)
            assert parameter is not None
            descriptions[parameter] = match.group(2)
        elif in_args and stripped and not line[:1].isspace():
            break
        elif parameter and stripped:
            descriptions[parameter] += " " + stripped
        elif not in_args:
            summary.append(line)
    return "\n".join(summary).strip(), descriptions


async def _invoke(fn: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(**arguments)

    # asyncio.run waits for its default executor at shutdown. A daemon worker
    # lets the synchronous public entry point return when its timeout expires.
    result: Future[Any] = Future()

    def work() -> None:
        if result.set_running_or_notify_cancel():
            try:
                result.set_result(fn(**arguments))
            except BaseException as exc:  # noqa: BLE001 - propagate across the thread boundary.
                result.set_exception(exc)

    Thread(target=copy_context().run, args=(work,), daemon=True).start()
    value = await asyncio.wrap_future(result)
    return await value if inspect.isawaitable(value) else value


def serialize(value: Any) -> str:
    """Convert a tool result to bounded model-facing text."""
    if isinstance(value, BaseModel):
        text = value.model_dump_json()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    marker = "...[truncated]"
    return text if len(text) <= 8000 else text[: 8000 - len(marker)] + marker


class FunctionTool:
    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None,
        description: str | None,
        timeout: float,
        retries: int,
    ) -> None:
        if timeout <= 0 or retries < 0:
            raise ValueError("Timeout must be positive and retries must be nonnegative")
        self._fn = fn
        self._timeout = timeout
        self._retries = retries
        fields: dict[str, Any] = {}
        hints = get_type_hints(fn, include_extras=True)
        for arg, param in inspect.signature(fn).parameters.items():
            if param.kind not in {param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD}:
                raise TypeError(f"Tool parameter {arg!r} must accept a named argument")
            default = ... if param.default is param.empty else param.default
            fields[arg] = (hints.get(arg, Any), default)
        summary, descriptions = _documentation(fn)
        self._model = create_model(
            f"{fn.__name__}Arguments", __config__=ConfigDict(extra="forbid"), **fields
        )
        parameters = self._model.model_json_schema()
        for arg, doc in descriptions.items():
            alias = self._model.model_fields[arg].alias or arg
            if alias in parameters["properties"]:
                parameters["properties"][alias]["description"] = doc
        self.spec = ToolSpec(
            name=name if name is not None else fn.__name__,
            description=description if description is not None else summary,
            parameters=parameters,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate a ReAct action as JSON before invoking the function."""
        self._model.model_validate_json(json.dumps(arguments, allow_nan=False), strict=True)

    def check_arguments(self, arguments: dict[str, Any]) -> None:
        """Raise if a call with these arguments would be rejected before the function runs."""
        self._model.model_validate(arguments)

    async def call(self, **arguments: Any) -> str:
        parsed = self._model.model_validate(arguments)
        values = {name: getattr(parsed, name) for name in self._model.model_fields}
        for attempt in range(self._retries + 1):
            try:
                async with asyncio.timeout(self._timeout):
                    output = await _invoke(self._fn, values)
                break
            except Exception:
                if attempt == self._retries:
                    raise
        return serialize(output)


@overload
def tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> Tool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate a typed sync or async function with a validated tool interface."""

    def decorate(function: Callable[..., Any]) -> Tool:
        return FunctionTool(function, name, description, timeout, retries)

    return decorate if fn is None else decorate(fn)
