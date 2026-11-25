"""The built-in arithmetic and date tools."""

import ast
import math
import operator
import sys
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .function import tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
}


def _math_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
    ):
        return node.attr
    raise ValueError("Only named math constants and functions are supported")


def _evaluate(node: ast.expr) -> int | float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    if isinstance(node, ast.BinOp) and (operation := _OPERATORS.get(type(node.op))):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("The absolute exponent must not exceed 1000")
            # Bound nested powers before allocating an enormous intermediate integer.
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and left.bit_length() * right > 65536
            ):
                raise ValueError("Integer power exceeds the calculator's size limit")
        value = operation(left, right)
        if not isinstance(value, (int, float)):
            raise ValueError("The calculation must produce a real number")
        return value
    if isinstance(node, ast.Call):
        function = _FUNCTIONS.get(_math_name(node.func))
        if function is not None and not node.keywords:
            return function(*(_evaluate(arg) for arg in node.args))
    if isinstance(node, (ast.Name, ast.Attribute)) and (name := _math_name(node)) in _CONSTANTS:
        return _CONSTANTS[name]
    raise ValueError("Unsupported calculator expression")


@tool
def calculator(expression: str) -> int | float:
    """Evaluate an arithmetic expression using numbers and common math functions.

    Args:
        expression: Arithmetic expression, for example sqrt(9) + 2 ** 3.
    """
    return _evaluate(ast.parse(expression.strip(), mode="eval").body)


@tool(name="datetime")
def current_datetime(timezone: str | None = None, offset_days: int = 0) -> dict[str, str]:
    """Return the current date, time, and weekday with an optional day offset.

    Args:
        timezone: IANA time zone name, or the machine's local zone when omitted.
        offset_days: Number of calendar days to add to the date.
    """
    instant = datetime.now(UTC)
    if timezone is None:
        local = instant.astimezone()
    elif timezone == "UTC":
        local = instant
    else:
        try:
            local = instant.astimezone(ZoneInfo(timezone))
        except ZoneInfoNotFoundError:
            if sys.platform != "win32":
                raise
            from ._windows_zones import local_time

            local = local_time(instant, timezone)
    shifted = local + timedelta(days=offset_days)
    return {
        "date": shifted.date().isoformat(),
        "time": shifted.strftime("%H:%M:%S"),
        "weekday": shifted.strftime("%A"),
    }
