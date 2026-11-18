"""Validate common JSON Schema assertions on structural tools without executing them."""

import re
from decimal import Decimal
from typing import Any

from ..protocols import Tool

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
    "integer": int,
    "number": (int, float),
}
_KEYWORDS = {
    "$schema",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$ref",
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "allOf",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minProperties",
    "maxProperties",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


class _Mismatch(ValueError):
    """The supplied argument fails a supported assertion."""


def _has_type(value: Any, kind: str) -> bool:
    if kind == "integer" and isinstance(value, float):
        return value.is_integer()
    if kind in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[kind])


def _equal(left: Any, right: Any) -> bool:
    # JSON booleans are distinct from numbers, including within enum literals.
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return isinstance(left, bool) == isinstance(right, bool) and left == right


def _validate(
    schema: Any, value: Any, root: dict[str, Any], path: str, active: set[tuple[int, int]]
) -> None:
    if schema is True:
        return
    if schema is False:
        raise _Mismatch(f"{path} is disallowed")
    if not isinstance(schema, dict):
        raise TypeError(f"Invalid tool schema at {path}")
    if unknown := schema.keys() - _KEYWORDS:
        raise ValueError(f"Unsupported tool schema keyword(s): {', '.join(sorted(unknown))}")
    key = (id(schema), id(value))
    if key in active:
        raise ValueError("Circular tool schema reference without a change of instance")
    active = active | {key}
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"Unsupported tool schema reference: {ref}")
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        _validate(target, value, root, path, active)
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            matched = 0
            for branch in schema[keyword]:
                try:
                    _validate(branch, value, root, path, active)
                    matched += 1
                except _Mismatch:
                    pass
            if (
                matched == 0
                or (keyword == "oneOf" and matched != 1)
                or (keyword == "allOf" and matched != len(schema[keyword]))
            ):
                raise _Mismatch(f"{path} does not match {keyword}")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if kinds and not any(_has_type(value, kind) for kind in kinds):
        raise _Mismatch(f"{path} must have type {' or '.join(kinds)}")
    if "const" in schema and not _equal(value, schema["const"]):
        raise _Mismatch(f"{path} must equal the declared constant")
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        raise _Mismatch(f"{path} is not among the allowed values")

    if isinstance(value, dict):
        if missing := set(schema.get("required", [])) - value.keys():
            raise _Mismatch(f"{path} is missing: {', '.join(sorted(missing))}")
        properties = schema.get("properties", {})
        for name, item in value.items():
            subschema = properties.get(name, schema.get("additionalProperties", True))
            _validate(subschema, item, root, f"{path}.{name}", active)
        unit = "Properties"
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate(schema.get("items", True), item, root, f"{path}[{index}]", active)
        unit = "Items"
    elif isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise _Mismatch(f"{path} does not match the required pattern")
        unit = "Length"
    else:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < schema.get("minimum", value) or value > schema.get("maximum", value):
                raise _Mismatch(f"{path} is outside the allowed range")
            if value <= schema.get("exclusiveMinimum", -float("inf")) or value >= schema.get(
                "exclusiveMaximum", float("inf")
            ):
                raise _Mismatch(f"{path} is outside the allowed range")
            if "multipleOf" in schema and Decimal(str(value)) % Decimal(str(schema["multipleOf"])):
                raise _Mismatch(f"{path} is not an allowed multiple")
        return
    size = len(value)
    if not schema.get("min" + unit, 0) <= size <= schema.get("max" + unit, size):
        raise _Mismatch(f"{path} violates {unit.lower()} bounds")


def validate_arguments(tool: Tool, arguments: dict[str, Any]) -> None:
    validate = getattr(tool, "validate_arguments", None)
    if callable(validate):
        validate(arguments)
    else:
        schema = tool.spec.parameters
        _validate(schema, arguments, schema, "arguments", set())
