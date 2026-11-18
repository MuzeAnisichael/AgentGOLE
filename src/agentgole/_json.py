"""Parse model-generated JSON, tolerating common quoting and fencing errors."""

import ast
import json
import re
from typing import Any

_TOKENS = re.compile(r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|,(?=\s*[}\]]))""", re.DOTALL)


def _repair(match: re.Match[str]) -> str:
    token = match.group()
    if token == ",":
        return ""
    if token.startswith("'"):
        return json.dumps(ast.literal_eval(token))
    return token


def parse_json(text: str, *, embedded: bool = False) -> Any:
    """Decode one JSON value; embedded mode reads the first action object."""
    payload = text.strip()
    if embedded:
        start = payload.find("{")
        if start < 0:
            raise ValueError("Expected a JSON action object")
        payload = payload[start:]
    elif payload.startswith("```"):
        payload = re.sub(r"^\s*```(?:json)?\s*", "", payload, count=1, flags=re.IGNORECASE)
        payload = payload.removesuffix("```").rstrip()
    try:
        return json.JSONDecoder().raw_decode(payload)[0] if embedded else json.loads(payload)
    except ValueError:
        repaired = _TOKENS.sub(_repair, payload)
        return json.JSONDecoder().raw_decode(repaired)[0] if embedded else json.loads(repaired)


def parse_object(text: str) -> dict[str, Any]:
    """Read the first action object from a ReAct reply."""
    return parse_json(text, embedded=True)
