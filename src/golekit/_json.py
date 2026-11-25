"""Parse model-generated JSON, tolerating common quoting and fencing errors."""

import ast
import json
import re
from typing import Any

from pydantic import TypeAdapter

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


class OutputSchema:
    """Translate a requested type into a prompt and validate its final value."""

    def __init__(self, output: type):
        self.adapter: TypeAdapter[Any] = TypeAdapter(output)
        self.schema: dict[str, Any] = self.adapter.json_schema()
        self.instruction = "最终回答只输出一个符合以下 JSON Schema 的 JSON 值。\n" + json.dumps(
            self.schema, ensure_ascii=False
        )

    def validate(self, text: str) -> Any:
        """Repair JSON syntax, then validate the decoded value."""
        return self.adapter.validate_python(parse_json(text))
