"""Plain-text run trees for replay and live output."""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..protocols.types import Event, Result, Trace


def _header(event: Event) -> str:
    return f"run {event.run_id[:4]}  {json.dumps(event.input, ensure_ascii=False)}"


class _Rows:
    def __init__(self) -> None:
        self.pending: Event | None = None

    def consume(self, event: Event) -> str | None:
        if event.type == "model_end":
            if event.tool_calls:
                names = ", ".join(call.name for call in event.tool_calls)
                return f"model  {event.model}  → tool_call {names}  {event.duration:.1f}s"
            # A prompt reply can select a tool, request a repair, or finish.
            self.pending = event
        elif self.pending is not None:
            if event.type == "tool_call":
                target = f"tool_call {event.tool}"
            elif event.type == "model_start":
                target = "重新调用"
            elif event.type == "step" and event.action == "finish":
                target = "最终回答"
            else:
                return None
            model, self.pending = self.pending, None
            return f"model  {model.model}  → {target}  {model.duration:.1f}s"
        elif event.type == "tool_result":
            args = ", ".join(
                f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in event.args.items()
            )
            output = json.dumps(event.output, ensure_ascii=False)
            return f"tool   {event.tool}({args}) → {output}  {event.duration:.1f}s"
        return None


class LiveTrace:
    """Write event lines to stderr as soon as their actions complete."""

    def __init__(self) -> None:
        self._rows = _Rows()

    def write(self, event: Event) -> None:
        """Render an event emitted during a verbose run."""
        if event.type == "run_start":
            print(_header(event), file=sys.stderr, flush=True)
        elif event.type == "run_end":
            print(
                f"└─ {event.duration:.1f}s   {event.result.usage.total_tokens:,} tok",
                file=sys.stderr,
                flush=True,
            )
        elif (row := self._rows.consume(event)) is not None:
            print(f"├─ {row}", file=sys.stderr, flush=True)


def show(trace: Trace) -> None:
    """Print the steps of a completed run."""
    start = next((ev for ev in trace.events if ev.type == "run_start"), None)
    if start is None:
        return
    elapsed = trace.events[-1].ts - start.ts
    tokens = sum(ev.usage.total_tokens for ev in trace.events if ev.type == "model_end")
    print(f"{_header(start)}  {elapsed:.1f}s   {tokens:,} tok")
    renderer = _Rows()
    rows = [row for ev in trace.events if (row := renderer.consume(ev)) is not None]
    for index, row in enumerate(rows):
        branch = "└─" if index == len(rows) - 1 else "├─"
        print(f"{branch} {row}")


def _plain(value: Any) -> Any:
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, BaseModel):
        # Fields are read one by one: a bulk dump would not let nested values be adapted.
        fields = type(value).model_fields
        data = {name: getattr(value, name) for name, field in fields.items() if not field.exclude}
        data.update(value.model_extra or {})
        if isinstance(value, Result):
            # The enclosing document already lists the events of this result.
            del data["trace"]
        return _plain(data)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_json(trace: Trace, path: str | Path | None = None) -> str:
    """Serialize a run's events to one JSON document and optionally write it to a file."""
    from .. import __version__

    document = {
        "golekit": __version__,
        "run_id": trace.events[0].run_id if trace.events else None,
        "events": [_plain(event) for event in trace.events],
    }
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text
