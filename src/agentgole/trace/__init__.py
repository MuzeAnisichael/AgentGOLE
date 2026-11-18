"""Plain-text run trees for replay and live output."""

import json
import sys

from ..protocols.types import Event, Trace


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
