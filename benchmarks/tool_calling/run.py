"""Run the tool-calling cases against real models and write RESULTS.md.

Usage:
    uv run python benchmarks/tool_calling/run.py --models ollama/qwen3:8b,ollama/qwen3:4b \\
        --modes native,prompt,constrained --out benchmarks/tool_calling/RESULTS.md

Every case runs once per model and mode at temperature 0. Replies that failed to parse or
triggered a retry are appended to failures.jsonl next to this file.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

import golekit as gole
from golekit.errors import GoleError
from golekit.protocols.types import Capabilities, Step, Trace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES, Call, Case

HERE = Path(__file__).resolve().parent
MODES = ("native", "prompt", "constrained")
_FORMAT_ERRORS = ("invalid arguments JSON", "ValidationError", "Unknown tool")


@dataclass
class Outcome:
    """What one case produced under one model and mode."""

    passed: bool
    model_calls: int
    retries: int
    seconds: float
    over_context: bool
    error: str | None = None


@dataclass
class Row:
    """Aggregated outcomes of one model and mode."""

    model: str
    mode: str
    supported: bool = True
    outcomes: dict[str, Outcome] = field(default_factory=dict)


def tool_calls(steps: list[Step]) -> list[Call]:
    """Flatten a run's tool steps into the calls the checks understand."""
    return [
        Call(call.name, call.arguments, result.error)
        for step in steps
        if step.action == "tool"
        for call, result in zip(step.calls, step.results, strict=True)
    ]


def format_failures(trace: Trace, protocol: str) -> list[dict[str, Any]]:
    """Collect replies that could not be parsed or had to be corrected."""
    found: list[dict[str, Any]] = []
    for event in trace.events:
        if event.type == "model_start":
            messages = event.messages
            last = messages[-1].content or "" if messages else ""
            if len(messages) > 1 and last.startswith("Error:") and "Reply again" in last:
                found.append({"protocol": protocol, "reply": messages[-2].content, "error": last})
        elif event.type == "tool_result" and event.error:
            if any(marker in event.error for marker in _FORMAT_ERRORS):
                reply = {"tool": event.tool, "arguments": event.args}
                found.append({"protocol": "native", "reply": reply, "error": event.error})
        elif event.type == "error" and event.recoverable and protocol != "native":
            # The paradigm gave up on a reply after its retries.
            found.append({"protocol": protocol, "reply": None, "error": str(event.exception)})
    return found


def run_case(case: Case, model: gole.Model, mode: str, log: Path) -> Outcome:
    """Run one case once and judge it."""
    tools = case.tools()
    agent = gole.Agent(model, tools=tools, memory=None, max_steps=12, timeout=300)
    started = monotonic()
    error: str | None = None
    try:
        result = agent.run(case.prompt)
        trace, steps, text = result.trace, result.steps, result.text
    except GoleError as exc:
        trace = getattr(exc, "trace", Trace())
        steps, text, error = [], "", f"{type(exc).__name__}: {exc}"
    seconds = monotonic() - started
    ends = [event for event in trace.events if event.type == "model_end"]
    failures = format_failures(trace, "native" if mode == "native" else mode)
    if failures:
        schemas = [tool.spec.model_dump() for tool in tools]
        with log.open("a", encoding="utf-8") as stream:
            for failure in failures:
                record = {"model": model._binding.client.name, "case": case.id, **failure}
                stream.write(json.dumps({**record, "tools": schemas}, ensure_ascii=False) + "\n")
    return Outcome(
        passed=error is None and case.check(tool_calls(steps), text),
        model_calls=len(ends),
        retries=len(failures),
        seconds=seconds,
        over_context=any((event.context_used or 0) >= 0.9 for event in ends),
        error=error,
    )


def build_model(spec: str, mode: str) -> gole.Model:
    """Configure a model so that exactly one tool-calling mode is exercised."""
    if mode == "native":
        return gole.Model(spec, temperature=0, tool_mode="native", timeout=300)
    return gole.Model(
        spec, temperature=0, tool_mode="prompt", constrained=mode == "constrained", timeout=300
    )


def observe_system_messages(spec: str) -> dict[str, Any]:
    """R9: ask one question with long-term memory, a summary and knowledge in context."""
    with tempfile.TemporaryDirectory(prefix="gole-bench-") as directory:
        root = Path(directory)
        (root / "policy.md").write_text("团建聚餐的报销上限是每人 150 元。", encoding="utf-8")
        kb = gole.Knowledge(root / "policy.md", store=root / "kb", retriever="keyword")
        memory = gole.memory.Layered(
            short=gole.memory.Summary(max_tokens=40, keep_turns=1),
            long=gole.memory.LongTerm(root / "facts.db"),
        )
        agent = gole.Agent(
            gole.Model(spec, temperature=0, timeout=300),
            instructions="你是公司的行政助理，回答要简短。",
            memory=memory,
            knowledge=kb,
            knowledge_mode="context",
            timeout=600,
        )
        try:
            agent.run("记住：我叫小林，我对花生过敏。", session="r9")
            agent.run("另外，我们组下周五要团建。", session="r9")
            result = agent.run("团建聚餐我个人要注意什么？每人能报销多少？", session="r9")
        except GoleError as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        request = next(e for e in result.trace.events if e.type == "model_start").messages
        return {
            "ok": True,
            "system_messages": sum(message.role == "system" for message in request),
            "used_memory": "花生" in result.text,
            "used_reference": "150" in result.text,
        }


def digest(spec: str) -> str:
    """Return the short digest of an Ollama model, or a dash for other providers."""
    provider, _, name = spec.partition("/")
    if provider != "ollama":
        return "—"
    try:
        models = httpx.get("http://localhost:11434/api/tags", timeout=5).json()["models"]
        return next(m["digest"][:12] for m in models if m["name"] == name)
    except (httpx.HTTPError, KeyError, ValueError, StopIteration, TypeError):
        return "未知"


def hardware() -> str:
    """Describe the machine in one line."""
    parts = [platform.platform(), platform.processor() or platform.machine()]
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        if gpu:
            parts.append(gpu.replace("\n", "; "))
    except (OSError, subprocess.SubprocessError):
        pass
    return " · ".join(parts)


def report(
    rows: list[Row], contexts: dict[str, int | None], observations: dict[str, dict[str, Any]]
) -> str:
    """Render the results as Markdown."""
    lines = [
        "# 工具调用基准结果",
        "",
        f"- 运行日期：{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        f"- GoleKit 版本：{gole.__version__}",
        f"- 硬件：{hardware()}",
        f"- 用例：{len(CASES)} 个，每个跑一次，temperature=0",
        "",
        "## 汇总",
        "",
        "| 模型 | 摘要 | 上下文 | 模式 | 成功率 | 平均模型调用次数 | 格式重试总数 | 平均耗时（秒） | 超限用例 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        context = contexts[row.model]
        head = f"| {row.model} | {digest(row.model)} | {context or '未检查上下文'} | {row.mode} |"
        if not row.supported:
            lines.append(f"{head} 不支持 | — | — | — | — |")
            continue
        counted = [o for o in row.outcomes.values() if not o.over_context]
        over = [case for case, o in row.outcomes.items() if o.over_context]
        runs = list(row.outcomes.values())
        passed = sum(o.passed for o in counted)
        rate = f"{passed}/{len(counted)}（{passed / len(counted):.0%}）" if counted else "—"
        lines.append(
            f"{head} {rate} | {sum(o.model_calls for o in runs) / len(runs):.1f} "
            f"| {sum(o.retries for o in runs)} | {sum(o.seconds for o in runs) / len(runs):.1f} "
            f"| {', '.join(over) or '无'} |"
        )
    lines += [
        "",
        "上下文用量达到 90% 的用例列在「超限用例」里，不计入成功率。",
        "",
        "## 逐用例结果",
        "",
    ]
    active = [row for row in rows if row.supported]
    lines.append("| 用例 | 类别 | " + " | ".join(f"{r.model} {r.mode}" for r in active) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in active) + " |")
    for case in CASES:
        marks = []
        for row in active:
            outcome = row.outcomes[case.id]
            marks.append("超限" if outcome.over_context else ("通过" if outcome.passed else "失败"))
        lines.append(f"| {case.id} | {case.category} | " + " | ".join(marks) + " |")
    lines += [
        "",
        "## R9 观察：多条系统消息",
        "",
        '每个模型用与 `memory="long"` 等价的分层记忆，加一个知识库（context 模式），各问一次。',
        "",
        "| 模型 | 请求成功 | 系统消息条数 | 用到了记忆 | 用到了参考资料 | 备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model, seen in observations.items():
        if seen["ok"]:
            lines.append(
                f"| {model} | 是 | {seen['system_messages']} | {'是' if seen['used_memory'] else '否'} "
                f"| {'是' if seen['used_reference'] else '否'} | |"
            )
        else:
            lines.append(f"| {model} | 否 | — | — | — | {seen['detail']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the selected models and modes and write the report."""
    parser = argparse.ArgumentParser(description="Run the tool-calling benchmark.")
    parser.add_argument("--models", required=True, help="comma-separated model specs")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--out", default=str(HERE / "RESULTS.md"))
    args = parser.parse_args(argv)
    modes = [mode for mode in args.modes.split(",") if mode]
    if unknown := set(modes) - set(MODES):
        parser.error(f"unknown modes: {', '.join(sorted(unknown))}")
    log = HERE / "failures.jsonl"
    rows: list[Row] = []
    contexts: dict[str, int | None] = {}
    observations: dict[str, dict[str, Any]] = {}
    for spec in [item.strip() for item in args.models.split(",") if item.strip()]:
        capabilities: Capabilities = gole.Model(spec).capabilities
        contexts[spec] = capabilities.context
        for mode in modes:
            row = Row(spec, mode)
            rows.append(row)
            if mode == "native" and not capabilities.tools:
                row.supported = False
                continue
            model = build_model(spec, mode)
            if mode == "constrained" and not model.capabilities.json_schema:
                row.supported = False
                continue
            for case in CASES:
                row.outcomes[case.id] = run_case(case, model, mode, log)
                print(
                    f"{spec} {mode} {case.id}: {'pass' if row.outcomes[case.id].passed else 'FAIL'}"
                )
            if mode == "constrained" and not model.capabilities.json_schema:
                # The server refused the response format, so these runs were plain prompt mode.
                row.supported = False
        observations[spec] = observe_system_messages(spec)
    Path(args.out).write_text(report(rows, contexts, observations), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
