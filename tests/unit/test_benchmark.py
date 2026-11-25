"""The tool-calling benchmark's checks and bookkeeping, driven by scripted models."""

import importlib.util
import json
import sys
from pathlib import Path

import golekit as gole
from golekit.testing import FakeModel, fake_tool_call

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "tool_calling"


def load(name: str):
    sys.path.insert(0, str(BENCH))
    try:
        spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCH))


def test_cases_cover_five_categories_and_judge_call_sequences():
    cases, run = load("cases"), load("run")
    by_category: dict[str, int] = {}
    for case in cases.CASES:
        by_category[case.category] = by_category.get(case.category, 0) + 1
    assert list(by_category.values()) == [4, 4, 4, 4, 4]
    chain = next(case for case in cases.CASES if case.id == "chain-manager")
    scripts = {
        True: [("find_employee", {"name": "王芳"}), ("get_manager", {"employee_id": "E-2087"})],
        False: [("get_manager", {"employee_id": "E-2087"})],
    }
    for expected, calls in scripts.items():
        model = FakeModel([*(fake_tool_call(*call) for call in calls), "她的经理是赵敏。"])
        result = gole.Agent(model, tools=chain.tools(), memory=None).run(chain.prompt)
        assert chain.check(run.tool_calls(result.steps), result.text) is expected
    flaky = next(case for case in cases.CASES if case.id == "recover-report")
    call = fake_tool_call("fetch_report", {"key": "R-17"})
    result = gole.Agent(FakeModel([call, call, "营收增长 12%"]), tools=flaky.tools()).run("x")
    assert flaky.check(run.tool_calls(result.steps), result.text)
    assert not flaky.check([], "营收增长 12%")


def test_run_case_counts_retries_and_logs_the_failed_reply(tmp_path, monkeypatch):
    run = load("run")
    fake = FakeModel(
        [
            '{"tool": "get_weather", "arguments": {"town": "上海"}}',
            'Action:\n{"tool": "get_weather", "arguments": {"city": "上海"}}',
            "Final Answer: 上海多云，28°C",
        ],
        tools_supported=False,
        name="fake/model",
    )
    monkeypatch.setattr("golekit.model.create_client", lambda *args, **kwargs: fake)
    case = next(case for case in run.CASES if case.id == "single-weather")
    log = tmp_path / "failures.jsonl"
    outcome = run.run_case(case, run.build_model("ollama/fake", "prompt"), "prompt", log)
    assert outcome.passed and outcome.model_calls == 3 and outcome.retries == 1
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["protocol"] == "prompt" and '"town"' in record["reply"]
    assert record["case"] == "single-weather" and record["tools"][0]["name"] == "get_weather"
    row = run.Row("ollama/fake", "prompt", outcomes={c.id: outcome for c in run.CASES})
    monkeypatch.setattr(run, "digest", lambda spec: "abc123")
    text = run.report(
        [row, run.Row("ollama/fake", "native", supported=False)], {"ollama/fake": None}, {}
    )
    assert "| ollama/fake | abc123 | 未检查上下文 | prompt | 20/20（100%） | 3.0 | 20 |" in text
    assert "| native | 不支持 |" in text
