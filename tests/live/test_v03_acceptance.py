"""Exercise v0.3 against the local Ollama service and, when a key is set, DeepSeek.

Run with GOLE_TRACE_DIR set; when it is not, traces go to traces/v0.3 in the repository.
Send that directory back when a test fails.
"""

import os
import runpy
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import BaseModel

import golekit as gole
from golekit.config import load_config

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


@pytest.fixture(scope="module")
def ollama_host():
    provider = load_config().get("providers", {}).get("ollama", {})
    host = provider.get("base_url", "http://localhost:11434").rstrip("/").removesuffix("/v1")
    try:
        response = httpx.get(f"{host}/api/tags", timeout=4, trust_env=False)
        response.raise_for_status()
        names = {entry["name"] for entry in response.json()["models"]}
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        pytest.skip(f"Ollama unavailable ({type(exc).__name__}).")
    if "qwen3:8b" not in names:
        pytest.skip("Install qwen3:8b with ollama pull.")
    return host


@pytest.fixture(autouse=True)
def local_setup(ollama_host, tmp_path, monkeypatch):
    traces = Path(os.environ.get("GOLE_TRACE_DIR") or REPO / "traces" / "v0.3").resolve()
    monkeypatch.setenv("GOLE_TRACE_DIR", str(traces))
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "GOLE_MODEL", "GOLE_EMBEDDING"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "gole.yaml"
    config = {"providers": {"ollama": {"base_url": ollama_host}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("GOLE_CONFIG", str(path))
    monkeypatch.setenv("PYTHONUTF8", "1")


def test_plan_example_uses_tools_offline(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = runpy.run_path(str(EXAMPLES / "l3_plan.py"))["main"]()
    tools = [call.name for step in result.steps for call in step.calls]
    assert "calculator" in tools and "21625" in result.text.replace(",", "")
    assert any(e.type == "step" and e.thought for e in result.trace.events)


def test_reflect_example_offline(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = runpy.run_path(str(EXAMPLES / "l3_reflect.py"))["main"]()
    assert "三楼" in result.text and "工牌" in result.text
    assert sum(e.type == "model_start" for e in result.trace.events) >= 2


def test_ollama_accepts_json_schema_response_format():
    class Paper(BaseModel):
        title: str
        year: int

    model = gole.Model("ollama/qwen3:8b", temperature=0)
    result = gole.Agent(model).run("提取：论文题目为海岸调查，发表于2024年。", output=Paper)
    assert isinstance(result.output, Paper) and result.output.year == 2024
    # A rejected response format would be reported as a recoverable error and remembered.
    assert not [e for e in result.trace.events if e.type == "error"]
    assert model.capabilities.json_schema


def test_constrained_prompt_mode_calls_a_tool():
    model = gole.Model("ollama/qwen3:8b", temperature=0, tool_mode="prompt", constrained=True)
    result = gole.Agent(model, tools=["calculator"]).run(
        "用 calculator 计算 247 * 13，告诉我结果。"
    )
    tools = [call.name for step in result.steps for call in step.calls]
    assert tools and set(tools) == {"calculator"} and "3211" in result.text.replace(",", "")
    assert model.capabilities.json_schema


def test_spec_l3_example_with_web_tools():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("Set DEEPSEEK_API_KEY to run the online L3 example.")
    agent = gole.Agent(
        model="deepseek/deepseek-chat",
        tools=["web_search", "web_fetch"],
        paradigm="plan",
        max_steps=40,
    )
    result = agent.run("调研三款开源向量数据库，比较部署难度并给出推荐")
    tools = {call.name for step in result.steps for call in step.calls}
    assert "web_search" in tools and len(result.text) > 100
