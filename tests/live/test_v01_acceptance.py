"""Run the example programs against each available real provider once."""

import os
import runpy
from pathlib import Path

import httpx
import pytest

import golekit as gole
from golekit.config import load_config

pytestmark = pytest.mark.live
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
OLLAMA = "ollama/qwen3:8b"
DEEPSEEK = "deepseek/deepseek-chat"


@pytest.fixture(scope="session")
def ollama_ready():
    provider = load_config().get("providers", {}).get("ollama", {})
    host = provider.get("base_url", "http://localhost:11434").rstrip("/").removesuffix("/v1")
    try:
        response = httpx.get(f"{host}/api/tags", timeout=4, trust_env=False)
        response.raise_for_status()
        names = {entry["name"] for entry in response.json()["models"]}
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return False, f"The configured Ollama endpoint is unavailable ({type(exc).__name__})."
    return "qwen3:8b" in names, "Install the live model with 'ollama pull qwen3:8b'."


def _require(spec, request):
    if spec == OLLAMA:
        available, reason = request.getfixturevalue("ollama_ready")
        if not available:
            pytest.skip(reason)
    elif not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("Set DEEPSEEK_API_KEY to run the DeepSeek acceptance tests.")


@pytest.mark.parametrize("spec", [OLLAMA, DEEPSEEK])
def test_quickstart(spec, request, monkeypatch, capsys):
    _require(spec, request)
    monkeypatch.setenv("GOLE_MODEL", spec)
    runpy.run_path(str(EXAMPLES / "quickstart.py"))
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("spec", [OLLAMA, DEEPSEEK])
def test_tools_example(spec, request, monkeypatch):
    _require(spec, request)
    monkeypatch.setenv("GOLE_MODEL", spec)
    result = runpy.run_path(str(EXAMPLES / "l1_tools.py"))["result"]
    names = {ev.tool for ev in result.trace.events if ev.type == "tool_call"}
    assert {"get_weather", "calculator"} <= names
    assert "18" in result.text and "3211" in result.text.replace(",", "")


def test_ollama_forced_prompt(request):
    _require(OLLAMA, request)
    agent = gole.Agent(OLLAMA, tools=["calculator"], tool_mode="prompt")
    result = agent.run("必须调用 calculator 计算 247 * 13，并告诉我计算结果。")
    assert any(ev.type == "tool_call" and ev.tool == "calculator" for ev in result.trace.events)
    assert "3211" in result.text.replace(",", "")
