"""Exercise v0.2 against the local Ollama service, with cloud credentials unset."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import BaseModel

import agentgole as gole
from agentgole.config import load_config

pytestmark = pytest.mark.live
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def local_models():
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
    return host, names


@pytest.fixture(autouse=True)
def offline_config(local_models, tmp_path, monkeypatch):
    host, _ = local_models
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "GOLE_EMBEDDING"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "gole.yaml"
    path.write_text(yaml.safe_dump({"providers": {"ollama": {"base_url": host}}}), encoding="utf-8")
    monkeypatch.setenv("GOLE_CONFIG", str(path))
    monkeypatch.setenv("PYTHONUTF8", "1")


def test_offline_expense_approvals(local_models):
    if not any(name.split(":")[0] == "bge-m3" for name in local_models[1]):
        pytest.skip("Install bge-m3 with ollama pull.")
    result = runpy.run_path(str(EXAMPLES / "knowledge_qa.py"))["main"]()
    assert "经理" in result.text and "财务负责人" in result.text
    assert any(Path(c.source).name == "approvals.md" for c in result.citations)


def test_persistent_meeting_across_processes():
    script = str(EXAMPLES / "assistant_memory.py")
    for question in ("记住：我周三下午固定开组会", "我哪天下午固定开组会？"):
        reply = subprocess.run(
            [sys.executable, script, question],
            check=True,
            capture_output=True,
            encoding="utf-8",
            env=os.environ.copy(),
            timeout=360,
        )
    assert "周三" in reply.stdout or "星期三" in reply.stdout


def test_structured_paper():
    class Paper(BaseModel):
        title: str
        year: int
        keywords: list[str]

    result = gole.Agent("ollama/qwen3:8b").run(
        "从摘要中提取：论文题目为海岸调查，发表于2024年，关键词为海岸、盐度。", output=Paper
    )
    assert isinstance(result.output, Paper)
    assert result.output.year == 2024 and "海岸" in result.output.title
