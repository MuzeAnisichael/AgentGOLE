import logging
from pathlib import Path

import pytest

from golekit.config import default_model, load_config
from golekit.errors import ConfigError


@pytest.fixture
def project(tmp_path, monkeypatch):
    cwd = tmp_path / "project"
    home = tmp_path / "user"
    cwd.mkdir()
    (home / ".gole").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("GOLE_CONFIG", raising=False)
    monkeypatch.delenv("GOLE_MODEL", raising=False)
    return cwd, home


def test_discovery_order_and_environment_default(project, monkeypatch):
    cwd, home = project
    (home / ".gole" / "gole.yaml").write_text("default_model: home")
    assert default_model(load_config()) == "home"
    (cwd / "gole.yaml").write_text("default_model: local")
    assert default_model(load_config()) == "local"
    override = home / "custom.yaml"
    override.write_text("default_model: custom")
    monkeypatch.setenv("GOLE_CONFIG", str(override))
    assert default_model(load_config()) == "custom"
    monkeypatch.setenv("GOLE_MODEL", "from-env")
    assert default_model(load_config()) == "from-env"


def test_aliases_providers_and_environment_references(project, monkeypatch):
    cwd, _ = project
    monkeypatch.setenv("GOLE_TEST_MODEL", "qwen")
    monkeypatch.setenv("GOLE_TEST_KEY", "value-from-env")
    (cwd / "gole.yaml").write_text(
        "default_model: small\nmodels:\n  small: ollama/${GOLE_TEST_MODEL}\n"
        "providers:\n  deepseek:\n    api_key: ${GOLE_TEST_KEY}\n"
        "  ollama:\n    base_url: http://localhost:11434"
    )
    config = load_config()
    assert default_model(config) == "small"
    assert config["models"]["small"] == "ollama/qwen"
    assert config["providers"] == {
        "deepseek": {"api_key": "value-from-env"},
        "ollama": {"base_url": "http://localhost:11434"},
    }


def test_optional_unset_key_and_unknown_fields(project, monkeypatch, caplog):
    cwd, _ = project
    monkeypatch.delenv("GOLE_TEST_ABSENT", raising=False)
    (cwd / "gole.yaml").write_text(
        "engine: native\nextra_feature: enabled\n"
        "providers:\n  deepseek:\n    api_key: ${GOLE_TEST_ABSENT}"
    )
    with caplog.at_level(logging.WARNING, logger="golekit.config"):
        assert load_config() == {"engine": "native", "providers": {"deepseek": {"api_key": ""}}}
    assert len(caplog.records) == 1 and "extra_feature" in caplog.text


def test_missing_config_has_actionable_error_only_on_model_selection(project):
    assert load_config() == {}
    with pytest.raises(ConfigError, match="GOLE_MODEL.*gole init"):
        default_model({})


def test_literal_key_rejected_without_echoing_value(project):
    cwd, _ = project
    (cwd / "gole.yaml").write_text("providers:\n  deepseek:\n    api_key: file-secret")
    with pytest.raises(ConfigError, match="environment variable") as err:
        load_config()
    assert "file-secret" not in str(err.value)
