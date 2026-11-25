"""Project configuration discovery and model defaults."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_logger = logging.getLogger("golekit.config")


def _expand(value: str) -> str:
    return _VARIABLE.sub(lambda match: os.environ.get(match[1], ""), value)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"Use a mapping with string keys for {field} in gole.yaml.")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"Use a string for {field} in gole.yaml.")
    return _expand(value)


def load_config() -> dict[str, Any]:
    """Read the first configuration file found and expand environment references."""
    paths = [Path.cwd() / "gole.yaml", Path.home() / ".gole" / "gole.yaml"]
    if override := os.environ.get("GOLE_CONFIG"):
        paths.insert(0, Path(override).expanduser())
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError):
        # YAML parser errors can quote the input line, including a secret.
        raise ConfigError(f"Check the YAML syntax and permissions of {path}.") from None
    config = _mapping({} if data is None else data, "the configuration")
    for key in config.keys() - {"default_model", "models", "providers", "engine", "tools"}:
        _logger.warning("Ignoring unknown gole.yaml key %r.", key)
    result: dict[str, Any] = {}
    if "engine" in config:
        if _string(config["engine"], "engine") != "native":
            raise ConfigError("Set engine to native in gole.yaml.")
        result["engine"] = "native"
    if "default_model" in config:
        result["default_model"] = _string(config["default_model"], "default_model")
    if "models" in config:
        result["models"] = {
            name: _string(value, f"models.{name}")
            for name, value in _mapping(config["models"], "models").items()
        }
    if "providers" in config:
        providers = {}
        for name, value in _mapping(config["providers"], "providers").items():
            fields = _mapping(value, f"providers.{name}")
            settings: dict[str, str] = {}
            if "base_url" in fields:
                settings["base_url"] = _string(fields["base_url"], f"providers.{name}.base_url")
            if "api_key" in fields:
                reference = fields["api_key"]
                if not isinstance(reference, str) or not _VARIABLE.fullmatch(reference):
                    raise ConfigError(
                        f"Reference an environment variable for providers.{name}.api_key "
                        "using ${VARIABLE_NAME}."
                    )
                settings["api_key"] = _expand(reference)
            providers[name] = settings
        result["providers"] = providers
    if "tools" in config:
        tools: dict[str, dict[str, str]] = {}
        for name, value in _mapping(config["tools"], "tools").items():
            fields = _mapping(value, f"tools.{name}")
            if "provider" in fields:
                provider = _string(fields["provider"], f"tools.{name}.provider")
                tools[name] = {"provider": provider}
        result["tools"] = tools
    return result


def default_model(config: dict[str, Any]) -> str:
    """Return the model selected by the environment or configuration file."""
    model = os.environ.get("GOLE_MODEL") or config.get("default_model")
    if not model or not isinstance(model, str):
        raise ConfigError("Set GOLE_MODEL or run 'gole init' to configure a default model.")
    return model
