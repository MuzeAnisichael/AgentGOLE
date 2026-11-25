"""Route model strings to the shared OpenAI-compatible transport."""

import os
from typing import Any

import httpx

from ..errors import ConfigError
from .client import CompatibleClient

_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "vllm": "",
}
_KEYS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
}


def create_client(
    spec: str,
    *,
    config: dict[str, Any],
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 60.0,
    retries: int = 2,
    transport: httpx.AsyncBaseTransport | None = None,
    embedding: bool = False,
    constrained: bool | None = None,
) -> CompatibleClient:
    """Resolve the model alias, provider endpoint and credentials without network I/O."""
    if "/" not in spec and base_url is None:
        alias = config.get("models", {}).get(spec)
        if not isinstance(alias, str) or "/" not in alias:
            raise ConfigError(f"Unknown model alias {spec!r}. Set GOLE_MODEL or run 'gole init'.")
        spec = alias
    provider, slash, model = spec.partition("/")
    if not slash:
        provider, model = "custom", spec
    elif provider not in _ENDPOINTS:
        raise ConfigError(f"Provider {provider!r} is not supported in this version.")
    if provider == "deepseek" and embedding:
        raise ConfigError(
            "DeepSeek does not serve embeddings. Choose another provider, such as Ollama."
        )
    if not model.strip():
        raise ConfigError("Select a model using provider/model, or run 'gole init'.")
    if retries < 0 or timeout <= 0:
        raise ConfigError("Set retries to zero or more and timeout to a positive number.")
    options = config.get("providers", {}).get(provider, {})
    endpoint = base_url or options.get("base_url") or _ENDPOINTS.get(provider)
    if not endpoint:
        raise ConfigError(f"Set base_url or providers.{provider}.base_url for this model.")
    try:
        url = httpx.URL(endpoint)
    except (httpx.InvalidURL, TypeError):
        raise ConfigError(f"Set a valid HTTP base_url for {provider}.") from None
    if not url.host or url.scheme not in {"http", "https"}:
        raise ConfigError(f"Set a valid HTTP base_url for {provider}.")
    endpoint = endpoint.rstrip("/")
    if provider == "ollama" and not endpoint.endswith("/v1"):
        endpoint += "/v1"
    key_name = _KEYS.get(provider)
    key = (
        api_key
        if api_key is not None
        else ((os.environ.get(key_name) if key_name else None) or options.get("api_key"))
    )
    if key_name and not key:
        raise ConfigError(f"Set {key_name} to use the {provider} provider.")
    defaults = {}
    if temperature is not None:
        defaults["temperature"] = temperature
    if max_tokens is not None:
        defaults["max_tokens"] = max_tokens
    return CompatibleClient(
        provider, model, spec, endpoint, key, defaults, timeout, retries, transport, constrained
    )
