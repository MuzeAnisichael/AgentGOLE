"""Keep unit tests independent of user model configuration."""

import pytest


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path, request):
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setenv("GOLE_CONFIG", str(tmp_path / "gole.yaml"))
        monkeypatch.delenv("GOLE_MODEL", raising=False)
