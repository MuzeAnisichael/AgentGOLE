"""Exceptions raised by AgentGOLE components."""


class AgentGoleError(Exception):
    """Base exception for AgentGOLE errors."""


class ConfigError(AgentGoleError):
    """A model or component could not be configured."""


class ModelError(AgentGoleError):
    """A model request failed."""


class UnsupportedToolsError(ModelError):
    """The server explicitly rejected native tools on a chat request."""


class ToolError(AgentGoleError):
    """A tool could not validate or execute a call."""


class MaxStepsError(AgentGoleError):
    """The run exhausted its action budget."""


class RunTimeoutError(AgentGoleError):
    """The run exceeded its time budget."""


class MissingDependencyError(AgentGoleError):
    """An optional dependency is missing."""
