"""Exceptions raised by GoleKit components."""


class GoleError(Exception):
    """Base exception for GoleKit errors."""


class ConfigError(GoleError):
    """A model or component could not be configured."""


class ModelError(GoleError):
    """A model request failed."""


class UnsupportedToolsError(ModelError):
    """The server explicitly rejected native tools on a chat request."""


class ToolError(GoleError):
    """A tool could not validate or execute a call."""


class MaxStepsError(GoleError):
    """The run exhausted its action budget."""


class RunTimeoutError(GoleError):
    """The run exceeded its time budget."""


class MissingDependencyError(GoleError):
    """An optional dependency is missing."""


class OutputValidationError(GoleError):
    """The final response still fails its requested schema after corrections."""
