"""Plain-text run traces."""

from ..protocols.types import Trace


def show(trace: Trace) -> None:
    """Print the steps of a completed run."""
    raise NotImplementedError
