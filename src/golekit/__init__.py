"""Run language model agents with local or hosted models."""

from . import memory, paradigms, tools
from .agent import Agent
from .knowledge import Knowledge
from .model import Model
from .models.embedding import Embedding
from .protocols.types import Event, Result
from .safety import approval
from .tools import tool

__version__ = "0.3.0.dev0"
__all__ = [
    "Agent",
    "Embedding",
    "Event",
    "Knowledge",
    "Model",
    "Result",
    "__version__",
    "approval",
    "memory",
    "paradigms",
    "tool",
    "tools",
]
