"""Public agent facade."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from uuid import uuid4

from ._sync import iter_sync, run_sync
from .config import default_model, load_config
from .errors import ConfigError
from .kernel.engine import execute
from .memory.window import Window
from .model import Model, _Binding, _check_mode
from .models import create_client
from .paradigms import Auto, React
from .protocols import EventCallback, EventSink, Memory, ModelClient, Paradigm, StepContext, Tool
from .protocols.types import Event, Result, ToolMode
from .tools import resolve_tools
from .trace import LiveTrace

logger = logging.getLogger("agentgole.agent")


class Agent:
    """Run a model with tools and optional session memory."""

    def __init__(
        self,
        model: str | Model | ModelClient | None = None,
        *,
        instructions: str | None = None,
        tools: Iterable[str | Tool] = (),
        memory: str | Memory | None = "session",
        paradigm: str | Paradigm = "auto",
        name: str | None = None,
        max_steps: int = 20,
        timeout: float | None = None,
        tool_mode: ToolMode | None = None,
        verbose: bool = False,
    ) -> None:
        _check_mode(tool_mode)
        if isinstance(model, Model):
            self._binding = model._binding
        elif model is None or isinstance(model, str):
            config = load_config()
            spec = default_model(config) if model is None else model
            self._binding = _Binding(create_client(spec, config=config), "auto")
        elif isinstance(model, ModelClient):
            self._binding = _Binding(model, "auto")
        else:
            raise ConfigError("Pass a model name, Model or asynchronous ModelClient.")
        self._memory: Memory | None
        if isinstance(memory, str):
            if memory != "session":
                raise ConfigError("Choose memory='session', None or a Memory object.")
            self._memory = Window()
        else:
            self._memory = memory
        self._paradigm: Paradigm
        if isinstance(paradigm, str):
            if paradigm not in ("auto", "react"):
                raise ConfigError("Choose paradigm='auto', 'react' or a Paradigm object.")
            self._paradigm = Auto() if paradigm == "auto" else React()
        else:
            self._paradigm = paradigm
        self.name = name
        self._instructions = instructions
        try:
            self._tools = resolve_tools(tools)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self._max_steps = max_steps
        self._timeout = timeout
        self._mode = tool_mode
        self._verbose = verbose
        self._listeners: dict[str, list[EventCallback]] = {}

    def run(self, input: str, *, session: str = "default") -> Result:
        """Run to completion and return the result."""
        return run_sync(self.arun(input, session=session))

    async def arun(self, input: str, *, session: str = "default") -> Result:
        """Run asynchronously to completion."""
        return await self._execute(input, session, streaming=False)

    def stream(
        self, input: str, *, session: str = "default", events: list[str] | None = None
    ) -> Iterator[Event]:
        """Iterate over events as they happen."""
        return iter_sync(self.astream(input, session=session, events=events))

    async def astream(
        self, input: str, *, session: str = "default", events: list[str] | None = None
    ) -> AsyncIterator[Event]:
        """Iterate asynchronously over events as they happen."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        selected = None if events is None else set(events)

        async def enqueue(event: Event) -> None:
            if selected is None or event.type in selected:
                await queue.put(event)

        async def produce() -> None:
            try:
                await self._execute(input, session, streaming=True, sink=enqueue)
            finally:
                queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        try:
            while (event := await queue.get()) is not None:
                yield event
            await producer
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    def on(self, event_type: str) -> Callable[[EventCallback], EventCallback]:
        """Register a callback for an event type or '*'."""

        def register(callback: EventCallback) -> EventCallback:
            self._listeners.setdefault(event_type, []).append(callback)
            return callback

        return register

    async def _execute(
        self, input: str, session: str, *, streaming: bool, sink: EventSink | None = None
    ) -> Result:
        run_id = uuid4().hex
        live = LiveTrace() if self._verbose else None

        async def emit(event: Event) -> None:
            if live is not None:
                live.write(event)
            for callback in (*self._listeners.get(event.type, ()), *self._listeners.get("*", ())):
                try:
                    value = callback(event)
                    if inspect.isawaitable(value):
                        await value
                except Exception:
                    logger.exception("Callback for %s failed", event.type)
            if sink is not None:
                await sink(event)

        async def report_error(error: Exception) -> None:
            await emit(Event(type="error", run_id=run_id, exception=error, recoverable=True))

        ctx = StepContext(
            model=self._binding.client,
            tools=self._tools,
            tool_mode="native",
            run_id=run_id,
            report_error=report_error,
            downgrade=self._binding.downgrade,
        )
        return await execute(
            input,
            session=session,
            instructions=self._instructions,
            memory=self._memory,
            paradigm=self._paradigm,
            ctx=ctx,
            max_steps=self._max_steps,
            timeout=self._timeout,
            streaming=streaming,
            emit=emit,
            resolve_mode=lambda: self._binding.resolve_mode(self._mode),
        )
