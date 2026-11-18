"""Run asynchronous agent operations from synchronous Python, including notebooks."""

import asyncio
import threading
from collections.abc import AsyncIterator, Coroutine, Generator
from concurrent.futures import CancelledError, Future
from contextlib import suppress
from queue import SimpleQueue
from typing import Any, TypeVar

T = TypeVar("T")
_workers: list[asyncio.AbstractEventLoop] = []
_worker_lock = threading.Lock()
_worker_thread = threading.local()


def _start_loop(ready: Future[asyncio.AbstractEventLoop], depth: int) -> None:
    # A synchronous call made by a callback already on this worker needs another loop.
    _worker_thread.depth = depth + 1
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        loop.call_soon(ready.set_result, loop)
        loop.run_forever()


def _background_loop() -> asyncio.AbstractEventLoop:
    depth = getattr(_worker_thread, "depth", 0)
    with _worker_lock:
        if depth == len(_workers):
            ready: Future[asyncio.AbstractEventLoop] = Future()
            threading.Thread(
                target=_start_loop, args=(ready, depth), name=f"agentgole-sync-{depth}", daemon=True
            ).start()
            _workers.append(ready.result())
        return _workers[depth]


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Wait for a coroutine on a suitable event loop and propagate its result."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
        try:
            return future.result()
        finally:
            if not future.done():
                future.cancel()
    return asyncio.run(coro)


def iter_sync(source: AsyncIterator[T]) -> Generator[T, None, None]:
    """Pull items from a background producer and close it when iteration stops."""
    loop = _background_loop()
    output: SimpleQueue[tuple[T] | None] = SimpleQueue()
    demand = asyncio.Event()
    started: Future[asyncio.Task[Any]] = Future()

    async def produce() -> None:
        task = asyncio.current_task()
        assert task is not None
        started.set_result(task)
        try:
            while True:
                await demand.wait()
                demand.clear()
                try:
                    item = await anext(source)
                except StopAsyncIteration:
                    return
                output.put((item,))
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    producer = asyncio.run_coroutine_threadsafe(produce(), loop)
    producer.add_done_callback(lambda future: output.put(None))
    try:
        while True:
            loop.call_soon_threadsafe(demand.set)
            entry = output.get()
            if entry is None:
                producer.result()
                return
            yield entry[0]
    finally:
        if not producer.done():
            # Cancel the task itself: its future then acknowledges async cleanup.
            loop.call_soon_threadsafe(started.result().cancel)
            with suppress(CancelledError):
                producer.result()
