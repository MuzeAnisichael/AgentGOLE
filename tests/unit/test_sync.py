import asyncio
import threading
from contextlib import closing

import pytest

from agentgole._sync import iter_sync, run_sync


async def identity():
    return threading.get_ident(), asyncio.get_running_loop()


def test_run_without_a_loop_uses_calling_thread():
    thread, loop = run_sync(identity())
    assert thread == threading.get_ident() and loop.is_closed()


async def test_running_loop_uses_a_reusable_worker_and_allows_reentry():
    first = run_sync(identity())
    assert run_sync(identity()) == first and first[0] != threading.get_ident()

    async def reenter():
        return run_sync(identity())

    assert run_sync(reenter())[0] != first[0]


def test_stream_early_close_waits_for_async_cleanup():
    events = []

    async def items():
        try:
            yield 1
            events.append("unexpected second pull")
            yield 2
        finally:
            await asyncio.sleep(0)
            events.append("closed")

    with closing(iter_sync(items())) as stream:
        assert next(stream) == 1
    assert events == ["closed"]


def test_stream_completion_and_producer_exception():
    async def items(fail):
        yield None
        if fail:
            raise ValueError("source failed")
        yield "last"

    assert list(iter_sync(items(False))) == [None, "last"]
    stream = iter_sync(items(True))
    assert next(stream) is None
    with pytest.raises(ValueError, match="source failed"):
        next(stream)
