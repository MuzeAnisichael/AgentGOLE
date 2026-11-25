"""Contracts for scripted model responses used by component tests."""

import pytest

from golekit.protocols.types import Message
from golekit.testing import FakeModel, fake_tool_call


async def test_scripted_calls_keep_a_snapshot_and_fail_when_exhausted():
    calls = [fake_tool_call("a", {}), fake_tool_call("b", {"x": 2})]
    model = FakeModel([calls])
    messages = [Message(role="user", content="first")]
    reply = await model.generate(messages)
    messages[0].content = "later"
    assert reply.message.tool_calls == calls
    assert model.requests[0]["messages"][0].content == "first"
    with pytest.raises(AssertionError, match="exhausted"):
        await model.generate(messages)


async def test_stream_emits_deltas_then_a_complete_reply():
    model = FakeModel(["scripted text"], tools_supported=False)
    chunks = [chunk async for chunk in model.stream([])]
    assert len(chunks) > 2
    assert "".join(chunk.text for chunk in chunks) == "scripted text"
    assert chunks[-1].response and chunks[-1].response.message.content == "scripted text"
    assert not (await model.probe()).tools


async def test_embedding_batches_are_reproducible_and_tokens_handle_mixed_text():
    from golekit.kernel.context import estimate_tokens
    from golekit.testing import FakeEmbedding

    model = FakeEmbedding(7)
    batch = await model.embed(["上海", "notes"])
    assert batch[0] == (await FakeEmbedding(7).embed(["上海"]))[0]
    assert len(batch[0]) == 7 and batch[0] != batch[1]
    assert model.calls == [["上海", "notes"]]
    assert estimate_tokens("会议notes") == 4
    assert estimate_tokens("😊😊😊😊") == 1
