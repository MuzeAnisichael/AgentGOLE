from agentgole.memory import Window
from agentgole.protocols.types import Message


def turn(value: str) -> list[Message]:
    return [Message(role="user", content=value), Message(role="assistant", content=value + "!")]


async def test_window_keeps_sessions_isolated():
    window = Window()
    await window.save("a", turn("one"))
    await window.save("b", turn("two"))
    assert await window.load("a", query="anything") == turn("one")
    assert await window.load("b") == turn("two")
    assert await window.load("new") == []


async def test_window_evicts_complete_turns_at_default_bound():
    window = Window()
    for i in range(21):
        await window.save("s", turn(str(i)))
    assert await window.load("s") == [message for i in range(1, 21) for message in turn(str(i))]


async def test_window_returns_independent_snapshots_and_copies_inputs():
    window = Window(turns=1)
    original = turn("one")
    await window.save("s", original)
    original[0].content = "edited"
    snapshot = await window.load("s")
    snapshot[1].content = "also edited"
    assert await window.load("s") == turn("one")
    await window.save("s", turn("two"))
    assert await window.load("s") == turn("two")
