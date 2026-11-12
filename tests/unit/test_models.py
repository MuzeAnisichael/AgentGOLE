import json

import httpx
import pytest

from agentgole.errors import ConfigError, ModelError, UnsupportedToolsError
from agentgole.models import create_client
from agentgole.protocols.types import Message, ToolCall, ToolSpec


def answer(content="ok"):
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


async def test_routes_and_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "regional")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = {
        "models": {"smart": "deepseek/m"},
        "providers": {
            "deepseek": {"api_key": "reference"},
            "openai": {"api_key": "shadowed"},
            "ollama": {"base_url": "http://localhost:11434"},
            "vllm": {"base_url": "http://vllm.test/v1"},
        },
    }
    cases = [
        ("smart", None, None, "https://api.deepseek.com", "reference", True),
        ("openai/m", None, None, "https://api.openai.com/v1", "environment", True),
        (
            "dashscope/m",
            None,
            None,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "regional",
            True,
        ),
        ("lmstudio/m", None, None, "http://localhost:1234/v1", None, False),
        ("vllm/m", None, None, "http://vllm.test/v1", None, False),
        ("m", "http://custom.test/v1", "explicit", "http://custom.test/v1", "explicit", False),
        ("ollama/m", None, None, "http://localhost:11434/v1", None, False),
        (
            "openai/m",
            "http://override.test/v1",
            "explicit",
            "http://override.test/v1",
            "explicit",
            True,
        ),
    ]
    for spec, base, key, expected_base, expected_key, native in cases:

        def serve(req, expected_base=expected_base, expected_key=expected_key):
            if req.url.path.endswith("/api/show"):
                return httpx.Response(200, json={})
            assert str(req.url) == expected_base + "/chat/completions"
            assert req.headers.get("authorization") == (
                f"Bearer {expected_key}" if expected_key else None
            )
            assert json.loads(req.content)["model"] == "m"
            assert json.loads(req.content)["max_tokens"] == 1
            return answer()

        client = create_client(
            spec, config=config, base_url=base, api_key=key, transport=httpx.MockTransport(serve)
        )
        assert (await client.probe()).tools is native
        await client.check_connection()


def test_unconfigured_models_raise_actionable_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for model, hint in [
        ("anthropic/m", "not supported"),
        ("other/m", "not supported"),
        ("vllm/m", "base_url"),
        ("no-alias", "gole init"),
        ("openai/m", "OPENAI_API_KEY"),
    ]:
        with pytest.raises(ConfigError, match=hint):
            create_client(model, config={})


async def test_request_and_complete_response_mapping():
    history = [
        Message(role="system", content="system"),
        Message(
            role="assistant",
            reasoning="earlier secret thoughts",
            tool_calls=[ToolCall(id="c", name="add", arguments={"n": 1})],
        ),
        Message(role="tool", tool_call_id="c", name="add", content="2"),
    ]
    spec = ToolSpec(name="add", description="Add one.", parameters={"type": "object"})

    def serve(req):
        body = json.loads(req.content)
        assert body["temperature"] == 0.1 and body["max_tokens"] == 12 and not body["stream"]
        assert body["tools"] == [{"type": "function", "function": spec.model_dump()}]
        assistant = body["messages"][1]
        assert "reasoning" not in assistant
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"n": 1}
        assert body["messages"][2] == {
            "role": "tool",
            "content": "2",
            "tool_call_id": "c",
            "name": "add",
        }
        return httpx.Response(
            200,
            json={
                "model": "resolved",
                "choices": [
                    {
                        "message": {
                            "content": "<think>private</think>\n\nanswer",
                            "tool_calls": [
                                {
                                    "id": "second",
                                    "function": {"name": "add", "arguments": '{"n": 2}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    client = create_client(
        "lmstudio/m",
        config={},
        temperature=0.4,
        max_tokens=12,
        transport=httpx.MockTransport(serve),
    )
    result = await client.generate(history, tools=[spec], temperature=0.1)
    assert (result.message.content, result.message.reasoning) == ("answer", "private")
    assert result.message.tool_calls == [ToolCall(id="second", name="add", arguments={"n": 2})]
    assert result.model == "resolved" and result.usage.total_tokens == 5


async def test_stream_reassembles_tools_by_index_and_only_emits_visible_text():
    deltas = [
        {"content": "<th"},
        {"content": "ink>secret</thi"},
        {"content": "nk>\n"},
        {
            "content": "v",
            "tool_calls": [
                {"index": 1, "id": "B", "function": {"name": "oth", "arguments": '{"q":'}}
            ],
        },
        {
            "content": "isible",
            "tool_calls": [{"index": 0, "id": "A", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"tool_calls": [{"index": 1, "function": {"name": "er", "arguments": "2}"}}]},
    ]
    frames: list[dict] = [{"choices": [{"index": 0, "delta": delta}]} for delta in deltas]
    frames.append({"choices": [], "usage": {"total_tokens": 17}})
    data = (
        "".join(f"data: {json.dumps(frame)}\r\n\r\n" for frame in frames) + "data: [DONE]\r\n\r\n"
    )

    class Segments(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(0, len(data), 11):
                yield data[i : i + 11].encode()

    def serve(req):
        assert json.loads(req.content)["stream_options"]["include_usage"]
        return httpx.Response(200, stream=Segments())

    client = create_client("lmstudio/m", config={}, transport=httpx.MockTransport(serve))
    chunks = [chunk async for chunk in client.stream([])]
    assert [chunk.text for chunk in chunks[:-1]] == ["v", "isible"]
    result = chunks[-1].response
    assert result is not None
    assert (result.message.content, result.message.reasoning) == ("visible", "secret")
    assert [call.model_dump() for call in result.message.tool_calls] == [
        {"id": "A", "name": "f", "arguments": {}},
        {"id": "B", "name": "other", "arguments": {"q": 2}},
    ]
    assert result.usage.total_tokens == 17


async def test_retryable_transport_rate_limit_and_server_failures():
    attempts = []

    def serve(req):
        attempts.append(req)
        match len(attempts):
            case 1:
                raise httpx.ConnectError("connection dropped")
            case 2:
                return httpx.Response(429, headers={"Retry-After": "0"})
            case 3:
                return httpx.Response(503, headers={"Retry-After": "0"})
            case _:
                return answer()

    client = create_client("lmstudio/m", config={}, retries=3, transport=httpx.MockTransport(serve))
    assert (await client.generate([])).message.content == "ok"
    assert len(attempts) == 4


async def test_status_errors_redact_credentials_and_identify_explicit_tool_rejection():
    credential = "value-from-test"
    spec = ToolSpec(name="a", description="test", parameters={})
    responses = iter(
        [
            (401, f"wrong key: {credential}"),
            (400, "invalid parameter type in tools"),
            (400, "This model does not support the tools parameter"),
        ]
    )

    def serve(req):
        status, message = next(responses)
        return httpx.Response(status, json={"error": {"message": message}})

    client = create_client(
        "openai/m", config={}, api_key=credential, transport=httpx.MockTransport(serve)
    )
    with pytest.raises(ModelError, match="openai HTTP 401: wrong key") as error:
        await client.check_connection()
    assert credential not in str(error.value)
    with pytest.raises(ModelError, match="parameter type") as error:
        await client.generate([], tools=[spec])
    assert not isinstance(error.value, UnsupportedToolsError)
    with pytest.raises(UnsupportedToolsError):
        _ = [chunk async for chunk in client.stream([], tools=[spec])]


async def test_ollama_model_probe_and_fallback():
    replies = iter(
        [
            httpx.Response(200, json={"capabilities": ["tools", "vision"]}),
            httpx.Response(200, json={}),
            httpx.Response(404),
        ]
    )

    def serve(req):
        assert str(req.url) == "http://example.test/ollama/api/show" and req.method == "POST"
        assert json.loads(req.content) == {"model": "qwen3:8b"}
        return next(replies)

    client = create_client(
        "ollama/qwen3:8b",
        config={},
        base_url="http://example.test/ollama/v1/",
        transport=httpx.MockTransport(serve),
    )
    assert (await client.probe()).tools
    assert not (await client.probe()).tools
    assert not (await client.probe()).tools


async def test_failed_stream_does_not_repeat_delivered_tokens():
    requests = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            raise httpx.ReadError("disconnected")

    def serve(req):
        requests.append(req)
        return httpx.Response(200, stream=Broken())

    client = create_client("lmstudio/m", config={}, transport=httpx.MockTransport(serve))
    received = []
    with pytest.raises(ModelError, match="connection failed"):
        async for chunk in client.stream([]):
            received.append(chunk.text)
    assert received == ["first"] and len(requests) == 1
