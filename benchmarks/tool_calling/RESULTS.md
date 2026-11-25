# 工具调用基准结果

**尚未运行（待人工验证）。** 这个文件由 `run.py` 生成；开发环境里没有 Ollama，下面没有任何实测数据。

在装有 Ollama 的机器上运行（模型可以按本机已有的替换，不少于 3 个）：

```powershell
$env:GOLE_TRACE_DIR = "traces\\bench"
uv run python benchmarks/tool_calling/run.py --models ollama/qwen3:8b,ollama/qwen3:4b,ollama/llama3.2:3b,ollama/gemma3:4b
```

运行结束后，本文件会被实测结果覆盖，解析失败和触发重试的原始回复写在同目录的 `failures.jsonl`。
