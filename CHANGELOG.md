# Changelog

## [Unreleased]

### Added

- Added the `direct`, `plan` and `reflect` paradigms and the `golekit.paradigms` namespace. `PlanExecute` and `Reflect` accept separate planner, executor and critic models.
- Added tool confirmation: `@gole.tool(confirm=True)`, `gole.Agent(approval=...)`, `gole.approval.Console` and `gole.approval.Callback`, and the `approval_required` event. A tool that needs confirmation does not run when no approval method is configured. `gole chat` asks on the terminal.
- Added the built-in tools `web_search` (DuckDuckGo or Tavily), `web_fetch`, `python`, `shell` and `gole.tools.files(root, write=False)`. `python` and `shell` need confirmation by default; `python` isolation is best effort, not a security boundary.
- `tools=` accepts lists of tools. gole.yaml accepts `tools.web_search.provider`; `GOLE_SEARCH_PROVIDER` overrides it.
- Added JSON Schema constrained decoding for structured answers, paradigm-internal calls and prompt-mode tool selection, with `gole.Model(constrained=...)` and `Capabilities.json_schema`. A server that rejects the response format is retried without it and remembered.
- Added context length awareness: `gole.Model(context=...)`, `GOLE_CONTEXT`, `providers.<name>.context` in gole.yaml, and Ollama's explicit `num_ctx`. `model_end` events carry `context_used`, and a prompt at 90% or more logs one warning per run. Nothing is compressed or truncated.
- `gole doctor` reports constrained decoding support and the context length of each chat model.
- Added the tool-calling benchmark in `benchmarks/tool_calling/` (20 cases, native, prompt and constrained modes) and a corpus of recorded model replies in `tests/corpus/` that the parsers are replayed against.
- Added the `l3_plan.py` and `l3_reflect.py` examples.
- Added `result.trace.to_json(path=None)`. A failed run's exception carries the recorded events as `exc.trace`. When `GOLE_TRACE_DIR` is set, every run writes `<run_id>.json` there. Traces contain the full conversation and tool arguments.
- Added summary, SQLite long-term and layered memory, with per-session management and memory tools.
- Added batched embedding clients and incremental local knowledge indexes with keyword, vector and hybrid retrieval.
- Added optional PDF and Word loaders via the rag extra.
- Added tool and context retrieval, source citations and retrieve events to Agent.
- Added schema-validated structured answers and correction attempts to all four run interfaces.
- Added local knowledge Q&A, two-process memory examples and the gole kb build command.

### Changed

- Retrieved passages shown to the model and `Citation.source` use paths relative to the working directory, or the file name alone for files outside it. URLs are unchanged.
- **BREAKING** Renamed the distribution and import package from `agentgole` to `golekit`. Use `import golekit as gole`. The `gole` command, `gole.yaml` and `GOLE_*` variables are unchanged.
- **BREAKING** Renamed the project to GoleKit and the base exception `AgentGoleError` to `GoleError`.
- Knowledge indexing and long-term facts use local SQLite files. Built-in stores are intended for single-machine use.
- A run validates structured answers before emitting their tokens.
- **BREAKING** Native tool-mode runs stream model text as it arrives, including text that precedes a tool call. The tokens after the last `tool_result` event (all tokens when no tool ran) concatenate to `result.text`. `gole chat` puts text written before a tool call on its own line.

### Fixed

- Word documents load table text in document order, one line per row with tab-separated cells.
- Directory and glob sources skip files and directories whose names start with a dot, unless the path is passed explicitly.
- A knowledge source path that does not exist raises `ConfigError` instead of building an empty index. A glob without matches logs a warning.
- Context retrieval no longer injects an empty reference message, places references after the instructions and memory preamble, and leaves the prompt-mode tool protocol on the instructions. A search tool with no match answers 「未找到相关内容。」.
- A failed memory save, such as a fact extraction error, no longer fails a run whose answer is ready. The run reports a recoverable `error` event and returns its result.
- Allow direct answers in prompt mode with or without tools.
- Emit missing final answer tokens and suppress intermediate tool text.
- Recover malformed tool arguments and generate missing call IDs.
- Supply the current query to session memory and keep interactive chat open after a failed turn.

## [0.1.0.dev0]

- Added the v0.1 native agent kernel, synchronous and asynchronous model and tool interfaces,
  session memory, native and prompt tool calling, event streams, and the gole CLI.
