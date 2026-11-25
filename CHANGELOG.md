# Changelog

## [Unreleased]

### Added

- Added summary, SQLite long-term and layered memory, with per-session management and memory tools.
- Added batched embedding clients and incremental local knowledge indexes with keyword, vector and hybrid retrieval.
- Added optional PDF and Word loaders via the rag extra.
- Added tool and context retrieval, source citations and retrieve events to Agent.
- Added schema-validated structured answers and correction attempts to all four run interfaces.
- Added local knowledge Q&A, two-process memory examples and the gole kb build command.

### Changed

- **BREAKING** Renamed the distribution and import package from `agentgole` to `golekit`. Use `import golekit as gole`. The `gole` command, `gole.yaml` and `GOLE_*` variables are unchanged.
- Knowledge indexing and long-term facts use local SQLite files. Built-in stores are intended for single-machine use.
- A run validates structured answers before emitting their tokens. Tool-assisted runs can buffer intermediate text.

### Fixed

- Allow direct answers in prompt mode with or without tools.
- Emit missing final answer tokens and suppress intermediate tool text.
- Recover malformed tool arguments and generate missing call IDs.
- Supply the current query to session memory and keep interactive chat open after a failed turn.

## [0.1.0.dev0]

- Added the v0.1 native agent kernel, synchronous and asynchronous model and tool interfaces,
  session memory, native and prompt tool calling, event streams, and the gole CLI.
