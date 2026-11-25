# Parsing corpus

Each file under `tool_calling/` is one model reply and the way GoleKit must interpret it.
`tests/unit/test_corpus.py` runs every file.

- `source`: where the reply was observed (a unit test, a benchmark run or a live run).
- `protocol`: `react` (Thought / Action / Final Answer text), `constrained` (JSON selection
  object) or `native` (the raw `arguments` string of a native tool call).
- `tools`: the tool schemas offered to the model.
- `reply`: the raw reply.
- `expected`: `{"tool", "arguments"}`, `{"final": text}`, `{"answer": true}` (the constrained
  answer signal), `{"retry": true}`, or for `native` `{"arguments"}` / `{"invalid": true}`.
- `xfail` (optional): why a sample cannot be interpreted yet.
