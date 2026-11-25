# AgentGOLE 开发计划与路线图（v0.1 → v1.0）

> 本文件导出自项目文档「开发计划与路线图」页（2026-11-12）。版本范围以本文为准；每个版本开工前，会另外下发 `docs/tasks/v0.X.md` 任务书，任务书里的细节优先于本文。

从 v0.1 到 v1.0 共 10 个版本，分三个阶段推进：阶段一用原生内核把 L0–L3 做扎实，阶段二接入框架后端并开放编排与内核，阶段三接入框架托管引擎并发布 1.0。每个阶段都以可运行的验收门槛收尾。

## 1. 开发原则

每个版本只做一个主题，并以可运行的验收用例收尾；门槛没过，不开下一版。

- **先 A 后 B、C**：原生内核先定义统一接口；模式 B 的后端和模式 C 的引擎都往这套接口上映射，而不是反过来。
- **测试先行**：用确定性的假模型（`FakeModel`，按脚本返回回复和工具调用）写单元测试，离线、可重复；真实模型测试单独标记，只在本地和夜间 CI 跑。
- **协议测试是准入条件**：任何内置实现、后端或引擎，都必须通过同一套 conformance 测试才能合入。
- **按 PR 粒度拆任务**：下文每个复选框对应一个可独立验收的 PR，适合直接交给编码 Agent。
- **v0.x 允许破坏性变更**：每次都写进 CHANGELOG；v0.9 进入 API 冻结候选，v1.0 起遵循语义化版本。
- **环境基线**：Python ≥ 3.11（3.10 已于 2026 年 10 月停止维护）；用 uv 管理环境，ruff + pyright + pytest 合并为一条 `make check` 命令。

## 2. 项目结构

仓库采用 `src/` 布局，每个模块对应使用文档里的一类组件；右侧标注的是该模块在哪个版本引入。v0.1 只建 `config`、`protocols`、`kernel`、`models`、`tools`、`paradigms`、`memory`、`trace`、`cli` 九个目录（外加 `_sync.py`、`errors.py`、`testing/`），其余目录到对应版本再建。

```text
golekit/
├── pyproject.toml              # 核心依赖 httpx / pydantic / pyyaml；extras: rag, mcp, mem0, llamaindex, langgraph, litellm, otel, all
├── Makefile                    # make check（唯一的本地/CI 检查入口）
├── AGENTS.md                   # 编码 Agent 的工作规则（必读）
├── README.md · README.zh.md · CHANGELOG.md · LICENSE (Apache-2.0) · NOTICE
├── docs/                       # SPEC.md · DEV_PLAN.md · DEV_GUIDE.md · REPORT_TEMPLATE.md · tasks/
├── reports/                    # 每个版本一份阶段报告
├── src/golekit/
│   ├── __init__.py             # 公共 API 出口：Agent, Model, Embedding, tool, mcp, Knowledge, prompt, Team, register ...
│   ├── agent.py                # 公共 API：Agent、Result                               v0.1
│   ├── model.py                # 公共 API：Model（包装 models/ 中的客户端）              v0.1
│   ├── _sync.py                # 同步包装：检测已有事件循环，必要时切到后台线程       v0.1
│   ├── errors.py               # 异常层级                                             v0.1
│   ├── testing/                # FakeModel 等测试工具（公开，方便用户写测试）          v0.1
│   ├── config/                 # 参数 > 环境变量 > gole.yaml；模型别名                 v0.1
│   ├── protocols/              # 公开协议：ModelClient · Tool · Paradigm · Memory · Retriever · Engine
│   ├── kernel/                 # 原生内核（NanoHarness 六关注点）
│   │   ├── engine.py           #   执行：异步运行循环、Run、Action                     v0.1
│   │   ├── context.py          #   上下文：组装、长度告警、token 预算、压缩             v0.1 / v0.3 / v0.8
│   │   ├── state.py            #   状态：检查点存储（SQLite / JSON）                   v0.5
│   │   ├── events.py           #   生命周期：事件总线、Hook                             v0.1
│   │   ├── middleware.py       #   中间件链                                              v0.6
│   │   └── evaluator.py        #   评估：轨迹记录、终止判定                             v0.1
│   ├── models/                 # openai_compat · ollama · anthropic；能力探测            v0.1
│   ├── tools/                  # @tool、注册表、调用路径（策略 → 审批 → 执行）、内置工具、mcp  v0.1–v0.4
│   ├── paradigms/              # direct · auto · react · plan_execute · reflect；降级解析  v0.1 / v0.3
│   ├── memory/                 # window · summary · longterm（SQLite）· layered         v0.1 / v0.2
│   ├── knowledge/              # loaders · chunking · stores · retrieval（向量 + BM25）· citations  v0.2
│   ├── flow/                   # >> · branch · parallel · loop · Context · YAML         v0.5
│   ├── team/                   # router · supervisor · handoff · debate                 v0.5
│   ├── backends/               # 模式 B 适配器：litellm · mem0 · llamaindex · chroma · qdrant ...  v0.4
│   ├── engines/                # 模式 C：native · langgraph ...；能力矩阵               v0.7
│   ├── trace/                  # 终端步骤树 · JSON · HTML · OTel / Langfuse 导出         v0.1 / v0.3 / v0.8
│   ├── safety/                 # 审批 · 沙箱 · 预算 · 注入防护                           v0.3 / v0.8
│   ├── serve/                  # ASGI：OpenAI 兼容端点、/runs + SSE                     v0.6
│   ├── registry.py             # 字符串 → 组件；entry points 插件发现                   v0.4 / v0.6
│   └── cli/                    # gole init · doctor · chat · kb · run · serve · engines · trace
├── tests/
│   ├── unit/                   # 默认运行，只用 FakeModel
│   ├── conformance/            # 协议测试：内置实现、后端、引擎都要通过                 v0.4
│   ├── crossmode/              # 同一套用例在 A / B / C 三种模式下运行                  v0.7
│   └── live/                   # 真实模型（Ollama、云端 API），手动或夜间运行
├── benchmarks/tool_calling/    # 小模型工具调用成功率基准                               v0.3
└── examples/                   # 与 SPEC §11 Cookbook 一一对应
```

模块依赖规则（v0.4 起在 CI 里用 import-linter 一类工具强制检查，之前靠评审）：

- `protocols/` 不依赖任何其他模块；`kernel/` 只依赖 `protocols/`。
- `backends/` 和 `engines/` 只依赖 `protocols/` 与公共 API；第三方库一律在函数内懒加载，`import golekit` 不得触发任何 extras 的导入。
- 组件之间不互相导入：Agent 通过 `registry` 按字符串解析组件，这是 `backend=`、`engine=` 和插件能工作的前提。

## 3. 路线图总览

10 个版本分三个阶段推进，阶段之间各有一道验收门槛；门槛没通过，就不进入下一阶段。

```text
阶段一 · 模式 A 打底（L0–L3）
  v0.1  内核骨架                 L0–L1
  v0.2  记忆与知识库             L2
  v0.3  推理范式与降级强化       L3 · 工具调用基准
  ◇ 门槛 1：Cookbook 11.1、11.2 全离线跑通，工具调用基准已发布

阶段二 · 模式 B 与编排（L4–L5）
  v0.4  后端适配层               模式 B · 协议测试
  v0.5  工作流与多智能体         L4
  v0.6  内核开放与服务化         L5
  ◇ 门槛 2：所有后端通过协议测试，Cookbook 11.3–11.5 跑通

阶段三 · 模式 C 与生态（生产加固）
  v0.7  首个运行引擎 LangGraph   模式 C
  v0.8  可观测与安全加固         故障注入测试
  v0.9  生态扩展与发布候选       1.0.0rc
  ◇ 门槛 3：Cookbook 5 个场景在 A / B / C 三种模式下跑通

1.0 发布
  v1.0  稳定版                   API 冻结 · 语义化版本
```

## 4. 阶段一：模式 A 打底（v0.1–v0.3）

阶段一结束时，只装核心包、只用本地模型，就能跑通 SPEC 中 L0–L3 的全部示例。

### v0.1 内核骨架（L0–L1）

目标：3 行代码跑起带工具的 Agent，本地模型与云端 API 用同一套代码。

- [ ] 脚手架：`pyproject.toml`（extras 到对应版本再加）、uv、ruff、pyright、pytest、GitHub Actions、`make check`
- [ ] `protocols/`：`ModelClient`、`Tool`、`Paradigm`、`Memory` 初版
- [ ] `kernel/engine.py`：异步运行循环、`Run`、`Action`（`CallModel` / `CallTool` / `Finish`），`max_steps` 与 `timeout`
- [ ] `kernel/events.py`：事件总线与 `agent.on()`；`stream()` / `astream()` 基于事件实现
- [ ] `_sync.py`：`run()` 等同步包装，兼容 Jupyter 已有的事件循环
- [ ] `models/`：OpenAI 兼容客户端 + Ollama 适配、模型字符串解析、能力探测（`capabilities`）
- [ ] `tools/`：`@gole.tool`（类型标注 → JSON Schema）、注册表、内置 `calculator` / `datetime`
- [ ] `paradigms/`：`auto`（原生工具调用循环）与 `react`（降级第一阶段：通用模板、损坏 JSON 修复、带报错重试）
- [ ] `memory/window.py`：默认 `memory="session"`，按 `session=` 隔离
- [ ] `config/`：参数 > 环境变量 > `gole.yaml`，模型别名
- [ ] `trace/` + `cli/`：`verbose=True` 步骤树、`result.trace`；`gole init` / `doctor` / `chat`
- [ ] `FakeModel`：可编排回复与工具调用的假模型，供全部单元测试使用

**验收**：SPEC §4 和 L0、L1 示例在 Ollama（qwen3:8b）与 DeepSeek 上跑通；同一组工具调用测试在 `tool_mode="native"` 和 `"prompt"` 下都通过。

### v0.2 记忆与知识库（L2）

目标：带记忆的助理和本地文档问答全离线可用。

- [ ] v0.1 审核遗留修复 R1–R5（见 `reports/v0.1-review.md`）
- [ ] `memory/summary.py`：滚动摘要，可单独指定小模型
- [ ] `memory/longterm.py`：SQLite 长期记忆，支持 `write="after_run"` 与 `"tool"` 两种写入时机
- [ ] `memory/layered.py` 与 `memory="long"` 预设；`add` / `search` / `export` / `clear`
- [ ] `knowledge/loaders`：md、txt、网页；pdf、docx 放进 `[rag]` extra
- [ ] `knowledge/`：切分、`gole.Embedding`、本地向量存储 + BM25 混合检索、增量索引
- [ ] Agent 接入知识库：`knowledge_mode` 三种模式、多知识库挂载、`result.citations`
- [ ] 结构化输出：`output=PydanticModel`，校验失败时带错误信息重试
- [ ] `gole kb build`

**验收**：Cookbook 11.1、11.2 断网跑通；对同一目录重复建索引时，未改动的文件不重新嵌入。

### v0.3 推理范式与降级强化（L3）

目标：多步任务可规划、可自检；小模型工具调用有数据支撑。

- [ ] v0.2 审核遗留修复 R1–R7（见 `reports/v0.2-review.md`）
- [ ] `paradigms/`：`direct`、`plan_execute`（含重规划）、`reflect`（评审轮数与标准）
- [ ] 多模型分工：范式、记忆可各自指定模型
- [ ] `safety/approval`：`confirm=True`、`approval.Console` / `Callback`、`approval_required` 事件；审批写成工具调用拦截链的一环（D-310）
- [ ] 内置工具补齐：`web_search`、`web_fetch`、`python`（子进程沙箱）、`files`、`shell`
- [ ] 降级第二阶段：接入约束解码（Ollama 结构化输出、vLLM guided decoding、llama.cpp GBNF）
- [ ] `benchmarks/tool_calling`：小模型工具调用成功率基准，结果写进文档；跑出的真实坏输出收进降级解析的语料测试（D-312）
- [ ] 上下文长度最小版：可配置、接近上限时告警、`gole doctor` 显示（D-309）；压缩仍在 v0.8
- [ ] `trace.to_json` 与 `GOLE_TRACE_DIR`（D-311，从 v0.8 提前）

**验收**：L3 示例跑通；基准覆盖至少 3 个本地模型 × native / prompt 两种模式，结果公开。

**阶段一门槛**：Cookbook 11.1、11.2 全离线跑通，工具调用基准已发布。

## 5. 阶段二：模式 B、编排与内核开放（v0.4–v0.6）

阶段二结束时，组件能在内置实现和主流框架之间无缝切换，L4–L5 的编排与扩展能力全部可用。

### v0.4 后端适配层（模式 B）

目标：`backend=` 一个参数切换实现，行为一致由协议测试保证。

- [ ] 模式 C 映射的抛弃式验证（D-010）：在定稿协议测试之前，用一到两天把一个带工具的 Agent 和一个两步流程手工映射到 LangGraph，逐项检查 D-301 的 token 语义、D-304 的默认拒绝、结构化输出纠正、prompt 降级能否对应。产出是一页结论，写明哪些能映射、哪些要列进能力矩阵的「不支持」；代码不合入
- [ ] 内核整形（D-011）：把运行循环改成可序列化、可逐步推进的 `Run`，模型调用和工具调用都经过同一条拦截链；对外行为不变，由现有单元测试守住。v0.5 的检查点、v0.6 的中间件与手动驱动、v0.7 的引擎都建立在它之上
- [ ] `user=`（D-012）：`run` 等四个入口增加可选的 `user=`，默认等于 `session`；长期记忆按 `user` 归属，会话记忆仍按 `session`
- [ ] `registry.py`：字符串 → 组件解析、懒加载，缺依赖时提示安装命令
- [ ] `tests/conformance/`：`ModelClient`、`Tool`、`Memory`、`Retriever` 协议测试套件，内置实现先全部通过
- [ ] `backends/`：LiteLLM（模型）、mem0（长期记忆）、LlamaIndex（知识库，含 `Knowledge.from_llamaindex`）
- [ ] `backends/`：Chroma、Qdrant 作为知识库存储
- [ ] `tools/mcp`：stdio 与 Streamable HTTP，`select()` 过滤；`tools.from_langchain`
- [ ] `.native` 逃生舱

**验收**：Cookbook 11.5 跑通；各后端在同一套协议测试下全部通过。

### v0.5 工作流与多智能体（L4）

目标：确定流程用 Workflow，开放协作用 Team，二者都能中断后恢复。

- [ ] `flow/`：`>>` 管道、`gole.prompt` 模板（`{input}` / `{task}` / `{步骤名}`）、`Context` 共享状态
- [ ] `flow/`：`branch`、`parallel`、`loop`
- [ ] `kernel/state.py`：SQLite 检查点；`flow.resume`、`agent.resume`；已完成的步骤恢复时不重新调用模型
- [ ] Workflow YAML 与 `gole run`
- [ ] `team/`：`router`、`supervisor`、`handoff`、`debate`；Team 提供与 Agent 相同的 `run / stream / as_tool`

**验收**：Cookbook 11.3 跑通；在任意一步强制中断后，都能从断点恢复。

### v0.6 内核开放与服务化（L5）

目标：开发者能以 AgentGOLE 为内核做扩展和部署。

- [ ] `protocols/` 定稿并写成文档；`Action` 补齐 `Delegate`、`AskHuman`
- [ ] `kernel/middleware.py`：`Retry`、`TokenBudget`、`Cache`、`PIIMask`；`@agent.middleware`
- [ ] 手动驱动：`agent.start()` 逐步迭代，`execute` / `reject`
- [ ] 插件机制：`gole.register` + entry points（`golekit.plugins`），附一个示例插件包
- [ ] 序列化：`agent.save` / `Agent.load`（YAML，不含密钥）
- [ ] `serve/`：`gole serve`（OpenAI 兼容 `/v1/chat/completions`、`/runs` + SSE、`/health`）与 `gole.asgi`

**验收**：Cookbook 11.4 跑通；示例插件包独立安装后，可以直接用字符串引用。

**阶段二门槛**：所有后端通过协议测试，Cookbook 11.3–11.5 跑通。

## 6. 阶段三：模式 C、生态与 1.0（v0.7–v1.0）

阶段三先打通「不用自有内核」这条路，再补齐生产所需的可观测与安全能力，最后冻结 API、发布 1.0。

### v0.7 首个运行引擎：LangGraph（模式 C）

目标：`engine="langgraph"` 下，同一段业务代码交给 LangGraph 运行。

- [ ] `engines/`：`Engine` 协议、能力矩阵、`EngineCapabilityError`、`gole engines`
- [ ] `engines/langgraph`：Agent 与 Workflow 映射为 LangGraph 图；工具、记忆映射；回调桥接为 AgentGOLE 事件
- [ ] `flow.compile("langgraph")` 与 `agent.native`
- [ ] `tests/crossmode/`：同一套 Cookbook 用例在 A / B / C 三种模式下运行

**验收**：Cookbook 5 个场景在三种模式下跑通；模式 C 不支持的能力，在构造时就给出明确报错。

### v0.8 可观测与安全加固

目标：长时间、无人值守地运行，也要可追溯、可控制。

- [ ] `trace/`：`to_html`、`gole trace`、OpenTelemetry 与 Langfuse 导出（`to_json` 已在 v0.3 提供）
- [ ] `safety/`：`python` 工具的 Docker 沙箱、提示词注入防护标记、`Webhook` 审批
- [ ] `kernel/context.py`：上下文预算，长工具结果换成摘要 + 引用，早期对话压缩（v0.3 已有上下文长度配置和告警）
- [ ] 并发工具调用、超时与取消
- [ ] 故障注入测试：工具超时、进程中断、模型返回坏 JSON

**验收**：在故障注入测试下，每次运行要么恢复完成，要么以明确的错误结束，不出现静默失败。

### v0.9 生态扩展与发布候选

目标：扩展生态并冻结接口，发布 1.0.0rc。

- [ ] 评估更多运行引擎（LlamaIndex Agent、smolagents、OpenAI Agents SDK），按评估结果接入一到两个
- [ ] 更多后端：Letta、LangChain 检索器、Milvus；评估 AutoGen、CrewAI 是否作为 Team 后端
- [ ] 文档站：使用文档、API 参考、L0–L5 分级教程
- [ ] API 冻结候选：发布 1.0.0rc，公布弃用策略

**验收**：不看源码、只按文档，就能从零完成 L0–L3 教程；rc 期间收到的问题全部分类处理。

### v1.0 稳定版

- [ ] 公共 API 冻结，此后遵循语义化版本
- [ ] 完整的 API 参考，以及从 0.x 升级的迁移指南
- [ ] 发布流程：打 tag 触发 CI，经 Trusted Publishing 发布到 PyPI

**验收**：rc 阶段没有破坏性变更；协议测试与跨模式测试全部通过。

**阶段三门槛**：Cookbook 5 个场景在 A / B / C 三种模式下跑通，即可进入 1.0 发布流程。

## 7. 测试、质量与发布流程

测试分五层，越往下越慢、越依赖外部环境，运行频率也越低。具体的测试纪律见 `docs/DEV_GUIDE.md` §4。

| 层级 | 位置 | 何时运行 | 依赖 |
| --- | --- | --- | --- |
| 单元测试 | `tests/unit` | 每次提交 | 仅 `FakeModel`，不联网 |
| 协议测试 | `tests/conformance` | 每次提交（内置实现）；合入 main 时跑全部后端 | 对应 extras |
| 跨模式测试 | `tests/crossmode` | 合入 main 时（v0.7 起） | LangGraph 等引擎 |
| 真实模型测试 | `tests/live` | 版本验收时或手动 | Ollama、云端 API 密钥 |
| 基准 | `benchmarks/` | 每个版本发布前（v0.3 起） | 本地模型 |

质量门槛：

- `make check` = ruff + pyright（`protocols/`、`kernel/` 用严格模式）+ 单元测试 + 内置实现的协议测试，本地和 CI 是同一条命令。
- CI：PR 跑 Linux 和 Windows + Python 3.11（D-013）；合入 main 时跑 Linux、macOS、Windows × {3.11、当前最新稳定版} 的矩阵。
- 覆盖率只观察、不追求：v0.9 起才把 `protocols/`、`kernel/` 覆盖率 ≥ 90% 作为门槛，之前不为凑覆盖率写测试。降级解析的语料测试是例外，样本数量不设上限（D-013）。
- 每个 PR 带测试和 CHANGELOG 条目；改动公共 API 时，在阶段报告里列出，由审核方同步更新 SPEC。

发布流程：

1. 当前版本的验收用例全部通过，阶段报告经审核确认。
2. CHANGELOG 写明破坏性变更（开发方维护）；SPEC 由审核方同步更新。
3. 由人打 tag，CI 构建并经 Trusted Publishing 发布到 PyPI。
4. GitHub Release 附上变更摘要；v0.3 起同时附上最新的工具调用基准结果。

版本号用 `0.MINOR.PATCH`：每个路线图版本对应一个 MINOR，版本内修 bug 只升 PATCH。

## 8. 风险与应对

最大的风险来自外部框架变化和范围蔓延，两者都靠「协议测试 + 每版只做一个主题」来控制。

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| LangGraph、LlamaIndex、mem0 等框架 API 变化快 | 后端和引擎频繁失效 | extras 锁定版本区间；协议测试按框架版本在 CI 矩阵中运行；适配代码只用框架的公开接口 |
| 模式 C 的映射复杂 | 进度拖延、行为不一致 | 只承诺能力矩阵内的功能，不支持就在构造时报错；1.0 之前只保证 LangGraph 一个引擎 |
| 小模型工具调用不稳定 | 「本地优先」体验变差 | 降级分阶段建设，用基准数据说话；文档给出推荐模型清单 |
| 范围蔓延 | 迟迟到不了 1.0 | 每版只做一个主题；新需求先进 backlog，门槛没过不开新版 |
| 个人开发精力有限 | 进度不可控 | 任务按 PR 粒度拆分，交给编码 Agent 并行实现；人专注于接口设计、评审和验收 |
