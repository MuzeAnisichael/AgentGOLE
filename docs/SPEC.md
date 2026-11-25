# AgentGOLE · Agent 内核库使用文档（草案 v0.2）

> 本文件导出自项目使用文档（草案 v0.2，2026-11-12）。它是 AgentGOLE 对外行为的**唯一事实来源**：公共 API、命名和默认值以本文为准。实现中如需偏离，必须在阶段报告的「偏差与决策」里写明。

## 1. 项目定位与目标

AgentGOLE（暂定名）是一个 Python Agent 内核库：配好一个模型，3 行代码就能跑起一个 Agent；之后用同一套接口，一路扩展到工具、记忆、知识库、工作流和多智能体。名字取自 LEGO 的字母重排，意思是像拼乐高一样拼装 Agent。

### 1.1 要解决的问题

- **框架多、范式杂**：LangChain、LlamaIndex、LangGraph、mem0 等各有一套抽象。初学者不知道从哪开始，开发者换一个组件就要改一大片代码。
- **上手门槛高**：想搭一个「工具 + 记忆 + 知识库」的最简 Agent，往往要先读懂 3–4 个库的文档。
- **本地模型跑不通**：很多范式默认模型支持原生函数调用，本地小模型容易卡在这一步。
- **难以复用**：每个项目都在重写同样的 Agent 循环、记忆管理和工具注册代码。

### 1.2 我们的回答

提供一套统一的组件接口（Model / Tool / Paradigm / Memory / Knowledge / Workflow / Team），每个组件都内置零额外依赖的最简实现，开箱可跑。更强的实现则按需接入现有框架，作为可替换的「后端」。

内核本身也可以自由选择：用 AgentGOLE 自有内核、自有内核 + 框架组件，或者不用自有内核、完全交给现有框架运行（见 §2.2）。

### 1.3 目标用户

| 用户 | 核心诉求 | 主要使用的层级 |
| --- | --- | --- |
| 初学者 / 学生 | 几行代码看到 Agent 跑起来，借此理解工具调用、记忆、RAG 是什么 | L0–L2 |
| 应用开发者 | 快速搭出业务 Agent 或 Workflow；换模型、换组件不改业务代码 | L1–L4 |
| 内核开发者 / 研究者 | 以 AgentGOLE 为内核自定义范式、记忆和运行时，或横向对比不同技术选型 | L4–L5 |

三类用户同时兼顾：v0.1–0.3 先保证 L0–L3 对初学者足够简单；之后每个版本逐步扩展更高层级的用法，以及可封装的框架和外部工具（见 §10.2）。

### 1.4 成功标准（建议值）

- 从 `pip install` 到第一个 Agent 回答：3 行代码，5 分钟以内。
- 本地模型和云端 API 之间切换：只改 1 个字符串。
- 切换记忆或知识库后端（如内置 → mem0 / LlamaIndex）：只改 1 个参数。
- 核心包只依赖 httpx、Pydantic、PyYAML 三个轻量库：`pip install agentgole` 不会自动装上 LangChain、LlamaIndex 等，需要时再用 extras 按需安装（见 §3）。
- 全离线可用：只靠 Ollama + 本地嵌入模型，就能跑通 L0–L3 的全部示例。

### 1.5 非目标

- **不重新造轮子**：不自研向量数据库，不训练模型，也不和 LangGraph 比拼复杂图编排能力。
- **不做低代码平台**：不提供拖拽式编排界面（后期最多提供一个调试用的轻量 UI）。
- **不追求覆盖所有功能**：只封装约 80% 场景的公共能力；剩下的通过「逃生舱」（§8.5）直接使用底层框架的原生对象。

## 2. 设计理念

AgentGOLE 的所有 API 形态都由五条原则决定：渐进式复杂度、统一接口可插拔后端、字符串即配置对象即控制、随时可逃生、本地优先。

1. **渐进式复杂度**：多一个需求，只多写一两行代码。初学者不必知道 Paradigm、Runtime 的存在，需要时它们一直都在。
2. **统一接口，可插拔后端**：每类组件只有一套接口。内置实现零额外依赖、开箱即用；LangChain、LlamaIndex、mem0 等作为可选后端，用 `backend=` 切换，业务代码不变；连内核本身也能替换（§2.2）。
3. **字符串即配置，对象即控制**：每个参数既接受简短字符串（`memory="summary"`），也接受完整对象（`memory=gole.memory.Summary(max_tokens=2000)`）。前者给初学者，后者给开发者。
4. **随时可逃生**：封装不是牢笼。由后端实现的组件都暴露 `.native`，可以拿到底层框架的原生对象；每个默认行为都能被 Hook 或自定义组件替换。
5. **本地优先，模型无关**：默认假设你用的是本地小模型。模型不支持原生工具调用时，自动降级为提示词解析式调用；云端 API 与本地模型用完全相同的代码。

一个例子看懂前四条：

```python
import agentgole as gole

# 原则 1 + 3：用字符串起步
agent = gole.Agent(model="ollama/qwen3:8b", memory="summary", knowledge="./docs")

# 原则 3：需要控制时，换成对象
agent = gole.Agent(
    model=gole.Model("ollama/qwen3:8b", temperature=0.2),
    memory=gole.memory.Summary(max_tokens=2000),
    knowledge=gole.Knowledge("./docs", backend="llamaindex"),  # 原则 2：换后端
)

# 原则 4：拿到 LlamaIndex 的原生索引对象
index = agent.knowledge.native
```

### 2.1 架构总览

AgentGOLE 分四层，你的代码只面对最上面两层；下面两层都可以替换。

```text
┌──────────────────────────────────────────────────────────────┐
│ 应用层 · 你写的代码只面对这一层                                  │
│   Agent（单个智能体） · Workflow（确定流程编排） · Team（多智能体协作）│
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ 核心层 · 统一组件接口与运行时（AgentGOLE 内核）                   │
│   Model · Tool · Paradigm · Memory · Knowledge · Runtime(事件·Hooks)│
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ 适配层 · 用 backend= 参数切换实现，业务代码不变                    │
│   内置实现（零额外依赖） | 可选后端：LiteLLM · MCP · mem0 ·          │
│                          LlamaIndex · LangChain · LangGraph        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ 模型层 · 本地与云端使用同一套代码                                 │
│   本地：Ollama · vLLM · LM Studio | 云端：OpenAI · Anthropic ·     │
│                                       DeepSeek · 通义千问         │
└──────────────────────────────────────────────────────────────┘
```

核心层是本项目真正要维护的部分；其余三层要么是你自己的代码，要么是可以替换的外部依赖。

### 2.2 三种运行模式

内核本身也可以替换：`engine=` 选谁来跑 Agent 循环，`backend=` 选组件由谁实现，两者自由组合，业务代码不变。

```text
            模式 A · 原生内核          模式 B · 内核 + 框架组件         模式 C · 框架托管
            engine=native（默认）      engine=native + 组件 backend=    engine=langgraph 等
            ┌──────────────┐          ┌──────────────┐               ┌──────────────┐
统一 API    │ Agent/Workflow/Team │   │ Agent/Workflow/Team │        │ Agent/Workflow/Team │
            └──────┬───────┘          └──────┬───────┘               └──────┬───────┘
运行循环    [ AgentGOLE 内核 ]         [ AgentGOLE 内核 ]               [ 框架运行时 ]
                   │                         │                              │
组件        内置组件（零额外依赖）      框架组件 mem0·LlamaIndex·MCP 等   框架自带组件（由 AgentGOLE 映射）
适合        入门学习、离线、自研范式    用成熟组件，保留内核控制力        复用已有框架资产与生态
```

方括号标出运行循环由谁驱动：模式 A 和 B 由 AgentGOLE 内核驱动，模式 C 把循环交给框架，AgentGOLE 只保留最上面的统一 API。

```python
# 模式 A：原生内核（默认），全部使用内置组件
agent = gole.Agent(model="ollama/qwen3:8b", memory="long", knowledge="./docs")

# 模式 B：AgentGOLE 内核 + 框架组件
agent = gole.Agent(
    model="ollama/qwen3:8b",
    memory=gole.memory.LongTerm(backend="mem0"),
    knowledge=gole.Knowledge("./docs", backend="llamaindex"),
)

# 模式 C：框架托管，AgentGOLE 只提供统一接口并负责映射
agent = gole.Agent(model="ollama/qwen3:8b", memory="long", knowledge="./docs", engine="langgraph")
agent.native   # LangGraph 的编译图，可继续使用它的生态工具
```

默认模式写在 `gole.yaml` 的 `engine:` 字段；`gole engines` 列出已安装的引擎及其能力。三种模式的能力差异：

| 能力 | A 原生内核 | B 内核 + 框架组件 | C 框架托管 |
| --- | --- | --- | --- |
| 统一 API（run / stream / as\_tool） | 支持 | 支持 | 支持 |
| 零额外依赖 | 是 | 否 | 否 |
| 事件粒度 | 完整 | 完整 | 取决于框架提供的回调 |
| 工具审批、检查点恢复 | AgentGOLE 实现 | AgentGOLE 实现 | 映射到框架自身机制 |
| 自定义范式、手动驱动（§8.3–8.4） | 支持 | 支持 | 视引擎而定 |
| 本地模型工具调用降级 | 支持 | 支持 | 视引擎而定 |

模式 C 下不可用的能力，在构造 Agent 时就直接报错（`EngineCapabilityError`）并提示改用模式 A 或 B，不会在运行中途才失败。

### 2.3 原生内核的设计参考

原生内核以 NanoHarness 的六关注点结构为骨架，运行时的安全与恢复机制参考 Minimal Local Agent。两者都不直接 fork，而是按本项目目标重新组织，必要时复用代码。

| 参考项目 | 借鉴 | 不照搬 |
| --- | --- | --- |
| [NanoHarness](https://github.com/HabitGraylight/NanoHarness) | 内核拆成六个关注点，每个一个小接口；最小默认实现 + 扩展包 + 声明式 Profile（YAML / TOML / JSON）；内核只依赖 Pydantic 和 PyYAML | 面向编码 Agent 的示例与扩展；入口 API 需要更简单 |
| [Minimal Local Agent](https://github.com/MuzeAnisichael/minimal-local-agent) | 运行时策略与预算、受限工具、可审查可回滚的编辑、可审计的运行记录；本地优先的模型适配 | 单工作区、编码任务导向的工具集；它要求模型支持工具调用，而 AgentGOLE 需要提示词降级 |

选 NanoHarness 做骨架，是因为它的六个接口几乎一一对应 AgentGOLE 的组件，且不绑定编码场景：

| NanoHarness 关注点 | NanoHarness 接口 | AgentGOLE 中的对应 |
| --- | --- | --- |
| 执行 Execution | `NanoEngine` | Runtime + Paradigm |
| 工具 Tools | `BaseToolRegistry` | Tool 注册表；调用路径：工具调用 → 策略 → 审批 → 执行 |
| 上下文 Context | `BaseContextManager` | 上下文组装与压缩，Memory、Knowledge 在此注入 |
| 状态 State | `BaseStateStore` | Checkpoint 存储 |
| 生命周期 Lifecycle | `BaseHookManager` | 事件、Hook、中间件 |
| 评估 Evaluation | `BaseEvaluator` | 运行轨迹、提前终止、Reflect 校验 |

两个项目都是 MIT 许可；AgentGOLE 采用 Apache-2.0，复用其代码时需保留原 MIT 版权声明（放入 `NOTICE` 或第三方许可目录）。

## 3. 安装与模型配置

核心包只依赖 httpx、Pydantic、PyYAML 三个轻量库；其他能力都用 extras 按需安装，缺少某个后端时，报错信息会直接给出安装命令。PyPI 包名为 `agentgole`，惯用导入写法是 `import agentgole as gole`，命令行工具叫 `gole`。

```bash
pip install agentgole                 # 核心：Agent、工具、内置记忆与轻量知识库
pip install "agentgole[rag]"          # 文档解析（PDF / Word）+ 本地向量库
pip install "agentgole[mcp]"          # MCP 工具
pip install "agentgole[mem0]"         # mem0 长期记忆后端
pip install "agentgole[llamaindex]"   # LlamaIndex 知识库后端
pip install "agentgole[langgraph]"    # LangGraph：工作流后端，或作为运行引擎（模式 C）
pip install "agentgole[all]"
```

### 3.1 模型字符串

模型用 `provider/model` 一个字符串指定。

| Provider | 示例 | 需要的配置 |
| --- | --- | --- |
| ollama | `ollama/qwen3:8b` | 本地运行 Ollama，默认 `http://localhost:11434` |
| vllm | `vllm/Qwen2.5-7B-Instruct` | `base_url` |
| lmstudio | `lmstudio/<model>` | 默认 `http://localhost:1234/v1` |
| openai | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| anthropic | `anthropic/<model>` | `ANTHROPIC_API_KEY` |
| deepseek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| dashscope（通义千问） | `dashscope/qwen-plus` | `DASHSCOPE_API_KEY` |
| 任意 OpenAI 兼容服务 | `gole.Model("<model>", base_url=...)` | `base_url`、`api_key` |

### 3.2 三种配置方式

优先级从高到低：代码参数 > 环境变量 > 配置文件。

1. **代码参数**：`gole.Agent(model="deepseek/deepseek-chat")`。
2. **环境变量**：`GOLE_MODEL=ollama/qwen3:8b`，以及各厂商的 `*_API_KEY`。
3. **配置文件**：`gole init` 生成 `gole.yaml`，放在项目目录或 `~/.gole/`。

```yaml
# gole.yaml
default_model: ollama/qwen3:8b
engine: native          # 默认运行模式（§2.2）

models:                 # 别名：代码里用 model="smart" 即可
  fast: ollama/qwen3:4b
  smart: deepseek/deepseek-chat
  embed: ollama/bge-m3

providers:
  ollama:
    base_url: http://localhost:11434
  deepseek:
    api_key: ${DEEPSEEK_API_KEY}   # 密钥只从环境变量读取
```

```python
agent = gole.Agent()                # 使用 default_model
agent = gole.Agent(model="smart")   # 使用别名
```

### 3.3 检查环境

`gole doctor` 逐个连接已配置的模型，并探测它们的能力。它会对每个模型发起一次只生成 1 个 token 的真实请求，云端 API 会因此产生极少量费用（下为示意输出）：

```text
$ gole doctor
[ok]   ollama/qwen3:8b          可连接   原生工具调用: 支持
[ok]   ollama/llama2:7b         可连接   原生工具调用: 不支持 → 将使用提示词模式
[ok]   deepseek/deepseek-chat   可连接   原生工具调用: 支持
[fail] openai/gpt-4o-mini       缺少 OPENAI_API_KEY
```

## 4. 快速开始

三行代码得到第一个 Agent：

```python
import agentgole as gole

agent = gole.Agent(model="ollama/qwen3:8b")
print(agent.run("用一句话介绍你自己"))
```

`run()` 返回一个 `Result`：直接 `print` 就是回答文本，需要时也能取出更多信息。

```python
result = agent.run("北京和上海哪个人口更多？")
result.text          # 最终回答
result.steps         # 每一步：模型调用、工具调用、检索……
result.usage         # token 用量
result.trace.show()  # 在终端打印步骤树
```

### 4.1 多轮对话

Agent 默认带会话记忆（`memory="session"`），连续调用就是多轮对话；不同 `session` 互相隔离。

```python
agent.run("我叫小林")
agent.run("我叫什么？")                      # → 你叫小林
agent.run("我叫什么？", session="user-42")   # → 我还不知道你的名字

stateless = gole.Agent(model="ollama/qwen3:8b", memory=None)   # 每次调用互不相关
```

### 4.2 流式输出

```python
for event in agent.stream("写一首关于秋天的短诗"):
    if event.type == "token":
        print(event.text, end="", flush=True)
```

`token` 事件的语义：文本一生成就发出。Agent 调用了工具时，工具调用之前的文本是中间文本；最后一个 `tool_result` 事件之后的 token 拼接起来等于 `result.text`。没有工具调用时，全部 token 拼接起来就是 `result.text`。指定了 `output=`（§4.4）时，回答通过校验后才发出 token。

### 4.3 异步

所有方法都有异步版本：内核本身是异步的，同步方法只是包装。在 Jupyter 这类已有事件循环的环境里，同步方法会自动切到后台线程执行，不需要额外处理。

```python
result = await agent.arun("...")
async for event in agent.astream("..."):
    ...
```

### 4.4 结构化输出

传入一个 Pydantic 模型，拿回的就是它的实例；`list[str]` 这类 Pydantic 能校验的类型也可以。模型输出不合规时，带着错误信息让模型重答，最多 2 次；仍不合规时抛 `OutputValidationError`。`result.text` 是原始的 JSON 文本。

```python
from pydantic import BaseModel

class Paper(BaseModel):
    title: str
    year: int
    keywords: list[str]

paper = agent.run("从这段摘要中提取信息：...", output=Paper).output   # Paper 实例
```

### 4.5 在终端里直接聊

```bash
gole chat --model ollama/qwen3:8b
gole chat my_agent.py:agent      # 和你在代码里定义的 Agent 聊天
```

对话中输入 `/reset` 开始新会话，输入 `/exit` 退出。

## 5. 复杂度阶梯 L0–L5

AgentGOLE 把能力分成六级台阶，每升一级只新增一两个概念；教程、示例和报错提示都按这个阶梯组织。

| 级别 | 新学的概念 | 新增代码量 | 典型场景 |
| --- | --- | --- | --- |
| L0 对话 | `Agent`、`model`、`instructions` | 3 行 | 聊天机器人、角色扮演 |
| L1 工具 | `@gole.tool`、`tools=` | +5 行 | 查天气、算数、调业务 API |
| L2 记忆与知识 | `memory=`、`knowledge=` | +2 行 | 个人助理、文档问答 |
| L3 推理范式 | `paradigm=` | +1 行 | 需要规划或自我纠错的多步任务 |
| L4 编排 | `>>`、`Workflow`、`Team` | +5–10 行 | 研究报告流水线、多角色协作 |
| L5 内核 | 事件、中间件、协议、Runtime | 按需 | 自研范式、嵌入已有系统 |

### L0 · 对话

```python
agent = gole.Agent(model="ollama/qwen3:8b", instructions="你是一位耐心的语文老师")
agent.run("解释一下“画蛇添足”")
```

### L1 · 工具

普通函数加一个装饰器就是工具；内置工具用字符串引用。

```python
@gole.tool
def get_weather(city: str) -> str:
    """查询城市的实时天气。city: 城市中文名"""
    return weather_api(city)

agent = gole.Agent(model="ollama/qwen3:8b", tools=[get_weather, "calculator", "web_search"])
agent.run("上海今天适合跑步吗？")
```

### L2 · 记忆与知识库

```python
agent = gole.Agent(
    model="ollama/qwen3:8b",
    memory="long",          # 会话摘要 + 跨进程持久的长期记忆
    knowledge="./papers",   # 自动解析、切分、向量化该目录
)
agent.run("我上周让你记住的实验参数是什么？")
agent.run("这些论文用了哪些评测指标？")
```

### L3 · 推理范式

```python
agent = gole.Agent(
    model="deepseek/deepseek-chat",
    tools=["web_search", "web_fetch"],
    paradigm="plan",        # 先拆解计划，再逐步执行
)
agent.run("调研三款开源向量数据库，比较部署难度并给出推荐")
```

### L4 · 编排

```python
researcher = gole.Agent(name="researcher", tools=["web_search"], instructions="负责检索资料")
writer = gole.Agent(name="writer", instructions="负责把资料写成报告")

pipeline = researcher >> writer                              # 确定流程
pipeline.run("端侧大模型的最新进展")

team = gole.Team([researcher, writer], mode="supervisor")   # 开放协作
team.run("端侧大模型的最新进展")
```

### L5 · 内核

```python
@agent.on("tool_call")
def audit(event):
    log.info(f"{event.tool} {event.args}")

class MyParadigm(gole.Paradigm):
    async def step(self, state, ctx):
        ...

agent = gole.Agent(model="smart", paradigm=MyParadigm())
```

## 6. 核心模块使用指南

五个核心模块都遵循同一套规则：字符串起步、对象细调、`backend=` 换实现。内置实现的定位是「单机可用」：SQLite 持久化、增量索引、混合检索，足够支撑个人工具、原型和小团队；多实例或生产部署时，换成模式 B / C 的外部后端。

### 6.1 Model 模型

`gole.Model` 封装生成参数、超时重试和能力探测，也可以脱离 Agent 单独使用。

```python
fast  = gole.Model("ollama/qwen3:4b")
smart = gole.Model("deepseek/deepseek-chat", temperature=0.3, max_tokens=4096, timeout=60, retries=2)
local = gole.Model("Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1", api_key="EMPTY")

fast.generate("你好")     # 不经过 Agent，直接调用模型
fast.capabilities          # Capabilities(tools=True, json_mode=False, json_schema=True, vision=False, context=None)，首次访问时探测并缓存

embed = gole.Embedding("ollama/bge-m3")   # 嵌入模型，供记忆与知识库使用
```

- **工具调用降级**：`tool_mode="auto"`（默认）先探测模型能力，支持原生函数调用就用原生，否则用提示词约定格式并解析；也可以强制指定 `"native"` 或 `"prompt"`。`gole.Model` 和 `gole.Agent` 都接受 `tool_mode`，Agent 上的设置优先；没有注册任何工具时，不使用提示词协议，直接对话。
- **降级分阶段建设**：v0.1 提供通用 ReAct 模板、损坏 JSON 修复和带报错信息的重试；v0.3 起接入推理后端的约束解码（Ollama 结构化输出、vLLM guided decoding、llama.cpp GBNF），并发布小模型工具调用成功率基准。后端支持按 JSON Schema 约束输出时自动启用，`gole.Model(..., constrained=True / False)` 可以强制开关。
- **上下文长度**：本地推理服务实际可用的上下文往往比模型标称的小，超出的部分会被静默截断。`gole.Model(..., context=8192)`、环境变量 `GOLE_CONTEXT` 或 gole.yaml 的 `providers.<provider>.context` 可以把实际值告诉 AgentGOLE；提示词用量达到 90% 时会记一条 warning，`model_end` 事件带 `context_used`，`gole doctor` 也会显示这个值。自动压缩在 v0.8 提供。
- **多模型分工**：记忆摘要、规划、评审等环节可以各自指定模型，例如用小模型做摘要：`gole.Agent(model=smart, memory=gole.memory.Summary(model=fast))`。

### 6.2 Tool 工具

类型标注和 docstring 自动生成工具的 JSON Schema；同步、异步函数和 Pydantic 参数都支持。

```python
@gole.tool(timeout=30, retries=1)
async def search_paper(query: str, limit: int = 5) -> list[dict]:
    """在 arXiv 上搜索论文。
    query: 检索词
    limit: 返回数量
    """
    ...

@gole.tool(confirm=True)            # 执行前需要人工确认
def delete_file(path: str) -> str:
    ...
```

内置工具（字符串引用）：

| 名称 | 作用 | 备注 |
| --- | --- | --- |
| `calculator` | 安全的算式求值 | 不执行任意代码 |
| `datetime` | 当前时间与日期计算 |  |
| `web_search` | 网页搜索 | 搜索提供方可配置：`duckduckgo`（默认，无需密钥）、`tavily` |
| `web_fetch` | 抓取网页正文 |  |
| `python` | 执行 Python 代码 | 子进程运行，带超时且禁网；默认需要人工确认；Docker 沙箱 v0.8 起 |
| `files` | 读写指定目录下的文件 | 用 `gole.tools.files(root=...)` 指定目录；默认只读，`write=True` 时可写 |
| `shell` | 执行命令行 | 默认需要人工确认 |

接入 MCP 服务、其他框架的工具，或把另一个 Agent 当作工具：

```python
fs = gole.mcp("npx -y @modelcontextprotocol/server-filesystem ./data")   # stdio
remote = gole.mcp(url="https://example.com/mcp")                         # Streamable HTTP

agent = gole.Agent(tools=[
    fs.select("read_file", "list_directory"),     # 只开放部分工具
    remote,                                        # 展开为该服务的全部工具
    gole.tools.from_langchain(lc_tool),            # 复用 LangChain 工具
    writer.as_tool(description="把要点写成报告"),  # Agent 即工具
])
```

### 6.3 Paradigm 推理范式

范式决定 Agent 怎样「思考—行动」；默认的 `auto` 能覆盖大多数任务，只有长任务或质量敏感任务才需要切换。

| 范式 | 字符串 | 工作方式 | 适合 | 模型调用次数 |
| --- | --- | --- | --- | --- |
| 直答 | `direct` | 不用工具，一次生成 | 纯对话、改写 | 1 |
| 自动 | `auto` | 原生工具调用循环；不支持时降级为 ReAct | 默认 | 少 |
| ReAct | `react` | 思考 → 行动 → 观察，循环直到得出答案 | 需要可读推理过程、本地小模型 | 中 |
| 规划-执行 | `plan` | 先拆计划，再逐步执行，必要时重新规划 | 长任务、多步调研 | 较多 |
| 反思 | `reflect` | 生成 → 自我评审 → 修订 | 写作、代码等质量敏感任务 | 较多 |

```python
from agentgole import paradigms

agent = gole.Agent(
    paradigm=paradigms.PlanExecute(planner="smart", executor="fast", max_replans=2),
    max_steps=20,
)
agent = gole.Agent(
    paradigm=paradigms.Reflect(critic="smart", rounds=2, criteria="事实准确、结构清晰"),
)
```

### 6.4 Memory 记忆

记忆分三层：工作记忆（当前上下文）、会话记忆（同一会话的历史）、长期记忆（跨进程持久的事实与偏好，按 `session` 隔离）。四个预设覆盖常见需求：

| 取值 | 含义 | 等价对象 |
| --- | --- | --- |
| `None` | 无状态，每次调用独立 | — |
| `"session"`（默认） | 保留最近 N 轮原文 | `memory.Window(turns=20)` |
| `"summary"` | 超出窗口的历史滚动压缩成摘要 | `memory.Summary(max_tokens=2000)` |
| `"long"` | 摘要 + 跨进程持久的长期记忆 | `memory.Layered(short=Summary(), long=LongTerm())` |

```python
mem = gole.memory.Layered(
    short=gole.memory.Summary(max_tokens=2000, model="fast"),
    long=gole.memory.LongTerm(backend="mem0"),   # 或 "builtin"（SQLite）| "letta"
)
agent = gole.Agent(memory=mem)

agent.memory.add("用户偏好简洁的回答", session="u42")
agent.memory.search("用户偏好", session="u42")
agent.memory.export(session="u42")    # 导出为 JSON，便于调试
agent.memory.clear(session="u42")
```

- **写入时机**：长期记忆默认在每轮结束后异步抽取（`write="after_run"`）；设为 `"tool"` 则把 remember / recall 交给模型自主调用。抽取在最终回答发出之后、`run` 返回之前完成，不阻塞流式输出；抽取失败只发一个可恢复的 `error` 事件，不影响本轮回答。
- **隔离**：长期记忆按 `session` 隔离，一个 session 写入的事实不会出现在另一个 session 里。把 `session` 设为用户或对话的标识即可。v0.4 起可以另外传 `user=`（默认等于 `session`）：长期记忆按 `user` 归属，同一个用户的多段对话就能共享长期记忆，会话记忆仍按 `session` 区分。
- **存储位置**：内置实现默认写入 `./.gole/memory.db`（SQLite），单机可用；多实例部署时应切换到外部后端。

### 6.5 Knowledge 知识库

`knowledge=` 只给一个路径就能完成 RAG：解析、切分、嵌入、入库、检索、标注引用，每一步都有默认值。

```python
kb = gole.Knowledge(
    sources=["./papers", "./notes/*.md", "https://example.com/faq"],
    embedding="ollama/bge-m3",
    store="./.gole/kb",            # 本地持久化，再次运行时增量更新
    chunk_size=512, chunk_overlap=64,   # 按字符计
    retriever="hybrid",            # "vector" | "keyword" | "hybrid"
    rerank=None,                   # 例如 "bge-reranker"（v0.4 起）
    backend="builtin",             # 或 "llamaindex" | "langchain"（v0.4 起）
)

kb.search("对比学习的损失函数", k=5)    # 直接检索，返回带来源的片段
kb.ask("这批论文用了哪些数据集？")       # 一行完成 RAG 问答
kb.add("./new_paper.pdf")

agent = gole.Agent(knowledge=kb)
result = agent.run("...")
result.citations   # [Citation(source="papers/xx.pdf", page=3, text="..."), ...]
```

知识库接入 Agent 的方式由 `knowledge_mode` 决定：

| `knowledge_mode` | 行为 | 适合 |
| --- | --- | --- |
| `"auto"`（默认） | 模型支持工具调用时用 `tool`，否则用 `context` | — |
| `"tool"` | 检索作为工具，由 Agent 决定何时检索、检索什么 | 问题多样、模型较强 |
| `"context"` | 每轮先检索，结果注入上下文 | 小模型、纯问答 |

多个知识库可以同时挂载：`knowledge=[kb_papers, kb_code]`，每个库会变成一个带名称和描述的独立检索工具。

## 7. 工作流与多智能体

一个 Agent 不够用时有两种编排：步骤固定、可预测，用 Workflow；分工明确但顺序不定，用 Team。两者对外都和 Agent 一样提供 `run / stream / as_tool`，可以任意嵌套。

### 7.1 Workflow：管道与控制流

一个步骤可以是 Agent、普通函数，或者提示词模板 `gole.prompt(...)`。模板中 `{input}` 是上一步的输出，`{task}` 是最初的输入，`{<步骤名>}` 是任意已完成步骤的输出。

```python
outline = gole.prompt("为主题「{task}」写一个 5 节的大纲", model="smart")
draft   = gole.Agent(name="writer", tools=["web_search"], instructions="按大纲逐节撰写，附来源")
polish  = gole.prompt("润色下面的文章，保持事实不变：\n{input}", model="fast")

flow = outline >> draft >> polish
flow.run("端侧大模型的现状")
```

分支、并行和循环用三个原语表达：

```python
from agentgole.flow import branch, parallel, loop

flow = (
    classify                                                        # 输出带 label 字段的结构化结果
    >> branch(lambda x: x.label, {"技术": tech_agent, "商务": biz_agent, "_": general_agent})
    >> parallel(fact_checker, style_checker)                       # 并行执行，结果合并为 dict
    >> loop(revise, until=lambda r: r.passed, max_iter=3)          # 不通过就修订，最多 3 次
)
```

普通函数步骤可以声明 `ctx` 参数，读写整条流程共享的状态：

```python
def save_draft(text: str, ctx: gole.Context) -> str:
    ctx.state["draft"] = text
    return text
```

- **断点续跑**：`flow.run(..., checkpoint="sqlite:///.gole/runs.db")`，中断后用 `flow.resume(run_id)` 从失败的步骤继续。
- **切换到 LangGraph**：`flow.compile("langgraph")` 得到 LangGraph 的编译图，可以继续使用它的检查点与可视化工具。
- **声明式定义**：同一条流程也可以写成 YAML，用 `gole run` 执行。

```yaml
# report.flow.yaml
name: report
steps:
  - id: outline
    prompt: "为主题「{task}」写一个 5 节的大纲"
    model: smart
  - id: draft
    agent: agents/writer.yaml
  - id: polish
    prompt: "润色下面的文章，保持事实不变：\n{input}"
    model: fast
```

```bash
gole run report.flow.yaml --input "端侧大模型的现状"
```

### 7.2 Team：多智能体协作

| `mode` | 协作方式 | 适合 |
| --- | --- | --- |
| `router` | 路由器把任务交给最合适的一个成员 | 客服分流、意图识别 |
| `supervisor` | 主管拆任务、分派、汇总 | 调研、报告 |
| `handoff` | 成员之间可以主动把对话移交给对方 | 多阶段对话（售前 → 售后） |
| `debate` | 成员轮流发言、互相质疑，最后由裁判给出结论 | 方案评审、降低幻觉 |

```python
team = gole.Team(
    members=[researcher, analyst, writer],
    mode="supervisor",
    supervisor=gole.Agent(model="smart", instructions="你是项目经理，负责拆解和验收"),
    max_rounds=8,
)
result = team.run("写一份开源 Agent 框架的选型报告")
result.transcript   # 成员之间的完整对话

# 组合：Team 可以是 Workflow 的一步，也可以是另一个 Team 的成员
flow = intake >> team >> gole.prompt("把报告压缩成 300 字摘要：\n{input}")
```

## 8. 作为内核使用

开发者可以把 AgentGOLE 当作 Agent 内核嵌入自己的产品：订阅事件、插入中间件、替换任意组件，或者手动驱动运行循环。

### 8.1 事件

每次 `run` 都会产生一串有序事件；`stream()`、Hook、轨迹记录和 `serve` 的 SSE 都建立在同一套事件之上。

| 事件 | 触发时机 | 关键字段 |
| --- | --- | --- |
| `run_start` / `run_end` | 一次运行开始 / 结束 | `input`、`session` / `result` |
| `memory_load` / `memory_save` | 读取 / 写入记忆 | `messages` |
| `retrieve` | 知识库检索 | `query`、`chunks` |
| `model_start` / `token` / `model_end` | 一次模型调用 | `messages` / `text` / `usage` |
| `tool_call` / `tool_result` | 一次工具调用 | `tool`、`args` / `output`、`error` |
| `step` | 范式完成一步 | `thought`、`action` |
| `approval_required` | 工具需要人工确认 | `tool`、`args` |
| `error` | 出错 | `exception`、`recoverable` |

```python
@agent.on("tool_call")
def audit(ev):
    print(f"[{ev.run_id}] 调用 {ev.tool}({ev.args})")

for ev in agent.stream("...", events=["step", "tool_call", "token"]):
    ...
```

### 8.2 中间件

中间件包裹模型调用和工具调用，可以修改请求、拦截或缓存。

```python
agent.use(
    gole.middleware.Retry(max_attempts=3),
    gole.middleware.TokenBudget(max_tokens=50_000),
    gole.middleware.PIIMask(),                         # 发给模型前脱敏
    gole.middleware.Cache("sqlite:///.gole/cache.db"),
)

@agent.middleware("tool")
async def guard_paths(call, next):
    if call.tool == "files" and ".." in call.args.get("path", ""):
        raise gole.ToolRejected("禁止访问上级目录")
    return await next(call)
```

### 8.3 自定义组件：实现协议即可

每类组件都是一个 Python Protocol，只需实现少数几个方法。

| 组件 | 需要实现的方法 |
| --- | --- |
| `ModelClient` | `generate(messages, tools, **params)`、`stream(...)` |
| `Tool` | `name`、`schema`、`call(**args)` |
| `Paradigm` | `step(state, ctx) -> Action` |
| `Memory` | `load(session, query)`、`save(session, messages)` |
| `Retriever` | `search(query, k)` |

```python
class RedisMemory(gole.protocols.Memory):
    def __init__(self, url: str): ...
    async def load(self, session, query=None): ...
    async def save(self, session, messages): ...

gole.register("memory", "redis", RedisMemory)
agent = gole.Agent(memory="redis")       # 注册后就能用字符串引用
```

第三方包通过 Python entry points（`agentgole.plugins`）自动注册，例如 `pip install agentgole-redis` 后直接写 `memory="redis"`。

自定义范式通过返回 `Action` 告诉运行时下一步做什么（`CallModel`、`CallTool`、`Delegate`、`AskHuman`、`Finish`）。范式自己不直接调用模型：每一次模型调用都以 `CallModel` 交给运行时，这样事件、用量统计和步数上限才完整。`CallModel` 可以用 `model=` 指定另一个模型，用 `visible=False` 表示这次的文本是中间产物、不作为 token 发给用户，用 `response_schema=` 要求回复满足某个 JSON Schema；回复在下一次 `step` 的 `state.last_response` 里。

```python
import json
from agentgole import actions

SUBQUESTIONS = {"type": "array", "items": {"type": "string"}}

class SelfAsk(gole.Paradigm):
    """先把问题拆成子问题，逐个回答后汇总"""
    async def step(self, state, ctx):
        phase = state.scratch.setdefault("phase", "split")
        if phase == "split":                       # 第 1 步：请模型拆解问题
            state.scratch["phase"] = "solve"
            ask = {"role": "user", "content": f"把问题拆成子问题：{state.input}"}
            return actions.CallModel(messages=[ask], visible=False, response_schema=SUBQUESTIONS)
        if phase == "solve":
            if "todo" not in state.scratch:        # 上一步的回复就是子问题列表
                state.scratch["todo"] = json.loads(state.last_response.content)
            if state.scratch["todo"]:
                q = state.scratch["todo"].pop(0)
                return actions.Delegate(q, paradigm="react")    # 子问题交给 ReAct 解决
            state.scratch["phase"] = "answer"
            ask = {"role": "user", "content": f"汇总以下答案：{state.history}"}
            return actions.CallModel(messages=[ask])            # 最终回答，逐 token 发出
        return actions.Finish(state.last_response.content)
```

### 8.4 手动驱动运行循环

`agent.start()` 返回一个可迭代的 Run：每次迭代拿到「将要执行」的动作，由你决定执行、修改还是拒绝。

```python
run = agent.start("清理 ./tmp 下 7 天前的日志")
for action in run:
    if action.kind == "tool_call" and action.tool == "shell" and not user_approves(action):
        run.reject(action, reason="用户拒绝")
        continue
    run.execute(action)
print(run.result)
```

### 8.5 逃生舱

```python
kb.native                    # backend="llamaindex" 时是 LlamaIndex 的索引对象
agent.memory.native          # backend="mem0" 时是 mem0 的 Memory 实例
agent.model.native           # 底层 HTTP / SDK 客户端
flow.compile("langgraph")    # LangGraph 的编译图
```

### 8.6 序列化

Agent 的配置可以导出为 YAML（不含密钥），便于版本管理和共享。

```python
agent.save("agents/writer.yaml")
agent = gole.Agent.load("agents/writer.yaml")
```

```yaml
# agents/writer.yaml
name: writer
model: smart
instructions: 按大纲逐节撰写，每个论断附来源链接
tools: [web_search, web_fetch]
memory: summary
paradigm: reflect
```

## 9. 可观测性、安全与部署

每次运行都可追溯、有危险的动作默认要确认、一条命令就能部署成服务。

### 9.1 运行轨迹

- `gole.Agent(..., verbose=True)`：在终端实时打印步骤树。
- `result.trace.show()` / `.to_json("trace.json")` / `.to_html("trace.html")`：事后查看或导出。运行失败时，轨迹在异常的 `trace` 属性上。
- 环境变量 `GOLE_TRACE_DIR`：设置后，每次运行结束或失败时自动把轨迹写成 `<目录>/<run_id>.json`，方便把出问题的运行发给别人分析。轨迹里有完整的对话内容，发出前先检查。
- `gole.trace.export("otel")` 或 `"langfuse"`：接入已有的观测平台。

```text
run 3f2a  "上海今天适合跑步吗？"                          2.8s   1,204 tok
├─ model  qwen3:8b  → tool_call get_weather               0.9s
├─ tool   get_weather(city="上海") → "多云 18°C AQI 42"     0.3s
└─ model  qwen3:8b  → 最终回答                             1.6s
```

（上为示意输出，数字不代表实测。）

### 9.2 安全与人工介入

- **工具审批**：`@gole.tool(confirm=True)` 的工具在执行前触发 `approval_required`；审批方式通过 `gole.Agent(approval=...)` 指定，可选 `gole.approval.Console()`（终端 y/n）、`Callback(fn)` 或 `Webhook(url)`（v0.8 起）。没有指定审批方式时，需要确认的工具不会执行，模型会收到拒绝原因；`gole chat` 默认使用终端确认。
- **资源上限**：`max_steps`、`timeout`、`TokenBudget` 中间件，防止死循环和费用失控。
- **沙箱**：`python` 工具在子进程中运行，带超时且禁网，但这不是安全边界，所以它和 `shell` 一样默认需要确认；v0.8 起可改为 `sandbox="docker"`。`files` 必须指定 `root`，默认只读。
- **提示词注入防护**：工具返回和检索片段默认用标记包裹，并在系统提示中声明为不可信数据。

### 9.3 持久化与恢复

`checkpoint=` 接受 SQLite、Redis 或 Postgres 地址。运行在等待人工审批时可以挂起，进程重启后用 `agent.resume(run_id)` 继续。

### 9.4 部署为服务

```bash
gole serve my_agent.py:agent --port 8000
```

- `POST /v1/chat/completions`：OpenAI 兼容接口，任何 OpenAI SDK 或聊天前端都能直接接入你的 Agent。
- `POST /runs` + `GET /runs/{id}/events`：完整事件流（SSE）。
- `GET /health`：健康检查。

也可以挂载到已有的 ASGI 应用：`app.mount("/agent", gole.asgi(agent))`。

### 9.5 CLI 速查

| 命令 | 作用 |
| --- | --- |
| `gole init` | 生成 `gole.yaml` 与示例代码 |
| `gole doctor` | 检查模型连通性与能力 |
| `gole chat [target]` | 终端对话 |
| `gole run <flow.yaml>` | 运行工作流 |
| `gole serve <target>` | 启动 HTTP 服务 |
| `gole engines` | 列出已安装的运行引擎及其能力 |
| `gole kb build <source...>` | 预先构建知识库索引；可选 `--embedding`、`--store`、`--name` |
| `gole trace <run_id>` | 查看某次运行的轨迹 |

## 10. 后端适配矩阵与路线图

每个模块都有一个零额外依赖的内置实现，现有框架只作为可选后端接入。所有后端必须通过同一套协议测试（conformance tests），保证切换后行为一致。

### 10.1 后端适配矩阵

| 模块 | 内置实现（零额外依赖） | 可选后端 | 安装 |
| --- | --- | --- | --- |
| Model | OpenAI 兼容 HTTP 客户端，Ollama / Anthropic 适配 | LiteLLM（覆盖更多厂商） | `agentgole[litellm]` |
| Tool | `@tool`、内置工具集 | MCP、LangChain Tools | `agentgole[mcp]`、`agentgole[langchain]` |
| Memory | Window、Summary、SQLite 长期记忆 | mem0、Letta | `agentgole[mem0]`、`agentgole[letta]` |
| Knowledge | 轻量 RAG（本地向量 + BM25） | LlamaIndex、LangChain；存储可选 Chroma、Qdrant、Milvus | `agentgole[llamaindex]`、`agentgole[chroma]` 等 |
| Workflow | 内置 DAG 运行时 | LangGraph | `agentgole[langgraph]` |
| Team | router / supervisor / handoff / debate | AutoGen、CrewAI（待调研） | — |
| Trace | 终端、JSON、HTML | OpenTelemetry、Langfuse | `agentgole[otel]` |
| Engine（运行引擎，模式 C） | native（AgentGOLE 原生内核） | LangGraph 首个接入；LlamaIndex、smolagents、OpenAI Agents SDK 等随路线图逐步评估接入 | `agentgole[langgraph]` 等 |

### 10.2 版本路线图

v0.1 到 v1.0 共 10 个版本，分三个阶段逐级解锁复杂度阶梯和运行模式；每个阶段都要通过验收门槛，才进入下一阶段。逐版本的任务清单和项目结构见 `docs/DEV_PLAN.md`。

```text
v0.1–0.3  模式 A 打底   内核骨架 · 工具 · 记忆 · 知识库 · plan/reflect 范式（L0–L3）
                       工具调用降级两阶段 · 小模型基准 · gole chat / doctor
   ◇ 门槛：Cookbook 11.1、11.2 全离线跑通，工具调用基准已发布
v0.4–0.6  模式 B 与编排 后端适配层 · 协议测试 · mem0 · LlamaIndex · MCP · LiteLLM
                       Workflow · Team · 中间件 · 插件机制 · gole serve（L4–L5）
   ◇ 门槛：所有后端通过协议测试，Cookbook 11.3–11.5 跑通
v0.7–0.9  模式 C 与生态 LangGraph 引擎 · 能力矩阵 · 跨模式测试
                       可观测与安全加固 · 更多引擎与后端 · 文档站 · 1.0.0rc
   ◇ 门槛：Cookbook 5 个场景在 A / B / C 三种模式下都能跑通
v1.0      稳定版       公共 API 冻结 · 语义化版本 · 完整 API 参考与迁移指南
```

路线图沿两条线同时推进：复杂度阶梯（L0 → L5）和运行模式（A → B → C）。v0.1 先把模式 A 下「3 行代码」的承诺做扎实，之后每个版本再加入新的用法、框架后端和运行引擎。

## 11. Cookbook：典型场景

五个场景覆盖了从初学者到内核开发者的主要用法，每个都能单独复制运行。

### 11.1 全离线的本地文档问答（L2）

```python
import agentgole as gole

kb = gole.Knowledge("./company_docs", embedding="ollama/bge-m3")
agent = gole.Agent(
    model="ollama/qwen3:8b",
    knowledge=kb,
    instructions="只根据知识库回答并标注来源；找不到就明确说不知道",
)

r = agent.run("报销流程需要哪些审批？")
print(r.text)
for c in r.citations:
    print(f"- {c.source} 第 {c.page} 页")
```

### 11.2 记得你的个人助理（L2）

```python
assistant = gole.Agent(
    model="deepseek/deepseek-chat",
    memory="long",
    tools=["web_search", "datetime"],
)

assistant.run("记住：我周三下午固定开组会", session="me")
# 一周后，在另一个进程里
assistant.run("这周哪天下午有空？", session="me")
```

### 11.3 研究报告流水线（L3 + L4）

```python
search = gole.Agent(name="search", tools=["web_search", "web_fetch"],
                    instructions="检索 5–8 条高质量资料，输出要点和链接")
outline = gole.prompt("主题：{task}\n资料：{input}\n写一个 5 节大纲", model="smart")
writer = gole.Agent(name="writer", model="smart", paradigm="reflect",
                    instructions="按大纲成文，每个论断附来源链接")

report = search >> outline >> writer
print(report.run("2026 年开源 Agent 框架格局", checkpoint="sqlite:///.gole/runs.db"))
```

### 11.4 把 AgentGOLE 嵌入已有的 FastAPI 项目（L5）

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import agentgole as gole

app = FastAPI()
agent = gole.Agent.load("agents/support.yaml")

@app.post("/chat")
async def chat(q: str, user_id: str):
    async def sse():
        async for ev in agent.astream(q, session=user_id, events=["token", "tool_call"]):
            yield f"data: {ev.to_json()}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
```

### 11.5 复用已有的 LangChain / LlamaIndex 资产

```python
agent = gole.Agent(
    model="smart",
    tools=[gole.tools.from_langchain(my_langchain_tool)],     # 已有的 LangChain 工具
    knowledge=gole.Knowledge.from_llamaindex(my_index),       # 已建好的 LlamaIndex 索引
)
```

## 12. 设计决策记录

以下决定会改变文档和后续开发的方向；10 项均已确定；名称为暂定，正式发布前再确认 PyPI 与 GitHub 无重名。

| # | 问题 | 可选方向 | 结论 |
| --- | --- | --- | --- |
| 1 | 内核路线 | A 自有内核；B 自有内核 + 框架组件；C 框架托管 | 已定：三种都支持，用户自由切换（§2.2）；原生内核以 NanoHarness 为骨架（§2.3） |
| 2 | v0.1 首要用户 | 初学者 / 应用开发者 / 内核开发者 | 已定：三者兼顾，用法与支持的框架随路线图逐步扩展 |
| 3 | 与 Minimal Local Agent 的关系 | 在其基础上演进 / 新仓库复用部分代码 / 完全独立 | 已定：从本项目目标出发，必要时复用其代码（§2.3） |
| 4 | API 风格 | 异步内核 + 同步包装 / 纯同步 | 已定：异步内核 + 同步包装（§4.3） |
| 5 | 内置实现做多厚 | 教学级最简版 / 单机可用 / 可生产 | 已定：单机可用，生产级交给后端（§6） |
| 6 | 本地模型工具调用降级 | 作为核心卖点 / 只做基础兼容 | 已定：分阶段投入（§6.1） |
| 7 | 声明式定义（YAML） | 提供 / 不提供 | 已定：提供，代码优先 |
| 8 | 项目名称 | 芥子 / 七巧 / 鹊桥 / AgentGOLE 等 | 已定（暂定）：AgentGOLE，取自 LEGO 的字母重排；PyPI 包名 agentgole，命令行 gole |
| 9 | 语言 | 仅 Python / 后续加 TypeScript | 已定：首发仅 Python |
| 10 | 开源协议 | Apache-2.0 / MIT | 已定：Apache-2.0 |
