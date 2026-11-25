"""Chinese copy and starter files used by the command-line interface."""

DESCRIPTION = "配置、检查和运行 GoleKit。"
INIT_HELP = "生成 gole.yaml 与示例代码"
DOCTOR_HELP = "检查模型连通性与能力"
CHAT_HELP = "在终端与 Agent 对话"
FORCE_HELP = "覆盖已有的模板文件"
MODEL_HELP = "使用指定的模型或别名"
TARGET_HELP = "从 path/to/file.py:attr 加载 Agent"
EXISTS = "文件已存在，请使用 --force 覆盖：{}"
CREATED = "已生成 gole.yaml 和 agent_example.py"
NO_MODEL = "请设置 GOLE_MODEL 或运行 gole init。"
OK = "[ok]   {:<24} 可连接   原生工具调用: {}   约束解码: {}   上下文: {}"
UNKNOWN = "未知"
CONTEXT_HINT = (
    "       上下文长度未知：可用 ollama ps 查看实际值，"
    "再通过 gole.Model(..., context=N) 或 gole.yaml 的 providers.ollama.context 告诉 GoleKit。"
)
EMBED_OK = "[ok]   {:<24} 可连接   文本嵌入"
FAIL = "[fail] {:<24} {}"
NATIVE = "支持"
UNSUPPORTED = "不支持"
PROMPT_MODE = "不支持 → 将使用提示词模式"
MISSING = "缺少 {}"
FAILED = "运行失败：{}"
BAD_TARGET = "请使用 path/to/file.py:attr，并确认该属性是 Agent 实例。"
PROMPT = "你> "
RESET = "已开始新会话。"
TEMPLATE = """\
default_model: ollama/qwen3:8b
engine: native

models:
  fast: ollama/qwen3:4b
  smart: deepseek/deepseek-chat
  embed: ollama/bge-m3

providers:
  ollama:
    base_url: http://localhost:11434
  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
"""
EXAMPLE = """\
import golekit as gole

agent = gole.Agent()
if __name__ == "__main__":
    print(agent.run("用一句话介绍你自己"))
"""

KB_HELP = "管理本地知识库"
BUILD_HELP = "按来源增量构建知识库"
BUILD_STATS = (
    "文件 {files}，新增块 {added}，跳过块 {skipped}，删除块 {deleted}，耗时 {elapsed:.2f} 秒"
)
