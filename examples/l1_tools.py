from os import getenv

import golekit as gole


@gole.tool
def get_weather(city: str) -> str:
    """Return the fixed weather report for the example city.

    Args:
        city: City to look up.
    """
    return "小雨，18°C"


agent = gole.Agent(model=getenv("GOLE_MODEL", "ollama/qwen3:8b"), tools=["calculator", get_weather])
result = agent.run(
    "请调用 get_weather 查询上海天气，并调用 calculator 计算 247 * 13。"
    "在最终回答中引用工具给出的气温数字和计算结果。"
)
print(result)
result.trace.show()
