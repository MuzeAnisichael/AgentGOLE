"""Plan a multi-step calculation offline, execute it with tools and summarize it."""

from os import getenv

import golekit as gole


def main() -> gole.Result:
    """Split the task into steps, run each step with the local tools and print the answer."""
    agent = gole.Agent(
        model=getenv("GOLE_MODEL", "ollama/qwen3:8b"),
        tools=["calculator", "datetime"],
        paradigm="plan",
        instructions="回答要简短，数字必须来自工具的结果。",
    )
    result = agent.run(
        "我每个工作日存 86.5 元，一年按 250 个工作日算能存多少？"
        "再告诉我今天是星期几，以及这笔钱平均到 12 个月每月是多少。"
    )
    plan = next(e.thought for e in result.trace.events if e.type == "step" and e.thought)
    print("计划：")
    print(plan)
    print("回答：")
    print(result.text)
    return result


if __name__ == "__main__":
    main()
