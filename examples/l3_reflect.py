"""Draft an answer offline, let a critic review it and revise it once if needed."""

from os import getenv

import golekit as gole


def main() -> gole.Result:
    """Write a short notice, review it against explicit criteria and print the final text."""
    agent = gole.Agent(
        model=getenv("GOLE_MODEL", "ollama/qwen3:8b"),
        paradigm=gole.paradigms.Reflect(
            rounds=1, criteria="必须写明时间、地点和需要携带的物品，全文不超过 120 字"
        ),
    )
    result = agent.run("写一则通知：本周五下午三点在三楼会议室做消防演练，请大家带上工牌。")
    reviews = sum(e.type == "model_start" for e in result.trace.events) - 1
    print(f"评审与修订共 {reviews} 次模型调用")
    print(result.text)
    return result


if __name__ == "__main__":
    main()
