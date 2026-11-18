"""Answer a question about the bundled fictional expense policy using Ollama."""

from pathlib import Path

import agentgole as gole


def main() -> gole.Result:
    """Retrieve local documents and print the answer and its sources."""
    docs = Path(__file__).resolve().parent / "data" / "company_docs"
    kb = gole.Knowledge(docs, embedding="ollama/bge-m3")
    agent = gole.Agent(
        "ollama/qwen3:8b",
        knowledge=kb,
        instructions="请先调用知识库检索工具，仅根据检索资料回答并给出审批人的完整职称。",
    )
    result = agent.run("报销流程需要哪些审批？")
    print(result.text)
    for citation in result.citations:
        page = "" if citation.page is None else f" 第 {citation.page} 页"
        print(f"- {citation.source}{page}")
    return result


if __name__ == "__main__":
    main()
