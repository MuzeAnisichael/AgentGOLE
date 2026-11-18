from os import getenv

import agentgole as gole

agent = gole.Agent(
    model=getenv("GOLE_MODEL", "ollama/qwen3:8b"), instructions="你是一位耐心的语文老师"
)
print(agent.run("我叫小林。请解释一下“画蛇添足”。"))
print(agent.run("我叫什么？"))
print(agent.run("我叫什么？", session="user-42"))
