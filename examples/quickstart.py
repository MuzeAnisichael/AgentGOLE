from os import getenv

import agentgole as gole

agent = gole.Agent(model=getenv("GOLE_MODEL", "ollama/qwen3:8b"))
print(agent.run("用一句话介绍你自己"))
