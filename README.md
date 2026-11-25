# GoleKit

在项目目录安装（Python 3.11 及以上）。

```sh
uv sync --dev
```

启动 Ollama 并安装模型（`ollama pull qwen3:8b`）后运行。

```python
import golekit as gole
agent = gole.Agent(model="ollama/qwen3:8b")
print(agent.run("用一句话介绍你自己"))
```
