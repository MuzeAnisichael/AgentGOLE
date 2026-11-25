"""Run twice from the same directory to recall a fact in a fresh process."""

import argparse

import golekit as gole


def main() -> None:
    """Store or recall facts using the same local session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message")
    args = parser.parse_args()
    agent = gole.Agent(
        "ollama/qwen3:8b", memory="long", instructions="结合已知信息，用中文简短回答。"
    )
    print(agent.run(args.message, session="me"))


if __name__ == "__main__":
    main()
