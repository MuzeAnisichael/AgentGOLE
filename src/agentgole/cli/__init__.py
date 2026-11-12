"""The gole init, doctor and chat commands."""

import argparse
import re
import runpy
import sys
from copy import copy
from pathlib import Path
from uuid import uuid4

from .. import Agent, Model
from .._sync import run_sync
from ..config import default_model, load_config
from ..errors import AgentGoleError, ConfigError
from ..models import create_client
from . import messages as msg


def _init(force: bool) -> int:
    templates = {Path("gole.yaml"): msg.TEMPLATE, Path("agent_example.py"): msg.EXAMPLE}
    existing = [str(path) for path in templates if path.exists()]
    if existing and not force:
        print(msg.EXISTS.format(", ".join(existing)), file=sys.stderr)
        return 1
    for path, template in templates.items():
        with path.open("w" if force else "x", encoding="utf-8") as file:
            file.write(template)
    print(msg.CREATED)
    return 0


async def _doctor() -> int:
    config = load_config()
    aliases = config.get("models", {})
    specs: list[str] = list(aliases)
    try:
        specs.insert(0, default_model(config))
    except ConfigError:
        if not specs:
            print(msg.NO_MODEL, file=sys.stderr)
            return 1
    failed = False
    seen: set[str] = set()
    for spec in specs:
        label = aliases.get(spec, spec)
        if label in seen:
            continue
        seen.add(label)
        try:
            client = create_client(spec, config=config, retries=0)
            await client.check_connection()
            caps = await client.probe()
            print(msg.OK.format(client.name, msg.NATIVE if caps.tools else msg.PROMPT_MODE))
        except AgentGoleError as exc:
            key = re.search(r"\b[A-Z][A-Z0-9_]+_API_KEY\b", str(exc))
            reason = (
                msg.MISSING.format(key.group())
                if key and isinstance(exc, ConfigError)
                else str(exc)
            )
            print(msg.FAIL.format(label, reason))
            failed = True
    return int(failed)


def _load_agent(target: str) -> Agent:
    path, separator, attr = target.rpartition(":")
    if not separator or not attr or not Path(path).is_file():
        raise ConfigError(msg.BAD_TARGET)
    directory = str(Path(path).resolve().parent)
    sys.path.insert(0, directory)
    try:
        value = runpy.run_path(path).get(attr)
    finally:
        sys.path.remove(directory)
    if not isinstance(value, Agent):
        raise ConfigError(msg.BAD_TARGET)
    return value


def _chat(target: str | None, model: str | None) -> int:
    agent = _load_agent(target) if target else Agent(model)
    if target and model:
        agent = copy(agent)
        agent._binding = Model(model)._binding
    session = "default"
    while True:
        try:
            prompt = input(msg.PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if prompt == "/exit":
            return 0
        if prompt == "/reset":
            session = uuid4().hex
            print(msg.RESET)
        elif prompt:
            wrote_text = False
            for ev in agent.stream(prompt, session=session, events=["token", "run_end"]):
                if ev.type == "token":
                    print(ev.text, end="", flush=True)
                    wrote_text = True
                elif not wrote_text:
                    print(ev.result.text, end="")
            print()


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command and return its exit status."""
    parser = argparse.ArgumentParser(prog="gole", description=msg.DESCRIPTION)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help=msg.INIT_HELP)
    init.add_argument("--force", action="store_true", help=msg.FORCE_HELP)
    commands.add_parser("doctor", help=msg.DOCTOR_HELP)
    chat = commands.add_parser("chat", help=msg.CHAT_HELP)
    chat.add_argument("--model", help=msg.MODEL_HELP)
    chat.add_argument("target", nargs="?", help=msg.TARGET_HELP)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _init(args.force)
        if args.command == "doctor":
            return run_sync(_doctor())
        return _chat(args.target, args.model)
    except (AgentGoleError, OSError) as exc:
        print(msg.FAILED.format(exc), file=sys.stderr)
        return 1
