"""The gole init, doctor and chat commands."""

import argparse
import os
import re
import runpy
import sys
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .. import Agent, Embedding, Knowledge, Model, approval
from .._sync import run_sync
from ..config import default_model, load_config
from ..errors import ConfigError, GoleError
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
    specs = [(name, name == "embed") for name in aliases]
    try:
        specs.insert(0, (default_model(config), False))
    except ConfigError:
        if not specs:
            print(msg.NO_MODEL, file=sys.stderr)
            return 1
    failed = False
    seen: set[tuple[str, bool]] = set()
    for spec, is_embedding in specs:
        label = aliases.get(spec, spec)
        identity = (label, is_embedding)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            if is_embedding:
                encoder = Embedding(spec, retries=0)
                await encoder.aembed(["ping"])
                print(msg.EMBED_OK.format(encoder.name))
                continue
            client = create_client(spec, config=config, retries=0)
            await client.check_connection()
            caps = await client.probe()
            print(
                msg.OK.format(
                    client.name,
                    msg.NATIVE if caps.tools else msg.PROMPT_MODE,
                    msg.NATIVE if caps.json_schema else msg.UNSUPPORTED,
                )
            )
        except GoleError as exc:
            key = re.search(r"\b[A-Z][A-Z0-9_]+_API_KEY\b", str(exc))
            reason = (
                msg.MISSING.format(key.group())
                if key and isinstance(exc, ConfigError)
                else str(exc)
            )
            print(msg.FAIL.format(label, reason))
            failed = True
    return int(failed)


def _load_agent(target: str, model: str | None) -> Agent:
    path, separator, attr = target.rpartition(":")
    if not separator or not attr or not Path(path).is_file():
        raise ConfigError(msg.BAD_TARGET)
    directory = str(Path(path).resolve().parent)
    sys.path.insert(0, directory)
    previous = os.environ.get("GOLE_MODEL")
    try:
        if model:
            os.environ["GOLE_MODEL"] = model
        value = runpy.run_path(path).get(attr)
    finally:
        sys.path.remove(directory)
        if model:
            if previous is None:
                os.environ.pop("GOLE_MODEL", None)
            else:
                os.environ["GOLE_MODEL"] = previous
    if not isinstance(value, Agent):
        raise ConfigError(msg.BAD_TARGET)
    return value


def _chat(target: str | None, model: str | None) -> int:
    agent = _load_agent(target, model) if target else Agent(model)
    if target and model:
        agent._use_model(Model(model)._binding)
    if agent._approval is None:
        agent._approval = approval.Console()
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
            try:
                kinds = ["token", "tool_call", "run_end"]
                for ev in agent.stream(prompt, session=session, events=kinds):
                    if ev.type == "token":
                        print(ev.text, end="", flush=True)
                        wrote_text = True
                    elif ev.type == "tool_call":
                        # Text streamed before a tool call is not part of the final answer.
                        if wrote_text:
                            print()
                        wrote_text = False
                    elif not wrote_text:
                        print(ev.result.text, end="")
            except (GoleError, OSError) as exc:
                print(msg.FAILED.format(exc), file=sys.stderr)
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
    kb = commands.add_parser("kb", help=msg.KB_HELP)
    sub = kb.add_subparsers(dest="kb_command", required=True)
    build = sub.add_parser("build", help=msg.BUILD_HELP)
    build.add_argument("sources", nargs="+")
    build.add_argument("--embedding")
    build.add_argument("--store", default="./.gole/kb")
    build.add_argument("--name")
    args = parser.parse_args(argv)
    try:
        if args.command == "kb":
            started = monotonic()
            knowledge = Knowledge(
                args.sources, embedding=args.embedding, store=args.store, name=args.name
            )
            counts = run_sync(knowledge.abuild())
            print(msg.BUILD_STATS.format(**counts, elapsed=monotonic() - started))
            return 0
        if args.command == "init":
            return _init(args.force)
        if args.command == "doctor":
            return run_sync(_doctor())
        return _chat(args.target, args.model)
    except (GoleError, OSError) as exc:
        print(msg.FAILED.format(exc), file=sys.stderr)
        return 1
