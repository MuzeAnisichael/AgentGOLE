"""Drive a paradigm to completion within a run's action and time budgets."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

from .._json import OutputSchema
from ..errors import (
    MaxStepsError,
    ModelError,
    OutputValidationError,
    RunTimeoutError,
    UnsupportedToolsError,
)
from ..protocols import EventSink, Memory, Paradigm, Retriever, StepContext
from ..protocols.types import (
    Action,
    CallModel,
    CallTool,
    Finish,
    Message,
    ResolvedMode,
    Result,
    RunState,
    Step,
    Trace,
    Usage,
)
from .context import append_results, initial_state, prepare_knowledge
from .evaluator import ModelCalls, call_model, call_tools
from .events import RunEvents

_logger = logging.getLogger("golekit.kernel.engine")


def _answer_tokens(events: RunEvents) -> str:
    """Join the tokens emitted since the last tool result."""
    texts: list[str] = []
    for event in reversed(events.trace.events):
        if event.type == "tool_result":
            break
        if event.type == "token":
            texts.append(event.text)
    return "".join(reversed(texts))


async def _actions(
    state: RunState,
    paradigm: Paradigm,
    ctx: StepContext,
    events: RunEvents,
    max_steps: int,
    streaming: bool,
    retrievers: Sequence[Retriever],
    knowledge_mode: str,
    schema: OutputSchema | None,
    calls: ModelCalls,
) -> tuple[str, Usage, Any]:
    usage = Usage()
    corrections = 0
    next_action: Action | None = None
    validated: Any = None
    # None while the latest visible reply was emitted as it arrived; otherwise its held deltas.
    pending_tokens: list[str] | None = []

    while state.turn < max_steps:
        events.step = state.turn + 1
        action = next_action or await paradigm.step(state, ctx)
        next_action = None
        started = monotonic()
        thought: str | None = None
        if isinstance(action, CallModel):
            try:
                # Native text is forwarded even when a tool call may follow it (D-301), and so
                # is text after an answer marker. A reply that may still turn out to be a
                # prompt-mode tool request, or that needs validation, is held until Finish.
                immediate = (
                    action.visible
                    and schema is None
                    and (
                        ctx.tool_mode == "native"
                        or bool(action.emit_tokens_after)
                        or not action.use_tools
                        or not ctx.tools
                    )
                )
                pending_tokens = None if immediate else []
                request = action
                if (
                    schema is not None
                    and action.visible
                    and action.response_schema is None
                    and not action.emit_tokens_after
                ):
                    # A visible reply outside a text protocol is a candidate final answer.
                    request = action.model_copy(update={"response_schema": schema.schema})
                response = await call_model(
                    request,
                    state.messages,
                    ctx,
                    streaming and action.visible,
                    events,
                    pending_tokens,
                    calls,
                )
            except UnsupportedToolsError as exc:
                if ctx.tool_mode != "native" or not ctx.tools or not action.use_tools:
                    raise
                state.last_response = None
                ctx.tool_mode = "prompt"
                ctx.downgrade()
                await events.recover(exc)
                await prepare_knowledge(
                    state, ctx, events, retrievers, knowledge_mode, after_downgrade=True
                )
                continue
            state.messages.append(response.message)
            state.last_response = response.message
            usage += response.usage
            thought = response.message.reasoning
            if corrections:
                next_action = Finish(output=response.message.content or "")
            step = Step(
                turn=events.step,
                action="model",
                output=response.message.content,
                calls=response.message.tool_calls,
            )
        elif isinstance(action, CallTool):
            results = await call_tools(action, ctx, events)
            append_results(state, results, action.as_observation)
            step = Step(turn=events.step, action="tool", calls=action.calls, results=results)
        else:
            if schema is not None:
                try:
                    validated = schema.validate(action.output)
                except (ValueError, SyntaxError, TypeError) as exc:
                    if corrections == 2:
                        raise OutputValidationError(
                            f"Invalid output after three attempts: {exc}"
                        ) from exc
                    corrections += 1
                    state.messages.append(
                        Message(
                            role="user",
                            content=f"Invalid output: {exc}. Return only the corrected JSON value.",
                        )
                    )
                    next_action = CallModel(use_tools=False)
            if streaming and next_action is None:
                if pending_tokens is None:
                    emitted = _answer_tokens(events)
                    if not action.output.startswith(emitted):
                        raise ModelError(
                            "Streamed text differs from the final output. "
                            "Mark intermediate CallModel actions visible=False."
                        )
                    pieces = [action.output[len(emitted) :]]
                else:
                    pieces = (
                        pending_tokens
                        if "".join(pending_tokens) == action.output
                        else [action.output]
                    )
                for piece in pieces:
                    if piece:
                        await events.send("token", text=piece)
            step = Step(turn=events.step, action="finish", output=action.output)
        step.duration = monotonic() - started
        state.turn = step.turn
        state.history.append(step)
        if isinstance(action, CallModel) and next_action is None and state.turn < max_steps:
            # The paradigm reads the reply first so this step's event can carry its thought.
            next_action = await paradigm.step(state, ctx)
            noted = state.scratch.pop("thought", None)
            thought = noted if isinstance(noted, str) and noted else thought
        await events.send("step", action=step.action, thought=thought)
        if isinstance(action, Finish) and next_action is None:
            events.step = None
            return action.output, usage, validated
    raise MaxStepsError(f"Action limit ({max_steps}) reached. Increase max_steps to continue.")


def _export(events: RunEvents) -> None:
    """Write the trace to GOLE_TRACE_DIR when it is set; a failed write never fails the run."""
    directory = os.environ.get("GOLE_TRACE_DIR")
    if not directory:
        return
    try:
        events.trace.to_json(Path(directory).expanduser() / f"{events.run_id}.json")
    except OSError as exc:
        _logger.warning("Could not write the run trace to %s: %s", directory, exc)


async def execute(
    input: str,
    *,
    session: str,
    instructions: str | None,
    memory: Memory | None,
    paradigm: Paradigm,
    ctx: StepContext,
    max_steps: int,
    timeout: float | None,
    streaming: bool,
    emit: EventSink,
    output: type | None = None,
    retrievers: Sequence[Retriever] = (),
    knowledge_mode: str = "auto",
    resolve_mode: Callable[[], Awaitable[ResolvedMode]] | None = None,
) -> Result:
    """Execute one run and publish its ordered events to the supplied sink."""
    events = RunEvents(ctx.run_id, emit)
    started = monotonic()
    original_report = ctx.report_error
    ctx.report_error = events.recover
    deadline = asyncio.timeout(timeout)
    try:
        try:
            async with deadline:
                await events.send("run_start", input=input, session=session)
                loaded: list[Message] = []
                if memory is not None:
                    loaded = await memory.load(session, query=input)
                    await events.send("memory_load", messages=list(loaded))
                schema = OutputSchema(output) if output is not None else None
                if schema is not None:
                    instructions = (
                        f"{instructions}\n\n{schema.instruction}"
                        if instructions
                        else schema.instruction
                    )
                state = initial_state(input, instructions, loaded)
                if schema is not None:
                    state.scratch["structured_output"] = True
                if resolve_mode is not None:
                    ctx.tool_mode = await resolve_mode()
                calls = ModelCalls(ctx)
                ctx.constrained = (await calls.capabilities(ctx.model)).json_schema
                await prepare_knowledge(state, ctx, events, retrievers, knowledge_mode)
                text, usage, validated = await _actions(
                    state,
                    paradigm,
                    ctx,
                    events,
                    max_steps,
                    streaming,
                    retrievers,
                    knowledge_mode,
                    schema,
                    calls,
                )
                if memory is not None:
                    turn = [
                        Message(role="user", content=input),
                        Message(role="assistant", content=text),
                    ]
                    # The answer already exists, so a failed save must not discard it.
                    failure: Exception | None = None
                    try:
                        await memory.save(session, turn)
                    except asyncio.CancelledError:
                        if not deadline.expired():
                            raise
                        failure = RunTimeoutError(
                            f"Run timed out after {timeout}s while saving memory."
                        )
                    except Exception as exc:  # noqa: BLE001 - reported as a recoverable event
                        failure = exc
                    if failure is None:
                        await events.send("memory_save", messages=turn)
                    else:
                        _logger.warning("Memory save failed; this turn was not stored: %s", failure)
                        await events.recover(failure)
                # The result inside run_end must not contain that same terminal event.
                result = Result(
                    text=text,
                    steps=state.history,
                    citations=events.citations,
                    output=validated,
                    usage=usage,
                    trace=Trace(events=events.trace.events),
                    session=session,
                    run_id=ctx.run_id,
                )
                await events.send("run_end", result=result, duration=monotonic() - started)
                _export(events)
                return result.model_copy(update={"trace": events.trace})
        except TimeoutError as exc:
            if deadline.expired():
                raise RunTimeoutError(
                    f"Run timed out after {timeout}s. Increase timeout to continue."
                ) from exc
            raise
    except Exception as exc:
        await events.send("error", exception=exc, recoverable=False)
        # A failed run returns no Result, so the exception carries what was recorded.
        try:
            setattr(exc, "trace", events.trace)  # noqa: B010 - not a declared attribute
        except AttributeError:
            pass
        _export(events)
        raise
    finally:
        ctx.report_error = original_report
