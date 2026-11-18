"""Drive a paradigm to completion within a run's action and time budgets."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from time import monotonic
from typing import Any

from .._json import OutputSchema
from ..errors import MaxStepsError, OutputValidationError, RunTimeoutError, UnsupportedToolsError
from ..paradigms import Auto, React
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
from .evaluator import call_model, call_tools
from .events import RunEvents


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
) -> tuple[str, Usage, Any]:
    usage = Usage()
    corrections = 0
    next_action: Action | None = None
    validated: Any = None
    # Tool requests and custom paradigms can replace intermediate prose at Finish.
    direct = type(paradigm) in (Auto, React) and not ctx.tools and schema is None
    pending_tokens: list[str] | None = None if direct else []

    while state.turn < max_steps:
        events.step = state.turn + 1
        action = next_action or await paradigm.step(state, ctx)
        next_action = None
        started = monotonic()
        thought: str | None = None
        if isinstance(action, CallModel):
            try:
                if pending_tokens is not None:
                    pending_tokens.clear()
                response = await call_model(
                    action, state.messages, ctx, streaming, events, pending_tokens
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
            if streaming and next_action is None and pending_tokens is not None:
                pieces = (
                    pending_tokens if "".join(pending_tokens) == action.output else [action.output]
                )
                for piece in pieces:
                    if piece:
                        await events.send("token", text=piece)
            step = Step(turn=events.step, action="finish", output=action.output)
        step.duration = monotonic() - started
        state.turn = step.turn
        state.history.append(step)
        await events.send("step", action=step.action, thought=thought)
        if isinstance(action, Finish) and next_action is None:
            events.step = None
            return action.output, usage, validated
    raise MaxStepsError(f"Action limit ({max_steps}) reached. Increase max_steps to continue.")


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
                )
                if memory is not None:
                    turn = [
                        Message(role="user", content=input),
                        Message(role="assistant", content=text),
                    ]
                    await memory.save(session, turn)
                    await events.send("memory_save", messages=turn)
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
                return result.model_copy(update={"trace": events.trace})
        except TimeoutError as exc:
            if deadline.expired():
                raise RunTimeoutError(
                    f"Run timed out after {timeout}s. Increase timeout to continue."
                ) from exc
            raise
    except Exception as exc:
        await events.send("error", exception=exc, recoverable=False)
        raise
    finally:
        ctx.report_error = original_report
