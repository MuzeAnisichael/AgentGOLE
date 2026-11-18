"""Drive a paradigm to completion within a run's action and time budgets."""

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic

from ..errors import MaxStepsError, RunTimeoutError, UnsupportedToolsError
from ..paradigms import Auto, React
from ..protocols import EventSink, Memory, Paradigm, StepContext
from ..protocols.types import (
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
from .context import append_results, initial_state
from .evaluator import call_model, call_tools
from .events import RunEvents


async def _actions(
    state: RunState,
    paradigm: Paradigm,
    ctx: StepContext,
    events: RunEvents,
    max_steps: int,
    streaming: bool,
) -> tuple[str, Usage]:
    usage = Usage()
    # Tool requests and custom paradigms can replace intermediate prose at Finish.
    direct = type(paradigm) in (Auto, React) and not ctx.tools
    pending_tokens: list[str] | None = None if direct else []

    while state.turn < max_steps:
        events.step = state.turn + 1
        action = await paradigm.step(state, ctx)
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
                continue
            state.messages.append(response.message)
            state.last_response = response.message
            usage += response.usage
            thought = response.message.reasoning
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
            if streaming and pending_tokens is not None:
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
        if isinstance(action, Finish):
            events.step = None
            return action.output, usage
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
                    loaded = await memory.load(session)
                    await events.send("memory_load", messages=list(loaded))
                state = initial_state(input, instructions, loaded)
                if resolve_mode is not None:
                    ctx.tool_mode = await resolve_mode()
                text, usage = await _actions(state, paradigm, ctx, events, max_steps, streaming)
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
