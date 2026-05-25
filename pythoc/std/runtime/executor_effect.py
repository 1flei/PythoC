"""
Effect-based executor integration.

Registers `effect.executor` so async/Future code can:
    effect.executor.spawn(fn, arg, stack_size)
    effect.executor.yield_now()
    effect.executor.join(task)

This follows the same pattern as pythoc.std.mem:
- Provide a default implementation
- Allow override via with effect(executor=..., suffix="custom")

Design:
    The executor effect binds scheduling operations to the active runtime.
    A global runtime pointer is stored as module state (set by runtime_start).
    This is a deliberate choice: one runtime per process is the normal case.
    For multiple runtimes, use effect override.
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from types import SimpleNamespace
from pythoc import (
    compile, effect, u64, ptr, void, func, linear, refined, assume, consume,
)

from .api import (
    Runtime,
    runtime_yield_now, runtime_set_current_executor, runtime_current_executor,
)
from .raw import (
    Task, task_destroy,
    runtime_spawn_raw, runtime_join_raw, runtime_detach_raw,
)


ExecutorProof = refined[linear, "executor_handle"]


@compile
class ExecutorHandle:
    token: ptr[void]
    _proof: ExecutorProof


@compile
def executor_handle_new(token: ptr[void]) -> ExecutorHandle:
    handle: ExecutorHandle
    handle.token = token
    handle._proof = assume(linear(), "executor_handle")
    return handle


@compile
def executor_handle_consume(handle: ExecutorHandle) -> ptr[void]:
    token: ptr[void] = handle.token
    consume(handle._proof)
    return token


@compile
def executor_set_runtime(rt: ptr[Runtime]) -> void:
    runtime_set_current_executor(rt)


# ============================================================
# Default executor effect implementation
# ============================================================

@compile
def _exec_spawn(
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64,
) -> ExecutorHandle:
    """Spawn via the global runtime."""
    task = runtime_spawn_raw(runtime_current_executor(), entry, arg, stack_size)
    return executor_handle_new(ptr[void](task))


@compile
def _exec_join(handle: ExecutorHandle) -> ptr[void]:
    """Join via the global runtime."""
    task = ptr[Task](executor_handle_consume(handle))
    result: ptr[void] = runtime_join_raw(runtime_current_executor(), task)
    task_destroy(task)
    return result


@compile
def _exec_yield() -> void:
    """Yield current task via the global runtime."""
    runtime_yield_now(runtime_current_executor())


@compile
def _exec_detach(handle: ExecutorHandle) -> void:
    """Detach via the global runtime."""
    task = ptr[Task](executor_handle_consume(handle))
    runtime_detach_raw(runtime_current_executor(), task)


# ============================================================
# Bundle as effect implementation
# ============================================================

DefaultExecutor = SimpleNamespace(
    spawn=_exec_spawn,
    join=_exec_join,
    yield_now=_exec_yield,
    detach=_exec_detach,
)


# ============================================================
# Register module default (overridable)
# ============================================================

effect.default(executor=DefaultExecutor)
