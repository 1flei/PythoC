"""
Effect-based executor integration.

Registers `effect.executor` so user code can:
    effect.executor.spawn(fn, arg)
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
    compile, effect, i32, i64, u64, ptr, void, nullptr, static, func, sizeof,
)
from pythoc.libc.string import memset

from .api import (
    Runtime, runtime_new, runtime_start, runtime_shutdown, runtime_free,
    runtime_spawn, runtime_join, runtime_join_and_free, runtime_detach,
    runtime_yield_now,
)
from .task import Task


# ============================================================
# Global runtime pointer (set by init_default_runtime)
# ============================================================

@compile
class _GlobalState:
    rt: ptr[Runtime]


@compile
def _global_rt() -> ptr[_GlobalState]:
    """Return pointer to global state (static allocation)."""
    g: static[ptr[_GlobalState]] = nullptr
    if g == nullptr:
        g = ptr[_GlobalState](effect.mem.malloc(u64(sizeof(_GlobalState))))
        memset(ptr[void](g), 0, i64(sizeof(_GlobalState)))
    return g


@compile
def executor_set_runtime(rt: ptr[Runtime]) -> void:
    g: ptr[_GlobalState] = _global_rt()
    g.rt = rt


# ============================================================
# Default executor effect implementation
# ============================================================

@compile
def _exec_spawn(entry: func[ptr[void], ptr[void]], arg: ptr[void]) -> ptr[Task]:
    """Spawn via the global runtime."""
    g: ptr[_GlobalState] = _global_rt()
    return runtime_spawn(g.rt, entry, arg, u64(0))


@compile
def _exec_join(task: ptr[Task]) -> ptr[void]:
    """Join via the global runtime."""
    g: ptr[_GlobalState] = _global_rt()
    return runtime_join(g.rt, task)


@compile
def _exec_yield() -> void:
    """Yield current task via the global runtime."""
    g: ptr[_GlobalState] = _global_rt()
    runtime_yield_now(g.rt)


@compile
def _exec_detach(task: ptr[Task]) -> void:
    """Detach via the global runtime."""
    g: ptr[_GlobalState] = _global_rt()
    runtime_detach(g.rt, task)


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
