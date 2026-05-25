"""Raw runtime escape hatch.

Import this module only when code needs direct ptr[Task] ownership.  The
normal user-facing runtime path is Future, which keeps task ownership linear
and typed.
"""
from __future__ import annotations

from .api import (
    Runtime,
    _runtime_new as runtime_new_raw,
    _runtime_start as runtime_start_raw,
    _runtime_shutdown as runtime_shutdown_raw,
    _runtime_free as runtime_free_raw,
    _runtime_spawn_task as runtime_spawn_raw,
    _runtime_join_task as runtime_join_raw,
    _runtime_detach_task as runtime_detach_raw,
)
from .task import Task, task_destroy


__all__ = [
    "Runtime",
    "Task",
    "runtime_new_raw",
    "runtime_start_raw",
    "runtime_shutdown_raw",
    "runtime_free_raw",
    "runtime_spawn_raw",
    "runtime_join_raw",
    "runtime_detach_raw",
    "task_destroy",
]
