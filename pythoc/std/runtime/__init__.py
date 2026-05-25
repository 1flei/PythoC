"""User-facing PythoC N:M runtime facade.

The root runtime package intentionally exposes one task API: typed linear
Future.  Lower-level TaskHandle and raw ptr[Task] entry points live in their
own modules so user code has a simple default path.
"""

from .api import (
    Runtime, RuntimeHandle, runtime_start, runtime_shutdown, runtime_yield_now,
)
from .channel import Channel
from .executor_effect import executor_set_runtime
from .future import Future
from .thread_pool_executor import ThreadPoolExecutor


__all__ = [
    "Runtime",
    "RuntimeHandle",
    "runtime_start",
    "runtime_shutdown",
    "runtime_yield_now",
    "executor_set_runtime",
    "Channel",
    "Future",
    "ThreadPoolExecutor",
]
