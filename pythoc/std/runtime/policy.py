"""
Runtime effect policies.

Registers per-module defaults for runtime compilation units.  Global
pythoc.std.mem keeps LibcMem; runtime modules bind PoolMem here.

Usage (automatic):
    Each runtime submodule calls bind_mem() at import before @compile defs.

Usage (override):
    from pythoc import effect
    from pythoc.std.mem import LibcMem
    from pythoc.std.mem_pool import PoolMem

    with effect(mem=LibcMem, suffix="libc_rt"):
        from pythoc.std.runtime import runtime_start
"""
from __future__ import annotations

from pythoc.std.mem_pool import PoolMem

# Default allocator policy for runtime modules.
RuntimeMem = PoolMem


def bind_mem() -> None:
    """Bind PoolMem as effect.mem default for the importing module."""
    import inspect

    from pythoc import effect as effect_mod

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return
    module_name = frame.f_back.f_globals.get('__name__', '__main__')
    with effect_mod._lock:
        if module_name not in effect_mod._defaults:
            effect_mod._defaults[module_name] = {}
        effect_mod._defaults[module_name]['mem'] = PoolMem
