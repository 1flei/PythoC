# -*- coding: utf-8 -*-
"""
Target functions for effect-propagation tests.
Uses effect.mem so that import-override can recompile with custom allocator.
"""

from pythoc import compile, u64, ptr, void, effect
from pythoc.std import mem  # Sets up default mem effect


@compile
def target_alloc(size: u64) -> ptr[void]:
    return effect.mem.malloc(size)


@compile
def target_free(p: ptr[void]) -> void:
    effect.mem.free(p)
