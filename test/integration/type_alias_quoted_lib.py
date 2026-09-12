# -*- coding: utf-8 -*-
"""Helper module for cross-TU type-alias tests (simulates a second
translation unit whose types are reached by name from another module)."""

from pythoc import compile, i32, i64, ptr


@compile
class LibPoint:
    x: i32
    y: i32


# Plain-assignment aliases (typedef emulation) living in this module's
# namespace.  LibPoint is also registered in the session forward-ref
# registry by its @compile decoration; the plain aliases below are not.
LibWide = i64
LibPointAlias = LibPoint
LibPointPtr = ptr["LibPoint"]


@compile
class RegOnlyPoint:
    """Never imported by name in the consumer test module; reached only
    through the session forward-ref registry."""
    v: i32


@compile
def lib_sum_point(p: LibPointPtr) -> i32:
    """Uses the module-level lazy alias defined above."""
    return p[0].x + p[0].y
