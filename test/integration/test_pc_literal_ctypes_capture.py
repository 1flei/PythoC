# -*- coding: utf-8 -*-
"""End-to-end checks for ctypes-backed pc_literal interop.

Capture into ``@compile`` is supported only for value-typed payloads
(scalars and value-typed struct/union/enum aggregates).  Pointer
captures are intentionally rejected -- a runtime heap address has no
stable identity to bake into a cacheable IR artefact, and the same
behaviour is trivially expressible by passing the pointer as an
explicit argument.

This file pins:

1. Argument-passing round-trips: pointer, by-value struct, and a
   struct with a pointer field can all be returned by a native
   function, held Python-side as a ctypes-backed pc_literal, then
   passed back into another @compile function with the original
   semantics intact.
2. Value-only captures: a by-value struct with no pointer fields can
   be captured as a free variable.  Field values are pinned into IR
   at decoration time.
3. Liveness: mutating the Python-side pc_literal writes through to
   the live ctypes buffer and is visible to the next native call on
   that value.
4. Rejection of unsupported captures: capturing a pointer pc_literal
   (or a struct that contains a pointer field) raises rather than
   silently embedding a process-local address into IR.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, i32, i64, u64, ptr, void
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free


# ---------------------------------------------------------------------------
# Named struct types (round-trip carriers)
# ---------------------------------------------------------------------------

@compile
class Pair:
    a: i32
    b: i64


@compile
class Handle:
    p: ptr[i64]
    extra: i32


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------

@compile
def alloc_tagged_buf(tag: i64) -> ptr[i64]:
    """Allocate an i64 cell, tag it, and return the pointer."""
    p = ptr[i64](libc_malloc(u64(8)))
    p[0] = tag
    return p


@compile
def free_tagged_buf(p: ptr[i64]) -> void:
    libc_free(ptr[void](p))


@compile
def make_pair() -> Pair:
    """Return a by-value, value-typed struct."""
    out: Pair
    out.a = 7
    out.b = 100
    return out


@compile
def make_handle(tag: i64) -> Handle:
    """Return a by-value struct that carries a pointer field."""
    out: Handle
    out.p = ptr[i64](libc_malloc(u64(8)))
    out.p[0] = tag
    out.extra = 42
    return out


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

@compile
def deref_i64(p: ptr[i64]) -> i64:
    return p[0]


@compile
def bump_pair(s: Pair) -> i64:
    return i64(s.a) + s.b


@compile
def use_handle(h: Handle) -> i64:
    return h.p[0] + i64(h.extra)


# ---------------------------------------------------------------------------
# Module-level value-only capture: Pair has only scalar fields, so its
# decoration-time content can be safely pinned into IR and survives
# disk caching across processes.
# ---------------------------------------------------------------------------

_CAPTURED_PAIR = make_pair()


@compile(suffix="capture_pair")
def read_captured_pair() -> i64:
    return i64(_CAPTURED_PAIR.a) + _CAPTURED_PAIR.b


class TestPcLiteralCtypesCaptureArgs(unittest.TestCase):
    """Argument-passing round-trips: every shape of ctypes-backed
    pc_literal that flows back across the Python<->native boundary."""

    def test_pointer_return_passthrough_argument(self):
        p = alloc_tagged_buf(0x1234)
        try:
            self.assertEqual(int(deref_i64(p)), 0x1234)
        finally:
            free_tagged_buf(p)

    def test_struct_return_passthrough_argument(self):
        pair = make_pair()
        self.assertEqual(int(bump_pair(pair)), 7 + 100)

    def test_struct_mutation_visible_to_native(self):
        # Live ctypes Structure rooted on the pc_literal: writing
        # through Python-side attribute access must be observable to
        # the next native call that consumes the same value.
        pair = make_pair()
        pair.a = 1000
        pair.b = 2000
        self.assertEqual(int(bump_pair(pair)), 1000 + 2000)

    def test_struct_with_pointer_field_passthrough(self):
        # Argument-passing route accepts pointer fields verbatim --
        # the pointer's lifetime is the user's responsibility.
        h = make_handle(0xABCD)
        try:
            self.assertEqual(int(use_handle(h)), 0xABCD + 42)
        finally:
            free_tagged_buf(h.p)


class TestPcLiteralCtypesCaptureValueOnly(unittest.TestCase):
    """Value-only capture: scalar-only struct survives disk caching."""

    def test_value_struct_capture(self):
        self.assertEqual(int(read_captured_pair()), 7 + 100)
        # Python-side handle still tracks the live ctypes buffer.
        self.assertEqual(int(_CAPTURED_PAIR.a), 7)
        self.assertEqual(int(_CAPTURED_PAIR.b), 100)


class TestPcLiteralCtypesCaptureRejected(unittest.TestCase):
    """Pointer captures are rejected -- they have no stable identity
    to bake into IR.  The user is expected to pass the pointer as an
    explicit argument instead (covered by the args test class)."""

    def test_pointer_capture_rejected(self):
        from pythoc.builtin_entities.pc_literal import pc_literal
        p = alloc_tagged_buf(0x42)
        try:
            with self.assertRaises((TypeError, SystemExit)):
                p.get_value()
        finally:
            free_tagged_buf(p)

    def test_struct_with_pointer_field_capture_rejected(self):
        h = make_handle(0x42)
        try:
            with self.assertRaises((TypeError, SystemExit)):
                h.get_value()
        finally:
            free_tagged_buf(h.p)


if __name__ == '__main__':
    unittest.main()
