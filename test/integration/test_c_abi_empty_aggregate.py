"""End-to-end ABI test for empty-aggregate parameters.

Background
----------
The x86-64 ABI classifier returns ``coerced_type = VoidType()`` for a
zero-sized aggregate *return* -- that is correct, because a function
that returns ``struct {}`` in C lowers to a ``void`` return in LLVM IR.
However the *argument* classifier must NOT produce ``VoidType()``: a
void-typed function parameter is illegal in LLVM IR.

Before the P1.7 fix, ``classify_argument_type`` simply re-used
``classify_return_type`` and flipped ``is_return = False``, which meant
an empty-struct argument carried ``coerced_type = VoidType()`` through
the classification. The bug was hidden at runtime by
``llvm_c_builder``: because the classification comes back with
``kind = DIRECT`` the builder ignores ``coerced_type`` and uses the
original type, so nothing crashed. This test locks in the invariant at
runtime and inspects the emitted ``.ll`` file for good measure, so a
regression at either layer is visible.

It is also the only place in the integration suite that exercises
``struct[()]`` (zero-field anonymous struct) through the full
``@compile`` pipeline. Adding it closed a real coverage gap: the
existing ``test_c_abi_comprehensive`` / ``test_c_abi_array_union_enum``
suites start at ``Small1 { int8_t }`` (1 byte) and never reach size 0.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TEST_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pythoc import compile, struct, i32


# ``struct[()]`` = anonymous zero-field struct. We use the anonymous
# form because ``@struct class Empty: pass`` currently fails at compile
# time due to an independent class-level caching bug (tracked as P0.3).
EmptyT = struct[()]


@compile
def _empty_first(e: EmptyT, x: i32) -> i32:
    return x + 1


@compile
def _empty_last(x: i32, e: EmptyT) -> i32:
    return x + 2


@compile
def _empty_middle(a: i32, e: EmptyT, b: i32) -> i32:
    return a + b + 3


@compile
def _empty_arg_driver() -> i32:
    e: EmptyT
    r1 = _empty_first(e, 41)           # 42
    r2 = _empty_last(40, e)            # 42
    r3 = _empty_middle(10, e, 20)      # 10 + 20 + 3 = 33
    return r1 + r2 + r3                # 42 + 42 + 33 = 117


def _find_ir_file_for(wrapper) -> str:
    """Return the path of the ``.ll`` file produced for ``wrapper``.

    We derive it from ``wrapper._so_file`` by swapping the shared-lib
    extension for ``.ll``. Using the ``.so`` path (rather than reading
    from the in-memory module) means the check is robust even when the
    build cache decides to skip re-emitting -- the on-disk ``.ll`` has
    to exist whenever the ``.so`` exists.
    """
    so_path = wrapper._so_file
    base, _ext = os.path.splitext(so_path)
    ll_path = base + ".ll"
    assert os.path.exists(ll_path), f"expected IR file missing: {ll_path}"
    return ll_path


def _extract_define_header(ir_text: str, func_name: str) -> str | None:
    """Return the ``define ... @func_name(<params>) ...`` header line."""
    pattern = re.compile(
        r"^define\s+[^\n]*?@" + re.escape(func_name) + r"\s*\([^)]*\)",
        re.MULTILINE,
    )
    m = pattern.search(ir_text)
    return m.group(0) if m else None


class TestEmptyAggregateArgumentABI(unittest.TestCase):
    """Runtime + on-disk IR checks for empty-aggregate parameters."""

    # Ensure compilation has happened once for the whole class.
    @classmethod
    def setUpClass(cls):
        cls.result = _empty_arg_driver()
        cls.ll_path = _find_ir_file_for(_empty_arg_driver)
        with open(cls.ll_path, "r") as f:
            cls.ir_text = f.read()

    def test_runtime_result_is_correct(self):
        """Empty-struct arguments must not corrupt argument order."""
        self.assertEqual(self.result, 117)

    def test_each_callee_header_has_no_void_parameter(self):
        """For every callee we emitted, its LLVM ``define`` header must
        not contain the token ``void`` in the parameter list -- a
        ``void``-typed parameter would be illegal LLVM IR.
        """
        checked = []
        for name in ("_empty_first", "_empty_last", "_empty_middle"):
            header = _extract_define_header(self.ir_text, name)
            if header is None:
                # Callee may have been inlined away; that's fine.
                continue
            checked.append((name, header))

            params_blob = header[header.index("(") + 1 : header.rindex(")")]
            # Split on commas that are at the top level of the parameter
            # list. Struct types use `{...}` so we need to be careful,
            # but empty struct `{}` has no nested commas so splitting on
            # ``,`` is safe here.
            params = [p.strip() for p in params_blob.split(",") if p.strip()]
            for idx, param in enumerate(params):
                type_token = param.split()[0]
                self.assertNotEqual(
                    type_token, "void",
                    msg=(
                        f"{name}: parameter #{idx} has LLVM type 'void' "
                        f"in header: {header!r} -- the ABI classifier "
                        f"leaked a VoidType into the function signature."
                    ),
                )

        # Driver is the entry point and therefore always present; make
        # sure at least *it* was inspected, otherwise we really didn't
        # check anything useful.
        drv_header = _extract_define_header(self.ir_text, "_empty_arg_driver")
        self.assertIsNotNone(drv_header, "driver missing from emitted IR")

        if not checked:
            self.skipTest(
                "all callees were inlined away; runtime test still "
                "covered the ABI path end-to-end"
            )

    def test_callees_accept_empty_struct_param(self):
        """At least one callee must survive inlining so we can verify
        the expected ``{}`` parameter literally appears in the IR. If
        this ever fails uniformly (all callees inlined), we need to
        disable inlining for this test to keep coverage.
        """
        for name in ("_empty_first", "_empty_last", "_empty_middle"):
            header = _extract_define_header(self.ir_text, name)
            if header is None:
                continue
            self.assertIn(
                "{}", header,
                msg=(
                    f"{name} header does not contain an empty-struct "
                    f"parameter literal '{{}}': {header!r}"
                ),
            )
            return
        self.skipTest(
            "all callees were inlined away; the empty-struct parameter "
            "type cannot be directly inspected"
        )


if __name__ == "__main__":
    unittest.main()
