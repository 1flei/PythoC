#!/usr/bin/env python3
"""Function linkage support: @compile(linkage=...).

Covers:
  (a) default linkage is unchanged (plain external definition);
  (b) 'internal' emits ``define internal`` (symbol local to the object);
  (c) 'weak_odr' / 'linkonce_odr' emit the corresponding ODR definitions
      (the multi-module duplicate-symbol scenario is exercised end-to-end
      outside the unit suite);
  (d) an unsupported linkage value is rejected loudly.

The address-of table is indexed with a runtime value so the optimizer
cannot devirtualize the calls or discard the internal definition.
"""

import re
import unittest

from pythoc import compile, i32, u64, static, array, func
from pythoc.decorators.compile import flush_all_pending_outputs
from pythoc.build.output_manager import get_output_manager
from pythoc.logger import set_raise_on_error


@compile
def linkage_default() -> i32:
    return 1


@compile(linkage='external')
def linkage_external() -> i32:
    return 5


@compile(linkage='internal')
def linkage_internal() -> i32:
    return 2


@compile(linkage='weak_odr')
def linkage_weak_odr() -> i32:
    return 3


@compile(linkage='linkonce_odr')
def linkage_linkonce_odr() -> i32:
    return 4


@compile
def linkage_addr_table(i: i32) -> u64:
    s: static[array[func[i32], 5]] = (
        linkage_default,
        linkage_external,
        linkage_internal,
        linkage_weak_odr,
        linkage_linkonce_odr,
    )
    return u64(s[i]())


def _ir_text_for(wrapper) -> str:
    """Return the optimised LLVM IR text for ``wrapper``'s group."""
    return get_output_manager().get_ir_text(wrapper._group_key)


def _extract_define_header(ir_text: str, func_name: str):
    """Return the ``define ... @func_name(<params>) ...`` header line."""
    pattern = re.compile(
        r"^define\s+[^\n]*?@" + re.escape(func_name) + r"\s*\([^)]*\)",
        re.MULTILINE,
    )
    m = pattern.search(ir_text)
    return m.group(0) if m else None


class TestFuncLinkage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flush_all_pending_outputs()
        cls.ir_text = _ir_text_for(linkage_default)

    def test_runtime_results(self):
        # internal / linkonce_odr symbols are not guaranteed to be
        # reachable through the group's shared library, so they are only
        # exercised indirectly through the in-module address table.
        self.assertEqual(linkage_default(), 1)
        self.assertEqual(linkage_external(), 5)
        self.assertEqual(linkage_weak_odr(), 3)
        self.assertEqual([linkage_addr_table(i) for i in range(5)],
                         [1, 5, 2, 3, 4])

    def test_default_linkage_is_plain_external(self):
        for name in ("linkage_default", "linkage_external"):
            header = _extract_define_header(self.ir_text, name)
            self.assertIsNotNone(header, f"{name} missing from emitted IR")
            for token in ("internal", "weak", "linkonce"):
                self.assertNotIn(
                    token, header,
                    f"{name} default linkage must stay external: {header!r}",
                )

    def test_internal_linkage_header(self):
        header = _extract_define_header(self.ir_text, "linkage_internal")
        self.assertIsNotNone(header, "linkage_internal missing from emitted IR")
        self.assertIn("define internal", header)

    def test_weak_odr_linkage_header(self):
        header = _extract_define_header(self.ir_text, "linkage_weak_odr")
        self.assertIsNotNone(header, "linkage_weak_odr missing from emitted IR")
        self.assertIn("define weak_odr", header)

    def test_linkonce_odr_linkage_header(self):
        header = _extract_define_header(self.ir_text, "linkage_linkonce_odr")
        self.assertIsNotNone(header, "linkage_linkonce_odr missing from emitted IR")
        self.assertIn("define linkonce_odr", header)

    def test_invalid_linkage_rejected(self):
        set_raise_on_error(True)
        with self.assertRaises(TypeError) as ctx:
            @compile(linkage='bogus')
            def bad_linkage() -> i32:
                return 0
        self.assertIn("Unsupported linkage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
