"""
Unit tests for lazy resolution of struct field annotations that reference
names not visible at decoration time.

A struct field annotation like ``f: binaryfunc`` used to raise NameError at
decoration time when ``binaryfunc`` was neither in the defining module's
globals nor in the session forward-ref registry yet.  The decorators must
instead keep the annotation as a string and let the existing lazy machinery
(forward-ref callbacks plus _resolve_field_types_locked, which folds the
session registry into the namespace) resolve it once the type is actually
defined via mark_type_defined.
"""

import unittest

from pythoc import i32, init, ptr
from pythoc.builtin_entities.func import func
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.builtin_entities.union import union
from pythoc.decorators import clear_registry
from pythoc.decorators.structs import compile_dynamic_class
from pythoc.forward_ref import clear_forward_ref_state, mark_type_defined


class TestLazyFieldAnnotationResolution(unittest.TestCase):
    """Decoration must tolerate field annotations naming unknown types."""

    def setUp(self):
        init()
        clear_registry()
        clear_forward_ref_state()

    def tearDown(self):
        clear_registry()
        clear_forward_ref_state()

    def test_struct_field_with_undefined_name_defers_resolution(self):
        """A field naming an unknown type stays a string until the type is marked."""

        class Struct:
            f: "binaryfunc"

        decorated = compile_dynamic_class(Struct)
        unified = decorated._struct_type

        self.assertTrue(unified._needs_type_resolution)
        self.assertIsInstance(unified._field_types[0], str)

        # Register the missing type the way another module would and force
        # lazy resolution: the registry snapshot is folded into the namespace
        # by _resolve_field_types_locked.
        mark_type_defined("binaryfunc", func[ptr[i32], i32])

        unified._ensure_field_types_resolved()
        self.assertFalse(isinstance(unified._field_types[0], str))

    def test_struct_field_with_genuinely_missing_name_stays_lazy(self):
        """Names that never resolve must not crash decoration; the deferred
        resolution path reports the failure when the type is first used."""

        class Struct:
            field: "NoSuchTypeAnywhere"

        decorated = compile_dynamic_class(Struct)
        self.assertTrue(decorated._struct_type._needs_type_resolution)
        self.assertIsInstance(decorated._struct_type._field_types[0], str)

    def test_union_field_with_undefined_name_defers_resolution(self):
        """@union shares the field parsing path and must defer as well."""

        @union
        class U:
            a: i32
            b: "LaterDefined"

        self.assertTrue(U._union_type._needs_type_resolution)
        self.assertTrue(any(isinstance(ft, str) for ft in U._union_type._field_types))

        # Simulate the defining module finishing later in the same session.
        later = create_struct_type([i32], ["x"])
        mark_type_defined("LaterDefined", later)

        U._union_type._ensure_field_types_resolved()
        self.assertFalse(any(isinstance(ft, str) for ft in U._union_type._field_types))


if __name__ == "__main__":
    unittest.main()
