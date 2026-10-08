"""Unit tests for lazy @extern annotation resolution.

C's textual ``#include`` semantics let two headers reference each other's
types, which under Python module semantics becomes an import cycle, so an
``@extern`` declaration may run while an annotated name is bound neither in
its module globals nor anywhere importable.  These tests pin the contract
that signature annotations resolve lazily on first access, with the session
forward-ref registry as the fallback for plain names.
"""

import unittest

import pythoc
from pythoc import extern, func, i32, ptr, void
from pythoc.forward_ref import (
    clear_forward_ref_state,
    mark_type_defined,
)


def _make_visitproc():
    class _PyObject:
        pass

    return _PyObject, func[ptr[_PyObject], ptr[void], i32]


class TestExternLazyAnnotations(unittest.TestCase):
    def setUp(self):
        self.session = pythoc.init()
        clear_forward_ref_state()

    def tearDown(self):
        clear_forward_ref_state()

    def test_undefined_at_decoration_resolves_via_registry(self):
        """A plain-name annotation unknown at decoration time resolves from
        the session forward-ref registry at first property access."""
        _obj, visitproc = _make_visitproc()

        @extern(lib="c")
        def managed_visit(obj: ptr[void], visit: "visitproc") -> i32:
            ...

        # Decoration succeeded despite ``visitproc`` being unbound; the name
        # becomes available only afterwards (import-cycle semantics).
        mark_type_defined("visitproc", visitproc)

        self.assertIs(managed_visit.return_type, i32)
        ptypes = dict(managed_visit.param_types)
        self.assertIs(ptypes["visit"], visitproc)

    def test_module_global_resolution_still_works(self):
        """Names bound in the function's own globals resolve as before."""
        _, visitproc = _make_visitproc()
        globals()["visitproc_alias"] = visitproc
        try:
            @extern(lib="c")
            def probe(visit: "visitproc_alias") -> i32:
                ...

            ptypes = dict(probe.param_types)
            self.assertIs(ptypes["visit"], visitproc)
        finally:
            del globals()["visitproc_alias"]

    def test_unresolvable_name_fails_loud_with_context(self):
        """A name absent everywhere fails at access time with a contextual
        error naming the extern function and the annotation."""
        @extern(lib="c")
        def broken(cb: "no_such_type_anywhere") -> i32:
            ...

        with self.assertRaises(NameError) as ctx:
            broken.param_types
        msg = str(ctx.exception)
        self.assertIn("broken", msg)
        self.assertIn("no_such_type_anywhere", msg)

    def test_annotation_resolution_is_cached(self):
        """Resolution happens once; later accesses reuse the cached tuple."""
        _obj, visitproc = _make_visitproc()

        @extern(lib="c")
        def cached_visit(visit: "visitproc_cached") -> i32:
            ...

        mark_type_defined("visitproc_cached", visitproc)
        first = cached_visit.param_types
        second = cached_visit.param_types
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
