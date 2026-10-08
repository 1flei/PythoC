"""
Unit tests for the weak-reference type ID cache.
"""

import gc
import unittest
import weakref

from pythoc.builtin_entities import i32, ptr
from pythoc.type_id import _cache_lookup, _type_id_cache, get_type_id


class TestTypeIdWeakCache(unittest.TestCase):
    """Cache entries disappear when the cached type object is collected."""

    def test_entry_removed_after_type_gc(self):
        # Warm up shared dependencies (i32) so only the new class adds an entry
        expected = f'P{get_type_id(i32)}'

        cls = type('TmpSpecialized', (ptr,), {
            '_pc_specialized': True,
            'pointee_type': i32,
        })
        result = get_type_id(cls)
        self.assertEqual(result, expected)
        self.assertEqual(_cache_lookup(cls), result)
        count_with_entry = len(_type_id_cache)

        ref = weakref.ref(cls)
        del cls
        gc.collect()

        # The class is collected and its weak cache entry is gone
        self.assertIsNone(ref())
        self.assertLessEqual(len(_type_id_cache), count_with_entry - 1)

    def test_none_type_id_not_cached(self):
        before = len(_type_id_cache)
        self.assertEqual(get_type_id(None), 'v')
        self.assertEqual(len(_type_id_cache), before)

    def test_cache_hit_returns_consistent_result(self):
        specialized = ptr[i32]
        first = get_type_id(specialized)
        second = get_type_id(specialized)
        self.assertEqual(first, second)
        self.assertEqual(_cache_lookup(specialized), first)


if __name__ == "__main__":
    unittest.main()


class TestTypeIdGenerationInvalidation(unittest.TestCase):
    """Forward-ref state changes must invalidate memoized type IDs."""

    def setUp(self):
        import pythoc
        pythoc.init()
        from pythoc.forward_ref import clear_forward_ref_state
        clear_forward_ref_state()

    def tearDown(self):
        from pythoc.forward_ref import clear_forward_ref_state
        clear_forward_ref_state()

    def test_unresolved_forward_ref_id_not_frozen(self):
        # A quoted name that is unresolvable today yields a name token;
        # once the type is defined, the same specialization must resolve.
        from pythoc.builtin_entities import i32
        from pythoc.builtin_entities.struct import create_struct_type
        from pythoc.forward_ref import mark_type_defined

        spec = ptr["GenTestNode"]
        first = get_type_id(spec)
        self.assertEqual(first, 'P11GenTestNode')
        # Unresolved name tokens are never memoized: the name may be
        # defined later in the session.
        self.assertIsNone(_cache_lookup(spec))

        node = create_struct_type([i32], ['value'])
        mark_type_defined("GenTestNode", node)
        second = get_type_id(spec)
        self.assertNotEqual(second, first)
        # Resolved results cache normally.
        self.assertEqual(_cache_lookup(spec), second)
