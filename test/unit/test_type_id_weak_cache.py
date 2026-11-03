"""
Unit tests for the weak-reference type ID cache.
"""

import gc
import unittest
import weakref

from pythoc.builtin_entities import i32, i64, ptr
from pythoc.type_id import _cache_lookup, _type_id_cache, get_type_id
from pythoc.type_descriptor import same_type


class TestTypeIdWeakCache(unittest.TestCase):
    """Cache entries disappear when the cached type object is collected."""

    def test_mutable_specialization_is_not_cached(self):
        cls = type('TmpSpecialized', (ptr,), {
            '_pc_specialized': True,
            'pointee_type': i32,
        })
        first = get_type_id(cls)
        self.assertEqual(first, get_type_id(ptr[i32]))
        self.assertIsNone(_cache_lookup(cls))
        cls.pointee_type = i64
        self.assertNotEqual(get_type_id(cls), first)

        ref = weakref.ref(cls)
        del cls
        gc.collect()

        # The class is collected and its weak cache entry is gone
        self.assertIsNone(ref())

    def test_none_type_id_not_cached(self):
        before = len(_type_id_cache)
        self.assertTrue(get_type_id(None).startswith('t2_'))
        self.assertEqual(len(_type_id_cache), before)

    def test_cache_hit_returns_consistent_result(self):
        first = get_type_id(i32)
        second = get_type_id(i32)
        self.assertEqual(first, second)
        self.assertEqual(_cache_lookup(i32), first)


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
        self.assertTrue(first.startswith('t2_'))
        # Unresolved name tokens are never memoized: the name may be
        # defined later in the session.
        self.assertIsNone(_cache_lookup(spec))

        node = create_struct_type([i32], ['value'])
        mark_type_defined("GenTestNode", node)
        second = get_type_id(spec)
        self.assertNotEqual(second, first)
        self.assertTrue(same_type(spec, ptr[node]))
        # Mutable pointer specializations are recomputed on each use.
        self.assertIsNone(_cache_lookup(spec))
