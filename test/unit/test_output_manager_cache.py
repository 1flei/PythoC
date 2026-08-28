"""Unit tests for OutputManager cached-object coverage checks.

Regression: the phase-split compile task (Phase 2) re-checks the cache
after the codegen phase has already drained the pending compilation queue.
With an empty pending queue the coverage check used to return True
unconditionally, discarding freshly generated IR and keeping a stale
on-disk object whose AST content hash no longer matches.
"""

import ast
import hashlib
import os
import tempfile
import unittest

from pythoc.build.deps import get_dependency_tracker
from pythoc.build.output_manager import OutputManager


def _combined_hash(hashes):
    combined = '|'.join(sorted(hashes))
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


class TestCachedObjectCoversPendingSymbols(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pythoc_om_cache_')
        self.source_file = os.path.join(self.tmpdir, 'mod.py')
        with open(self.source_file, 'w') as f:
            f.write('# fake source\n')
        self.obj_file = os.path.join(self.tmpdir, 'mod.o')
        with open(self.obj_file, 'wb') as f:
            f.write(b'fake object')
        self.group_key = (self.source_file, None, None, None)
        self.om = OutputManager()
        self.tracker = get_dependency_tracker()

    def _save_cached_deps(self, hashes, symbols=('f',)):
        group_deps = self.tracker.get_or_create_group_deps(self.group_key)
        group_deps.ast_content_hash = _combined_hash(hashes)
        group_deps.compiled_symbols = sorted(symbols)
        self.tracker.save_deps(self.group_key, self.obj_file)

    def _group(self, hashes):
        return {
            'obj_file': self.obj_file,
            'source_file': self.source_file,
            '_ast_content_hashes': list(hashes),
        }

    def test_no_pending_no_hashes_is_covered(self):
        group = self._group([])
        self.assertTrue(
            self.om._cached_object_covers_pending_symbols(self.group_key, group)
        )

    def test_drained_pending_with_matching_hash_is_covered(self):
        # Phase-2 re-check after codegen drained the queue: same content.
        self._save_cached_deps(['aaa'])
        group = self._group(['aaa'])
        self.assertTrue(
            self.om._cached_object_covers_pending_symbols(self.group_key, group)
        )

    def test_drained_pending_with_changed_hash_is_not_covered(self):
        # Phase-2 re-check after codegen drained the queue: the on-disk
        # object was built from a different AST and must not be reused.
        self._save_cached_deps(['old-hash'])
        group = self._group(['new-hash'])
        self.assertFalse(
            self.om._cached_object_covers_pending_symbols(self.group_key, group)
        )

    def test_current_hashes_without_cached_hash_is_not_covered(self):
        # An older .deps file that never recorded an AST hash must not
        # cover a group that now tracks one (captured-constant fingerprint).
        group = self._group(['aaa'])
        self.assertFalse(
            self.om._cached_object_covers_pending_symbols(self.group_key, group)
        )


class TestFunctionContentHash(unittest.TestCase):
    def _hash(self, src, user_globals):
        from pythoc.build.cache import function_content_hash
        fn_ast = ast.parse(src).body[0]
        return function_content_hash(fn_ast, user_globals)

    def test_captured_int_is_part_of_hash(self):
        src = "def f():\n    return ptr[i8](addr)\n"
        h1 = self._hash(src, {'ptr': object(), 'i8': object(), 'addr': 0x1000})
        h2 = self._hash(src, {'ptr': object(), 'i8': object(), 'addr': 0x2000})
        self.assertIsNotNone(h1)
        self.assertNotEqual(h1, h2)

    def test_same_captured_int_is_stable(self):
        src = "def f():\n    return ptr[i8](addr)\n"
        g = {'ptr': object(), 'i8': object(), 'addr': 0x1000}
        self.assertEqual(self._hash(src, g), self._hash(src, g))

    def test_types_are_not_fingerprinted(self):
        src = "def f():\n    return ptr[i8](addr)\n"
        h1 = self._hash(src, {'ptr': object(), 'i8': object(), 'addr': 1})
        h2 = self._hash(src, {'ptr': object(), 'i8': object(), 'addr': 1})
        self.assertEqual(h1, h2)

    def test_captured_int_is_reported(self):
        from pythoc.build.cache import fingerprint_function_content
        src = "def f():\n    return ptr[i8](addr)\n"
        fn_ast = ast.parse(src).body[0]
        fp = fingerprint_function_content(
            fn_ast, {'ptr': object(), 'i8': object(), 'addr': 0x1000}
        )
        self.assertEqual(fp.captured, ('addr=4096',))

    def test_uncaptured_names_are_empty(self):
        from pythoc.build.cache import fingerprint_function_content
        src = "def f():\n    return ptr[i8](addr)\n"
        fn_ast = ast.parse(src).body[0]
        fp = fingerprint_function_content(
            fn_ast, {'ptr': object(), 'i8': object()}
        )
        self.assertEqual(fp.captured, ())


class TestNestedCapturedHashFoldsIntoParent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pythoc_om_nested_')
        self.source_file = os.path.join(self.tmpdir, 'mod.py')
        with open(self.source_file, 'w') as f:
            f.write('# fake source\n')
        self.obj_file = os.path.join(self.tmpdir, 'mod.o')
        with open(self.obj_file, 'wb') as f:
            f.write(b'fake object')
        self.group_key = (self.source_file, None, None, None)
        self.om = OutputManager()
        self.tracker = get_dependency_tracker()

    def _save_cached_deps(self, hashes, symbols=('f',)):
        group_deps = self.tracker.get_or_create_group_deps(self.group_key)
        group_deps.ast_content_hash = _combined_hash(hashes)
        group_deps.compiled_symbols = sorted(symbols)
        self.tracker.save_deps(self.group_key, self.obj_file)

    def test_record_nested_captured_hash_appends_to_active_parent(self):
        parent = {
            'obj_file': self.obj_file,
            'source_file': self.source_file,
            '_ast_content_hashes': ['parent-hash'],
        }
        self.om._all_groups[self.group_key] = parent
        self.om._active_build_groups.add(self.group_key)
        self.om.record_nested_captured_hash('child-addr-hash')
        self.assertEqual(
            parent['_ast_content_hashes'],
            ['parent-hash', 'child-addr-hash'],
        )

    def test_parent_without_nested_digest_misses_cached_object(self):
        # Process-2 cache check: the parent only has its own AST hash,
        # but the on-disk object was published after a nested captured
        # constant was folded in during codegen.
        self._save_cached_deps(['parent-hash', 'child-addr-hash'])
        group = {
            'obj_file': self.obj_file,
            'source_file': self.source_file,
            '_ast_content_hashes': ['parent-hash'],
        }
        self.assertFalse(
            self.om._cached_object_covers_pending_symbols(self.group_key, group)
        )


if __name__ == '__main__':
    unittest.main()
