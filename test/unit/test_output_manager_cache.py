"""Unit tests for OutputManager cached-object coverage checks.

Regression: the phase-split compile task (Phase 2) re-checks the cache
after the codegen phase has already drained the pending compilation queue.
With an empty pending queue the coverage check used to return True
unconditionally, discarding freshly generated IR and keeping a stale
on-disk object whose AST content hash no longer matches.
"""

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


if __name__ == '__main__':
    unittest.main()
