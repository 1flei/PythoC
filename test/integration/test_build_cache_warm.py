"""Integration tests for warm-build cache behavior across processes.

Regressions pinned here:

1. Compile-only flows (no native call, hence no .so artifacts) must still
   hit the object cache on a second run.  The dependent-output check used
   to require the dependency's .so, which only exists after native
   execution, so every dependent group recompiled on every warm run.
2. An imported scalar constant baked into IR must invalidate the importer's
   cached object when the *defining* module changes, even though the
   importer's own source file is untouched (the AST content fingerprint's
   captured-constant component).
3. Same for imported constant containers (tuples).
4. Meta-generated functions (compile_ast) must be invalidated when the
   generated body changes even if the driving .py file is untouched
   (the AST component of the fingerprint).

Each round runs in a fresh subprocess so the in-process state cannot leak
between rounds; bytecode caches are disabled and edited files get an
explicit future mtime to stay clear of timestamp-granularity races.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_CONSTS_PY = """\
VALUE = {value}
TABLE = {table}
"""

_LIB_PY = """\
from pythoc import compile, i32
from consts import VALUE, TABLE


@compile
def lib_value() -> i32:
    return VALUE + TABLE[1]
"""

_APP_PY = """\
from pythoc import compile, i32
from lib_mod import lib_value


@compile
def compute() -> i32:
    return lib_value() * 2
"""

_DRIVER_PY = """\
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythoc

pythoc.init()
import app_mod
from pythoc.decorators.compile import flush_all_pending_outputs

flush_all_pending_outputs()
print("compute:", app_mod.compute())
"""

# Compile-only driver: imports and flushes but never calls a compiled
# function, so no .so artifacts are produced (the cpython_pc gate shape).
_DRIVER_COMPILE_ONLY_PY = """\
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythoc

pythoc.init()
import app_mod
from pythoc.decorators.compile import flush_all_pending_outputs

flush_all_pending_outputs()
print("compiled ok")
"""

_META_APP_PY = """\
import ast
import os

from pythoc import i32, meta

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, 'variant.txt')) as f:
    _VARIANT = int(f.read().strip())

_fn_ast = ast.parse("def generated():\\n    return %d\\n" % _VARIANT).body[0]
ast.fix_missing_locations(_fn_ast)

generated_fn = meta.compile_ast(
    _fn_ast,
    param_types={},
    return_type=i32,
    user_globals=globals(),
)
"""

_META_DRIVER_PY = """\
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythoc

pythoc.init()
import meta_app
from pythoc.decorators.compile import flush_all_pending_outputs

flush_all_pending_outputs()
print("generated:", meta_app.generated_fn())
"""


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


class _WarmCacheBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pythoc_warm_cache_')

    def _env(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = _PROJECT_ROOT + os.pathsep + env.get('PYTHONPATH', '')
        # Keep bytecode caches out of the picture: timestamp pyc validation
        # has 1-second granularity and can mask same-size source edits.
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return env

    def _run(self, script_name):
        result = subprocess.run(
            [sys.executable, os.path.join(self.tmpdir, script_name)],
            capture_output=True, text=True, timeout=300,
            cwd=self.tmpdir, env=self._env(), stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"round failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}")
        return result.stdout

    def _object_mtimes(self):
        build_dir = os.path.join(self.tmpdir, 'build')
        mtimes = {}
        for root, _dirs, files in os.walk(build_dir):
            for name in files:
                if name.endswith('.o'):
                    path = os.path.join(root, name)
                    mtimes[os.path.relpath(path, build_dir)] = os.path.getmtime(path)
        return mtimes

    def _bump_mtime(self, path):
        # Move the file's mtime clearly into the future so any mtime-based
        # staleness check sees the change regardless of clock granularity.
        future = time.time() + 5
        os.utime(path, (future, future))


class TestWarmCompileOnlyCache(_WarmCacheBase):
    """A warm compile-only run must not recompile any group."""

    def setUp(self):
        super().setUp()
        _write(os.path.join(self.tmpdir, 'consts.py'), _CONSTS_PY.format(value=4, table=(1, 2)))
        _write(os.path.join(self.tmpdir, 'lib_mod.py'), _LIB_PY)
        _write(os.path.join(self.tmpdir, 'app_mod.py'), _APP_PY)
        _write(os.path.join(self.tmpdir, 'driver.py'), _DRIVER_PY)
        _write(os.path.join(self.tmpdir, 'driver_compile.py'), _DRIVER_COMPILE_ONLY_PY)

    def test_warm_compile_only_run_reuses_all_objects(self):
        # Compile-only rounds: no native calls, so no .so artifacts exist.
        # A warm round must still reuse every cached object (regression: the
        # dependent-output check used to require the dependency's .so).
        out1 = self._run('driver_compile.py')
        self.assertIn('compiled ok', out1)
        cold_mtimes = self._object_mtimes()
        self.assertTrue(cold_mtimes)

        out2 = self._run('driver_compile.py')
        self.assertIn('compiled ok', out2)
        self.assertEqual(cold_mtimes, self._object_mtimes())

        # The cached artifacts must actually work when called.
        out3 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out3)  # (4 + 2) * 2

    def test_warm_run_reuses_all_objects(self):
        out1 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out1)  # (4 + 2) * 2
        cold_mtimes = self._object_mtimes()
        self.assertTrue(cold_mtimes)

        out2 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out2)
        self.assertEqual(cold_mtimes, self._object_mtimes())

    def test_imported_scalar_change_rebuilds_importer(self):
        out1 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out1)

        _write(os.path.join(self.tmpdir, 'consts.py'),
               _CONSTS_PY.format(value=10, table=(1, 2)))
        self._bump_mtime(os.path.join(self.tmpdir, 'consts.py'))

        out2 = self._run('driver.py')
        self.assertIn('compute: i32(24)', out2)  # (10 + 2) * 2

    def test_imported_container_change_rebuilds_importer(self):
        out1 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out1)

        _write(os.path.join(self.tmpdir, 'consts.py'),
               _CONSTS_PY.format(value=4, table=(1, 9)))
        self._bump_mtime(os.path.join(self.tmpdir, 'consts.py'))

        out2 = self._run('driver.py')
        self.assertIn('compute: i32(26)', out2)  # (4 + 9) * 2

    def test_importer_source_edit_rebuilds(self):
        out1 = self._run('driver.py')
        self.assertIn('compute: i32(12)', out1)

        _write(os.path.join(self.tmpdir, 'app_mod.py'),
               _APP_PY.replace('lib_value() * 2', 'lib_value() * 3'))
        self._bump_mtime(os.path.join(self.tmpdir, 'app_mod.py'))

        out2 = self._run('driver.py')
        self.assertIn('compute: i32(18)', out2)  # (4 + 2) * 3


class TestMetaGeneratedCache(_WarmCacheBase):
    """compile_ast functions key on generated content, not the driving file."""

    def setUp(self):
        super().setUp()
        _write(os.path.join(self.tmpdir, 'variant.txt'), '7\n')
        _write(os.path.join(self.tmpdir, 'meta_app.py'), _META_APP_PY)
        _write(os.path.join(self.tmpdir, 'meta_driver.py'), _META_DRIVER_PY)

    def test_generated_content_change_rebuilds(self):
        out1 = self._run('meta_driver.py')
        self.assertIn('generated: i32(7)', out1)

        # Only the data file changes; meta_app.py keeps its old mtime.
        app_path = os.path.join(self.tmpdir, 'meta_app.py')
        old_mtime = os.path.getmtime(app_path)
        _write(os.path.join(self.tmpdir, 'variant.txt'), '42\n')
        os.utime(app_path, (old_mtime, old_mtime))

        out2 = self._run('meta_driver.py')
        self.assertIn('generated: i32(42)', out2)

    def test_unchanged_generated_content_hits_cache(self):
        out1 = self._run('meta_driver.py')
        self.assertIn('generated: i32(7)', out1)
        cold_mtimes = self._object_mtimes()
        self.assertTrue(cold_mtimes)

        out2 = self._run('meta_driver.py')
        self.assertIn('generated: i32(7)', out2)
        self.assertEqual(cold_mtimes, self._object_mtimes())


if __name__ == '__main__':
    unittest.main()
