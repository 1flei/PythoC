"""Reuse generic native code across processes without replacing mapped images."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


FIXTURES = Path(__file__).parent / 'fixtures' / 'parametric_cache'
WORKER = (FIXTURES / 'worker.py').resolve()


class ParametricSharedCacheTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='pc-parametric-cache-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.definition = self.root / 'shared_definition.py'
        shutil.copyfile(FIXTURES / 'shared_definition.py', self.definition)
        self.env = os.environ.copy()
        import pythoc
        compiler = str(Path(pythoc.__file__).resolve().parent.parent)
        self.env['PYTHONPATH'] = os.pathsep.join(
            [compiler, self.env.get('PYTHONPATH', '')])

    def command(self, mode, offset=0, library='generic'):
        return [sys.executable, str(WORKER), str(self.root), mode, str(offset), library]

    def run_worker(self, mode, offset=0, library='generic'):
        result = subprocess.run(self.command(mode, offset, library), cwd=self.root,
                                env=self.env, capture_output=True, text=True,
                                timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines()
                    if line.startswith('CACHE_STATE='))
        return json.loads(line.removeprefix('CACHE_STATE='))

    def test_existing_specializations_are_reused_while_another_process_holds_them(self):
        self.check_reuse('generic', 'generic')

    def test_map_and_set_share_their_loaded_scalar_helpers(self):
        self.check_reuse('set', 'map')

    def check_reuse(self, holder_library, reader_library):
        holder = subprocess.Popen(self.command('holder', library=holder_library), cwd=self.root,
                                  env=self.env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
        try:
            deadline = time.monotonic() + 90
            while not (self.root / 'ready.json').exists():
                if holder.poll() is not None:
                    self.fail(holder.stdout.read() + holder.stderr.read())
                self.assertLess(time.monotonic(), deadline, 'holder timed out')
                time.sleep(0.02)
            self.run_worker('reader', library=reader_library)
        finally:
            stdout, stderr = holder.communicate(input='release\n', timeout=90)
        self.assertEqual(holder.returncode, 0, stdout + stderr)

    def test_edits_to_the_physical_template_invalidate_specializations(self):
        before = self.run_worker('single')
        shutil.copyfile(FIXTURES / 'updated_definition.py', self.definition)
        after = self.run_worker('single', offset=1)
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(after[path] > before[path] for path in before))


if __name__ == '__main__':
    unittest.main()
