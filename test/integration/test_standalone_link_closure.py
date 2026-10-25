"""Standalone dependency closure survives process-local registry loss."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from pythoc.utils.cc_utils import compile_c_to_object
from pythoc.utils.link_utils import (
    get_shared_lib_extension, link_files,
)
from test.utils.pe_imports import read_pe_imports


class TestStandaloneLinkClosure(unittest.TestCase):
    def test_cold_and_cached_processes_run_without_host_python(self):
        source_root = Path(__file__).resolve().parents[2]
        sample = Path(__file__).with_name("standalone_link_sample.py")
        c_source = Path(__file__).parent / "fixtures/link_support.c"
        with tempfile.TemporaryDirectory(prefix="pythoc-link-test-") as temp:
            library = Path(temp) / ("liblinkprobe" + get_shared_lib_extension())
            object_file = compile_c_to_object(
                str(c_source), str(Path(temp) / "linkprobe.o"),
            )
            link_files([object_file], str(library), shared=True,
                       link_objects=[], link_libraries=[])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(source_root)
            env["PC_TEST_LINK_LIBRARY"] = str(library)

            def run_sample():
                result = subprocess.run(
                    [sys.executable, str(sample)], cwd=temp, env=env,
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)
                line = next(line for line in result.stdout.splitlines()
                            if line.startswith("LINK_STATE="))
                return json.loads(line.partition("=")[2])

            first = run_sample()
            second = run_sample()
            self.assertGreater(first["runtime_group_count"], 0)
            self.assertEqual(first["objects"], second["objects"])
            self.assertIn(str(library), second["libraries"])
            self.assertGreaterEqual(len(second["objects"]), 2)
            if sys.platform == "win32":
                imports = read_pe_imports(second["executable"])
                self.assertTrue(imports)
                self.assertFalse(any("python" in name.lower()
                                     for name in imports))
                symbols = "\n".join(symbol for names in imports.values()
                                    for symbol in names)
            else:
                symbols = subprocess.check_output(
                    ["nm", "-u", second["executable"]], text=True,
                )
            self.assertIsNone(re.search(r"\b_?Py[A-Z_]", symbols), symbols)
            if sys.platform == "darwin":
                libraries = subprocess.check_output(
                    ["otool", "-L", second["executable"]], text=True,
                )
                self.assertNotIn("libpython", libraries.lower())


if __name__ == "__main__":
    unittest.main()
