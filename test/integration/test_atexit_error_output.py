"""Test that compilation failures during atexit produce clean output.

When a module only defines @compile functions and relies on the atexit flush,
compilation errors should not emit an "Exception ignored in atexit callback"
traceback. Only the formatted compiler error should be printed.
"""

import os
import sys
import subprocess
import tempfile
import unittest


def _run_source_in_subprocess(source_code: str) -> tuple[int, str]:
    """Run source code in a subprocess and return (returncode, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source_code)
        temp_file = f.name

    workspace = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = workspace + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = workspace

    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode, result.stderr
    finally:
        os.unlink(temp_file)


class TestAtexitErrorOutput(unittest.TestCase):
    def test_atexit_compile_error_has_no_python_traceback(self):
        source = """
from pythoc import compile, i32
from pythoc.builtin_entities import linear

@compile
def bad() -> i32:
    t = linear()
    return 0
"""
        returncode, stderr = _run_source_in_subprocess(source)

        self.assertNotEqual(returncode, 0)
        self.assertIn("[ERROR]", stderr)
        self.assertIn("Linear tokens not consumed before function exit", stderr)
        self.assertNotIn("Exception ignored in atexit callback", stderr)
        self.assertNotIn("Traceback (most recent call last):", stderr)
