"""
Unit tests for AST debug helpers.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pythoc.utils.ast_debug import ASTDebugger


class TestASTDebugger(unittest.TestCase):
    """Test AST debugger setup behavior."""

    def test_setup_output_dir_does_not_call_get_build_paths(self):
        """AST debug output should use the fixed build/debug_ast directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {"PC_DEBUG_AST": "1"}, clear=False):
                    with patch("pythoc.utils.path_utils.get_build_paths") as mock_get_build_paths:
                        debugger = ASTDebugger()
            finally:
                os.chdir(old_cwd)

            mock_get_build_paths.assert_not_called()

            expected_dir = Path(tmpdir) / "build" / "debug_ast"
            self.assertEqual(debugger.output_dir, Path("build/debug_ast"))
            self.assertTrue(expected_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
