"""
Test linear state compatibility across branches (if/else, match)

This tests all combinations of linear states to ensure the compatibility
logic is semantically correct:
- active: linear token is still active (not consumed), has ownership
- consumed: linear token was consumed or never initialized (no ownership)

Compatibility rules:
1. States must match exactly across all branches
2. 'active' in one branch and 'consumed' in another is an error
3. All paths must be tracked in all branches (no missing states)
"""

import unittest
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i32, i8, void
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs


# =============================================================================
# Valid cases - these are defined at module level and compiled immediately
# =============================================================================

@compile
def test_active_active() -> i32:
    """Both branches leave token active - should pass"""
    t = linear()
    flag: i8 = 1
    if flag == 1:
        pass  # t remains active
    else:
        pass  # t remains active
    consume(t)  # Consume after if
    return 0


@compile
def test_consumed_consumed() -> i32:
    """Both branches consume token - should pass"""
    t = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t)
    else:
        consume(t)
    return 0


@compile
def test_different_tokens_consumed() -> i32:
    """Different tokens consumed in different branches - should pass"""
    t1 = linear()
    t2 = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t1)
        consume(t2)
    else:
        consume(t2)
        consume(t1)
    return 0


@compile
def test_match_all_consume() -> i32:
    """All match cases consume the token - should pass"""
    t = linear()
    val: i32 = 1
    match val:
        case 1:
            consume(t)
        case 2:
            consume(t)
        case _:
            consume(t)
    return 0


@compile
def test_simple_if_return_consumes() -> i32:
    """Simple if with return that consumes - should pass"""
    t = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t)
        return 0  # Early return - code after if only runs when flag != 1
    consume(t)  # This runs only when flag != 1
    return 0


@compile
def test_nested_if_consistent() -> i32:
    """Nested if with consistent linear handling - should pass"""
    t = linear()
    flag1: i8 = 1
    flag2: i8 = 0
    if flag1 == 1:
        if flag2 == 1:
            consume(t)
        else:
            consume(t)
    else:
        consume(t)
    return 0


@compile
def test_multiple_tokens_mixed() -> i32:
    """Multiple tokens with correct handling in all branches"""
    t1 = linear()
    t2 = linear()
    flag: i8 = 1
    if flag == 1:
        # Both consumed in then
        consume(t1)
        consume(t2)
    else:
        # Both consumed in else
        consume(t1)
        consume(t2)
    return 0


@compile
def test_partial_consume_consistent() -> i32:
    """One token consumed in both branches, another left for later"""
    t1 = linear()
    t2 = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t1)
        # t2 remains active
    else:
        consume(t1)
        # t2 remains active
    consume(t2)  # Consume t2 after if
    return 0


def run_invalid_code(source_code: str) -> tuple:
    """
    Run invalid code in a subprocess and return (returncode, stderr).
    This avoids polluting the current process state.
    """
    # Create a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source_code)
        temp_file = f.name
    
    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}
        )
        return result.returncode, result.stderr
    finally:
        os.unlink(temp_file)


# =============================================================================
# Invalid cases - source code strings to be compiled in subprocess
# =============================================================================

INVALID_ACTIVE_CONSUMED = '''
from pythoc import compile, i32, i8
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_active_consumed() -> i32:
    t = linear()
    flag: i8 = 1
    if flag == 1:
        pass  # t remains active
    else:
        consume(t)  # t consumed
    return 0

flush_all_pending_outputs()
'''

INVALID_CONSUMED_ACTIVE = '''
from pythoc import compile, i32, i8
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_consumed_active() -> i32:
    t = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t)  # t consumed
    else:
        pass  # t remains active
    return 0

flush_all_pending_outputs()
'''

INVALID_MATCH_INCONSISTENT = '''
from pythoc import compile, i32
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_match_inconsistent() -> i32:
    t = linear()
    val: i32 = 1
    match val:
        case 1:
            consume(t)
        case 2:
            pass  # t remains active - inconsistent!
        case _:
            consume(t)
    return 0

flush_all_pending_outputs()
'''

INVALID_SIMPLE_IF_CONSUMES = '''
from pythoc import compile, i32, i8
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_simple_if_consumes() -> i32:
    t = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t)  # Consumes in then branch, but no else!
    return 0

flush_all_pending_outputs()
'''

INVALID_NESTED_IF_INCONSISTENT = '''
from pythoc import compile, i32, i8
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_nested_if_inconsistent() -> i32:
    t = linear()
    flag1: i8 = 1
    flag2: i8 = 0
    if flag1 == 1:
        if flag2 == 1:
            consume(t)
        else:
            pass  # t remains active
    else:
        consume(t)
    return 0

flush_all_pending_outputs()
'''

INVALID_PARTIAL_CONSUME = '''
from pythoc import compile, i32, i8
from pythoc.builtin_entities import linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

@compile
def test_partial_consume_inconsistent() -> i32:
    t1 = linear()
    t2 = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t1)
        consume(t2)
    else:
        consume(t1)
        # t2 NOT consumed - inconsistent!
    return 0

flush_all_pending_outputs()
'''


class TestLinearStateCompatibility(unittest.TestCase):
    """Test suite for linear state compatibility across branches"""
    
    @classmethod
    def setUpClass(cls):
        """Compile all valid functions before running tests"""
        flush_all_pending_outputs()
    
    # =========================================================================
    # Tests that should pass (compatible states)
    # =========================================================================
    
    def test_active_active(self):
        """Both branches leave token active"""
        result = test_active_active()
        self.assertEqual(result, 0)
    
    def test_consumed_consumed(self):
        """Both branches consume token"""
        result = test_consumed_consumed()
        self.assertEqual(result, 0)
    
    def test_different_tokens_consumed(self):
        """Different tokens consumed in different order"""
        result = test_different_tokens_consumed()
        self.assertEqual(result, 0)
    
    def test_match_all_consume(self):
        """All match cases consume the token"""
        result = test_match_all_consume()
        self.assertEqual(result, 0)
    
    def test_simple_if_return_consumes(self):
        """Simple if with return that consumes"""
        result = test_simple_if_return_consumes()
        self.assertEqual(result, 0)
    
    def test_nested_if_consistent(self):
        """Nested if with consistent handling"""
        result = test_nested_if_consistent()
        self.assertEqual(result, 0)
    
    def test_multiple_tokens_mixed(self):
        """Multiple tokens with correct handling"""
        result = test_multiple_tokens_mixed()
        self.assertEqual(result, 0)
    
    def test_partial_consume_consistent(self):
        """One token consumed, another left active consistently"""
        result = test_partial_consume_consistent()
        self.assertEqual(result, 0)
    
    # =========================================================================
    # Tests that should fail (incompatible states)
    # These run in subprocess to avoid state pollution
    # =========================================================================
    
    def test_active_consumed_fails(self):
        """active vs consumed should fail"""
        returncode, stderr = run_invalid_code(INVALID_ACTIVE_CONSUMED)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("handled consistently", stderr)
    
    def test_consumed_active_fails(self):
        """consumed vs active should fail"""
        returncode, stderr = run_invalid_code(INVALID_CONSUMED_ACTIVE)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("handled consistently", stderr)
    
    def test_match_inconsistent_fails(self):
        """Match with inconsistent handling should fail"""
        returncode, stderr = run_invalid_code(INVALID_MATCH_INCONSISTENT)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("handled consistently", stderr)
    
    def test_simple_if_consumes_fails(self):
        """Simple if without else that consumes should fail"""
        returncode, stderr = run_invalid_code(INVALID_SIMPLE_IF_CONSUMES)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("if without else", stderr)
    
    def test_nested_if_inconsistent_fails(self):
        """Nested if with inconsistent handling should fail"""
        returncode, stderr = run_invalid_code(INVALID_NESTED_IF_INCONSISTENT)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("handled consistently", stderr)
    
    def test_partial_consume_inconsistent_fails(self):
        """Partial consume inconsistent should fail"""
        returncode, stderr = run_invalid_code(INVALID_PARTIAL_CONSUME)
        self.assertNotEqual(returncode, 0, "Expected compilation to fail")
        self.assertIn("handled consistently", stderr)


if __name__ == '__main__':
    unittest.main()
