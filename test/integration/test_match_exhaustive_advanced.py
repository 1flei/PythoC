#!/usr/bin/env python3
"""
Comprehensive corner case tests for match exhaustiveness checking

These tests target specific edge cases and potential bugs in the current
exhaustiveness checking implementation, consolidated from multiple test suites.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, i8, i16, compile, enum, ptr, struct
from pythoc import bool as pc_bool
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error
import unittest

# Enable exception raising for tests
set_raise_on_error(True)


# ============================================================
# Advanced type definitions
# ============================================================

@struct
class TripleBool:
    """Struct with 3 bool fields - 8 combinations total"""
    x: pc_bool
    y: pc_bool
    z: pc_bool

@struct
class InfiniteStruct:
    """Struct with infinite field"""
    flag: pc_bool
    value: i32

@struct
class NestedStruct:
    """Struct containing another struct"""
    outer: pc_bool
    inner: TripleBool

@enum(i32)
class ComplexEnum:
    """Enum with various payload types"""
    Unit: None
    Bool: pc_bool
    Int: i32
    Struct: TripleBool

@enum(i8)
class NonSequentialEnum:
    """Enum with non-sequential tag values"""
    First: None = 10
    Second: None = 20
    Third: None = 5

@enum(i16)
class PayloadEnum:
    """Enum for testing recursive payload analysis"""
    Empty: None
    BoolPayload: pc_bool
    IntPayload: i32


# ============================================================
# Compiled functions for positive tests (must be at top level)
# ============================================================

@compile
def non_sequential_exhaustive() -> i32:
    e: NonSequentialEnum = NonSequentialEnum(NonSequentialEnum.First)
    match e:
        case (NonSequentialEnum.First):   # tag 10
            return 1
        case (NonSequentialEnum.Second):  # tag 20
            return 2
        case (NonSequentialEnum.Third):   # tag 5
            return 3
    # Should be exhaustive despite non-sequential tags


@compile
def variable_binding_test() -> i32:
    x: pc_bool = True
    match x:
        case True:
            return 1
        case false:  # Variable binding (not False literal!)
            return 2
    # Should be exhaustive - True literal + variable binding covers all


# ============================================================
# Test 1: Complex struct field correlation (the classic bug)
# ============================================================

def test_struct_field_correlation_soundness():
    """Test the critical struct field correlation issue from the docs"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'struct_correlation_bug')
    try:
        @compile(suffix="struct_correlation_bug")
        def struct_correlation_bug() -> i32:
            s: TripleBool = TripleBool()
            s.x = True
            s.y = False
            s.z = True
            match s:
                # Classic counterexample: per-field subtraction would say this is exhaustive
                # because x covers {True, False}, y covers {True, False}, z covers {True, False}
                # But pattern matrix should correctly identify missing combinations
                case TripleBool(x=True, y=True, z=True):
                    return 1
                case TripleBool(x=False, y=False, z=False):
                    return 2
                # Missing 6 out of 8 combinations!
        
        flush_all_pending_outputs()
        return False, "should have detected missing struct field combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 2: Infinite struct handling
# ============================================================

def test_infinite_struct_requires_wildcard():
    """Test that struct with infinite fields requires catch-all"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'infinite_struct')
    try:
        @compile(suffix="infinite_struct")
        def infinite_struct_test() -> i32:
            s: InfiniteStruct = InfiniteStruct()
            s.flag = True
            s.value = 42
            match s:
                case InfiniteStruct(flag=True, value=0):
                    return 1
                case InfiniteStruct(flag=False, value=0):
                    return 2
                # Missing: all other i32 values for both flag states
        
        flush_all_pending_outputs()
        return False, "should require catch-all for infinite field"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 3: Complex OR pattern edge cases
# ============================================================

def test_or_pattern_gap_detection():
    """Test OR patterns with subtle gaps"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'or_pattern_gap')
    try:
        @compile(suffix="or_pattern_gap")
        def or_pattern_gap() -> i32:
            x: pc_bool = True
            y: pc_bool = False
            match x, y:
                case (True, True) | (False, False):
                    return 1
                case (True, False):
                    return 2
                # Missing: (False, True)
        
        flush_all_pending_outputs()
        return False, "should have detected missing (False, True)"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_complex_or_pattern_overlap():
    """Test complex OR patterns with overlapping coverage"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'complex_or_overlap')
    try:
        @compile(suffix="complex_or_overlap")
        def complex_or_overlap() -> i32:
            s: TripleBool = TripleBool()
            s.x = True
            s.y = False
            s.z = True
            match s:
                case TripleBool(x=True, y=True, z=_) | TripleBool(x=False, y=False, z=_):
                    return 1
                case TripleBool(x=True, y=False, z=True):
                    return 2
                # Missing: (True, False, False) and (False, True, _)
        
        flush_all_pending_outputs()
        return False, "should have detected missing OR pattern combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 4: Nested struct exhaustiveness
# ============================================================

def test_nested_struct_partial_coverage():
    """Test nested struct with partial inner coverage"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'nested_struct_partial')
    try:
        @compile(suffix="nested_struct_partial")
        def nested_struct_partial() -> i32:
            s: NestedStruct = NestedStruct()
            s.outer = True
            s.inner = TripleBool()
            s.inner.x = True
            s.inner.y = False
            s.inner.z = True
            match s:
                case NestedStruct(outer=True, inner=TripleBool(x=True, y=True, z=True)):
                    return 1
                case NestedStruct(outer=False, inner=_):
                    return 2
                # Missing: outer=True with other inner combinations
        
        flush_all_pending_outputs()
        return False, "should have detected missing nested struct combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 5: Non-sequential enum tags
# ============================================================

def test_non_sequential_enum_coverage():
    """Test enum with non-sequential tags"""
    return non_sequential_exhaustive()


def test_non_sequential_enum_partial():
    """Test incomplete coverage of non-sequential enum"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'non_sequential_partial')
    try:
        @compile(suffix="non_sequential_partial")
        def non_sequential_partial() -> i32:
            e: NonSequentialEnum = NonSequentialEnum(NonSequentialEnum.First)
            match e:
                case (NonSequentialEnum.First):   # tag 10
                    return 1
                case (NonSequentialEnum.Third):   # tag 5
                    return 3
                # Missing NonSequentialEnum.Second (tag 20)
        
        flush_all_pending_outputs()
        return False, "should have detected missing NonSequentialEnum.Second"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 6: Guard interaction with finite types
# ============================================================

def test_guards_with_finite_types():
    """Test that guards with finite types still require catch-all"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'guards_finite')
    try:
        @compile(suffix="guards_finite")
        def guards_finite() -> i32:
            x: pc_bool = True
            match x:
                case True if True:   # Guard always true
                    return 1
                case False if True:  # Guard always true
                    return 0
                # Even with "always true" guards, should require catch-all
        
        flush_all_pending_outputs()
        return False, "should require catch-all despite guards covering all values"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_guards_with_complex_conditions():
    """Test guards with complex boolean conditions"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'guards_complex')
    try:
        @compile(suffix="guards_complex")
        def guards_complex() -> i32:
            s: TripleBool = TripleBool()
            s.x = True
            s.y = False
            s.z = True
            match s:
                case TripleBool(x=True, y=y_val, z=z_val) if y_val and z_val:
                    return 1
                case TripleBool(x=False, y=_, z=_):
                    return 2
                # Missing: x=True with other y,z combinations where not (y and z)
        
        flush_all_pending_outputs()
        return False, "should have detected missing guarded combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 7: Multi-subject pattern combinations
# ============================================================

def test_multi_subject_different_enums():
    """Test multi-subject with different enum types"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'multi_different_enums')
    try:
        @compile(suffix="multi_different_enums")
        def multi_different_enums() -> i32:
            e1: ComplexEnum = ComplexEnum(ComplexEnum.Unit)
            e2: NonSequentialEnum = NonSequentialEnum(NonSequentialEnum.First)
            match e1, e2:
                case (ComplexEnum.Unit), (NonSequentialEnum.First):
                    return 1
                case (ComplexEnum.Bool, True), (NonSequentialEnum.Second):
                    return 2
                # Missing many combinations across different enum types
        
        flush_all_pending_outputs()
        return False, "should have detected missing combinations across different enum types"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_multi_subject_partial_enum():
    """Test multi-subject with partial enum coverage"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'multi_partial_enum')
    try:
        @compile(suffix="multi_partial_enum")
        def multi_partial_enum() -> i32:
            e1: PayloadEnum = PayloadEnum(PayloadEnum.Empty)
            e2: PayloadEnum = PayloadEnum(PayloadEnum.Empty)
            match e1, e2:
                case (PayloadEnum.Empty), (PayloadEnum.Empty):
                    return 1
                case (PayloadEnum.BoolPayload, True), (PayloadEnum.IntPayload, _):
                    return 2
                case (PayloadEnum.IntPayload, _), (PayloadEnum.Empty):
                    return 3
                # Missing many cross-combinations
        
        flush_all_pending_outputs()
        return False, "should have detected missing multi-subject enum combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 8: Edge cases with variable bindings
# ============================================================

def test_variable_binding_exhaustiveness():
    """Test that variable bindings provide exhaustiveness"""
    return variable_binding_test()


def test_mixed_literal_variable_binding():
    """Test mixed literal and variable binding patterns"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'mixed_binding')
    try:
        @compile(suffix="mixed_binding")
        def mixed_binding() -> i32:
            s: TripleBool = TripleBool()
            s.x = True
            s.y = False
            s.z = True
            match s:
                case TripleBool(x=True, y=True, z=z_val):  # Variable binding for z
                    return 1
                case TripleBool(x=False, y=y_val, z=True):  # Variable binding for y
                    return 2
                # Missing: (True, False, _) and (False, _, False)
        
        flush_all_pending_outputs()
        return False, "should have detected missing mixed binding combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test 9: Algorithmic stress tests
# ============================================================

def test_large_struct_partial_coverage():
    """Test partial coverage of large struct pattern space"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'large_struct_partial')
    try:
        @compile(suffix="large_struct_partial")
        def large_struct_partial() -> i32:
            s: TripleBool = TripleBool()
            s.x = True
            s.y = False
            s.z = True
            match s:
                # Only cover 5 out of 8 combinations
                case TripleBool(x=True, y=True, z=True):
                    return 1
                case TripleBool(x=True, y=True, z=False):
                    return 2
                case TripleBool(x=True, y=False, z=True):
                    return 3
                case TripleBool(x=False, y=True, z=True):
                    return 4
                case TripleBool(x=False, y=False, z=False):
                    return 5
                # Missing: (True, False, False), (False, True, False), (False, False, True)
        
        flush_all_pending_outputs()
        return False, "should have detected missing struct combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_enum_payload_correlation():
    """Test enum payload field correlation"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'enum_payload_correlation')
    try:
        @compile(suffix="enum_payload_correlation")
        def enum_payload_correlation() -> i32:
            e: ComplexEnum = ComplexEnum(ComplexEnum.Struct, TripleBool())
            match e:
                case (ComplexEnum.Unit):
                    return 1
                case (ComplexEnum.Bool, True):
                    return 2
                case (ComplexEnum.Bool, False):
                    return 3
                case (ComplexEnum.Int, _):
                    return 4
                case (ComplexEnum.Struct, TripleBool(x=True, y=True, z=True)):
                    return 5
                case (ComplexEnum.Struct, TripleBool(x=False, y=False, z=False)):
                    return 6
                # Missing: 6 out of 8 struct payload combinations
        
        flush_all_pending_outputs()
        return False, "should have detected missing enum struct payload combinations"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


# ============================================================
# Test execution
# ============================================================

class TestMatchExhaustiveAdvanced(unittest.TestCase):
    """Comprehensive test cases for match exhaustiveness checking"""

    def test_struct_correlation_soundness(self):
        """Struct field correlation should be preserved (critical soundness test)"""
        passed, msg = test_struct_field_correlation_soundness()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_infinite_struct_wildcard(self):
        """Infinite struct should require catch-all"""
        passed, msg = test_infinite_struct_requires_wildcard()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_or_pattern_gaps(self):
        """OR patterns with gaps should be detected"""
        passed, msg = test_or_pattern_gap_detection()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_complex_or_overlap(self):
        """Complex OR patterns with overlaps should detect missing cases"""
        passed, msg = test_complex_or_pattern_overlap()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_nested_struct_partial(self):
        """Nested struct partial coverage should be detected"""
        passed, msg = test_nested_struct_partial_coverage()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_non_sequential_exhaustive(self):
        """Non-sequential enum should work when exhaustive"""
        self.assertEqual(test_non_sequential_enum_coverage(), 1)

    def test_non_sequential_partial(self):
        """Non-sequential enum should detect missing variants"""
        passed, msg = test_non_sequential_enum_partial()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_guards_finite(self):
        """Guards with finite types should still require catch-all"""
        passed, msg = test_guards_with_finite_types()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_guards_complex(self):
        """Guards with complex conditions should be conservative"""
        passed, msg = test_guards_with_complex_conditions()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_multi_different_enums(self):
        """Multi-subject with different enums should detect missing combinations"""
        passed, msg = test_multi_subject_different_enums()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_multi_partial_enum(self):
        """Multi-subject partial enum coverage should be detected"""
        passed, msg = test_multi_subject_partial_enum()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_variable_binding(self):
        """Variable bindings should provide exhaustiveness"""
        self.assertEqual(test_variable_binding_exhaustiveness(), 1)

    def test_mixed_binding(self):
        """Mixed literal and variable binding should detect gaps"""
        passed, msg = test_mixed_literal_variable_binding()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_large_struct_partial(self):
        """Large struct partial coverage should be detected"""
        passed, msg = test_large_struct_partial_coverage()
        self.assertTrue(passed, f"Expected failure but got: {msg}")

    def test_enum_payload_correlation(self):
        """Enum payload field correlation should be preserved"""
        passed, msg = test_enum_payload_correlation()
        self.assertTrue(passed, f"Expected failure but got: {msg}")


if __name__ == '__main__':
    unittest.main(verbosity=2)