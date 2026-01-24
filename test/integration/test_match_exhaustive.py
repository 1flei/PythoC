#!/usr/bin/env python3
"""
Match/case exhaustiveness checking tests

Tests for the exhaustiveness checking feature that ensures all match
statements cover all possible cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, i8, compile, enum, ptr, struct
from pythoc import bool as pc_bool
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error
import unittest

# Enable exception raising for tests
set_raise_on_error(True)


# ============================================================
# Test enums
# ============================================================

@enum(i8)
class Color:
    Red: None
    Green: None
    Blue: None


@enum(i32)
class Status:
    Ok: i32
    Error: ptr[i8]
    Pending: None


@enum(i32)
class Wrapper:
    Flag: pc_bool


# ============================================================
# Test structs with finite fields
# ============================================================

@struct
class BoolPair:
    x: pc_bool
    y: pc_bool


# ============================================================
# Tests that SHOULD PASS (exhaustive matches)
# ============================================================

@compile
def exhaustive_bool(x: pc_bool) -> i32:
    """Exhaustive bool match - covers True and False"""
    match x:
        case True:
            return 1
        case False:
            return 0


@compile
def exhaustive_bool_wildcard(x: pc_bool) -> i32:
    """Bool match with wildcard"""
    match x:
        case True:
            return 1
        case _:
            return 0


@compile
def exhaustive_int_wildcard(x: i32) -> i32:
    """Int match with wildcard (required for non-enumerable)"""
    match x:
        case 0:
            return 0
        case 1:
            return 1
        case _:
            return 99


@compile
def exhaustive_int_binding(x: i32) -> i32:
    """Int match with variable binding (equivalent to wildcard)"""
    match x:
        case 0:
            return 0
        case n:
            return n


@compile
def exhaustive_guard_with_wildcard(x: i32) -> i32:
    """Guards with final wildcard"""
    match x:
        case n if n > 0:
            return 1
        case n if n < 0:
            return -1
        case _:
            return 0


# Enum tests - create enum internally to avoid ctypes conversion issues
@compile
def test_enum_all_variants_red() -> i32:
    """Enum match covering all variants - test Red"""
    c: Color = Color(Color.Red)
    match c:
        case (Color.Red):
            return 1
        case (Color.Green):
            return 2
        case (Color.Blue):
            return 3


@compile
def test_enum_all_variants_green() -> i32:
    """Enum match covering all variants - test Green"""
    c: Color = Color(Color.Green)
    match c:
        case (Color.Red):
            return 1
        case (Color.Green):
            return 2
        case (Color.Blue):
            return 3


@compile
def test_enum_all_variants_blue() -> i32:
    """Enum match covering all variants - test Blue"""
    c: Color = Color(Color.Blue)
    match c:
        case (Color.Red):
            return 1
        case (Color.Green):
            return 2
        case (Color.Blue):
            return 3


@compile
def test_enum_wildcard_red() -> i32:
    """Enum match with wildcard - test Red"""
    c: Color = Color(Color.Red)
    match c:
        case (Color.Red):
            return 1
        case _:
            return 99


@compile
def test_enum_wildcard_blue() -> i32:
    """Enum match with wildcard - test Blue"""
    c: Color = Color(Color.Blue)
    match c:
        case (Color.Red):
            return 1
        case _:
            return 99


@compile
def test_status_all_ok() -> i32:
    """Status enum with all variants covered - test Ok"""
    s: Status = Status(Status.Ok, 42)
    match s:
        case (Status.Ok, _):
            return 1
        case (Status.Error, _):
            return 2
        case (Status.Pending):
            return 3


@compile
def test_status_all_pending() -> i32:
    """Status enum with all variants covered - test Pending"""
    s: Status = Status(Status.Pending)
    match s:
        case (Status.Ok, _):
            return 1
        case (Status.Error, _):
            return 2
        case (Status.Pending):
            return 3


# ============================================================
# Struct pattern tests (product types with finite fields)
# ============================================================

@compile
def test_struct_all_cases_tt() -> i32:
    """Struct match covering all 4 combinations - returns 1 for (True, True)"""
    p: BoolPair = BoolPair()
    p.x = True
    p.y = True
    match p:
        case BoolPair(x=True, y=True):
            return 1
        case BoolPair(x=True, y=False):
            return 2
        case BoolPair(x=False, y=True):
            return 3
        case BoolPair(x=False, y=False):
            return 4


@compile
def test_struct_all_cases_tf() -> i32:
    """Struct match covering all 4 combinations - returns 2 for (True, False)"""
    p: BoolPair = BoolPair()
    p.x = True
    p.y = False
    match p:
        case BoolPair(x=True, y=True):
            return 1
        case BoolPair(x=True, y=False):
            return 2
        case BoolPair(x=False, y=True):
            return 3
        case BoolPair(x=False, y=False):
            return 4


@compile
def test_struct_all_cases_ft() -> i32:
    """Struct match covering all 4 combinations - returns 3 for (False, True)"""
    p: BoolPair = BoolPair()
    p.x = False
    p.y = True
    match p:
        case BoolPair(x=True, y=True):
            return 1
        case BoolPair(x=True, y=False):
            return 2
        case BoolPair(x=False, y=True):
            return 3
        case BoolPair(x=False, y=False):
            return 4


@compile
def test_struct_all_cases_ff() -> i32:
    """Struct match covering all 4 combinations - returns 4 for (False, False)"""
    p: BoolPair = BoolPair()
    p.x = False
    p.y = False
    match p:
        case BoolPair(x=True, y=True):
            return 1
        case BoolPair(x=True, y=False):
            return 2
        case BoolPair(x=False, y=True):
            return 3
        case BoolPair(x=False, y=False):
            return 4


@compile
def test_struct_with_wildcard() -> i32:
    """Struct match with wildcard"""
    p: BoolPair = BoolPair()
    p.x = True
    p.y = False
    match p:
        case BoolPair(x=True, y=True):
            return 1
        case _:
            return 99


# ============================================================
# Multi-subject exhaustiveness tests
# ============================================================

@compile
def test_multi_subject_all_cases_tt() -> i32:
    """Multi-subject match covering all 4 combinations"""
    x: pc_bool = True
    y: pc_bool = True
    match x, y:
        case (True, True):
            return 1
        case (True, False):
            return 2
        case (False, True):
            return 3
        case (False, False):
            return 4


@compile
def test_multi_subject_all_cases_tf() -> i32:
    """Multi-subject match covering all 4 combinations"""
    x: pc_bool = True
    y: pc_bool = False
    match x, y:
        case (True, True):
            return 1
        case (True, False):
            return 2
        case (False, True):
            return 3
        case (False, False):
            return 4


@compile
def test_multi_subject_with_wildcard() -> i32:
    """Multi-subject match with wildcard"""
    x: pc_bool = False
    y: pc_bool = True
    match x, y:
        case (True, True):
            return 1
        case _:
            return 99


# ============================================================
# Enum with bool payload exhaustiveness tests
# ============================================================

@compile
def test_enum_bool_payload_all_tt() -> i32:
    """Enum with bool payload - exhaustive match covering both payload values"""
    w: Wrapper = Wrapper(Wrapper.Flag, True)
    match w:
        case (Wrapper.Flag, True):
            return 1
        case (Wrapper.Flag, False):
            return 0


@compile
def test_enum_bool_payload_all_ff() -> i32:
    """Enum with bool payload - exhaustive match covering both payload values"""
    w: Wrapper = Wrapper(Wrapper.Flag, False)
    match w:
        case (Wrapper.Flag, True):
            return 1
        case (Wrapper.Flag, False):
            return 0


@compile
def test_enum_bool_payload_wildcard() -> i32:
    """Enum with bool payload - wildcard for payload"""
    w: Wrapper = Wrapper(Wrapper.Flag, True)
    match w:
        case (Wrapper.Flag, _):
            return 99


class TestExhaustiveMatches(unittest.TestCase):
    """Tests for matches that should pass exhaustiveness checking"""

    def test_bool_exhaustive(self):
        self.assertEqual(exhaustive_bool(True), 1)
        self.assertEqual(exhaustive_bool(False), 0)

    def test_bool_wildcard(self):
        self.assertEqual(exhaustive_bool_wildcard(True), 1)
        self.assertEqual(exhaustive_bool_wildcard(False), 0)

    def test_int_wildcard(self):
        self.assertEqual(exhaustive_int_wildcard(0), 0)
        self.assertEqual(exhaustive_int_wildcard(1), 1)
        self.assertEqual(exhaustive_int_wildcard(42), 99)

    def test_int_binding(self):
        self.assertEqual(exhaustive_int_binding(0), 0)
        self.assertEqual(exhaustive_int_binding(42), 42)

    def test_guard_with_wildcard(self):
        self.assertEqual(exhaustive_guard_with_wildcard(10), 1)
        self.assertEqual(exhaustive_guard_with_wildcard(-5), -1)
        self.assertEqual(exhaustive_guard_with_wildcard(0), 0)

    def test_enum_all_variants(self):
        self.assertEqual(test_enum_all_variants_red(), 1)
        self.assertEqual(test_enum_all_variants_green(), 2)
        self.assertEqual(test_enum_all_variants_blue(), 3)

    def test_enum_wildcard(self):
        self.assertEqual(test_enum_wildcard_red(), 1)
        self.assertEqual(test_enum_wildcard_blue(), 99)

    def test_status_all(self):
        self.assertEqual(test_status_all_ok(), 1)
        self.assertEqual(test_status_all_pending(), 3)

    def test_struct_all_cases(self):
        self.assertEqual(test_struct_all_cases_tt(), 1)
        self.assertEqual(test_struct_all_cases_tf(), 2)
        self.assertEqual(test_struct_all_cases_ft(), 3)
        self.assertEqual(test_struct_all_cases_ff(), 4)

    def test_struct_wildcard(self):
        self.assertEqual(test_struct_with_wildcard(), 99)

    def test_multi_subject_all_cases(self):
        self.assertEqual(test_multi_subject_all_cases_tt(), 1)
        self.assertEqual(test_multi_subject_all_cases_tf(), 2)

    def test_multi_subject_wildcard(self):
        self.assertEqual(test_multi_subject_with_wildcard(), 99)

    def test_enum_bool_payload(self):
        self.assertEqual(test_enum_bool_payload_all_tt(), 1)
        self.assertEqual(test_enum_bool_payload_all_ff(), 0)
        self.assertEqual(test_enum_bool_payload_wildcard(), 99)


# ============================================================
# Tests that SHOULD FAIL (non-exhaustive matches)
# ============================================================

def test_struct_incomplete_should_fail():
    """Test that incomplete struct match fails compilation"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_struct_incomplete')
    try:
        @compile(suffix="bad_struct_incomplete")
        def bad_struct_incomplete() -> i32:
            p: BoolPair = BoolPair()
            p.x = True
            p.y = False
            match p:
                case BoolPair(x=True, y=True):
                    return 1
                case BoolPair(x=False, y=False):
                    return 2
                # Missing: (True, False) and (False, True)

        flush_all_pending_outputs()
        return False, "should have raised error"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_multi_subject_incomplete_should_fail():
    """Test that incomplete multi-subject match fails compilation"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_multi_incomplete')
    try:
        @compile(suffix="bad_multi_incomplete")
        def bad_multi_incomplete() -> i32:
            x: pc_bool = True
            y: pc_bool = True
            match x, y:
                case (True, True):
                    return 1
                case (False, False):
                    return 2
                # Missing: (True, False) and (False, True)

        flush_all_pending_outputs()
        return False, "should have raised error"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_guards_only_should_fail():
    """Test that match with only guarded patterns (no catch-all) fails"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_guards_only')
    try:
        @compile(suffix="bad_guards_only")
        def bad_guards_only(x: pc_bool) -> i32:
            match x:
                case b if b == True:
                    return 1
                case b if b == False:
                    return 0
                # ERROR: guards could all be False - should require catch-all!

        flush_all_pending_outputs()
        return False, "should have raised error"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_enum_bool_payload_incomplete_should_fail():
    """Test that incomplete enum bool payload match fails"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_enum_bool_payload')
    try:
        @compile(suffix="bad_enum_bool_payload")
        def bad_enum_bool_payload() -> i32:
            w: Wrapper = Wrapper(Wrapper.Flag, True)
            match w:
                case (Wrapper.Flag, True):
                    return 1
                # Missing: (Wrapper.Flag, False)

        flush_all_pending_outputs()
        return False, "should have raised error"
    except (ValueError, SystemExit) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


class TestNonExhaustiveMatches(unittest.TestCase):
    """Tests for matches that should fail exhaustiveness checking"""

    def test_struct_incomplete(self):
        """Incomplete struct match should fail"""
        passed, msg = test_struct_incomplete_should_fail()
        self.assertTrue(passed, msg)

    def test_multi_subject_incomplete(self):
        """Incomplete multi-subject match should fail"""
        passed, msg = test_multi_subject_incomplete_should_fail()
        self.assertTrue(passed, msg)

    def test_guards_only(self):
        """Guards-only match without catch-all should fail"""
        passed, msg = test_guards_only_should_fail()
        self.assertTrue(passed, msg)

    def test_enum_bool_payload_incomplete(self):
        """Incomplete enum bool payload match should fail"""
        passed, msg = test_enum_bool_payload_incomplete_should_fail()
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
