#!/usr/bin/env python3
"""
Integration tests for pythoc.regex.

Tests the compile-time regex system:
  - Python-level matching (match/search API)
  - @compile function generation for native execution

IMPORTANT: All @compile and regex generate_*_fn() calls happen at module
level to avoid "Cannot define new compiled function after native execution
has started" errors.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i8, i32, i64, u8, u64, ptr
from pythoc import libc
from pythoc.regex import compile as regex_compile, CompiledRegex


# ============================================================================
# Python-level regex tests
# ============================================================================

class TestRegexLiteralMatch(unittest.TestCase):

    def test_simple_literal(self):
        r = regex_compile("abc")
        ok, _ = r.match(b"abc");    self.assertTrue(ok)
        ok, _ = r.match(b"abcx");   self.assertTrue(ok)
        ok, _ = r.match(b"xabcx");  self.assertFalse(ok)
        ok, _ = r.match(b"ab");     self.assertFalse(ok)
        ok, _ = r.match(b"xyz");    self.assertFalse(ok)

    def test_single_char(self):
        r = regex_compile("x")
        ok, _ = r.match(b"x");    self.assertTrue(ok)
        ok, _ = r.match(b"xab");  self.assertTrue(ok)
        ok, _ = r.match(b"axb");  self.assertFalse(ok)
        ok, _ = r.match(b"abc");  self.assertFalse(ok)

    def test_fullmatch_via_anchors(self):
        r = regex_compile("^abc$")
        ok, _ = r.match(b"abc");   self.assertTrue(ok)
        ok, _ = r.match(b"abcd");  self.assertFalse(ok)
        ok, _ = r.match(b"xabc");  self.assertFalse(ok)


class TestRegexAlternation(unittest.TestCase):

    def test_two_way(self):
        r = regex_compile("cat|dog")
        self.assertTrue(r.match(b"cat")[0])
        self.assertTrue(r.match(b"dog")[0])
        self.assertFalse(r.match(b"car")[0])

    def test_three_way(self):
        r = regex_compile("cat|dog|bird")
        self.assertTrue(r.match(b"cat")[0])
        self.assertTrue(r.match(b"dog")[0])
        self.assertTrue(r.match(b"bird")[0])
        self.assertFalse(r.match(b"fish")[0])

    def test_in_context(self):
        r = regex_compile("cat|dog")
        self.assertFalse(r.match(b"I have a cat")[0])
        self.assertFalse(r.match(b"my dog")[0])
        self.assertTrue(r.match(b"cat is here")[0])
        self.assertTrue(r.match(b"dog is here")[0])


class TestRegexQuantifiers(unittest.TestCase):

    def test_star(self):
        r = regex_compile("ab*c")
        self.assertTrue(r.match(b"ac")[0])
        self.assertTrue(r.match(b"abc")[0])
        self.assertTrue(r.match(b"abbc")[0])
        self.assertTrue(r.match(b"abbbc")[0])
        self.assertFalse(r.match(b"adc")[0])

    def test_plus(self):
        r = regex_compile("ab+c")
        self.assertFalse(r.match(b"ac")[0])
        self.assertTrue(r.match(b"abc")[0])
        self.assertTrue(r.match(b"abbc")[0])

    def test_question(self):
        r = regex_compile("ab?c")
        self.assertTrue(r.match(b"ac")[0])
        self.assertTrue(r.match(b"abc")[0])
        self.assertFalse(r.match(b"abbc")[0])

    def test_dot_star(self):
        r = regex_compile("a.*b")
        self.assertTrue(r.match(b"ab")[0])
        self.assertTrue(r.match(b"axxb")[0])
        self.assertTrue(r.match(b"axyzb")[0])
        self.assertFalse(r.match(b"a")[0])

    def test_group_plus(self):
        r = regex_compile("(ab)+")
        self.assertTrue(r.match(b"ab")[0])
        self.assertTrue(r.match(b"abab")[0])
        self.assertTrue(r.match(b"ababab")[0])
        self.assertFalse(r.match(b"aa")[0])


class TestRegexCharClasses(unittest.TestCase):

    def test_simple_class(self):
        r = regex_compile("[abc]")
        self.assertTrue(r.match(b"a")[0])
        self.assertTrue(r.match(b"b")[0])
        self.assertTrue(r.match(b"c")[0])
        self.assertFalse(r.match(b"d")[0])

    def test_range(self):
        r = regex_compile("[a-z]+")
        self.assertTrue(r.match(b"hello")[0])
        self.assertFalse(r.match(b"123")[0])

    def test_digit_range(self):
        r = regex_compile("[0-9]+")
        self.assertTrue(r.match(b"123")[0])
        self.assertTrue(r.match(b"0")[0])
        self.assertFalse(r.match(b"abc")[0])

    def test_negated_class(self):
        r = regex_compile("[^0-9]+")
        self.assertTrue(r.match(b"abc")[0])
        self.assertFalse(r.match(b"123")[0])

    def test_multi_range(self):
        r = regex_compile("[a-zA-Z]+")
        self.assertTrue(r.match(b"Hello")[0])
        self.assertTrue(r.match(b"WORLD")[0])
        self.assertFalse(r.match(b"123")[0])

    def test_identifier_pattern(self):
        r = regex_compile("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertTrue(r.match(b"hello")[0])
        self.assertTrue(r.match(b"_var")[0])
        self.assertTrue(r.match(b"x123")[0])
        self.assertFalse(r.match(b"123")[0])


class TestRegexAnchors(unittest.TestCase):

    def test_start_anchor(self):
        r = regex_compile("^hello")
        self.assertTrue(r.match(b"hello world")[0])
        self.assertFalse(r.match(b"say hello")[0])

    def test_end_anchor(self):
        r = regex_compile("world$")
        self.assertTrue(r.match(b"world")[0])
        self.assertFalse(r.match(b"hello world")[0])
        self.assertFalse(r.match(b"world cup")[0])

    def test_both_anchors(self):
        r = regex_compile("^hello$")
        self.assertTrue(r.match(b"hello")[0])
        self.assertFalse(r.match(b"hello world")[0])
        self.assertFalse(r.match(b"say hello")[0])

    def test_anchored_with_quantifier(self):
        r = regex_compile("^[a-z]+$")
        self.assertTrue(r.match(b"hello")[0])
        self.assertFalse(r.match(b"Hello")[0])
        self.assertFalse(r.match(b"hello world")[0])
        self.assertFalse(r.match(b"123")[0])


class TestRegexEscapeSequences(unittest.TestCase):

    def test_digit_class(self):
        r = regex_compile("\\d+")
        self.assertTrue(r.match(b"123")[0])
        self.assertFalse(r.match(b"abc")[0])

    def test_word_class(self):
        r = regex_compile("\\w+")
        self.assertTrue(r.match(b"hello_123")[0])
        self.assertFalse(r.match(b"   ")[0])

    def test_space_class(self):
        r = regex_compile("\\s+")
        self.assertTrue(r.match(b"  \t\n")[0])
        self.assertFalse(r.match(b"abc")[0])

    def test_escaped_dot(self):
        r = regex_compile("a\\.b")
        self.assertTrue(r.match(b"a.b")[0])
        self.assertFalse(r.match(b"axb")[0])

    def test_escaped_backslash(self):
        r = regex_compile("a\\\\b")
        self.assertTrue(r.match(b"a\\b")[0])
        self.assertFalse(r.match(b"a/b")[0])


class TestRegexSearch(unittest.TestCase):

    def test_search_found(self):
        r = regex_compile("foo")
        ok, info = r.search(b"xxfooxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)

    def test_search_not_found(self):
        r = regex_compile("foo")
        ok, info = r.search(b"xxbarxx")
        self.assertFalse(ok)

    def test_search_at_start(self):
        r = regex_compile("foo")
        ok, info = r.search(b"foobar")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)

    def test_search_at_end(self):
        r = regex_compile("foo")
        ok, info = r.search(b"barfoo")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_digit_sequence(self):
        r = regex_compile("\\d+")
        ok, info = r.search(b"abc123def")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_span(self):
        r = regex_compile("foo")
        ok, info = r.search(b"xxfooxx")
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 5))

    def test_search_not_found_empty(self):
        r = regex_compile("foo")
        ok, info = r.search(b"xxbarxx")
        self.assertFalse(ok)
        self.assertEqual(info, {})

    def test_search_span_lazy_dotstar(self):
        r = regex_compile("a.*b")
        ok, info = r.search(b"xxaxbxbxx")
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 5))


class TestRegexTags(unittest.TestCase):

    def test_variable_width_tag(self):
        r = regex_compile("a*{mid}b")
        ok, info = r.search(b"aaab")
        self.assertTrue(ok)
        self.assertEqual(info, {"start": 0, "end": 4, "mid": 3})

    def test_alternation_tags(self):
        r = regex_compile("{x}a|{y}bc")
        ok_a, info_a = r.search(b"za")
        self.assertTrue(ok_a)
        self.assertEqual(info_a, {"start": 1, "end": 2, "x": 1})
        ok_bc, info_bc = r.search(b"zbc")
        self.assertTrue(ok_bc)
        self.assertEqual(info_bc, {"start": 1, "end": 3, "y": 1})

    def test_loop_phase_tags(self):
        r = regex_compile("(ab)*{mid}cde")
        ok, info = r.search(b"zzababcdezz")
        self.assertTrue(ok)
        self.assertEqual(info, {"start": 2, "end": 9, "mid": 6})


class TestRegexComplex(unittest.TestCase):

    def test_email_like(self):
        r = regex_compile("[a-z]+@[a-z]+\\.[a-z]+")
        self.assertTrue(r.match(b"foo@bar.com")[0])
        self.assertFalse(r.match(b"email: a@b.cc")[0])
        self.assertFalse(r.match(b"foobar")[0])

    def test_ip_like(self):
        r = regex_compile("\\d+\\.\\d+\\.\\d+\\.\\d+")
        self.assertTrue(r.match(b"192.168.1.1")[0])
        self.assertFalse(r.match(b"addr: 10.0.0.1 here")[0])
        self.assertFalse(r.match(b"abc")[0])

    def test_hex_color(self):
        r = regex_compile("#[0-9a-f]+")
        self.assertTrue(r.match(b"#ff0000")[0])
        self.assertFalse(r.match(b"color: #abc")[0])
        self.assertFalse(r.match(b"no hash")[0])

    def test_nested_alternation_group(self):
        r = regex_compile("((a|b)c)+")
        self.assertTrue(r.match(b"ac")[0])
        self.assertTrue(r.match(b"bc")[0])
        self.assertTrue(r.match(b"acbc")[0])
        self.assertFalse(r.match(b"cc")[0])

    def test_optional_prefix(self):
        r = regex_compile("https?://")
        self.assertTrue(r.match(b"http://")[0])
        self.assertTrue(r.match(b"https://")[0])
        self.assertFalse(r.match(b"ftp://")[0])

    def test_empty_match(self):
        r = regex_compile("a*")
        self.assertTrue(r.match(b"")[0])
        self.assertTrue(r.match(b"aaa")[0])


# ============================================================================
# @compile function generation + native execution tests
#
# All compiled regex functions must be created at module level.
# Match functions have signature (n: u64, data: ptr[i8]) -> u8.
# ============================================================================

_r_abc = regex_compile("abc")
_abc_match = _r_abc.generate_match_fn()

_r_abpc = regex_compile("ab+c")
_abpc_match = _r_abpc.generate_match_fn()

_r_hello = regex_compile("^hello$")
_hello_match = _r_hello.generate_match_fn()

_r_az = regex_compile("^[a-z]+$")
_az_match = _r_az.generate_match_fn()

_r_catdog = regex_compile("cat|dog")
_catdog_match = _r_catdog.generate_match_fn()

_r_digits = regex_compile("\\d+")
_digits_match = _r_digits.generate_match_fn()

_r_dotstar = regex_compile("a.*b")
_dotstar_match = _r_dotstar.generate_match_fn()

_r_start = regex_compile("^hello")
_start_match = _r_start.generate_match_fn()

_r_diffalt = regex_compile("ab|cdef")
_r_suffix = regex_compile("abcd.*efg")
_suffix_match = _r_suffix.generate_match_fn()

_r_needle = regex_compile("needle")


@compile
def native_abc_match_exact() -> u8:
    return _abc_match(u64(3), "abc")

@compile
def native_abc_match_embedded() -> u8:
    return _abc_match(u64(7), "xxabcxx")

@compile
def native_abc_no_match() -> u8:
    return _abc_match(u64(3), "xyz")

@compile
def native_abc_partial_no_match() -> u8:
    return _abc_match(u64(2), "ab")

@compile
def native_abpc_match_abc() -> u8:
    return _abpc_match(u64(3), "abc")

@compile
def native_abpc_match_abbc() -> u8:
    return _abpc_match(u64(4), "abbc")

@compile
def native_abpc_no_match_ac() -> u8:
    return _abpc_match(u64(2), "ac")

@compile
def native_hello_exact() -> u8:
    return _hello_match(u64(5), "hello")

@compile
def native_hello_extra() -> u8:
    return _hello_match(u64(11), "hello world")

@compile
def native_hello_prefix() -> u8:
    return _hello_match(u64(9), "say hello")

@compile
def native_az_lower() -> u8:
    return _az_match(u64(5), "hello")

@compile
def native_az_upper() -> u8:
    return _az_match(u64(5), "Hello")

@compile
def native_az_digits() -> u8:
    return _az_match(u64(6), "abc123")

@compile
def native_catdog_cat() -> u8:
    return _catdog_match(u64(3), "cat")

@compile
def native_catdog_dog() -> u8:
    return _catdog_match(u64(3), "dog")

@compile
def native_catdog_fish() -> u8:
    return _catdog_match(u64(4), "fish")

@compile
def native_digits_numbers() -> u8:
    return _digits_match(u64(5), "12345")

@compile
def native_digits_letters() -> u8:
    return _digits_match(u64(5), "abcde")

@compile
def native_dotstar_ab() -> u8:
    return _dotstar_match(u64(2), "ab")

@compile
def native_dotstar_axxb() -> u8:
    return _dotstar_match(u64(4), "axxb")

@compile
def native_dotstar_no_b() -> u8:
    return _dotstar_match(u64(3), "axx")

@compile
def native_start_yes() -> u8:
    return _start_match(u64(11), "hello world")

@compile
def native_start_no() -> u8:
    return _start_match(u64(9), "say hello")

@compile
def native_suffix_no_match() -> u8:
    return _suffix_match(u64(10), "abcdxxxxxx")

@compile
def native_suffix_match() -> u8:
    return _suffix_match(u64(9), "abcdxxefg")


class TestRegexCodegen(unittest.TestCase):

    # --- Literal "abc" ---

    def test_match_literal_exact(self):
        self.assertEqual(native_abc_match_exact(), 1)

    def test_match_literal_embedded(self):
        self.assertEqual(native_abc_match_embedded(), 0)

    def test_match_literal_no_match(self):
        self.assertEqual(native_abc_no_match(), 0)

    def test_match_literal_partial(self):
        self.assertEqual(native_abc_partial_no_match(), 0)

    # --- Quantifier "ab+c" ---

    def test_match_plus_abc(self):
        self.assertEqual(native_abpc_match_abc(), 1)

    def test_match_plus_abbc(self):
        self.assertEqual(native_abpc_match_abbc(), 1)

    def test_match_plus_no_b(self):
        self.assertEqual(native_abpc_no_match_ac(), 0)

    # --- Anchored "^hello$" ---

    def test_match_anchored_exact(self):
        self.assertEqual(native_hello_exact(), 1)

    def test_match_anchored_too_long(self):
        self.assertEqual(native_hello_extra(), 0)

    def test_match_anchored_wrong_start(self):
        self.assertEqual(native_hello_prefix(), 0)

    # --- Char class "^[a-z]+$" ---

    def test_match_charclass_lower(self):
        self.assertEqual(native_az_lower(), 1)

    def test_match_charclass_upper(self):
        self.assertEqual(native_az_upper(), 0)

    def test_match_charclass_digits(self):
        self.assertEqual(native_az_digits(), 0)

    # --- Alternation "cat|dog" ---

    def test_match_alt_cat(self):
        self.assertEqual(native_catdog_cat(), 1)

    def test_match_alt_dog(self):
        self.assertEqual(native_catdog_dog(), 1)

    def test_match_alt_fish(self):
        self.assertEqual(native_catdog_fish(), 0)

    # --- Escape class "\\d+" ---

    def test_match_digits_numbers(self):
        self.assertEqual(native_digits_numbers(), 1)

    def test_match_digits_letters(self):
        self.assertEqual(native_digits_letters(), 0)

    # --- Dot-star "a.*b" ---

    def test_match_dotstar_ab(self):
        self.assertEqual(native_dotstar_ab(), 1)

    def test_match_dotstar_axxb(self):
        self.assertEqual(native_dotstar_axxb(), 1)

    def test_match_dotstar_no_b(self):
        self.assertEqual(native_dotstar_no_b(), 0)

    # --- Start-anchored "^hello" ---

    def test_match_start_anchored_yes(self):
        self.assertEqual(native_start_yes(), 1)

    def test_match_start_anchored_no(self):
        self.assertEqual(native_start_no(), 0)

    # --- Suffix guard "abcd.*efg" ---

    def test_suffix_guard_no_match(self):
        self.assertEqual(native_suffix_no_match(), 0)

    def test_suffix_guard_match(self):
        self.assertEqual(native_suffix_match(), 1)

    # --- Search (Python-level, verified against native pipeline) ---

    def test_search_abc_found(self):
        ok, info = _r_abc.search(b"xxabcxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)

    def test_search_abc_not_found(self):
        ok, _ = _r_abc.search(b"xxbarxx")
        self.assertFalse(ok)

    def test_search_abc_at_start(self):
        ok, info = _r_abc.search(b"abcdef")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)

    def test_search_catdog_embedded(self):
        ok, info = _r_catdog.search(b"my dog runs")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_digits_mixed(self):
        ok, info = _r_digits.search(b"abc123def")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_diffalt_ab(self):
        ok, info = _r_diffalt.search(b"xxxabxxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_diffalt_cdef(self):
        ok, info = _r_diffalt.search(b"xxxcdefx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_diffalt_none(self):
        ok, _ = _r_diffalt.search(b"xxxxxxxx")
        self.assertFalse(ok)

    def test_search_suffix_found(self):
        ok, info = _r_suffix.search(b"xxabcdxefgx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)

    def test_search_suffix_none(self):
        ok, _ = _r_suffix.search(b"xxabcdxxxxx")
        self.assertFalse(ok)

    def test_skip_search_end(self):
        ok, info = _r_needle.search(
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxneedle")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 50)

    def test_skip_search_start(self):
        ok, info = _r_needle.search(
            b"needlexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)

    def test_skip_search_none(self):
        ok, _ = _r_needle.search(
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.assertFalse(ok)

    def test_skip_search_mid(self):
        ok, info = _r_needle.search(
            b"xxxxxxxxxxxxxxxxxxxxxxxxxneedlexxxxxxxxxxxxxxxxxxxxxxxxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 25)


if __name__ == "__main__":
    unittest.main()
