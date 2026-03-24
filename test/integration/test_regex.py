#!/usr/bin/env python3
"""
Integration tests for pythoc.regex.

Tests the compile-time regex system:
  - Python-level matching (DFA simulation)
  - @compile function generation for native execution

The Python-level tests verify correctness of the regex engine.
The @compile tests verify the generated native code produces correct results.

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
# Python-level regex tests (DFA simulation)
# ============================================================================

class TestRegexLiteralMatch(unittest.TestCase):
    """Test basic literal pattern matching."""

    def test_simple_literal(self):
        r = regex_compile("abc")
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abcx"))   # matches at start
        self.assertFalse(r.is_match(b"xabcx"))  # no match at start
        self.assertFalse(r.is_match(b"ab"))
        self.assertFalse(r.is_match(b"xyz"))

    def test_single_char(self):
        r = regex_compile("x")
        self.assertTrue(r.is_match(b"x"))
        self.assertTrue(r.is_match(b"xab"))    # matches at start
        self.assertFalse(r.is_match(b"axb"))   # no match at start
        self.assertFalse(r.is_match(b"abc"))

    def test_fullmatch_literal(self):
        r = regex_compile("abc")
        self.assertTrue(r.fullmatch(b"abc"))
        self.assertFalse(r.fullmatch(b"abcd"))
        self.assertFalse(r.fullmatch(b"xabc"))


class TestRegexAlternation(unittest.TestCase):
    """Test alternation (|) patterns."""

    def test_two_way(self):
        r = regex_compile("cat|dog")
        self.assertTrue(r.is_match(b"cat"))
        self.assertTrue(r.is_match(b"dog"))
        self.assertFalse(r.is_match(b"car"))

    def test_three_way(self):
        r = regex_compile("cat|dog|bird")
        self.assertTrue(r.is_match(b"cat"))
        self.assertTrue(r.is_match(b"dog"))
        self.assertTrue(r.is_match(b"bird"))
        self.assertFalse(r.is_match(b"fish"))

    def test_in_context(self):
        r = regex_compile("cat|dog")
        self.assertFalse(r.is_match(b"I have a cat"))  # no match at start
        self.assertFalse(r.is_match(b"my dog"))          # no match at start
        self.assertTrue(r.is_match(b"cat is here"))
        self.assertTrue(r.is_match(b"dog is here"))


class TestRegexQuantifiers(unittest.TestCase):
    """Test quantifier patterns (?, *, +)."""

    def test_star(self):
        r = regex_compile("ab*c")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abbc"))
        self.assertTrue(r.is_match(b"abbbc"))
        self.assertFalse(r.is_match(b"adc"))

    def test_plus(self):
        r = regex_compile("ab+c")
        self.assertFalse(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abbc"))

    def test_question(self):
        r = regex_compile("ab?c")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertFalse(r.is_match(b"abbc"))

    def test_dot_star(self):
        r = regex_compile("a.*b")
        self.assertTrue(r.is_match(b"ab"))
        self.assertTrue(r.is_match(b"axxb"))
        self.assertTrue(r.is_match(b"axyzb"))
        self.assertFalse(r.is_match(b"a"))

    def test_group_plus(self):
        r = regex_compile("(ab)+")
        self.assertTrue(r.is_match(b"ab"))
        self.assertTrue(r.is_match(b"abab"))
        self.assertTrue(r.is_match(b"ababab"))
        self.assertFalse(r.is_match(b"aa"))


class TestRegexCharClasses(unittest.TestCase):
    """Test character class patterns."""

    def test_simple_class(self):
        r = regex_compile("[abc]")
        self.assertTrue(r.is_match(b"a"))
        self.assertTrue(r.is_match(b"b"))
        self.assertTrue(r.is_match(b"c"))
        self.assertFalse(r.is_match(b"d"))

    def test_range(self):
        r = regex_compile("[a-z]+")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"123"))

    def test_digit_range(self):
        r = regex_compile("[0-9]+")
        self.assertTrue(r.is_match(b"123"))
        self.assertTrue(r.is_match(b"0"))
        self.assertFalse(r.is_match(b"abc"))

    def test_negated_class(self):
        r = regex_compile("[^0-9]+")
        self.assertTrue(r.is_match(b"abc"))
        self.assertFalse(r.is_match(b"123"))

    def test_multi_range(self):
        r = regex_compile("[a-zA-Z]+")
        self.assertTrue(r.is_match(b"Hello"))
        self.assertTrue(r.is_match(b"WORLD"))
        self.assertFalse(r.is_match(b"123"))

    def test_identifier_pattern(self):
        r = regex_compile("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertTrue(r.is_match(b"hello"))
        self.assertTrue(r.is_match(b"_var"))
        self.assertTrue(r.is_match(b"x123"))
        self.assertFalse(r.is_match(b"123"))


class TestRegexAnchors(unittest.TestCase):
    """Test anchor patterns (^, $)."""

    def test_start_anchor(self):
        r = regex_compile("^hello")
        self.assertTrue(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"say hello"))

    def test_end_anchor(self):
        r = regex_compile("world$")
        self.assertTrue(r.is_match(b"world"))
        self.assertFalse(r.is_match(b"hello world"))  # no match at start
        self.assertFalse(r.is_match(b"world cup"))

    def test_both_anchors(self):
        r = regex_compile("^hello$")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"say hello"))

    def test_anchored_with_quantifier(self):
        r = regex_compile("^[a-z]+$")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"Hello"))
        self.assertFalse(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"123"))


class TestRegexEscapeSequences(unittest.TestCase):
    """Test escape sequence patterns."""

    def test_digit_class(self):
        r = regex_compile("\\d+")
        self.assertTrue(r.is_match(b"123"))
        self.assertFalse(r.is_match(b"abc"))

    def test_word_class(self):
        r = regex_compile("\\w+")
        self.assertTrue(r.is_match(b"hello_123"))
        self.assertFalse(r.is_match(b"   "))

    def test_space_class(self):
        r = regex_compile("\\s+")
        self.assertTrue(r.is_match(b"  \t\n"))
        self.assertFalse(r.is_match(b"abc"))

    def test_escaped_dot(self):
        r = regex_compile("a\\.b")
        self.assertTrue(r.is_match(b"a.b"))
        self.assertFalse(r.is_match(b"axb"))

    def test_escaped_backslash(self):
        r = regex_compile("a\\\\b")
        self.assertTrue(r.is_match(b"a\\b"))
        self.assertFalse(r.is_match(b"a/b"))


class TestRegexSearch(unittest.TestCase):
    """Test search functionality."""

    def test_search_found(self):
        r = regex_compile("foo")
        self.assertEqual(r.search(b"xxfooxx"), 2)

    def test_search_not_found(self):
        r = regex_compile("foo")
        self.assertEqual(r.search(b"xxbarxx"), -1)

    def test_search_at_start(self):
        r = regex_compile("foo")
        self.assertEqual(r.search(b"foobar"), 0)

    def test_search_at_end(self):
        r = regex_compile("foo")
        self.assertEqual(r.search(b"barfoo"), 3)

    def test_search_digit_sequence(self):
        r = regex_compile("\\d+")
        self.assertEqual(r.search(b"abc123def"), 3)

    def test_find_span(self):
        r = regex_compile("foo")
        span = r.find_span(b"xxfooxx")
        self.assertEqual(span, (2, 5))

    def test_find_span_not_found(self):
        r = regex_compile("foo")
        self.assertIsNone(r.find_span(b"xxbarxx"))

    def test_find_span_lazy_dotstar(self):
        r = regex_compile("a.*b")
        self.assertEqual(r.find_span(b"xxaxbxbxx"), (2, 5))


class TestRegexTags(unittest.TestCase):
    """Test zero-width tag reporting."""

    def test_variable_width_tag(self):
        r = regex_compile("a*{mid}b")
        result = r.find_with_tags(b"aaab")
        self.assertEqual(result, {"start": 0, "end": 4, "mid": 3})

    def test_alternation_tags(self):
        r = regex_compile("{x}a|{y}bc")
        self.assertEqual(r.find_with_tags(b"za"), {"start": 1, "end": 2, "x": 1})
        self.assertEqual(r.find_with_tags(b"zbc"), {"start": 1, "end": 3, "y": 1})

    def test_loop_phase_tags(self):
        r = regex_compile("(ab)*{mid}cde")
        self.assertEqual(r.find_with_tags(b"zzababcdezz"),
                         {"start": 2, "end": 9, "mid": 6})


class TestRegexComplex(unittest.TestCase):
    """Test complex/real-world-like patterns."""

    def test_email_like(self):
        r = regex_compile("[a-z]+@[a-z]+\\.[a-z]+")
        self.assertTrue(r.is_match(b"foo@bar.com"))
        self.assertFalse(r.is_match(b"email: a@b.cc"))  # no match at start
        self.assertFalse(r.is_match(b"foobar"))

    def test_ip_like(self):
        r = regex_compile("\\d+\\.\\d+\\.\\d+\\.\\d+")
        self.assertTrue(r.is_match(b"192.168.1.1"))
        self.assertFalse(r.is_match(b"addr: 10.0.0.1 here"))  # no match at start
        self.assertFalse(r.is_match(b"abc"))

    def test_hex_color(self):
        r = regex_compile("#[0-9a-f]+")
        self.assertTrue(r.is_match(b"#ff0000"))
        self.assertFalse(r.is_match(b"color: #abc"))  # no match at start
        self.assertFalse(r.is_match(b"no hash"))

    def test_nested_alternation_group(self):
        r = regex_compile("((a|b)c)+")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"bc"))
        self.assertTrue(r.is_match(b"acbc"))
        self.assertFalse(r.is_match(b"cc"))

    def test_optional_prefix(self):
        r = regex_compile("https?://")
        self.assertTrue(r.is_match(b"http://"))
        self.assertTrue(r.is_match(b"https://"))
        self.assertFalse(r.is_match(b"ftp://"))

    def test_empty_match(self):
        r = regex_compile("a*")
        self.assertTrue(r.is_match(b""))
        self.assertTrue(r.is_match(b"aaa"))


# ============================================================================
# @compile function generation + native execution tests
#
# All compiled regex functions and their @compile callers must be created
# at module level before any test method runs.
# ============================================================================

# --- Pattern: "abc" (literal, unanchored) ---
_r_abc = regex_compile("abc")
_abc_is_match = _r_abc.generate_is_match_fn()
_abc_search = _r_abc.generate_search_fn()


@compile
def native_abc_match_exact() -> u8:
    """Match "abc" against "abc" -- should match (1)."""
    return _abc_is_match(u64(3), "abc")


@compile
def native_abc_match_embedded() -> u8:
    """Match "abc" in "xxabcxx" -- no match at start (0)."""
    return _abc_is_match(u64(7), "xxabcxx")


@compile
def native_abc_no_match() -> u8:
    """Match "abc" against "xyz" -- should not match (0)."""
    return _abc_is_match(u64(3), "xyz")


@compile
def native_abc_partial_no_match() -> u8:
    """Match "abc" against "ab" -- partial, should not match (0)."""
    return _abc_is_match(u64(2), "ab")


@compile
def native_abc_search_found() -> i64:
    """Search "abc" in "xxabcxx" -- should find at position 2."""
    return _abc_search(u64(7), "xxabcxx")


@compile
def native_abc_search_not_found() -> i64:
    """Search "abc" in "xxbarxx" -- should return -1."""
    return _abc_search(u64(7), "xxbarxx")


@compile
def native_abc_search_at_start() -> i64:
    """Search "abc" in "abcdef" -- should find at position 0."""
    return _abc_search(u64(6), "abcdef")


# --- Pattern: "ab+c" (quantifier +, unanchored) ---
_r_abpc = regex_compile("ab+c")
_abpc_is_match = _r_abpc.generate_is_match_fn()


@compile
def native_abpc_match_abc() -> u8:
    """Match "ab+c" against "abc" -- should match (1)."""
    return _abpc_is_match(u64(3), "abc")


@compile
def native_abpc_match_abbc() -> u8:
    """Match "ab+c" against "abbc" -- should match (1)."""
    return _abpc_is_match(u64(4), "abbc")


@compile
def native_abpc_no_match_ac() -> u8:
    """Match "ab+c" against "ac" -- no 'b', should not match (0)."""
    return _abpc_is_match(u64(2), "ac")


# --- Pattern: "^hello$" (anchored both, literal) ---
_r_hello = regex_compile("^hello$")
_hello_is_match = _r_hello.generate_is_match_fn()


@compile
def native_hello_exact() -> u8:
    """Match "^hello$" against "hello" -- exact match (1)."""
    return _hello_is_match(u64(5), "hello")


@compile
def native_hello_extra() -> u8:
    """Match "^hello$" against "hello world" -- too long (0)."""
    return _hello_is_match(u64(11), "hello world")


@compile
def native_hello_prefix() -> u8:
    """Match "^hello$" against "say hello" -- wrong start (0)."""
    return _hello_is_match(u64(9), "say hello")


# --- Pattern: "^[a-z]+$" (anchored, char class + quantifier) ---
_r_az = regex_compile("^[a-z]+$")
_az_is_match = _r_az.generate_is_match_fn()


@compile
def native_az_lower() -> u8:
    """Match "^[a-z]+$" against "hello" -- all lowercase (1)."""
    return _az_is_match(u64(5), "hello")


@compile
def native_az_upper() -> u8:
    """Match "^[a-z]+$" against "Hello" -- has uppercase (0)."""
    return _az_is_match(u64(5), "Hello")


@compile
def native_az_digits() -> u8:
    """Match "^[a-z]+$" against "abc123" -- has digits (0)."""
    return _az_is_match(u64(6), "abc123")


# --- Pattern: "cat|dog" (alternation, unanchored) ---
_r_catdog = regex_compile("cat|dog")
_catdog_is_match = _r_catdog.generate_is_match_fn()
_catdog_search = _r_catdog.generate_search_fn()


@compile
def native_catdog_cat() -> u8:
    """Match "cat|dog" against "cat" -- should match (1)."""
    return _catdog_is_match(u64(3), "cat")


@compile
def native_catdog_dog() -> u8:
    """Match "cat|dog" against "dog" -- should match (1)."""
    return _catdog_is_match(u64(3), "dog")


@compile
def native_catdog_fish() -> u8:
    """Match "cat|dog" against "fish" -- should not match (0)."""
    return _catdog_is_match(u64(4), "fish")


@compile
def native_catdog_search_embedded() -> i64:
    """Search "cat|dog" in "my dog runs" -- should find at 3."""
    return _catdog_search(u64(11), "my dog runs")


# --- Pattern: "\\d+" (escape class, unanchored) ---
_r_digits = regex_compile("\\d+")
_digits_is_match = _r_digits.generate_is_match_fn()
_digits_search = _r_digits.generate_search_fn()


@compile
def native_digits_numbers() -> u8:
    """Match "\\d+" against "12345" -- all digits (1)."""
    return _digits_is_match(u64(5), "12345")


@compile
def native_digits_letters() -> u8:
    """Match "\\d+" against "abcde" -- no digits (0)."""
    return _digits_is_match(u64(5), "abcde")


@compile
def native_digits_search_mixed() -> i64:
    """Search "\\d+" in "abc123def" -- should find at 3."""
    return _digits_search(u64(9), "abc123def")


# --- Pattern: "a.*b" (dot-star, unanchored) ---
_r_dotstar = regex_compile("a.*b")
_dotstar_is_match = _r_dotstar.generate_is_match_fn()


@compile
def native_dotstar_ab() -> u8:
    """Match "a.*b" against "ab" -- zero dots (1)."""
    return _dotstar_is_match(u64(2), "ab")


@compile
def native_dotstar_axxb() -> u8:
    """Match "a.*b" against "axxb" -- two dots (1)."""
    return _dotstar_is_match(u64(4), "axxb")


@compile
def native_dotstar_no_b() -> u8:
    """Match "a.*b" against "axx" -- no trailing b (0)."""
    return _dotstar_is_match(u64(3), "axx")


# --- Pattern: "^hello" (start-anchored only) ---
_r_start = regex_compile("^hello")
_start_is_match = _r_start.generate_is_match_fn()


@compile
def native_start_yes() -> u8:
    """Match "^hello" against "hello world" -- starts with hello (1)."""
    return _start_is_match(u64(11), "hello world")


@compile
def native_start_no() -> u8:
    """Match "^hello" against "say hello" -- doesn't start (0)."""
    return _start_is_match(u64(9), "say hello")


# --- 1-pass search: different-length alternation (Opt #1) ---
_r_diffalt = regex_compile("ab|cdef")
_diffalt_search = _r_diffalt.generate_search_fn()

@compile
def native_diffalt_search_ab() -> i64:
    """Search "ab|cdef" in "xxxabxxx" -- expect 3."""
    return _diffalt_search(u64(8), "xxxabxxx")

@compile
def native_diffalt_search_cdef() -> i64:
    """Search "ab|cdef" in "xxxcdefx" -- expect 3."""
    return _diffalt_search(u64(8), "xxxcdefx")

@compile
def native_diffalt_search_none() -> i64:
    """Search "ab|cdef" in "xxxxxxxx" -- expect -1."""
    return _diffalt_search(u64(8), "xxxxxxxx")


# --- Suffix guard (Opt #3) ---
_r_suffix = regex_compile("abcd.*efg")
_suffix_is_match = _r_suffix.generate_is_match_fn()
_suffix_search = _r_suffix.generate_search_fn()

@compile
def native_suffix_no_match() -> u8:
    """Match "abcd.*efg" against "abcdxxxxxx" -- no efg, expect 0."""
    return _suffix_is_match(u64(10), "abcdxxxxxx")

@compile
def native_suffix_match() -> u8:
    """Match "abcd.*efg" against "abcdxxefg" -- expect 1."""
    return _suffix_is_match(u64(9), "abcdxxefg")

@compile
def native_suffix_search_found() -> i64:
    """Search "abcd.*efg" in "xxabcdxefgx" -- expect 2."""
    return _suffix_search(u64(11), "xxabcdxefgx")

@compile
def native_suffix_search_none() -> i64:
    """Search "abcd.*efg" in "xxabcdxxxxx" -- no efg, expect -1."""
    return _suffix_search(u64(11), "xxabcdxxxxx")


# --- Skip optimization: "needle" (literal, W=6 skip table) ---
_r_needle = regex_compile("needle")
_needle_search = _r_needle.generate_search_fn()

@compile
def native_skip_search_end() -> i64:
    """Search 'needle' at end of 50 x's -- skip should fire."""
    return _needle_search(u64(56), "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxneedle")

@compile
def native_skip_search_start() -> i64:
    """Search 'needle' at position 0."""
    return _needle_search(u64(56), "needlexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

@compile
def native_skip_search_none() -> i64:
    """Search 'needle' in all x's -- no match."""
    return _needle_search(u64(50), "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

@compile
def native_skip_search_mid() -> i64:
    """Search 'needle' in the middle of data."""
    return _needle_search(u64(56), "xxxxxxxxxxxxxxxxxxxxxxxxxneedlexxxxxxxxxxxxxxxxxxxxxxxxx")


class TestRegexCodegen(unittest.TestCase):
    """Test that @compile function generation works and produces correct results."""

    # --- Literal "abc" ---

    def test_is_match_literal_exact(self):
        self.assertEqual(native_abc_match_exact(), 1)

    def test_is_match_literal_embedded(self):
        self.assertEqual(native_abc_match_embedded(), 0)

    def test_is_match_literal_no_match(self):
        self.assertEqual(native_abc_no_match(), 0)

    def test_is_match_literal_partial(self):
        self.assertEqual(native_abc_partial_no_match(), 0)

    def test_search_literal_found(self):
        self.assertEqual(native_abc_search_found(), 2)

    def test_search_literal_not_found(self):
        self.assertEqual(native_abc_search_not_found(), -1)

    def test_search_literal_at_start(self):
        self.assertEqual(native_abc_search_at_start(), 0)

    # --- Quantifier "ab+c" ---

    def test_is_match_plus_abc(self):
        self.assertEqual(native_abpc_match_abc(), 1)

    def test_is_match_plus_abbc(self):
        self.assertEqual(native_abpc_match_abbc(), 1)

    def test_is_match_plus_no_b(self):
        self.assertEqual(native_abpc_no_match_ac(), 0)

    # --- Anchored "^hello$" ---

    def test_is_match_anchored_exact(self):
        self.assertEqual(native_hello_exact(), 1)

    def test_is_match_anchored_too_long(self):
        self.assertEqual(native_hello_extra(), 0)

    def test_is_match_anchored_wrong_start(self):
        self.assertEqual(native_hello_prefix(), 0)

    # --- Char class "^[a-z]+$" ---

    def test_is_match_charclass_lower(self):
        self.assertEqual(native_az_lower(), 1)

    def test_is_match_charclass_upper(self):
        self.assertEqual(native_az_upper(), 0)

    def test_is_match_charclass_digits(self):
        self.assertEqual(native_az_digits(), 0)

    # --- Alternation "cat|dog" ---

    def test_is_match_alt_cat(self):
        self.assertEqual(native_catdog_cat(), 1)

    def test_is_match_alt_dog(self):
        self.assertEqual(native_catdog_dog(), 1)

    def test_is_match_alt_fish(self):
        self.assertEqual(native_catdog_fish(), 0)

    def test_search_alt_embedded(self):
        self.assertEqual(native_catdog_search_embedded(), 3)

    # --- Escape class "\\d+" ---

    def test_is_match_digits_numbers(self):
        self.assertEqual(native_digits_numbers(), 1)

    def test_is_match_digits_letters(self):
        self.assertEqual(native_digits_letters(), 0)

    def test_search_digits_mixed(self):
        self.assertEqual(native_digits_search_mixed(), 3)

    # --- Dot-star "a.*b" ---

    def test_is_match_dotstar_ab(self):
        self.assertEqual(native_dotstar_ab(), 1)

    def test_is_match_dotstar_axxb(self):
        self.assertEqual(native_dotstar_axxb(), 1)

    def test_is_match_dotstar_no_b(self):
        self.assertEqual(native_dotstar_no_b(), 0)

    # --- Start-anchored "^hello" ---

    def test_is_match_start_anchored_yes(self):
        self.assertEqual(native_start_yes(), 1)

    def test_is_match_start_anchored_no(self):
        self.assertEqual(native_start_no(), 0)

    # --- 1-pass search: different-length alternation (Opt #1) ---

    def test_search_diffalt_ab(self):
        self.assertEqual(native_diffalt_search_ab(), 3)

    def test_search_diffalt_cdef(self):
        self.assertEqual(native_diffalt_search_cdef(), 3)

    def test_search_diffalt_none(self):
        self.assertEqual(native_diffalt_search_none(), -1)

    # --- Suffix guard (Opt #3) ---

    def test_suffix_guard_no_match(self):
        self.assertEqual(native_suffix_no_match(), 0)

    def test_suffix_guard_match(self):
        self.assertEqual(native_suffix_match(), 1)

    def test_suffix_guard_search_found(self):
        self.assertEqual(native_suffix_search_found(), 2)

    def test_suffix_guard_search_none(self):
        self.assertEqual(native_suffix_search_none(), -1)

    # --- Skip optimization "needle" ---

    def test_skip_search_end(self):
        self.assertEqual(native_skip_search_end(), 50)

    def test_skip_search_start(self):
        self.assertEqual(native_skip_search_start(), 0)

    def test_skip_search_none(self):
        self.assertEqual(native_skip_search_none(), -1)

    def test_skip_search_mid(self):
        self.assertEqual(native_skip_search_mid(), 25)


if __name__ == "__main__":
    unittest.main()
