#!/usr/bin/env python3
"""
Unit tests for regex internals: parser, NFA, DFA.

These are pure Python tests — no @compile needed.
"""

import ctypes
import unittest
import sys
import os
import warnings
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.regex.parse import (
    parse, ParseError,
    Literal, Dot, CharClass, Concat, Alternate, Repeat, Group, Anchor, Tag,
)
from pythoc.regex.nfa import build_nfa, epsilon_closure, NFA
from pythoc.regex.dfa import build_dfa, DFA
from pythoc.regex import opcs
from pythoc.regex.codegen import CompiledRegex, compile_pattern, _pattern_digest

# ---------------------------------------------------------------------------
# Pre-compile all patterns before any native execution starts.
# PythoC requires all @compile functions to be defined before the first
# native call.  Collecting them here ensures deterministic ordering.
# ---------------------------------------------------------------------------
_ALL_PATTERNS = [
    "abc", "cat|dog", "ab*c", "ab+c", "ab?c", "a.c", "a.*b",
    "[abc]", "[a-z]+", "[^0-9]+", "^hello", "world$", "^hello$",
    "\\d+", "\\w+", "\\s+", "(ab)+", "[a-zA-Z_][a-zA-Z0-9_]*",
    "[a-z]+@[a-z]+\\.[a-z]+", "foo", "a*", "cat|dog|bird",
    "((a|b)c)+", "a\\.b", "^a|bc", "ab|c$", "^ab$", "(^a|b)c",
    "c$", "hello", "a{mid}b", "{beg}hello{end_tag}", "^{s}hello",
    "a*{mid}b", "{x}a|{y}bc", "{x}a|a{y}", "{s}abc{e}", "abcd", "(ab)*cde",
    ".*{tag}.*b", "a*{tag}a*b", "(ab)*{tag}abc", "(ab)*{mid}cde",
    "needle", "https?://",
]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _p in _ALL_PATTERNS:
        CompiledRegex(_p)
    CompiledRegex("abc", mode="match")
    CompiledRegex("abc", mode="search")
del _p

# ============================================================================
# Parser tests
# ============================================================================

class TestParser(unittest.TestCase):

    def test_single_literal(self):
        node = parse("a")
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.byte, ord('a'))

    def test_concatenation(self):
        node = parse("abc")
        self.assertIsInstance(node, Concat)
        self.assertEqual(len(node.children), 3)
        self.assertEqual(node.children[0].byte, ord('a'))
        self.assertEqual(node.children[1].byte, ord('b'))
        self.assertEqual(node.children[2].byte, ord('c'))

    def test_alternation(self):
        node = parse("a|b")
        self.assertIsInstance(node, Alternate)
        self.assertEqual(len(node.children), 2)

    def test_alternation_three(self):
        node = parse("a|b|c")
        self.assertIsInstance(node, Alternate)
        self.assertEqual(len(node.children), 3)

    def test_quantifier_star(self):
        node = parse("a*")
        self.assertIsInstance(node, Repeat)
        self.assertEqual(node.min_count, 0)
        self.assertIsNone(node.max_count)
        self.assertIsInstance(node.child, Literal)

    def test_quantifier_plus(self):
        node = parse("a+")
        self.assertIsInstance(node, Repeat)
        self.assertEqual(node.min_count, 1)
        self.assertIsNone(node.max_count)

    def test_quantifier_question(self):
        node = parse("a?")
        self.assertIsInstance(node, Repeat)
        self.assertEqual(node.min_count, 0)
        self.assertEqual(node.max_count, 1)

    def test_dot(self):
        node = parse(".")
        self.assertIsInstance(node, Dot)

    def test_group(self):
        node = parse("(ab)")
        self.assertIsInstance(node, Group)
        self.assertIsInstance(node.child, Concat)

    def test_group_with_alternation(self):
        node = parse("(a|b)c")
        self.assertIsInstance(node, Concat)
        self.assertIsInstance(node.children[0], Group)
        self.assertIsInstance(node.children[0].child, Alternate)

    def test_anchor_start(self):
        node = parse("^a")
        self.assertIsInstance(node, Concat)
        self.assertIsInstance(node.children[0], Anchor)
        self.assertEqual(node.children[0].kind, 'start')

    def test_anchor_end(self):
        node = parse("a$")
        self.assertIsInstance(node, Concat)
        self.assertIsInstance(node.children[1], Anchor)
        self.assertEqual(node.children[1].kind, 'end')

    def test_charclass_simple(self):
        node = parse("[abc]")
        self.assertIsInstance(node, CharClass)
        self.assertFalse(node.negated)
        bytes_in_class = set()
        for lo, hi in node.ranges:
            for b in range(lo, hi + 1):
                bytes_in_class.add(b)
        self.assertEqual(bytes_in_class, {ord('a'), ord('b'), ord('c')})

    def test_charclass_range(self):
        node = parse("[a-z]")
        self.assertIsInstance(node, CharClass)
        self.assertEqual(len(node.ranges), 1)
        self.assertEqual(node.ranges[0], (ord('a'), ord('z')))

    def test_charclass_negated(self):
        node = parse("[^abc]")
        self.assertIsInstance(node, CharClass)
        self.assertTrue(node.negated)

    def test_escape_digit(self):
        node = parse("\\d")
        self.assertIsInstance(node, CharClass)
        bytes_in_class = set()
        for lo, hi in node.ranges:
            for b in range(lo, hi + 1):
                bytes_in_class.add(b)
        self.assertEqual(bytes_in_class, set(range(ord('0'), ord('9') + 1)))

    def test_escape_word(self):
        node = parse("\\w")
        self.assertIsInstance(node, CharClass)

    def test_escape_dot(self):
        node = parse("\\.")
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.byte, ord('.'))

    def test_escape_backslash(self):
        node = parse("\\\\")
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.byte, ord('\\'))

    def test_invalid_trailing_backslash(self):
        with self.assertRaises(ParseError):
            parse("\\")

    def test_invalid_unmatched_paren(self):
        with self.assertRaises(ParseError):
            parse("(abc")

    def test_invalid_unexpected_rparen(self):
        with self.assertRaises(ParseError):
            parse(")")

    def test_complex_pattern(self):
        node = parse("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertIsInstance(node, Concat)

    def test_alternation_in_group(self):
        node = parse("(cat|dog)")
        self.assertIsInstance(node, Group)
        self.assertIsInstance(node.child, Alternate)

    # --- Lazy repeat syntax ---

    def test_lazy_star(self):
        with self.assertRaises(ParseError):
            parse("a*?")

    def test_lazy_plus(self):
        with self.assertRaises(ParseError):
            parse("a+?")

    def test_lazy_question(self):
        with self.assertRaises(ParseError):
            parse("a??")

    def test_plain_star_is_lazy_by_default(self):
        node = parse("a*")
        self.assertIsInstance(node, Repeat)
        self.assertTrue(node.lazy)

    # --- Tag syntax ---

    def test_tag_simple(self):
        node = parse("{foo}")
        self.assertIsInstance(node, Tag)
        self.assertEqual(node.name, "foo")

    def test_tag_underscore(self):
        node = parse("{_beg}")
        self.assertIsInstance(node, Tag)
        self.assertEqual(node.name, "_beg")

    def test_tag_in_pattern(self):
        node = parse("a{mid}b")
        self.assertIsInstance(node, Concat)
        self.assertIsInstance(node.children[1], Tag)
        self.assertEqual(node.children[1].name, "mid")

    def test_brace_as_literal(self):
        """{ not followed by alpha/underscore is a literal."""
        node = parse("{")
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.byte, ord('{'))

    def test_numeric_brace_repeat_rejected(self):
        with self.assertRaises(ParseError):
            parse("a{2}")

    def test_duplicate_tags_rejected(self):
        with self.assertRaises(ParseError):
            parse("{x}a{x}")

    def test_reserved_internal_tag_prefix_rejected(self):
        with self.assertRaises(ParseError):
            parse("{__pythoc_internal_beg}")


# ============================================================================
# NFA tests
# ============================================================================

class TestNFA(unittest.TestCase):

    def test_literal_nfa(self):
        ast = parse("a")
        nfa = build_nfa(ast)
        self.assertEqual(nfa.state_count(), 2)
        start = nfa.states[nfa.start]
        self.assertIn(ord('a'), start.byte_transitions)

    def test_concat_nfa(self):
        ast = parse("ab")
        nfa = build_nfa(ast)
        self.assertGreaterEqual(nfa.state_count(), 4)

    def test_alternation_nfa(self):
        ast = parse("a|b")
        nfa = build_nfa(ast)
        self.assertGreaterEqual(nfa.state_count(), 6)

    def test_star_nfa(self):
        ast = parse("a*")
        nfa = build_nfa(ast)
        self.assertGreaterEqual(nfa.state_count(), 4)

    def test_epsilon_closure(self):
        ast = parse("a*")
        nfa = build_nfa(ast)
        closure = epsilon_closure(nfa, {nfa.start})
        self.assertIn(nfa.accept, closure)

    def test_dot_nfa(self):
        ast = parse(".")
        nfa = build_nfa(ast)
        start = nfa.states[nfa.start]
        self.assertEqual(len(start.byte_transitions), 256)


# ============================================================================
# DFA tests
# ============================================================================

class TestDFA(unittest.TestCase):

    def test_literal_dfa(self):
        ast = parse("a")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertGreaterEqual(dfa.num_states, 2)
        self.assertIn(dfa.start_state, range(dfa.num_states))
        self.assertTrue(len(dfa.accept_states) > 0)

    def test_dfa_has_class_map(self):
        ast = parse("[a-z]")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertEqual(len(dfa.class_map), 256)
        cls_a = dfa.class_map[ord('a')]
        for c in range(ord('a'), ord('z') + 1):
            self.assertEqual(dfa.class_map[c], cls_a)

    def test_anchor_start(self):
        ast = parse("^abc")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertNotEqual(dfa.sentinel_256_class, -1)

    def test_anchor_end(self):
        ast = parse("abc$")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertNotEqual(dfa.sentinel_257_class, -1)

    def test_dfa_dead_state(self):
        ast = parse("a")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertNotEqual(dfa.dead_state, -1)


# ============================================================================
# CompiledRegex match tests
# ============================================================================

class TestCompiledRegexMatch(unittest.TestCase):

    def _match(self, cr, data):
        ok, _ = cr.match(data)
        return ok

    def test_literal_match(self):
        r = CompiledRegex("abc")
        self.assertTrue(self._match(r, b"abc"))
        self.assertTrue(self._match(r, b"abcy"))
        self.assertFalse(self._match(r, b"xabcy"))
        self.assertFalse(self._match(r, b"ab"))
        self.assertFalse(self._match(r, b"xyz"))

    def test_alternation(self):
        r = CompiledRegex("cat|dog")
        self.assertTrue(self._match(r, b"cat"))
        self.assertTrue(self._match(r, b"dog"))
        self.assertFalse(self._match(r, b"my cat"))
        self.assertFalse(self._match(r, b"car"))

    def test_quantifier_star(self):
        r = CompiledRegex("ab*c")
        self.assertTrue(self._match(r, b"ac"))
        self.assertTrue(self._match(r, b"abc"))
        self.assertTrue(self._match(r, b"abbc"))
        self.assertTrue(self._match(r, b"abbbc"))
        self.assertFalse(self._match(r, b"adc"))

    def test_quantifier_plus(self):
        r = CompiledRegex("ab+c")
        self.assertFalse(self._match(r, b"ac"))
        self.assertTrue(self._match(r, b"abc"))
        self.assertTrue(self._match(r, b"abbc"))

    def test_quantifier_question(self):
        r = CompiledRegex("ab?c")
        self.assertTrue(self._match(r, b"ac"))
        self.assertTrue(self._match(r, b"abc"))
        self.assertFalse(self._match(r, b"abbc"))

    def test_dot(self):
        r = CompiledRegex("a.c")
        self.assertTrue(self._match(r, b"abc"))
        self.assertTrue(self._match(r, b"axc"))
        self.assertFalse(self._match(r, b"ac"))

    def test_dot_star(self):
        r = CompiledRegex("a.*b")
        self.assertTrue(self._match(r, b"ab"))
        self.assertTrue(self._match(r, b"axxb"))
        self.assertFalse(self._match(r, b"a"))

    def test_charclass(self):
        r = CompiledRegex("[abc]")
        self.assertTrue(self._match(r, b"a"))
        self.assertTrue(self._match(r, b"b"))
        self.assertTrue(self._match(r, b"c"))
        self.assertFalse(self._match(r, b"d"))

    def test_charclass_range(self):
        r = CompiledRegex("[a-z]+")
        self.assertTrue(self._match(r, b"hello"))
        self.assertFalse(self._match(r, b"123"))

    def test_charclass_negated(self):
        r = CompiledRegex("[^0-9]+")
        self.assertTrue(self._match(r, b"hello"))
        self.assertFalse(self._match(r, b"123"))

    def test_anchor_start(self):
        r = CompiledRegex("^hello")
        self.assertTrue(self._match(r, b"hello world"))
        self.assertFalse(self._match(r, b"say hello"))

    def test_anchor_end(self):
        r = CompiledRegex("world$")
        self.assertTrue(self._match(r, b"world"))
        self.assertFalse(self._match(r, b"hello world"))
        self.assertFalse(self._match(r, b"world cup"))

    def test_anchor_both(self):
        r = CompiledRegex("^hello$")
        self.assertTrue(self._match(r, b"hello"))
        self.assertFalse(self._match(r, b"hello world"))
        self.assertFalse(self._match(r, b"say hello"))

    def test_escape_digit(self):
        r = CompiledRegex("\\d+")
        self.assertTrue(self._match(r, b"123"))
        self.assertFalse(self._match(r, b"abc"))

    def test_escape_word(self):
        r = CompiledRegex("\\w+")
        self.assertTrue(self._match(r, b"hello_123"))
        self.assertFalse(self._match(r, b"   "))

    def test_escape_space(self):
        r = CompiledRegex("\\s+")
        self.assertTrue(self._match(r, b"  \t\n"))
        self.assertFalse(self._match(r, b"abc"))

    def test_group_precedence(self):
        r = CompiledRegex("(ab)+")
        self.assertTrue(self._match(r, b"ab"))
        self.assertTrue(self._match(r, b"abab"))
        self.assertFalse(self._match(r, b"aa"))

    def test_complex_identifier(self):
        r = CompiledRegex("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertTrue(self._match(r, b"hello"))
        self.assertTrue(self._match(r, b"_var"))
        self.assertTrue(self._match(r, b"x123"))
        self.assertFalse(self._match(r, b"123"))

    def test_email_like(self):
        r = CompiledRegex("[a-z]+@[a-z]+\\.[a-z]+")
        self.assertTrue(self._match(r, b"foo@bar.com"))
        self.assertFalse(self._match(r, b"foobar"))

    def test_search_found(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"xxfooxx")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)

    def test_search_not_found(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"xxbarxx")
        self.assertFalse(ok)
        self.assertEqual(info, {})

    def test_search_at_start(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"foobar")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)

    def test_search_at_end(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"barfoo")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 3)

    def test_search_span(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"xxfooxx")
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 5))

    def test_search_not_found_returns_empty(self):
        r = CompiledRegex("foo")
        ok, info = r.search(b"xxbarxx")
        self.assertFalse(ok)
        self.assertEqual(info, {})

    def test_empty_string_match(self):
        r = CompiledRegex("a*")
        self.assertTrue(self._match(r, b""))
        self.assertTrue(self._match(r, b"aaa"))

    def test_complex_alternation(self):
        r = CompiledRegex("cat|dog|bird")
        self.assertTrue(self._match(r, b"cat"))
        self.assertTrue(self._match(r, b"dog"))
        self.assertTrue(self._match(r, b"bird"))
        self.assertFalse(self._match(r, b"fish"))

    def test_nested_groups(self):
        r = CompiledRegex("((a|b)c)+")
        self.assertTrue(self._match(r, b"ac"))
        self.assertTrue(self._match(r, b"bc"))
        self.assertTrue(self._match(r, b"acbc"))
        self.assertFalse(self._match(r, b"ab"))

    def test_escaped_dot(self):
        r = CompiledRegex("a\\.b")
        self.assertTrue(self._match(r, b"a.b"))
        self.assertFalse(self._match(r, b"axb"))

    def test_match_returns_tuple(self):
        r = CompiledRegex("abc")
        result = r.match(b"abc")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], True)
        self.assertIsInstance(result[1], dict)

    def test_match_no_match_returns_false_empty(self):
        r = CompiledRegex("abc")
        ok, tags = r.match(b"xyz")
        self.assertFalse(ok)
        self.assertEqual(tags, {})

    def test_search_returns_tuple(self):
        r = CompiledRegex("abc")
        result = r.search(b"xxabcxx")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], True)
        self.assertIn('start', result[1])
        self.assertIn('end', result[1])


# ============================================================================
# Public API tests
# ============================================================================

class TestPublicAPI(unittest.TestCase):

    def test_compile_returns_compiled_regex(self):
        from pythoc.regex import compile
        r = compile("abc")
        self.assertIsInstance(r, CompiledRegex)

    def test_compile_has_methods(self):
        from pythoc.regex import compile
        r = compile("abc")
        self.assertTrue(hasattr(r, 'match'))
        self.assertTrue(hasattr(r, 'search'))

    def test_parse_error_on_invalid(self):
        from pythoc.regex import compile, ParseError
        with self.assertRaises(ParseError):
            compile("(abc")


# ============================================================================
# Per-branch anchor tests
# ============================================================================

class TestPerBranchAnchors(unittest.TestCase):

    def _match(self, cr, data):
        ok, _ = cr.match(data)
        return ok

    def test_caret_a_or_bc_match(self):
        r = CompiledRegex("^a|bc")
        self.assertTrue(self._match(r, b"a"))
        self.assertTrue(self._match(r, b"bc"))
        self.assertFalse(self._match(r, b"xa"))
        self.assertFalse(self._match(r, b"xbc"))

    def test_caret_a_or_bc_search(self):
        r = CompiledRegex("^a|bc")
        ok, info = r.search(b"a")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)
        ok, info = r.search(b"xxbc")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        ok, info = r.search(b"xxa")
        self.assertFalse(ok)

    def test_ab_or_c_dollar_match(self):
        r = CompiledRegex("ab|c$")
        self.assertTrue(self._match(r, b"ab"))
        self.assertTrue(self._match(r, b"c"))
        self.assertFalse(self._match(r, b"cx"))

    def test_caret_ab_dollar(self):
        r = CompiledRegex("^ab$")
        self.assertTrue(self._match(r, b"ab"))
        self.assertFalse(self._match(r, b"abc"))
        ok, info = r.search(b"ab")
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)
        ok, _ = r.search(b"xab")
        self.assertFalse(ok)

    def test_group_caret_a_or_b_c(self):
        r = CompiledRegex("(^a|b)c")
        self.assertTrue(self._match(r, b"ac"))
        self.assertTrue(self._match(r, b"bc"))

    def test_fullmatch_via_anchors(self):
        r = CompiledRegex("^ab$")
        self.assertTrue(self._match(r, b"ab"))
        self.assertFalse(self._match(r, b"abc"))
        self.assertFalse(self._match(r, b"a"))

    def test_no_anchor_unchanged(self):
        r = CompiledRegex("abc")
        self.assertTrue(self._match(r, b"abc"))
        self.assertTrue(self._match(r, b"abcx"))
        self.assertFalse(self._match(r, b"xabc"))

    def test_dollar_only(self):
        r = CompiledRegex("c$")
        self.assertTrue(self._match(r, b"c"))
        self.assertFalse(self._match(r, b"cx"))

    def test_search_span_mixed_anchors(self):
        r = CompiledRegex("^a|bc")
        ok, info = r.search(b"a")
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (0, 1))
        ok, info = r.search(b"xxbc")
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 4))
        ok, _ = r.search(b"xx")
        self.assertFalse(ok)


# ============================================================================
# Tag propagation tests
# ============================================================================

class TestTagPropagation(unittest.TestCase):

    def test_tag_in_nfa(self):
        ast_node = parse("{foo}a")
        nfa = build_nfa(ast_node)
        tag_states = [s for s in nfa.states if s.tag == 'foo']
        self.assertEqual(len(tag_states), 1)

    def test_tag_in_dfa(self):
        from pythoc.regex.dfa import build_dfa
        ast_node = parse("{foo}a")
        nfa = build_nfa(ast_node)
        dfa = build_dfa(nfa)
        all_tags = set()
        for tags in dfa.tag_map.values():
            all_tags.update(tags)
        self.assertIn('foo', all_tags)

    def test_search_dfa_has_beg_end_tags(self):
        from pythoc.regex.codegen import compile_search_dfa_from_ast
        ast_node = parse("abc")
        search_dfa = compile_search_dfa_from_ast(ast_node)
        all_tags = set()
        for tags in search_dfa.tag_map.values():
            all_tags.update(tags)
        self.assertIn('__pythoc_internal_beg', all_tags)
        self.assertIn('__pythoc_internal_end', all_tags)


# ============================================================================
# Greedy/lazy repeat semantics tests
# ============================================================================

class TestGreedyLazy(unittest.TestCase):

    def test_lazy_dotstar(self):
        cr = CompiledRegex('a.*b')
        ok, info = cr.search(b'axbxb')
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (0, 3))

    def test_lazy_dotstar_single(self):
        cr = CompiledRegex('a.*b')
        ok, info = cr.search(b'ab')
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (0, 2))

    def test_lazy_dotstar_search(self):
        cr = CompiledRegex('a.*b')
        ok, info = cr.search(b'xxaxbxbxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)

    def test_lazy_dotstar_span_embedded(self):
        cr = CompiledRegex('a.*b')
        ok, info = cr.search(b'xxaxbxbxx')
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 5))

    def test_no_repeat_unaffected(self):
        cr = CompiledRegex('abc')
        ok, info = cr.search(b'xxabcxx')
        self.assertTrue(ok)
        self.assertEqual((info['start'], info['end']), (2, 5))


# ============================================================================
# Tag results tests
# ============================================================================

class TestTagResults(unittest.TestCase):

    def test_basic_no_tags(self):
        cr = CompiledRegex('hello')
        ok, info = cr.search(b'xxhelloxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        self.assertEqual(info['end'], 7)

    def test_mid_tag(self):
        cr = CompiledRegex('a{mid}b')
        ok, info = cr.search(b'xxabxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        self.assertEqual(info['end'], 4)
        self.assertEqual(info['mid'], 3)

    def test_start_end_tags(self):
        cr = CompiledRegex('{beg}hello{end_tag}')
        ok, info = cr.search(b'xxhelloxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        self.assertEqual(info['end'], 7)
        self.assertEqual(info['beg'], 2)
        self.assertEqual(info['end_tag'], 7)

    def test_no_match_returns_false(self):
        cr = CompiledRegex('hello')
        ok, info = cr.search(b'xxxx')
        self.assertFalse(ok)
        self.assertEqual(info, {})

    def test_anchored_pattern_tags(self):
        cr = CompiledRegex('^{s}hello')
        ok, info = cr.search(b'helloxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)
        self.assertEqual(info['end'], 5)
        self.assertEqual(info['s'], 0)

    def test_anchored_no_match(self):
        cr = CompiledRegex('^hello')
        ok, _ = cr.search(b'xxhello')
        self.assertFalse(ok)

    def test_variable_width_tag(self):
        cr = CompiledRegex('a*{mid}b')
        ok, info = cr.search(b'aaab')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 0)
        self.assertEqual(info['end'], 4)
        self.assertEqual(info['mid'], 3)

    def test_alternation_branch_tags(self):
        cr = CompiledRegex('{x}a|{y}bc')
        ok_a, info_a = cr.search(b'za')
        self.assertTrue(ok_a)
        self.assertEqual(info_a, {'start': 1, 'end': 2, 'x': 1})
        ok_bc, info_bc = cr.search(b'zbc')
        self.assertTrue(ok_bc)
        self.assertEqual(info_bc, {'start': 1, 'end': 3, 'y': 1})

    def test_model_keeps_coexisting_branch_tags(self):
        cr = CompiledRegex('{x}a|a{y}')
        ok, info = cr.search(b'a')
        self.assertTrue(ok)
        self.assertEqual(info, {'start': 0, 'end': 1, 'x': 0, 'y': 1})

    def test_match_with_user_tags(self):
        cr = CompiledRegex('{s}abc{e}')
        ok, tags = cr.match(b'abcdef')
        self.assertTrue(ok)
        self.assertEqual(tags['s'], 0)
        self.assertEqual(tags['e'], 3)

    def test_match_without_tags_empty_dict(self):
        cr = CompiledRegex('abc')
        ok, tags = cr.match(b'abc')
        self.assertTrue(ok)
        self.assertEqual(tags, {})

    def test_loop_phase_search_uses_leftmost_hidden_beg_write(self):
        cr = CompiledRegex('(ab)*{mid}cde')
        ok, info = cr.search(b'zzababcdezz')
        self.assertTrue(ok)
        self.assertEqual(info, {'start': 2, 'end': 9, 'mid': 6})


class TestTagWarnings(unittest.TestCase):

    def test_warns_for_sliding_search_start(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CompiledRegex('.*b', mode='search')
        self.assertTrue(any(
            'search start boundary' in str(w.message)
            for w in caught
        ))

    def test_warns_for_sliding_user_tag(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CompiledRegex('.*{tag}.*b', mode='match')
        self.assertTrue(any(
            'tag {tag}' in str(w.message)
            for w in caught
        ))

    def test_fixed_suffix_tag_pattern_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CompiledRegex('(ab)*{tag}abc', mode='match')
        self.assertEqual(caught, [])


class TestOPCSBMAArtifacts(unittest.TestCase):

    def test_normalize_opcs_commands_collapses_equivalent_copy_chains(self):
        commands = (
            opcs.OPCSCommand(kind="copy", lhs=2, rhs=1),
            opcs.OPCSCommand(kind="copy", lhs=3, rhs=2),
        )
        self.assertEqual(
            opcs._normalize_opcs_commands(commands),
            (
                opcs.OPCSCommand(kind="copy", lhs=2, rhs=1),
                opcs.OPCSCommand(kind="copy", lhs=3, rhs=1),
            ),
        )

    def test_literal_match_bma_has_root_gadget(self):
        cr = CompiledRegex('abcd')
        bma = cr._match_bma
        root_control = bma.effective_start_control
        self.assertEqual(len(bma.shadow_control_states), len(bma.runtime_control_states))
        self.assertIn(root_control, bma.control_entry_states)
        self.assertTrue(any(state.kind == 'gadget' for state in bma.states))
        ok, info = cr.search(b'zzabcdzz')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        self.assertEqual((info['start'], info['end']), (2, 6))

    def test_literal_match_bma_uses_interior_interval_summaries(self):
        cr = CompiledRegex('abcd')
        bma = cr._match_bma
        self.assertTrue(any(
            state.kind == 'gadget' and
            0 < sum(byte_val >= 0 for byte_val in state.known_bytes) < state.block_len
            for state in bma.states
        ))

    def test_loop_match_bma_has_block_local_gadgets(self):
        cr = CompiledRegex('(ab)*cde')
        bma = cr._match_bma
        loop_gadgets = [
            state for state in bma.states if state.kind == 'gadget'
        ]
        self.assertTrue(loop_gadgets)
        self.assertTrue(any(state.block_len >= 2 for state in loop_gadgets))
        ok1, _ = cr.match(b'ababcde')
        self.assertTrue(ok1)
        ok2, _ = cr.match(b'ababcde')
        self.assertTrue(ok2)

    def test_loop_match_bma_has_overlap_edges(self):
        cr = CompiledRegex('(ab)*cde')
        bma = cr._match_bma
        self.assertTrue(any(
            state.kind == 'gadget' and any(
                edge.shift != 1
                for edge in state.edges
            )
            for state in bma.states
        ))

    def test_literal_search_bma_has_root_gadget_with_tags(self):
        cr = CompiledRegex('needle')
        bma = cr._search_bma
        self.assertGreaterEqual(bma.register_count, len(bma.tag_names))
        self.assertTrue(any(
            edge.commands
            for state in bma.states
            for edge in state.edges
            if state.kind != 'control_shadow'
        ) or bma.initial_commands or any(
            state.accept_commands or state.eof_commands
            for state in bma.states
        ))

        ok, info = cr.search(b'xxxxxxxxxxneedle')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 10)
        self.assertEqual(info['end'], 16)

    def test_literal_bma_codegen_all_entrypoints_lower_to_fsm_blocks(self):
        import re as stdlib_re
        CompiledRegex('needle')
        from pythoc.build.output_manager import get_output_manager
        get_output_manager().flush_all()
        digest = _pattern_digest('needle')
        candidate_paths = [
            os.path.abspath(os.path.join(
                os.getcwd(),
                'build/pythoc/regex',
                'codegen.meta.{}.ll'.format(digest),
            )),
            os.path.abspath(os.path.join(
                os.getcwd(),
                'build/external',
                'codegen.meta.{}.ll'.format(digest),
            )),
            os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                '../../build/pythoc/regex',
                'codegen.meta.{}.ll'.format(digest),
            )),
            os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                '../../build/external',
                'codegen.meta.{}.ll'.format(digest),
            )),
        ]
        ir_path = next((path for path in candidate_paths if os.path.exists(path)), candidate_paths[0])
        self.assertTrue(os.path.exists(ir_path))
        with open(ir_path, 'r', encoding='utf-8') as f:
            ir = f.read()
        self.assertRegex(ir, stdlib_re.compile(
            r'define\b.*\bi8 @regex_match_'))
        self.assertRegex(ir, stdlib_re.compile(
            r'define\b.*\bi8 @regex_search_'))
        self.assertNotIn('while_true_body', ir)
        self.assertNotIn('switch i32 %state_addr', ir)

    def test_generate_entrypoints_funnel_through_unified_bma_builder(self):
        cr = CompiledRegex('needle')
        cases = [
            ('_compile_match_fn', cr._match_bma, 'match'),
            ('_compile_search_fn', cr._search_bma, 'search'),
        ]
        for method_name, bma, kind in cases:
            sentinel = object()
            with mock.patch('pythoc.regex.codegen._build_bma_fn',
                            return_value=sentinel) as build_bma_fn:
                result = getattr(cr, method_name)()
                self.assertIs(result, sentinel)
                build_bma_fn.assert_called_once_with(bma, cr._digest, kind)


# ============================================================================
# Native execution API tests
# ============================================================================

class TestNativeAPI(unittest.TestCase):

    def test_match_returns_tuple_bool(self):
        cr = CompiledRegex('abc')
        ok, tags = cr.match(b'abc')
        self.assertIsInstance(ok, bool)
        self.assertTrue(ok)
        self.assertIsInstance(tags, dict)
        ok2, _ = cr.match(b'xyz')
        self.assertFalse(ok2)

    def test_search_returns_tuple(self):
        cr = CompiledRegex('abc')
        ok, info = cr.search(b'xxabcxx')
        self.assertIsInstance(ok, bool)
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        self.assertEqual(info['end'], 5)

    def test_native_fn_cached(self):
        cr = CompiledRegex('abc')
        fn1 = cr._native_match_fn
        fn2 = cr._native_match_fn
        self.assertIs(fn1, fn2)

    def test_compiled_regex_cached(self):
        cr1 = CompiledRegex('abc')
        cr2 = CompiledRegex('abc')
        self.assertIs(cr1, cr2)

    def test_search_uses_native_fn(self):
        cr = CompiledRegex('abc')
        original = cr._native_search_fn
        calls = []
        end_tag = cr._search_bma.tag_names[
            cr._search_bma.tag_slots[
                '__pythoc_internal_end']]
        try:
            def fake_search(n, data_ptr, out):
                calls.append(("search", n))
                setattr(out.contents, end_tag, 5)
                return 1
            cr._native_search_fn = fake_search
            ok, info = cr.search(b'xxabcxx')
            self.assertTrue(ok)
            self.assertEqual(info['end'], 5)
            self.assertEqual(calls, [("search", 7)])
        finally:
            cr._native_search_fn = original

    def test_mode_match_only(self):
        cr = CompiledRegex('abc', mode='match')
        ok, _ = cr.match(b'abc')
        self.assertTrue(ok)
        with self.assertRaises(RuntimeError):
            cr.search(b'abc')

    def test_mode_search_only(self):
        cr = CompiledRegex('abc', mode='search')
        ok, info = cr.search(b'xxabcxx')
        self.assertTrue(ok)
        self.assertEqual(info['start'], 2)
        with self.assertRaises(RuntimeError):
            cr.match(b'abc')


# ============================================================================
# Chain optimization correctness tests
# ============================================================================

class TestChainOptimization(unittest.TestCase):

    def _match(self, cr, data):
        ok, _ = cr.match(data)
        return ok

    def test_optional_in_chain(self):
        cr = CompiledRegex('https?://')
        self.assertTrue(self._match(cr, b'http://'))
        self.assertTrue(self._match(cr, b'https://'))
        self.assertFalse(self._match(cr, b'ftp://'))

    def test_repeat_in_chain(self):
        cr = CompiledRegex('(ab)+')
        self.assertTrue(self._match(cr, b'ab'))
        self.assertTrue(self._match(cr, b'abab'))
        self.assertFalse(self._match(cr, b'a'))

    def test_alternation_with_shared_prefix(self):
        cr = CompiledRegex('^a|bc')
        self.assertTrue(self._match(cr, b'a'))
        self.assertTrue(self._match(cr, b'bc'))
        self.assertFalse(self._match(cr, b'c'))


if __name__ == "__main__":
    unittest.main()
