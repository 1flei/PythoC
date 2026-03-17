#!/usr/bin/env python3
"""
Unit tests for regex internals: parser, NFA, DFA.

These are pure Python tests — no @compile needed.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.regex.parse import (
    parse, ParseError,
    Literal, Dot, CharClass, Concat, Alternate, Repeat, Group, Anchor,
)
from pythoc.regex.nfa import build_nfa, epsilon_closure, NFA
from pythoc.regex.dfa import build_dfa, DFA
from pythoc.regex.codegen import CompiledRegex


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
        # Should have 3 ranges, each single char
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
        # Should not raise
        node = parse("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertIsInstance(node, Concat)

    def test_alternation_in_group(self):
        node = parse("(cat|dog)")
        self.assertIsInstance(node, Group)
        self.assertIsInstance(node.child, Alternate)


# ============================================================================
# NFA tests
# ============================================================================

class TestNFA(unittest.TestCase):

    def test_literal_nfa(self):
        ast = parse("a")
        nfa = build_nfa(ast)
        self.assertEqual(nfa.state_count(), 2)
        # Start state should have transition on 'a'
        start = nfa.states[nfa.start]
        self.assertIn(ord('a'), start.byte_transitions)

    def test_concat_nfa(self):
        ast = parse("ab")
        nfa = build_nfa(ast)
        # 2 states per literal + epsilon connections
        self.assertGreaterEqual(nfa.state_count(), 4)

    def test_alternation_nfa(self):
        ast = parse("a|b")
        nfa = build_nfa(ast)
        # Thompson: 2 for each literal + 2 for split/join = 6
        self.assertGreaterEqual(nfa.state_count(), 6)

    def test_star_nfa(self):
        ast = parse("a*")
        nfa = build_nfa(ast)
        self.assertGreaterEqual(nfa.state_count(), 4)

    def test_epsilon_closure(self):
        ast = parse("a*")
        nfa = build_nfa(ast)
        closure = epsilon_closure(nfa, {nfa.start})
        # Start should reach accept via epsilon (since a* matches empty)
        self.assertIn(nfa.accept, closure)

    def test_dot_nfa(self):
        ast = parse(".")
        nfa = build_nfa(ast)
        start = nfa.states[nfa.start]
        # Should have transitions for all 256 bytes
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
        # All a-z bytes should have the same class
        cls_a = dfa.class_map[ord('a')]
        for c in range(ord('a'), ord('z') + 1):
            self.assertEqual(dfa.class_map[c], cls_a)

    def test_anchor_start(self):
        ast = parse("^abc")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertTrue(dfa.anchored_start)

    def test_anchor_end(self):
        ast = parse("abc$")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertTrue(dfa.anchored_end)

    def test_dfa_dead_state(self):
        ast = parse("a")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        self.assertNotEqual(dfa.dead_state, -1)


# ============================================================================
# CompiledRegex (Python-level matching) tests
# ============================================================================

class TestCompiledRegexMatch(unittest.TestCase):
    """Test the Python-level DFA simulation for correctness."""

    def test_literal_match(self):
        r = CompiledRegex("abc")
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"xabcy"))
        self.assertFalse(r.is_match(b"ab"))
        self.assertFalse(r.is_match(b"xyz"))

    def test_literal_fullmatch(self):
        r = CompiledRegex("abc")
        self.assertTrue(r.fullmatch(b"abc"))
        self.assertFalse(r.fullmatch(b"abcd"))
        self.assertFalse(r.fullmatch(b"xabc"))

    def test_alternation(self):
        r = CompiledRegex("cat|dog")
        self.assertTrue(r.is_match(b"cat"))
        self.assertTrue(r.is_match(b"dog"))
        self.assertTrue(r.is_match(b"my cat"))
        self.assertFalse(r.is_match(b"car"))

    def test_quantifier_star(self):
        r = CompiledRegex("ab*c")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abbc"))
        self.assertTrue(r.is_match(b"abbbc"))
        self.assertFalse(r.is_match(b"adc"))

    def test_quantifier_plus(self):
        r = CompiledRegex("ab+c")
        self.assertFalse(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abbc"))

    def test_quantifier_question(self):
        r = CompiledRegex("ab?c")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"abc"))
        self.assertFalse(r.is_match(b"abbc"))

    def test_dot(self):
        r = CompiledRegex("a.c")
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"axc"))
        self.assertFalse(r.is_match(b"ac"))

    def test_dot_star(self):
        r = CompiledRegex("a.*b")
        self.assertTrue(r.is_match(b"ab"))
        self.assertTrue(r.is_match(b"axxb"))
        self.assertFalse(r.is_match(b"a"))

    def test_charclass(self):
        r = CompiledRegex("[abc]")
        self.assertTrue(r.is_match(b"a"))
        self.assertTrue(r.is_match(b"b"))
        self.assertTrue(r.is_match(b"c"))
        self.assertFalse(r.is_match(b"d"))

    def test_charclass_range(self):
        r = CompiledRegex("[a-z]+")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"123"))

    def test_charclass_negated(self):
        r = CompiledRegex("[^0-9]+")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"123"))

    def test_anchor_start(self):
        r = CompiledRegex("^hello")
        self.assertTrue(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"say hello"))

    def test_anchor_end(self):
        r = CompiledRegex("world$")
        self.assertTrue(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"world cup"))

    def test_anchor_both(self):
        r = CompiledRegex("^hello$")
        self.assertTrue(r.is_match(b"hello"))
        self.assertFalse(r.is_match(b"hello world"))
        self.assertFalse(r.is_match(b"say hello"))

    def test_escape_digit(self):
        r = CompiledRegex("\\d+")
        self.assertTrue(r.is_match(b"123"))
        self.assertFalse(r.is_match(b"abc"))

    def test_escape_word(self):
        r = CompiledRegex("\\w+")
        self.assertTrue(r.is_match(b"hello_123"))
        self.assertFalse(r.is_match(b"   "))

    def test_escape_space(self):
        r = CompiledRegex("\\s+")
        self.assertTrue(r.is_match(b"  \t\n"))
        self.assertFalse(r.is_match(b"abc"))

    def test_group_precedence(self):
        r = CompiledRegex("(ab)+")
        self.assertTrue(r.is_match(b"ab"))
        self.assertTrue(r.is_match(b"abab"))
        self.assertFalse(r.is_match(b"aa"))

    def test_complex_identifier(self):
        r = CompiledRegex("[a-zA-Z_][a-zA-Z0-9_]*")
        self.assertTrue(r.is_match(b"hello"))
        self.assertTrue(r.is_match(b"_var"))
        self.assertTrue(r.is_match(b"x123"))
        self.assertFalse(r.is_match(b"123"))

    def test_email_like(self):
        r = CompiledRegex("[a-z]+@[a-z]+\\.[a-z]+")
        self.assertTrue(r.is_match(b"foo@bar.com"))
        self.assertFalse(r.is_match(b"foobar"))

    def test_search_found(self):
        r = CompiledRegex("foo")
        self.assertEqual(r.search(b"xxfooxx"), 2)

    def test_search_not_found(self):
        r = CompiledRegex("foo")
        self.assertEqual(r.search(b"xxbarxx"), -1)

    def test_search_at_start(self):
        r = CompiledRegex("foo")
        self.assertEqual(r.search(b"foobar"), 0)

    def test_search_at_end(self):
        r = CompiledRegex("foo")
        self.assertEqual(r.search(b"barfoo"), 3)

    def test_find_span(self):
        r = CompiledRegex("foo")
        span = r.find_span(b"xxfooxx")
        self.assertEqual(span, (2, 5))

    def test_find_span_not_found(self):
        r = CompiledRegex("foo")
        self.assertIsNone(r.find_span(b"xxbarxx"))

    def test_empty_string_match(self):
        r = CompiledRegex("a*")
        self.assertTrue(r.is_match(b""))
        self.assertTrue(r.is_match(b"aaa"))

    def test_complex_alternation(self):
        r = CompiledRegex("cat|dog|bird")
        self.assertTrue(r.is_match(b"cat"))
        self.assertTrue(r.is_match(b"dog"))
        self.assertTrue(r.is_match(b"bird"))
        self.assertFalse(r.is_match(b"fish"))

    def test_nested_groups(self):
        r = CompiledRegex("((a|b)c)+")
        self.assertTrue(r.is_match(b"ac"))
        self.assertTrue(r.is_match(b"bc"))
        self.assertTrue(r.is_match(b"acbc"))
        self.assertFalse(r.is_match(b"ab"))

    def test_escaped_dot(self):
        r = CompiledRegex("a\\.b")
        self.assertTrue(r.is_match(b"a.b"))
        self.assertFalse(r.is_match(b"axb"))


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
        self.assertTrue(hasattr(r, 'is_match'))
        self.assertTrue(hasattr(r, 'search'))
        self.assertTrue(hasattr(r, 'fullmatch'))
        self.assertTrue(hasattr(r, 'find_span'))

    def test_parse_error_on_invalid(self):
        from pythoc.regex import compile, ParseError
        with self.assertRaises(ParseError):
            compile("(abc")


if __name__ == "__main__":
    unittest.main()
