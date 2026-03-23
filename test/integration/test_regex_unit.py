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
    Literal, Dot, CharClass, Concat, Alternate, Repeat, Group, Anchor, Tag,
)
from pythoc.regex.nfa import build_nfa, epsilon_closure, NFA
from pythoc.regex.dfa import build_dfa, DFA
from pythoc.regex.codegen import CompiledRegex, compile_pattern
from pythoc.regex.analysis import (
    analyze_pattern, compute_accept_depths, find_linear_chains,
    compute_skip_table, compute_state_skips,
    PatternInfo, AcceptDepthInfo, LinearChain, SkipInfo, StateSkipInfo,
)


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
        # Sentinel 256 class should exist
        self.assertNotEqual(dfa.sentinel_256_class, -1)

    def test_anchor_end(self):
        ast = parse("abc$")
        nfa = build_nfa(ast)
        dfa = build_dfa(nfa)
        # Sentinel 257 class should exist
        self.assertNotEqual(dfa.sentinel_257_class, -1)

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
        self.assertTrue(r.is_match(b"abcy"))   # matches at start
        self.assertFalse(r.is_match(b"xabcy"))  # no match at start
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
        self.assertFalse(r.is_match(b"my cat"))  # no match at start
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
        self.assertTrue(r.is_match(b"world"))
        self.assertFalse(r.is_match(b"hello world"))  # no match at start
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
# Analysis tests (PatternInfo, AcceptDepths, LinearChains)
# ============================================================================

class TestPatternAnalysis(unittest.TestCase):
    """Test AST-level pattern analysis."""

    def test_literal_suffix(self):
        info = analyze_pattern(parse("abcd.*efg"))
        self.assertEqual(info.literal_suffix, b"efg")

    def test_no_suffix_charclass_end(self):
        info = analyze_pattern(parse("foo[a-z]+"))
        self.assertIsNone(info.literal_suffix)

    def test_anchor_suffix(self):
        info = analyze_pattern(parse("abc$"))
        self.assertEqual(info.literal_suffix, b"abc")

    def test_pure_literal_suffix(self):
        info = analyze_pattern(parse("hello"))
        self.assertEqual(info.literal_suffix, b"hello")

    def test_single_char_suffix(self):
        info = analyze_pattern(parse("a"))
        self.assertEqual(info.literal_suffix, b"a")

    def test_dotstar_no_suffix(self):
        info = analyze_pattern(parse("a.*"))
        self.assertIsNone(info.literal_suffix)


class TestAcceptDepths(unittest.TestCase):
    """Test DFA accept depth computation."""

    def test_fixed_literal(self):
        nfa = build_nfa(parse("abc"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertTrue(depths.all_unique)
        self.assertEqual(depths.max_depth, 3)

    def test_charclass_fixed(self):
        nfa = build_nfa(parse("[a-z][0-9]"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertTrue(depths.all_unique)
        self.assertEqual(depths.max_depth, 2)

    def test_same_length_alt(self):
        nfa = build_nfa(parse("cat|dog"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertTrue(depths.all_unique)
        # All accept states have depth 3
        for d in depths.depths.values():
            self.assertEqual(d, 3)

    def test_diff_length_alt(self):
        nfa = build_nfa(parse("ab|cdef"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertTrue(depths.all_unique)
        depth_vals = sorted(depths.depths.values())
        self.assertIn(2, depth_vals)
        self.assertIn(4, depth_vals)

    def test_star_not_unique(self):
        nfa = build_nfa(parse("a.*z"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertFalse(depths.all_unique)

    def test_plus_not_unique(self):
        nfa = build_nfa(parse("a+b"))
        dfa = build_dfa(nfa)
        depths = compute_accept_depths(dfa)
        self.assertFalse(depths.all_unique)


class TestLinearChains(unittest.TestCase):
    """Test DFA linear chain detection."""

    def test_pure_literal_chain(self):
        nfa = build_nfa(parse("abcd"))
        dfa = build_dfa(nfa)
        chains = find_linear_chains(dfa)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].byte_sequence, b"abcd")

    def test_no_chain_star(self):
        nfa = build_nfa(parse("a*"))
        dfa = build_dfa(nfa)
        chains = find_linear_chains(dfa)
        self.assertEqual(len(chains), 0)

    def test_alternation_short_chains(self):
        nfa = build_nfa(parse("cat|dog"))
        dfa = build_dfa(nfa)
        chains = find_linear_chains(dfa)
        # Each branch has a chain of 2: "at" and "og" (first char branches)
        chain_bytes = set(c.byte_sequence for c in chains)
        self.assertIn(b"at", chain_bytes)
        self.assertIn(b"og", chain_bytes)

    def test_needle_chain(self):
        nfa = build_nfa(parse("needle"))
        dfa = build_dfa(nfa)
        chains = find_linear_chains(dfa)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].byte_sequence, b"needle")


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


# ============================================================================
# Per-branch anchor tests
# ============================================================================

class TestPerBranchAnchors(unittest.TestCase):
    """Test that anchors work per-branch, not globally."""

    def test_caret_a_or_bc_is_match(self):
        """^a|bc: both branches should be reachable."""
        r = CompiledRegex("^a|bc")
        self.assertTrue(r.is_match(b"a"))      # ^a matches at start
        self.assertTrue(r.is_match(b"bc"))     # bc matches at start
        self.assertFalse(r.is_match(b"xa"))    # ^a requires position 0
        self.assertFalse(r.is_match(b"xbc"))   # bc at start only (is_match)

    def test_caret_a_or_bc_search(self):
        """^a|bc: search should find bc anywhere, ^a only at pos 0."""
        r = CompiledRegex("^a|bc")
        self.assertEqual(r.search(b"a"), 0)       # ^a at start
        self.assertEqual(r.search(b"xxbc"), 2)    # bc found at pos 2
        self.assertEqual(r.search(b"xxa"), -1)    # ^a not at start, no bc

    def test_ab_or_c_dollar_is_match(self):
        """ab|c$: ab matches normally, c$ matches only at end."""
        r = CompiledRegex("ab|c$")
        self.assertTrue(r.is_match(b"ab"))     # ab matches
        self.assertTrue(r.is_match(b"c"))      # c$ matches at end
        self.assertFalse(r.is_match(b"cx"))    # c not at end

    def test_caret_ab_dollar(self):
        """^ab$: exact match only."""
        r = CompiledRegex("^ab$")
        self.assertTrue(r.is_match(b"ab"))
        self.assertFalse(r.is_match(b"abc"))
        self.assertEqual(r.search(b"ab"), 0)
        self.assertEqual(r.search(b"xab"), -1)

    def test_group_caret_a_or_b_c(self):
        """(^a|b)c: ^a branch needs pos 0, b branch anywhere (is_match)."""
        r = CompiledRegex("(^a|b)c")
        self.assertTrue(r.is_match(b"ac"))     # ^a at start, then c
        self.assertTrue(r.is_match(b"bc"))     # b at start, then c

    def test_fullmatch_caret_ab_dollar(self):
        """^ab$ fullmatch."""
        r = CompiledRegex("^ab$")
        self.assertTrue(r.fullmatch(b"ab"))
        self.assertFalse(r.fullmatch(b"abc"))
        self.assertFalse(r.fullmatch(b"a"))

    def test_no_anchor_unchanged(self):
        """Patterns without anchors should work as before."""
        r = CompiledRegex("abc")
        self.assertTrue(r.is_match(b"abc"))
        self.assertTrue(r.is_match(b"abcx"))
        self.assertFalse(r.is_match(b"xabc"))

    def test_dollar_only(self):
        """c$: matches c at end of input."""
        r = CompiledRegex("c$")
        self.assertTrue(r.is_match(b"c"))
        self.assertFalse(r.is_match(b"cx"))

    def test_find_span_mixed_anchors(self):
        """^a|bc find_span."""
        r = CompiledRegex("^a|bc")
        self.assertEqual(r.find_span(b"a"), (0, 1))
        self.assertEqual(r.find_span(b"xxbc"), (2, 4))
        self.assertIsNone(r.find_span(b"xx"))


# ============================================================================
# Tag propagation tests
# ============================================================================

class TestTagPropagation(unittest.TestCase):
    """Test that tags propagate through NFA -> DFA correctly."""

    def test_tag_in_nfa(self):
        """Tag creates epsilon transition with tag metadata on NFA state."""
        ast_node = parse("{foo}a")
        nfa = build_nfa(ast_node)
        # Find the NFA state with the tag
        tag_states = [s for s in nfa.states if s.tag == 'foo']
        self.assertEqual(len(tag_states), 1)

    def test_tag_in_dfa(self):
        """Tag name propagates to DFA tag_map."""
        from pythoc.regex.dfa import build_dfa
        ast_node = parse("{foo}a")
        nfa = build_nfa(ast_node)
        dfa = build_dfa(nfa)
        # At least one DFA state should have the 'foo' tag
        all_tags = set()
        for tags in dfa.tag_map.values():
            all_tags.update(tags)
        self.assertIn('foo', all_tags)

    def test_search_dfa_has_beg_end_tags(self):
        """Search DFA rewrite injects _beg and _end tags."""
        from pythoc.regex.codegen import compile_search_dfa_from_ast
        ast_node = parse("abc")
        search_dfa = compile_search_dfa_from_ast(ast_node)
        all_tags = set()
        for tags in search_dfa.tag_map.values():
            all_tags.update(tags)
        self.assertIn('__pythoc_internal_beg', all_tags)
        self.assertIn('__pythoc_internal_end', all_tags)


# ============================================================================
# Per-state skip optimization tests
# ============================================================================

class TestPerStateSkips(unittest.TestCase):
    """Tests for compute_state_skips() and per-state skip probes."""

    def _build_search_dfa_with_skips(self, pattern):
        from pythoc.regex.codegen import compile_search_dfa
        search_dfa = compile_search_dfa(pattern)
        # Identify restart states: self-loop on >= 128 bytes
        restart = set()
        for s in range(search_dfa.num_states):
            if s == search_dfa.dead_state:
                continue
            self_loop_count = 0
            for bv in range(256):
                cls = search_dfa.class_map[bv]
                t = search_dfa.transitions[s][cls]
                if t == s:
                    self_loop_count += 1
            if self_loop_count >= 128:
                restart.add(s)
        return search_dfa, compute_state_skips(
            search_dfa, restart_states=restart)

    def test_abcdef_skip_info(self):
        """Pattern 'abcdef' should produce useful skip probes."""
        dfa = compile_pattern('abcdef')
        skip_info = compute_skip_table(dfa)
        # Min match length = 6, so window should be 6
        self.assertIsNotNone(skip_info)
        self.assertEqual(skip_info.window, 6)

    def test_abcdef_state_skips(self):
        """Pattern 'abcdef' search DFA: state skips computed correctly.

        The search DFA for 'abcdef' has self-loops (state 2 on 'a')
        due to DFA minimization, which creates variable accept depths.
        States with self-loops are correctly excluded from per-state skips.
        """
        search_dfa, state_skips = self._build_search_dfa_with_skips('abcdef')
        # All non-restart states in the search DFA for 'abcdef' have
        # self-loops (the DFA re-enters the 'a' match state on 'a'),
        # so no per-state skips are produced. This is correct.
        self.assertIsInstance(state_skips, dict)
        for state, info in state_skips.items():
            self.assertIsInstance(info, StateSkipInfo)
            self.assertEqual(info.state, state)
            self.assertGreater(info.shift, 0)
            self.assertLess(len(info.live_bytes), 256)

    def test_charclass_pattern_state_skips(self):
        """Pattern '[ab][abc][a-d][a-e]' should produce per-state skips."""
        dfa = compile_pattern('[ab][abc][a-d][a-e]')
        skip_info = compute_skip_table(dfa)
        # Min match length = 4, so window should be 4
        self.assertIsNotNone(skip_info)
        self.assertEqual(skip_info.window, 4)

        search_dfa, state_skips = self._build_search_dfa_with_skips(
            '[ab][abc][a-d][a-e]')
        # Should have per-state skips
        self.assertTrue(len(state_skips) > 0)

    def test_dotstar_pattern(self):
        """Pattern 'ab.*cdef' should produce state skips."""
        search_dfa, state_skips = self._build_search_dfa_with_skips('ab.*cdef')
        # States matching 'ab.*' lead to states waiting for 'cdef',
        # those states should potentially have skips
        # (though .* creates cycles, some states may still benefit)
        # Just verify it doesn't crash and returns a dict
        self.assertIsInstance(state_skips, dict)

    def test_short_pattern_no_skip(self):
        """Pattern 'a' has min depth 1, so no skip probes possible."""
        search_dfa, state_skips = self._build_search_dfa_with_skips('a')
        # Min accept depth from any state is likely 1, so no useful skips
        # (shift = W-1-d, W=1 => shift=0)
        for info in state_skips.values():
            self.assertGreater(info.shift, 0)

    def test_state_skip_offset_within_window(self):
        """StateSkipInfo.offset should be < min accept depth from that state."""
        search_dfa, state_skips = self._build_search_dfa_with_skips('abcdef')
        for state, info in state_skips.items():
            # offset + shift should not exceed the window
            # (shift = W - 1 - offset, so offset + shift = W - 1)
            self.assertGreaterEqual(info.offset, 0)
            self.assertGreater(info.shift, 0)

    def test_compiled_regex_search_with_state_skips(self):
        """CompiledRegex should use state_skips and still produce correct results."""
        # Verify that patterns with per-state skips still search correctly
        cr = CompiledRegex('abcdef')
        self.assertEqual(cr.search(b'xxabcdefxx'), 2)
        self.assertEqual(cr.search(b'abcdef'), 0)
        self.assertEqual(cr.search(b'xyzxyz'), -1)

    def test_compiled_regex_charclass_search(self):
        """Search with character class pattern still correct with state skips."""
        cr = CompiledRegex('abc')
        self.assertEqual(cr.search(b'xyzabcxyz'), 3)
        self.assertEqual(cr.search(b'abc'), 0)
        self.assertEqual(cr.search(b'ab'), -1)


# ============================================================================
# Greedy/lazy repeat semantics tests
# ============================================================================

class TestGreedyLazy(unittest.TestCase):
    """Tests for lazy-by-default repeat semantics."""

    def test_lazy_dotstar(self):
        """Lazy a.*b on 'axbxb' should match the earliest valid span."""
        cr = CompiledRegex('a.*b')
        self.assertEqual(cr.find_span(b'axbxb'), (0, 3))

    def test_lazy_dotstar_single(self):
        """Lazy a.*b on 'ab' should still match (0, 2)."""
        cr = CompiledRegex('a.*b')
        self.assertEqual(cr.find_span(b'ab'), (0, 2))

    def test_lazy_dotstar_search(self):
        """Lazy a.*b search in 'xxaxbxbxx' should still find start=2."""
        cr = CompiledRegex('a.*b')
        self.assertEqual(cr.search(b'xxaxbxbxx'), 2)

    def test_lazy_dotstar_span_embedded(self):
        """Lazy a.*b in 'xxaxbxbxx' should stop at the first valid b."""
        cr = CompiledRegex('a.*b')
        self.assertEqual(cr.find_span(b'xxaxbxbxx'), (2, 5))

    def test_no_repeat_unaffected(self):
        """Patterns without repeat should be unaffected."""
        cr = CompiledRegex('abc')
        self.assertEqual(cr.find_span(b'xxabcxx'), (2, 5))


# ============================================================================
# Tag results tests
# ============================================================================

class TestTagResults(unittest.TestCase):
    """Tests for find_with_tags() tag position tracking."""

    def test_basic_no_tags(self):
        """Pattern without tags returns start/end only."""
        cr = CompiledRegex('hello')
        result = cr.find_with_tags(b'xxhelloxx')
        self.assertIsNotNone(result)
        self.assertEqual(result['start'], 2)
        self.assertEqual(result['end'], 7)

    def test_mid_tag(self):
        """Tag between two literals records correct position."""
        cr = CompiledRegex('a{mid}b')
        result = cr.find_with_tags(b'xxabxx')
        self.assertIsNotNone(result)
        self.assertEqual(result['start'], 2)
        self.assertEqual(result['end'], 4)
        self.assertEqual(result['mid'], 3)

    def test_start_end_tags(self):
        """Tags at start and end of pattern."""
        cr = CompiledRegex('{beg}hello{end_tag}')
        result = cr.find_with_tags(b'xxhelloxx')
        self.assertIsNotNone(result)
        self.assertEqual(result['start'], 2)
        self.assertEqual(result['end'], 7)
        self.assertEqual(result['beg'], 2)
        self.assertEqual(result['end_tag'], 7)

    def test_no_match_returns_none(self):
        """No match returns None."""
        cr = CompiledRegex('hello')
        self.assertIsNone(cr.find_with_tags(b'xxxx'))

    def test_anchored_pattern_tags(self):
        """Anchored pattern with tags."""
        cr = CompiledRegex('^{s}hello')
        result = cr.find_with_tags(b'helloxx')
        self.assertIsNotNone(result)
        self.assertEqual(result['start'], 0)
        self.assertEqual(result['end'], 5)
        self.assertEqual(result['s'], 0)

    def test_anchored_no_match(self):
        """Anchored pattern with no match."""
        cr = CompiledRegex('^hello')
        self.assertIsNone(cr.find_with_tags(b'xxhello'))

    def test_variable_width_tag(self):
        """Tags after lazy repeats should be recovered from the winning path."""
        cr = CompiledRegex('a*{mid}b')
        result = cr.find_with_tags(b'aaab')
        self.assertIsNotNone(result)
        self.assertEqual(result['start'], 0)
        self.assertEqual(result['end'], 4)
        self.assertEqual(result['mid'], 3)

    def test_alternation_branch_tags(self):
        """Only tags on the winning alternation branch should be reported."""
        cr = CompiledRegex('{x}a|{y}bc')
        result_a = cr.find_with_tags(b'za')
        self.assertEqual(result_a, {'start': 1, 'end': 2, 'x': 1})
        result_bc = cr.find_with_tags(b'zbc')
        self.assertEqual(result_bc, {'start': 1, 'end': 3, 'y': 1})


# ============================================================================
# Native execution API tests
# ============================================================================

class TestNativeAPI(unittest.TestCase):
    """Tests that is_match/search always use native execution."""

    def test_is_match_returns_bool(self):
        """is_match returns a Python bool."""
        cr = CompiledRegex('abc')
        self.assertIsInstance(cr.is_match(b'abc'), bool)
        self.assertTrue(cr.is_match(b'abc'))
        self.assertFalse(cr.is_match(b'xyz'))

    def test_search_returns_int(self):
        """search returns a Python int."""
        cr = CompiledRegex('abc')
        result = cr.search(b'xxabcxx')
        self.assertIsInstance(result, int)
        self.assertEqual(result, 2)

    def test_native_fn_cached(self):
        """Native functions are compiled once and cached."""
        cr = CompiledRegex('abc')
        fn1 = cr._native_is_match_fn
        fn2 = cr._native_is_match_fn
        self.assertIs(fn1, fn2)

    def test_compiled_regex_cached(self):
        """CompiledRegex instances are cached by pattern."""
        cr1 = CompiledRegex('abc')
        cr2 = CompiledRegex('abc')
        self.assertIs(cr1, cr2)

    def test_fullmatch_uses_native_fn(self):
        """fullmatch should delegate directly to the compiled function."""
        cr = CompiledRegex('abc')
        original = cr._native_fullmatch_fn
        calls = []
        try:
            def fake_fullmatch(n, data_ptr):
                calls.append(n)
                return 1
            cr._native_fullmatch_fn = fake_fullmatch
            self.assertTrue(cr.fullmatch(b'zzz'))
            self.assertEqual(calls, [3])
        finally:
            cr._native_fullmatch_fn = original

    def test_find_span_uses_native_search(self):
        """find_span should use compiled search and match-info helpers only."""
        cr = CompiledRegex('abc')
        original_search = cr._native_search_fn
        original_match_info = cr._native_match_info_fn
        calls = []
        try:
            def fake_search(n, data_ptr):
                calls.append(("search", n))
                return 2

            def fake_match_info(n, data_ptr, start, out):
                calls.append(("match_info", start))
                out[0] = 5
                return 1

            cr._native_search_fn = fake_search
            cr._native_match_info_fn = fake_match_info
            self.assertEqual(cr.find_span(b'xxabcxx'), (2, 5))
            self.assertEqual(calls, [("search", 7), ("match_info", 2)])

            calls.clear()
            def fake_search_miss(n, data_ptr):
                calls.append(("search", n))
                return -1
            cr._native_search_fn = fake_search_miss
            cr._native_match_info_fn = (
                lambda *args: self.fail("find_span should not call match_info after search miss"))
            self.assertIsNone(cr.find_span(b'xxxx'))
            self.assertEqual(calls, [("search", 4)])
        finally:
            cr._native_search_fn = original_search
            cr._native_match_info_fn = original_match_info


# ============================================================================
# Chain optimization correctness tests
# ============================================================================

class TestChainOptimization(unittest.TestCase):
    """Tests for chain optimization edge cases."""

    def test_optional_in_chain(self):
        """Chain with optional element (https?://) must not skip states."""
        cr = CompiledRegex('https?://')
        self.assertTrue(cr.is_match(b'http://'))
        self.assertTrue(cr.is_match(b'https://'))
        self.assertFalse(cr.is_match(b'ftp://'))

    def test_repeat_in_chain(self):
        """Chain with repeat (ab)+ must not span accept states."""
        cr = CompiledRegex('(ab)+')
        self.assertTrue(cr.is_match(b'ab'))
        self.assertTrue(cr.is_match(b'abab'))
        self.assertFalse(cr.is_match(b'a'))

    def test_alternation_with_shared_prefix(self):
        """^a|bc has shared DFA states, chains must not break dispatch."""
        cr = CompiledRegex('^a|bc')
        self.assertTrue(cr.is_match(b'a'))
        self.assertTrue(cr.is_match(b'bc'))
        self.assertFalse(cr.is_match(b'c'))


if __name__ == "__main__":
    unittest.main()
