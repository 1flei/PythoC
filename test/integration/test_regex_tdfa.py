#!/usr/bin/env python3
"""
Python-level TNFA/TDFA consistency tests for pythoc.regex.

These tests pin the TDFA frontend to the prioritized TNFA reference
runner. They run purely in Python (no ``@compile`` / native code) and
serve as the oracle the native pipeline (T-BMA + codegen) is
checked against elsewhere.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.regex import (
    build_tnfa,
    compile_search_tdfa,
    compile_tdfa,
    format_search_result,
    run_tdfa,
    run_tnfa,
)
from pythoc.regex.codegen import rewrite_for_search
from pythoc.regex.parse import parse


def _build_match_at_middle(size, needle):
    mid = size // 2
    return b'x' * mid + needle + b'x' * (size - mid - len(needle))


class TestRegexTDFAConsistency(unittest.TestCase):

    def assert_match_consistent(self, pattern, data, expected=None):
        re_ast = parse(pattern)
        tnfa = build_tnfa(re_ast)
        tdfa = compile_tdfa(pattern)

        res_tnfa = run_tnfa(tnfa, data)
        res_tdfa = run_tdfa(tdfa, data)

        self.assertEqual(res_tnfa, res_tdfa)
        if expected is not None:
            self.assertEqual(res_tdfa, expected)

    def assert_search_consistent(self, pattern, data, expected=None):
        re_ast = parse(pattern)
        search_tnfa = build_tnfa(rewrite_for_search(re_ast))
        search_tdfa = compile_search_tdfa(pattern)

        ok_tnfa, tags_tnfa = run_tnfa(search_tnfa, data, include_internal=True)
        ok_tdfa, tags_tdfa = run_tdfa(search_tdfa, data, include_internal=True)

        self.assertEqual((ok_tnfa, tags_tnfa), (ok_tdfa, tags_tdfa))
        if expected is not None:
            self.assertEqual((ok_tdfa, format_search_result(tags_tdfa)), (True, expected))

    def test_match_basic_and_anchored_patterns(self):
        self.assert_match_consistent("abc", b"abcx", (True, {}))
        self.assert_match_consistent("^abc$", b"abc", (True, {}))
        self.assert_match_consistent("^abc$", b"abcx", (False, {}))
        self.assert_match_consistent("abc$", b"zabc", (False, {}))
        self.assert_match_consistent("a.*b", b"axxb", (True, {}))

    def test_match_tags_and_loop_positions(self):
        self.assert_match_consistent("{x}a|a{y}", b"a", (True, {"x": 0, "y": 1}))
        self.assert_match_consistent("a|{x}a{y}", b"a", (True, {"x": 0, "y": 1}))
        self.assert_match_consistent("({x}a|a)b{z}", b"ab", (True, {"x": 0, "z": 2}))
        self.assert_match_consistent("(ab)*{mid}cde", b"ababcde", (True, {"mid": 4}))
        self.assert_match_consistent(
            ".*{t1}a{t2}.*{t3}b{t4}",
            b"aaab",
            (True, {"t1": 0, "t2": 1, "t3": 3, "t4": 4}),
        )
        self.assert_match_consistent(
            ".*{t1}a{t2}.*{t3}b{t4}",
            b"baab",
            (True, {"t1": 1, "t2": 2, "t3": 3, "t4": 4}),
        )
        self.assert_match_consistent(
            ".*{t1}a{t2}.*{t3}b{t4}",
            b"aaaabbbb",
            (True, {"t1": 0, "t2": 1, "t3": 4, "t4": 5}),
        )
        self.assert_match_consistent("^{beg}a{end}$", b"a", (True, {"beg": 0, "end": 1}))
        self.assert_match_consistent("a{end}$", b"a", (True, {"end": 1}))

    def test_search_tag_regressions(self):
        self.assert_search_consistent(
            "a*{mid}b",
            b"aaab",
            {"start": 0, "end": 4, "mid": 3},
        )
        self.assert_search_consistent(
            "{x}a|a{y}",
            b"a",
            {"start": 0, "end": 1, "x": 0, "y": 1},
        )
        self.assert_search_consistent(
            "a|{x}a{y}",
            b"a",
            {"start": 0, "end": 1, "x": 0, "y": 1},
        )

    def test_search_internal_tags_and_empty_match(self):
        ok, tags = run_tdfa(compile_search_tdfa("a*"), b"bbb", include_internal=True)
        self.assertTrue(ok)
        self.assertEqual(format_search_result(tags), {"start": 0, "end": 0})

        ok, tags = run_tdfa(compile_search_tdfa("^abc$"), b"abc", include_internal=True)
        self.assertTrue(ok)
        self.assertEqual(format_search_result(tags), {"start": 0, "end": 3})

    def test_search_middle_alternation_matches_reference(self):
        data = _build_match_at_middle(50_000, b"gamma")
        self.assert_search_consistent("alpha|beta|gamma", data)

        ok, tags = run_tdfa(compile_search_tdfa("alpha|beta|gamma"), data, include_internal=True)
        self.assertTrue(ok)
        self.assertEqual(format_search_result(tags)["start"], 25_000)

    def test_search_email_middle_matches_reference(self):
        data = _build_match_at_middle(50_000, b"user@host.com")
        self.assert_search_consistent("[a-z]+@[a-z]+\\.[a-z]+", data)

        ok, tags = run_tdfa(
            compile_search_tdfa("[a-z]+@[a-z]+\\.[a-z]+"),
            data,
            include_internal=True,
        )
        self.assertTrue(ok)
        self.assertEqual(format_search_result(tags)["start"], 0)
