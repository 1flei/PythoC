"""
pythoc.regex — Compile-time regex for PythoC.

This package provides a compile-time regex system that:
  1. Parses regex patterns at compile time (pure Python)
  2. Builds NFA/DFA data structures
  3. Can generate @compile functions for native PythoC execution
  4. Also provides Python-level match/search for testing

Usage:
    from pythoc.regex import compile

    r = compile("[a-z]+@[a-z]+\\.[a-z]+")
    r.is_match(b"foo@bar.com")   # True
    r.search(b"email: a@b.c")   # 7
"""

from .api import compile
from .codegen import CompiledRegex
from .parse import ParseError

__all__ = ['compile', 'CompiledRegex', 'ParseError']
