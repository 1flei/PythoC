"""
pythoc.regex — Compile-time regex for PythoC.

This package provides a compile-time regex system that:
  1. Parses regex patterns at compile time (pure Python)
  2. Builds NFA/DFA data structures
  3. Generates @compile functions with the universal ABI
  4. Provides match() and search() for native execution

Usage:
    from pythoc.regex import compile

    r = compile("[a-z]+@[a-z]+\\.[a-z]+")
    r.match(b"foo@bar.com")             # (True, {})
    r.search(b"email: a@b.c")           # (True, {'start': 7, 'end': 12})
"""

from .api import compile
from .codegen import CompiledRegex
from .parse import ParseError

__all__ = ['compile', 'CompiledRegex', 'ParseError']
