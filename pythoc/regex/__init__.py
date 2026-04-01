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
from .codegen import (
    CompiledRegex,
    compile_search_tdfa,
    compile_tdfa,
)
from .parse import ParseError
from .tnfa import TNFA, build_tnfa
from .tdfa import TDFA, format_search_result, run_tdfa, run_tnfa

__all__ = [
    'compile',
    'CompiledRegex',
    'ParseError',
    'TNFA',
    'TDFA',
    'build_tnfa',
    'compile_tdfa',
    'compile_search_tdfa',
    'run_tdfa',
    'run_tnfa',
    'format_search_result',
]
