"""
Public API for compile-time regex.

Usage:
    from pythoc.regex import compile

    r = compile("ab+c")
    r.is_match(b"abbc")        # True
    r.search(b"xxabbcxx")      # 2
    r.find_span(b"xxabbcxx")   # (2, 6)
    r.fullmatch(b"abbc")       # True

For @compile function generation:
    is_match_fn = r.generate_is_match_fn()
    search_fn = r.generate_search_fn()
"""

from __future__ import annotations

from .codegen import CompiledRegex


def compile(pattern: str) -> CompiledRegex:
    """Compile a regex pattern at compile time.

    The pattern must be a compile-time constant string.
    Returns a CompiledRegex object with match/search methods.

    Supported syntax:
      - Literal characters: a, b, 0
      - Concatenation: abc
      - Alternation: a|b
      - Quantifiers: ? (0-1), * (0+), + (1+), all lazy by default
      - Character classes: [abc], [a-z], [^abc]
      - Dot: . (any byte)
      - Grouping: (...) for precedence
      - Anchors: ^, $
      - Zero-width tags: {name}
      - Escape sequences: \\d, \\w, \\s, \\., \\\\, etc.
    """
    return CompiledRegex(pattern)
