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


def compile(pattern: str, mode: str = "both") -> CompiledRegex:
    """Compile a regex pattern at compile time.

    Args:
        pattern: The regex pattern string (must be a compile-time constant).
        mode: Which execution modes to compile.
            ``"match"``  — only ``is_match`` / ``fullmatch``
            ``"search"`` — only ``search`` / ``find_span`` / ``find_with_tags``
            ``"both"``   — (default) all methods available

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
    return CompiledRegex(pattern, mode=mode)
