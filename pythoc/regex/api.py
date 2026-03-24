"""
Public API for compile-time regex.

Usage:
    from pythoc.regex import compile

    r = compile("ab+c")
    r.match(b"abbc")            # (True, {})
    r.search(b"xxabbcxx")       # (True, {'start': 2, 'end': 6})

    # With tags:
    r = compile("{s}a+{e}")
    r.match(b"aaa")             # (True, {'s': 0, 'e': 3})
    r.search(b"xxaaaxx")        # (True, {'start': 2, 'end': 5, 's': 2, 'e': 5})
"""

from __future__ import annotations

from .codegen import CompiledRegex


def compile(pattern: str, mode: str = "both") -> CompiledRegex:
    """Compile a regex pattern at compile time.

    Args:
        pattern: The regex pattern string (must be a compile-time constant).
        mode: Which execution modes to compile.
            ``"match"``  — only ``match()`` available (anchored)
            ``"search"`` — only ``search()`` available (unanchored)
            ``"both"``   — (default) all methods available

    Both ``match`` and ``search`` return ``(bool, dict)``.  The bool
    indicates whether a match was found, and the dict maps tag names
    to byte positions.  ``search`` additionally includes ``'start'``
    and ``'end'`` keys in the result dict.

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
