"""
Compile-time regex parser.

Parses regex pattern strings into an AST representation.
Supports: literals, concatenation, alternation, quantifiers (?*+),
character classes ([abc], [a-z], [^abc]), dot (.), grouping (()),
anchors (^$), zero-width tags ({name}), escape sequences
(\\d, \\w, \\s, \\\\, \\., etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class Literal:
    """Matches a single literal byte."""
    byte: int  # 0–255

@dataclass
class Dot:
    """Matches any single byte."""
    pass

@dataclass
class CharClass:
    """Matches a byte in (or not in) a set of ranges.

    Each range is (lo, hi) inclusive.  Negated means match bytes NOT in the set.
    """
    ranges: List[Tuple[int, int]]
    negated: bool = False

@dataclass
class Concat:
    """Concatenation of sub-expressions."""
    children: List[object]  # list of AST nodes

@dataclass
class Alternate:
    """Alternation (|) of sub-expressions."""
    children: List[object]

@dataclass
class Repeat:
    """Quantifier applied to a sub-expression.

    min_count / max_count: None means unbounded.
    lazy: if True, prefer shortest match. Public syntax is lazy by default.
    """
    child: object
    min_count: int
    max_count: Optional[int]  # None = unbounded
    lazy: bool = True

@dataclass
class Group:
    """Non-capturing group – used only for precedence."""
    child: object

@dataclass
class Anchor:
    """^ (start) or $ (end) anchor."""
    kind: str  # 'start' or 'end'


@dataclass
class Tag:
    """Zero-width named position marker: {tag_name}."""
    name: str


# ---------------------------------------------------------------------------
# Escape-sequence helpers
# ---------------------------------------------------------------------------

_SHORTHAND_CLASSES = {
    'd': [(ord('0'), ord('9'))],
    'D': [(0, ord('0') - 1), (ord('9') + 1, 255)],
    'w': [(ord('0'), ord('9')), (ord('A'), ord('Z')),
          (ord('a'), ord('z')), (ord('_'), ord('_'))],
    'W': [],  # computed below
    's': [(ord(' '), ord(' ')), (ord('\t'), ord('\t')),
          (ord('\n'), ord('\n')), (ord('\r'), ord('\r')),
          (ord('\f'), ord('\f')), (0x0b, 0x0b)],
    'S': [],  # computed below
}

def _negate_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Compute complement of a set of byte ranges within 0–255."""
    # Merge and sort first
    sorted_ranges = sorted(ranges)
    merged: List[Tuple[int, int]] = []
    for lo, hi in sorted_ranges:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    result: List[Tuple[int, int]] = []
    prev = 0
    for lo, hi in merged:
        if prev < lo:
            result.append((prev, lo - 1))
        prev = hi + 1
    if prev <= 255:
        result.append((prev, 255))
    return result

# Fill in negated shorthand classes
_SHORTHAND_CLASSES['W'] = _negate_ranges(_SHORTHAND_CLASSES['w'])
_SHORTHAND_CLASSES['S'] = _negate_ranges(_SHORTHAND_CLASSES['s'])

_SIMPLE_ESCAPES = {
    'n': ord('\n'),
    'r': ord('\r'),
    't': ord('\t'),
    'f': ord('\f'),
    'v': 0x0b,
    'a': 0x07,
    '0': 0,
}

# Characters that, when escaped, produce a literal of themselves
_LITERAL_ESCAPES = set('\\.|*+?()[]{}^$-/')


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the regex pattern is invalid."""
    pass


class _Parser:
    """Recursive-descent regex parser.

    Grammar (simplified precedence, low→high):
        regex     = alternate
        alternate = concat ('|' concat)*
        concat    = repeat+
        repeat    = atom ('?' | '*' | '+')?
        atom      = literal | '.' | charclass | '(' regex ')' | anchor
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    # -- helpers --

    def _peek(self) -> Optional[str]:
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def _expect(self, ch: str):
        if self.pos >= len(self.pattern) or self.pattern[self.pos] != ch:
            raise ParseError(
                f"Expected '{ch}' at position {self.pos} in /{self.pattern}/")
        self.pos += 1

    def _at_end(self) -> bool:
        return self.pos >= len(self.pattern)

    # -- grammar rules --

    def parse(self) -> object:
        node = self._parse_alternate()
        if not self._at_end():
            raise ParseError(
                f"Unexpected character '{self.pattern[self.pos]}' "
                f"at position {self.pos} in /{self.pattern}/")
        return node

    def _parse_alternate(self) -> object:
        children = [self._parse_concat()]
        while self._peek() == '|':
            self._advance()
            children.append(self._parse_concat())
        if len(children) == 1:
            return children[0]
        return Alternate(children=children)

    def _parse_concat(self) -> object:
        children: List[object] = []
        while True:
            pk = self._peek()
            if pk is None or pk in ('|', ')'):
                break
            children.append(self._parse_repeat())
        if len(children) == 0:
            # Empty alternative – matches empty string
            return Concat(children=[])
        if len(children) == 1:
            return children[0]
        return Concat(children=children)

    def _parse_repeat(self) -> object:
        child = self._parse_atom()
        pk = self._peek()
        if pk == '?':
            self._advance()
            if self._peek() == '?':
                raise ParseError(
                    f"Explicit lazy suffix '??' is not supported at position "
                    f"{self.pos - 1} in /{self.pattern}/; '?' is already lazy by default")
            return Repeat(child=child, min_count=0, max_count=1, lazy=True)
        elif pk == '*':
            self._advance()
            if self._peek() == '?':
                raise ParseError(
                    f"Explicit lazy suffix '*?' is not supported at position "
                    f"{self.pos - 1} in /{self.pattern}/; '*' is already lazy by default")
            return Repeat(child=child, min_count=0, max_count=None, lazy=True)
        elif pk == '+':
            self._advance()
            if self._peek() == '?':
                raise ParseError(
                    f"Explicit lazy suffix '+?' is not supported at position "
                    f"{self.pos - 1} in /{self.pattern}/; '+' is already lazy by default")
            return Repeat(child=child, min_count=1, max_count=None, lazy=True)
        return child

    def _parse_atom(self) -> object:
        pk = self._peek()
        if pk is None:
            raise ParseError(f"Unexpected end of pattern /{self.pattern}/")

        if pk == '(':
            return self._parse_group()
        elif pk == '[':
            return self._parse_charclass()
        elif pk == '.':
            self._advance()
            return Dot()
        elif pk == '^':
            self._advance()
            return Anchor(kind='start')
        elif pk == '$':
            self._advance()
            return Anchor(kind='end')
        elif pk == '{':
            return self._parse_brace()
        elif pk == '\\':
            return self._parse_escape()
        elif pk in ('*', '+', '?', '|', ')'):
            raise ParseError(
                f"Unexpected '{pk}' at position {self.pos} in /{self.pattern}/")
        else:
            self._advance()
            return Literal(byte=ord(pk))

    def _parse_brace(self) -> object:
        """Parse {tag_name} or treat { as literal.

        If the first char after { is alpha or underscore, parse as a tag
        name until }. Numeric brace repeats are rejected explicitly in v1.
        """
        save_pos = self.pos
        self._advance()  # consume '{'
        pk = self._peek()
        if pk is not None and pk.isdigit():
            raise ParseError(
                f"Numeric brace repeats are not supported at position {save_pos} "
                f"in /{self.pattern}/; use {{name}} for zero-width tags")
        if pk is not None and (pk.isalpha() or pk == '_'):
            name_chars = []
            while not self._at_end():
                ch = self._peek()
                if ch == '}':
                    self._advance()  # consume '}'
                    return Tag(name=''.join(name_chars))
                if ch.isalnum() or ch == '_':
                    name_chars.append(self._advance())
                else:
                    break
        # Not a tag: restore position and treat { as literal.
        self.pos = save_pos
        self._advance()
        return Literal(byte=ord('{'))

    def _parse_group(self) -> object:
        self._expect('(')
        child = self._parse_alternate()
        self._expect(')')
        return Group(child=child)

    def _parse_escape(self) -> object:
        """Parse \\X escape sequence."""
        self._expect('\\')
        if self._at_end():
            raise ParseError(
                f"Trailing backslash in /{self.pattern}/")
        ch = self._advance()
        # Shorthand character classes
        if ch in _SHORTHAND_CLASSES:
            return CharClass(ranges=list(_SHORTHAND_CLASSES[ch]))
        # Simple escapes (\n, \t, etc.)
        if ch in _SIMPLE_ESCAPES:
            return Literal(byte=_SIMPLE_ESCAPES[ch])
        # Literal escapes (\., \\, etc.)
        if ch in _LITERAL_ESCAPES:
            return Literal(byte=ord(ch))
        raise ParseError(
            f"Unknown escape sequence '\\{ch}' at position {self.pos - 1} "
            f"in /{self.pattern}/")

    def _parse_charclass(self) -> object:
        """Parse [...] character class."""
        self._expect('[')
        negated = False
        if self._peek() == '^':
            negated = True
            self._advance()

        ranges: List[Tuple[int, int]] = []
        # First character can be ']' as literal
        first = True
        while True:
            pk = self._peek()
            if pk is None:
                raise ParseError(
                    f"Unterminated character class in /{self.pattern}/")
            if pk == ']' and not first:
                self._advance()
                break
            first = False
            lo = self._parse_cc_atom()
            # Check for range
            if self._peek() == '-' and self.pos + 1 < len(self.pattern) \
                    and self.pattern[self.pos + 1] != ']':
                self._advance()  # consume '-'
                hi = self._parse_cc_atom()
                if isinstance(lo, int) and isinstance(hi, int):
                    if lo > hi:
                        raise ParseError(
                            f"Invalid range {chr(lo)}-{chr(hi)} "
                            f"in /{self.pattern}/")
                    ranges.append((lo, hi))
                else:
                    # Either side was a shorthand class – treat '-' as literal
                    if isinstance(lo, list):
                        ranges.extend(lo)
                    else:
                        ranges.append((lo, lo))
                    ranges.append((ord('-'), ord('-')))
                    if isinstance(hi, list):
                        ranges.extend(hi)
                    else:
                        ranges.append((hi, hi))
            else:
                if isinstance(lo, list):
                    ranges.extend(lo)
                else:
                    ranges.append((lo, lo))

        if not ranges:
            raise ParseError(
                f"Empty character class in /{self.pattern}/")
        return CharClass(ranges=ranges, negated=negated)

    def _parse_cc_atom(self):
        """Parse a single element inside a character class.

        Returns int (a single byte value) or list of ranges (for shorthand classes).
        """
        if self._at_end():
            raise ParseError(
                f"Unterminated character class in /{self.pattern}/")
        ch = self._advance()
        if ch == '\\':
            if self._at_end():
                raise ParseError(
                    f"Trailing backslash in character class in /{self.pattern}/")
            esc = self._advance()
            if esc in _SHORTHAND_CLASSES:
                return list(_SHORTHAND_CLASSES[esc])
            if esc in _SIMPLE_ESCAPES:
                return _SIMPLE_ESCAPES[esc]
            if esc in _LITERAL_ESCAPES or esc == ']':
                return ord(esc)
            raise ParseError(
                f"Unknown escape '\\{esc}' in character class "
                f"in /{self.pattern}/")
        return ord(ch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(pattern: str) -> object:
    """Parse a regex pattern string into an AST.

    Raises ParseError on invalid patterns.
    """
    node = _Parser(pattern).parse()
    _validate_tags(node, pattern)
    return node


def _validate_tags(node, pattern: str) -> None:
    """Validate public tag naming rules on the parsed AST."""
    seen = set()
    reserved_prefix = "__pythoc_internal_"

    def walk(cur):
        if isinstance(cur, Tag):
            if cur.name in seen:
                raise ParseError(
                    f"Duplicate tag {{{cur.name}}} in /{pattern}/")
            if cur.name.startswith(reserved_prefix):
                raise ParseError(
                    f"Tag name {{{cur.name}}} uses reserved internal prefix "
                    f"'{reserved_prefix}' in /{pattern}/")
            seen.add(cur.name)
            return
        if isinstance(cur, Group):
            walk(cur.child)
            return
        if isinstance(cur, Repeat):
            walk(cur.child)
            return
        if isinstance(cur, (Concat, Alternate)):
            for child in cur.children:
                walk(child)

    walk(node)
