"""
Shared regex pattern fixtures for tests and benchmarks.

This module is a flat catalogue of regex patterns together with
positive / negative sample inputs.  Any test-specific interpretation
(shape expectations, T-BMA artifact checks, benchmark data sizes, ...)
lives in the test that consumes a case, not here.  The goal is to
have one place to grow a stress catalogue of patterns so that tests
stop relying on toy inputs like ``abc`` or ``a.*b``.

Scope
-----

The catalogue only contains regex syntax the pythoc parser currently
accepts:

- literals, concatenation, alternation
- ``.``, ``[abc]``, ``[a-z]``, ``[^...]`` character classes
- ``?``, ``*``, ``+`` quantifiers
- ``^`` / ``$`` anchors
- ``{name}`` zero-width tags (pythoc extension)
- ``\\d`` / ``\\w`` / ``\\s`` shorthand classes and their uppercase
  complements
- escaped metacharacters such as ``\\.``

Anything that requires features pythoc does not support yet (numeric
``{n,m}`` quantifiers, backreferences, lookaround, non-capturing
groups, named groups, inline flags, etc.) is intentionally left out.

Each ``RegexCase`` carries:

- ``name`` -- stable identifier (``lowercase_with_underscores``).
- ``pattern`` -- the regex string.
- ``tier`` -- rough difficulty class; see below.
- ``tag_names`` -- user tags the pattern declares (tuple of strings).
- ``positive`` -- bytestrings that must be found via ``search`` (and
  must match via anchored ``match`` when explicitly marked with
  ``anchored_positive``).
- ``negative`` -- bytestrings that must not be found.
- ``anchored_positive`` -- inputs that the pattern must match at
  position 0.  Kept separate so a plain ``search`` catalogue can still
  include anchored-only inputs without bogus cross-checks.
- ``search_at`` -- optional dict mapping a positive input to the
  expected ``start`` offset from ``search``.  Only filled in when the
  offset is uniquely determined by leftmost semantics.
- ``portable_re`` -- when False the pattern is only meant to be run
  through pythoc (e.g. it uses the ``{tag}`` extension, or it is a
  known catastrophic-backtracking stressor whose Python ``re`` timing
  is not useful as a baseline).  Defaults to True.
- ``notes`` -- free-form comment.

Tiers
-----

* ``"trivial"``     -- single-byte literal classes / tiny alternations.
* ``"basic"``       -- features a typical regex user touches daily:
  literals with quantifiers, character classes, anchors, optional
  prefixes.
* ``"complex"``     -- non-trivial TDFA state counts: multi-byte
  alternations, long trie-like unions, real-world-ish patterns,
  long fixed/variable segment chains.
* ``"adversarial"`` -- patterns crafted to stress the compiled
  engine: redundant alternations, heavy ``.*`` sliding, patterns that
  drive backtracking engines (Python's ``re``) into quadratic /
  exponential behaviour on crafted inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Tuple


_VALID_TIERS: FrozenSet[str] = frozenset({
    "trivial",
    "basic",
    "complex",
    "adversarial",
})


@dataclass(frozen=True)
class RegexCase:
    """A single regex fixture, reusable across tests and benchmarks."""

    name: str
    pattern: str
    tier: str
    tag_names: Tuple[str, ...] = ()
    positive: Tuple[bytes, ...] = ()
    negative: Tuple[bytes, ...] = ()
    anchored_positive: Tuple[bytes, ...] = ()
    search_at: Mapping[bytes, int] = field(default_factory=dict)
    portable_re: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in _VALID_TIERS:
            raise ValueError(
                f"RegexCase {self.name!r} has unknown tier "
                f"{self.tier!r}; expected one of {sorted(_VALID_TIERS)}"
            )


# ---------------------------------------------------------------------------
# Pattern-builder helpers
#
# A few stress patterns are only interesting at non-trivial sizes
# (hundreds of alternation branches, DFA size that scales with a k
# parameter, etc.).  Keeping them spelled out literally would bury
# the structure, so we expose small builders.
# ---------------------------------------------------------------------------


def _longest_first_alt(words: Tuple[str, ...]) -> str:
    """Alternation of literal branches, longest first, deduped."""
    ordered = sorted(set(words), key=lambda s: (-len(s), s))
    return "|".join(ordered)


def _dotstar_a_dot_k(k: int) -> str:
    """``.*a.{k+1}`` with ``{k+1}`` expanded manually.

    Textbook DFA blow-up: minimal DFA has 2**(k+1) states because it
    must track which of the last k+1 bytes were ``a``. We spell
    ``.{k+1}`` out as ``k+1`` concatenated ``.`` because pythoc does
    not support brace-count quantifiers.
    """
    return ".*a" + ("." * (k + 1))


def _a_opt_n_a_n(n: int) -> str:
    """``a?`` repeated n times followed by ``a`` repeated n times.

    Classic ReDoS for backtracking engines: matching ``a`` * n runs in
    exponential time under naive backtracking.  Linear for TDFA.
    """
    return ("a?" * n) + ("a" * n)


def _shared_prefix_words(prefix: str,
                         suffixes: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(prefix + s for s in suffixes)


# Python 3.12 reserved words (soft keywords included).
_PYTHON_KEYWORDS: Tuple[str, ...] = (
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
)


# C11 keywords with the underscored "_Generic"-style additions.
_C11_KEYWORDS: Tuple[str, ...] = (
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic",
    "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch",
    "typedef", "union", "unsigned", "void", "volatile", "while",
)


# Trimmed ANSI SQL reserved-word set; 80+ branches, heavy prefix
# sharing (CREATE/CROSS, SELECT/SET, TABLE/TRIGGER).
_SQL_RESERVED: Tuple[str, ...] = (
    "ABSOLUTE", "ACTION", "ADD", "ALL", "ALLOCATE", "ALTER", "AND",
    "ANY", "ARE", "AS", "ASC", "ASSERTION", "AT", "AUTHORIZATION",
    "BEGIN", "BETWEEN", "BIT", "BOTH", "BY", "CASCADE", "CASCADED",
    "CASE", "CAST", "CATALOG", "CHAR", "CHARACTER", "CHECK", "CLOSE",
    "COALESCE", "COLLATE", "COLLATION", "COLUMN", "COMMIT", "CONNECT",
    "CONNECTION", "CONSTRAINT", "CONSTRAINTS", "CONTINUE", "CONVERT",
    "CORRESPONDING", "CREATE", "CROSS", "CURRENT", "CURSOR", "DATE",
    "DAY", "DEALLOCATE", "DECIMAL", "DECLARE", "DEFAULT", "DEFERRABLE",
    "DEFERRED", "DELETE", "DESC", "DESCRIBE", "DESCRIPTOR", "DIAGNOSTICS",
    "DISCONNECT", "DISTINCT", "DOMAIN", "DOUBLE", "DROP", "ELSE",
    "END", "ESCAPE", "EXCEPT", "EXCEPTION", "EXEC", "EXECUTE", "EXISTS",
    "EXTERNAL", "EXTRACT", "FALSE", "FETCH", "FIRST", "FLOAT", "FOR",
    "FOREIGN", "FOUND", "FROM", "FULL", "GET", "GLOBAL", "GO", "GOTO",
    "GRANT", "GROUP", "HAVING", "HOUR", "IDENTITY", "IMMEDIATE", "IN",
    "INDICATOR", "INITIALLY", "INNER", "INPUT", "INSENSITIVE", "INSERT",
    "INT", "INTEGER", "INTERSECT", "INTERVAL", "INTO", "IS", "ISOLATION",
    "JOIN", "KEY", "LANGUAGE", "LAST", "LEADING", "LEFT", "LEVEL", "LIKE",
    "LOCAL", "MATCH", "MINUTE", "MODULE", "MONTH", "NAMES", "NATIONAL",
    "NATURAL", "NCHAR", "NEXT", "NO", "NOT", "NULL", "NULLIF", "NUMERIC",
    "OCTET_LENGTH", "OF", "ON", "ONLY", "OPEN", "OPTION", "OR", "ORDER",
    "OUTER", "OUTPUT", "OVERLAPS", "PAD", "PARTIAL", "POSITION",
    "PRECISION", "PREPARE", "PRESERVE", "PRIMARY", "PRIOR", "PRIVILEGES",
    "PROCEDURE", "PUBLIC", "READ", "REAL", "REFERENCES", "RELATIVE",
    "RESTRICT", "REVOKE", "RIGHT", "ROLLBACK", "ROWS", "SCHEMA",
    "SCROLL", "SECOND", "SECTION", "SELECT", "SESSION", "SESSION_USER",
    "SET", "SIZE", "SMALLINT", "SOME", "SPACE", "SQL", "SQLCODE",
    "SQLERROR", "SQLSTATE", "SUBSTRING", "SUM", "SYSTEM_USER", "TABLE",
    "TEMPORARY", "THEN", "TIME", "TIMESTAMP", "TIMEZONE_HOUR",
    "TIMEZONE_MINUTE", "TO", "TRAILING", "TRANSACTION", "TRANSLATE",
    "TRANSLATION", "TRIGGER", "TRUE", "UNION", "UNIQUE", "UNKNOWN",
    "UPDATE", "UPPER", "USAGE", "USER", "USING", "VALUE", "VALUES",
    "VARCHAR", "VARYING", "VIEW", "WHEN", "WHENEVER", "WHERE", "WITH",
    "WORK", "WRITE", "YEAR", "ZONE",
)


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------

_TRIVIAL_CASES: Tuple[RegexCase, ...] = (
    RegexCase(
        name="single_literal",
        pattern="a",
        tier="trivial",
        positive=(b"a", b"bab"),
        negative=(b"", b"bb"),
        search_at={b"bab": 1},
    ),
    RegexCase(
        name="literal_abc",
        pattern="abc",
        tier="trivial",
        positive=(b"abc", b"xxabcxx"),
        negative=(b"ab", b"abd", b"xyz"),
        anchored_positive=(b"abc", b"abcd"),
        search_at={b"xxabcxx": 2, b"abc": 0},
    ),
    RegexCase(
        name="dot_any",
        pattern="a.c",
        tier="trivial",
        positive=(b"abc", b"a_c", b"a c"),
        negative=(b"ac", b"abd"),
    ),
    RegexCase(
        name="anchored_start",
        pattern="^go",
        tier="trivial",
        positive=(b"go", b"going", b"go home"),
        negative=(b"let go", b"nogo"),
        anchored_positive=(b"go", b"going"),
    ),
    RegexCase(
        name="anchored_end",
        pattern="done$",
        tier="trivial",
        positive=(b"done", b"all done"),
        negative=(b"done it", b"doneness"),
    ),
)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

_BASIC_CASES: Tuple[RegexCase, ...] = (
    RegexCase(
        name="alt_pair",
        pattern="cat|dog",
        tier="basic",
        positive=(b"cat", b"dog", b"my dog runs"),
        negative=(b"cow", b"do", b"ca"),
        search_at={b"my dog runs": 3},
    ),
    RegexCase(
        name="alt_three",
        pattern="cat|dog|bird",
        tier="basic",
        positive=(b"cat", b"dog", b"bird", b"  bird call"),
        negative=(b"fish", b"do"),
    ),
    RegexCase(
        name="quant_plus",
        pattern="ab+c",
        tier="basic",
        positive=(b"abc", b"abbc", b"abbbc"),
        negative=(b"ac", b"adc"),
    ),
    RegexCase(
        name="quant_star",
        pattern="ab*c",
        tier="basic",
        positive=(b"ac", b"abc", b"abbc"),
        negative=(b"adc",),
    ),
    RegexCase(
        name="dotstar_bracket",
        pattern="a.*b",
        tier="basic",
        positive=(b"ab", b"axb", b"axxb", b"zzazzbzz"),
        negative=(b"a", b"bbbb"),
        search_at={b"zzazzbzz": 2},
    ),
    RegexCase(
        name="dotplus_bracket",
        pattern="BEGIN.+END",
        tier="basic",
        positive=(b"BEGINxEND", b"BEGIN 1 END", b"xBEGIN...ENDy"),
        negative=(b"BEGINEND", b"BEGIN only", b"END BEGIN"),
        notes="`.+` requires at least one byte between the fixed ends.",
    ),
    RegexCase(
        name="charclass_word",
        pattern="\\w+",
        tier="basic",
        positive=(b"hello_123", b"  token  ", b"x"),
        negative=(b"", b"   "),
    ),
    RegexCase(
        name="digit_run",
        pattern="\\d+",
        tier="basic",
        positive=(b"123", b"abc123def", b"0"),
        negative=(b"abc",),
        search_at={b"abc123def": 3},
    ),
    RegexCase(
        name="anchor_exact",
        pattern="^hello$",
        tier="basic",
        positive=(b"hello",),
        anchored_positive=(b"hello",),
        negative=(b"hello world", b"say hello"),
    ),
    RegexCase(
        name="optional_prefix",
        pattern="https?://",
        tier="basic",
        positive=(b"http://", b"https://", b"see https://x"),
        negative=(b"ftp://", b"htps://"),
        search_at={b"see https://x": 4},
    ),
    RegexCase(
        name="optional_chain",
        pattern="colou?rs?",
        tier="basic",
        positive=(
            b"color", b"colour", b"colors", b"colours",
            b"the colours are nice",
        ),
        negative=(b"colur", b"colr", b"cloor"),
        notes="Two independent optional groups; small Cartesian set.",
    ),
    RegexCase(
        name="alt_shared_suffix",
        pattern="ab|cb",
        tier="basic",
        positive=(b"ab", b"cb", b"xxcb"),
        negative=(b"ac", b"bb", b"bbc"),
    ),
    RegexCase(
        name="alt_unequal_len",
        pattern="ab|cde",
        tier="basic",
        positive=(b"ab", b"cde", b"xxab", b"cdeyy"),
        negative=(b"ac", b"cd", b"cdf"),
    ),
    RegexCase(
        name="nested_group_star",
        pattern="(ab)*c",
        tier="basic",
        positive=(b"c", b"abc", b"ababc", b"abababc"),
        negative=(b"ab", b"abab", b"xyzw"),
        notes=("Negative inputs must avoid 'c' entirely: '(ab)*c' "
               "matches a bare 'c' via zero repetitions."),
    ),
)


# ---------------------------------------------------------------------------
# Complex cases
# ---------------------------------------------------------------------------

_BIG_ALT_PROGLANG = (
    "auto|bool|break|byte|case|char|class|const|continue|default|do|"
    "double|else|enum|extern|false|float|for|goto|if|inline|int|long|"
    "namespace|new|nullptr|operator|private|protected|public|register|"
    "return|short|signed|sizeof|static|struct|switch|template|this|"
    "throw|true|try|typedef|typeid|typename|union|unsigned|using|"
    "virtual|void|volatile|while"
)


_COMPLEX_CASES: Tuple[RegexCase, ...] = (
    RegexCase(
        name="identifier",
        pattern="[a-zA-Z_][a-zA-Z0-9_]*",
        tier="complex",
        positive=(b"hello", b"_var", b"x123", b"  ident_42"),
        negative=(b"123", b"--", b""),
        anchored_positive=(b"hello", b"_v", b"x123"),
    ),
    RegexCase(
        name="email_like",
        pattern="[a-z]+@[a-z]+\\.[a-z]+",
        tier="complex",
        positive=(
            b"foo@bar.com",
            b"contact: user@host.io for help",
            b"a@b.c",
        ),
        negative=(b"foobar", b"@x.y", b"foo@bar", b"foo.bar@com"),
        search_at={b"contact: user@host.io for help": 9},
    ),
    RegexCase(
        name="ipv4_like",
        pattern="\\d+\\.\\d+\\.\\d+\\.\\d+",
        tier="complex",
        positive=(
            b"192.168.1.1",
            b"addr: 10.0.0.1 here",
            b"255.255.255.0",
        ),
        negative=(b"1.2.3", b"abc.def.ghi.jkl"),
        search_at={b"addr: 10.0.0.1 here": 6},
    ),
    RegexCase(
        name="hex_color",
        pattern="#[0-9a-fA-F]+",
        tier="complex",
        positive=(b"#ff0000", b"color: #AbCdEf;", b"#0"),
        negative=(b"no hash", b"#", b"ffffff"),
        search_at={b"color: #AbCdEf;": 7},
    ),
    RegexCase(
        name="url_prefix",
        pattern="https?://[a-zA-Z0-9.]+",
        tier="complex",
        positive=(
            b"http://example.com",
            b"visit https://pythoc.io now",
            b"https://a.b",
        ),
        negative=(b"http:/", b"htps://x"),
        search_at={b"visit https://pythoc.io now": 6},
    ),
    RegexCase(
        name="quoted_string",
        pattern='"[^"]*"',
        tier="complex",
        positive=(
            b'"hello"',
            b'prefix "quoted" suffix',
            b'""',
        ),
        negative=(b'unquoted', b'"missing', b'trailing"'),
        search_at={b'prefix "quoted" suffix': 7},
    ),
    RegexCase(
        name="log_level",
        pattern="\\[(INFO|WARN|ERROR|DEBUG)\\] ",
        tier="complex",
        positive=(
            b"[INFO] started",
            b"---[ERROR] boom",
            b"[WARN] x",
            b"[DEBUG] y",
        ),
        negative=(b"[TRACE] x", b"[INFO]no-space", b"[info] lowercase"),
        search_at={b"---[ERROR] boom": 3},
    ),
    RegexCase(
        name="c_keywords",
        pattern=_BIG_ALT_PROGLANG,
        tier="complex",
        positive=(
            b"return",
            b"  namespace foo",
            b"typename T",
            b"unsigned int",
        ),
        negative=(b"retur", b"reeturn", b"", b"foozz"),
        search_at={b"  namespace foo": 2},
        notes=("Large alternation of C++-shaped keywords; exercises "
               "big-alt compile throughput and prefix sharing."),
    ),
    RegexCase(
        name="python_keywords",
        pattern=_longest_first_alt(_PYTHON_KEYWORDS),
        tier="complex",
        positive=(
            b"return",
            b"continue going",
            b"  async def foo():",
            b"nonlocal_xyz",
        ),
        negative=(b"retur", b"", b"zzz"),
        search_at={b"continue going": 0, b"  async def foo():": 2},
        notes=("Python 3.12 reserved words.  Branch count is ~35 with "
               "mixed lengths; a good TDFA throughput probe."),
    ),
    RegexCase(
        name="c11_keywords",
        pattern=_longest_first_alt(_C11_KEYWORDS),
        tier="complex",
        positive=(
            b"_Static_assert",
            b"typedef struct",
            b"  _Atomic int x",
            b"while (1)",
        ),
        negative=(b"_Stat", b"typedf", b"wh"),
        notes=("C11 keywords including the ``_Generic``-family; mixed "
               "lengths and underscore prefixes stress class-partition "
               "sharing."),
    ),
    RegexCase(
        name="sql_reserved",
        pattern=_longest_first_alt(_SQL_RESERVED),
        tier="complex",
        positive=(
            b"SELECT",
            b"INNER JOIN",
            b"  CREATE TABLE t (x INT)",
            b"SESSION_USER",
        ),
        negative=(b"SELEC", b"", b"zzz"),
        search_at={b"  CREATE TABLE t (x INT)": 2},
        notes=("~180-branch ANSI SQL reserved-word alternation.  "
               "Very heavy prefix sharing (CREATE/CROSS, SELECT/SET)."),
    ),
    RegexCase(
        name="long_fv_chain_5_segments",
        pattern="a.*b.*c.*d.*e",
        tier="complex",
        positive=(b"abcde", b"a1b2c3d4e", b"xxa__b__c__d__eyy"),
        negative=(b"abcd", b"aedcb", b"a b c d"),
        notes=("Five fixed anchors threaded through four variable "
               "gaps; exercises shape segment linearisation."),
    ),
    RegexCase(
        name="long_fv_chain_cc_gaps",
        pattern="[A-Z]+[0-9]+[A-Z]+[0-9]+[A-Z]+",
        tier="complex",
        positive=(b"A1B2C", b"HELLO42WORLD7X", b"xxAB12CD34EFyy"),
        negative=(b"a1b2c", b"A1B2", b"12A34B"),
    ),
    RegexCase(
        name="dotstar_sandwich_chain",
        pattern="BEGIN.*\\{.*\\}.*END",
        tier="complex",
        positive=(
            b"BEGIN{}END",
            b"BEGIN foo{x=1}bar END",
            b"xx BEGIN header {body} footer END yy",
        ),
        negative=(b"BEGIN END", b"BEGIN {END", b"BEGIN }END"),
    ),
    RegexCase(
        name="many_dotstar_gaps",
        pattern="x.*x.*x.*x.*x",
        tier="complex",
        positive=(b"xxxxx", b"x_x_x_x_x", b"aXAxBxCxDxEx"),
        negative=(b"xxxx", b"xyxyx"),
        notes=("Five 'x's separated by greedy gaps; match requires "
               "five x occurrences in order."),
    ),
    RegexCase(
        name="dotted_identifier_chain",
        pattern="[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_][A-Za-z0-9_]*)+",
        tier="complex",
        positive=(
            b"a.b",
            b"module.sub.func",
            b"name.with_123.parts",
            b"see pkg.mod.name here",
        ),
        negative=(b"a", b".b", b"a..b"),
        search_at={b"see pkg.mod.name here": 4},
        portable_re=False,
        notes=("Repeated grouped ``.ident`` segments.  On large "
               "alphabetic haystacks Python's ``re`` runs in "
               "polynomial time, which is not a useful baseline."),
    ),
    RegexCase(
        name="iso_timestamp",
        pattern=(
            "\\d\\d\\d\\d-\\d\\d-\\d\\dT"
            "\\d\\d:\\d\\d:\\d\\d"
            "(\\.\\d+)?(Z|[+-]\\d\\d:\\d\\d)?"
        ),
        tier="complex",
        positive=(
            b"2024-01-02T03:04:05Z",
            b"2024-01-02T03:04:05.678Z",
            b"2024-01-02T03:04:05+08:00",
            b"log: 2024-01-02T03:04:05 here",
        ),
        negative=(b"2024-1-2T3:4:5", b"2024-01-02 03:04:05"),
        search_at={b"log: 2024-01-02T03:04:05 here": 5},
        notes=("ISO-8601 timestamp with optional fractional second "
               "and optional timezone; long fixed digit chain."),
    ),
    RegexCase(
        name="cc_big_negated_class",
        pattern="[^\\sA-Za-z0-9_]",
        tier="complex",
        positive=(b"!", b"@", b"abc!", b"x;y"),
        negative=(b"abc", b"XYZ", b"123", b"_", b" \t\n"),
        notes="Everything that is neither word nor whitespace.",
    ),
)


# ---------------------------------------------------------------------------
# Adversarial cases
# ---------------------------------------------------------------------------

_ADVERSARIAL_CASES: Tuple[RegexCase, ...] = (
    RegexCase(
        name="needle_in_haystack",
        pattern="needle",
        tier="adversarial",
        positive=(b"needle",),
        negative=(b"hay",),
        notes=("Literal-only search.  Benchmarks feed large "
               "haystacks with the needle at start / middle / end."),
    ),
    RegexCase(
        name="long_needle",
        pattern="supercalifragilisticexpialidocious",
        tier="adversarial",
        positive=(b"supercalifragilisticexpialidocious",),
        negative=(b"supercalifragil", b""),
        notes=("34-byte literal: big |F_1|, exercises the leading-"
               "anchor gadget's long-block shift ceiling."),
    ),
    RegexCase(
        name="dotstar_sliding",
        pattern="a.*z",
        tier="adversarial",
        positive=(b"az", b"amz", b"ammz", b"axxz", b"azzzzzzz"),
        negative=(b"aaa", b"zzz"),
        portable_re=False,
        notes=("Anchored ``a.*z`` -- classic leftmost tag write "
               "discipline stressor.  On haystacks with many 'a's "
               "and no 'z' Python's ``re`` is polynomial."),
    ),
    RegexCase(
        name="alt_long_prefix",
        pattern="alpha|alpine|alumni|alpaca|almond|always|alphabet",
        tier="adversarial",
        positive=(
            b"alpha",
            b"alpine mountain",
            b"  alphabet",
            b"alumni",
        ),
        negative=(b"alph", b"alp", b"al"),
        search_at={b"alpine mountain": 0, b"  alphabet": 2},
        notes="Alternatives share long 'al' prefix; deep TDFA sharing.",
    ),
    RegexCase(
        name="alt_shared_body",
        pattern="(axxc|byyc)",
        tier="adversarial",
        positive=(b"axxc", b"byyc", b" xxaxxcxx "),
        negative=(b"axxd", b"ayyc", b"bxxc"),
        notes=("Two branches differ only in the 'corner' bytes; "
               "exercises set-BM column reasoning on mixed columns."),
    ),
    RegexCase(
        name="sliding_tag",
        pattern=".*{tag}.*b",
        tier="adversarial",
        tag_names=("tag",),
        positive=(b"b", b"xb", b"xxxb", b"abbb"),
        negative=(b"aaa", b"c"),
        portable_re=False,
        notes=("Tag position depends on which ``.*`` split the engine "
               "picks; triggers the tag-rewrite warning path."),
    ),
    RegexCase(
        name="loop_then_fixed",
        pattern="(ab)*cde",
        tier="adversarial",
        positive=(b"cde", b"abcde", b"ababcde", b"abababcde"),
        negative=(b"cd", b"aabbcde", b"abab"),
        anchored_positive=(b"abababcde",),
        notes=("Cycle-shaped V_0 followed by a fixed F; exercises the "
               "restricted-Sigma_V branch of anchor selection."),
    ),
    RegexCase(
        name="chained_search_markers",
        pattern=".*foo.*bar",
        tier="adversarial",
        positive=(b"foobar", b"xxfoo___bar", b"fooxxxxxxxxxxbar"),
        negative=(b"foo", b"bar", b"foofoo", b"barbar"),
        portable_re=False,
        notes=("Two sliding segments; Python's ``re`` spends O(n^2) "
               "exploring ``.*`` splits on non-matching haystacks."),
    ),
    RegexCase(
        name="catastrophic_email",
        pattern="[a-z]+@[a-z]+\\.[a-z]+",
        tier="adversarial",
        positive=(
            b"user@host.com",
            b"x@y.z" + b"x" * 200 + b"user@host.com",
        ),
        negative=(
            b"a" * 4096,
            b"a" * 2048 + b"@" + b"b" * 2048,
        ),
        notes=("On non-matching haystacks Python's ``re`` hits "
               "catastrophic backtracking; the compiled TDFA stays "
               "linear."),
    ),
    RegexCase(
        name="catastrophic_nested",
        pattern="(a+)+b",
        tier="adversarial",
        positive=(b"ab", b"aaaab", b"xxaaab"),
        negative=(b"a" * 30,),
        portable_re=False,
        notes=("Classic ReDoS: ``(a+)+b`` vs ``aaaa...`` runs "
               "Python ``re`` into exponential behaviour.  TDFA "
               "handles it in a single linear pass."),
    ),
    RegexCase(
        name="redos_a_opt_n_a_n",
        pattern=_a_opt_n_a_n(20),
        tier="adversarial",
        positive=(b"a" * 40,),
        negative=(b"a" * 39, b""),
        anchored_positive=(b"a" * 40,),
        portable_re=False,
        notes=("``a?{n}a{n}`` for n=20 -- exponential for backtracking "
               "engines, linear for TDFA."),
    ),
    RegexCase(
        name="dotstar_a_dot_k",
        pattern=_dotstar_a_dot_k(6),
        tier="adversarial",
        positive=(
            b"aXXXXXXX",
            b"xxxaabcdefg",
            b"padding....axxxxxxx",
        ),
        negative=(b"axx", b""),
        portable_re=False,
        notes=("``.*a.{k+1}`` for k=6: minimal DFA has 2**(k+1)=128 "
               "states.  Textbook subset-construction stressor; "
               "``re`` backtracks exponentially on non-matching "
               "haystacks."),
    ),
    RegexCase(
        name="long_trie_shared_prefix",
        pattern=_longest_first_alt(
            _shared_prefix_words(
                "pre",
                (
                    "fix", "view", "pare", "tend", "dict", "dicate",
                    "mium", "mise", "text", "lude", "fect", "cept",
                    "cede", "serve", "scribe", "sent", "sume",
                ),
            )
        ),
        tier="adversarial",
        positive=(
            b"prefix",
            b"preview",
            b"predicate",
            b"  preserve",
            b"presumption",
        ),
        negative=(b"pre", b"pry", b"zzz"),
        notes=("17 branches sharing the 'pre' prefix.  Stresses trie "
               "factoring during subset construction."),
    ),
    RegexCase(
        name="dotstar_then_anchored_tail",
        pattern=".*[0-9]+end$",
        tier="adversarial",
        positive=(b"123end", b"abc42end", b"0end"),
        negative=(b"end", b"123end trailing", b"endend"),
        portable_re=False,
        notes=("Greedy ``.*`` must swallow everything up to the last "
               "digit run.  Python's ``re`` does quadratic work on "
               "large haystacks regardless of match outcome."),
    ),
    RegexCase(
        name="alt_with_shared_variable_interior",
        pattern="a.*b|c.*b",
        tier="adversarial",
        positive=(b"ab", b"cb", b"aXXb", b"cYYb", b"xxaQQbyy"),
        negative=(b"axx", b"bbb", b"a"),
        portable_re=False,
        notes=("Branches differ in prefix but share trailing 'b'.  "
               "Exercises pumping-lemma-style unrolling during "
               "determinisation; ``re`` is polynomial on 'a'-heavy "
               "inputs."),
    ),
    RegexCase(
        name="many_optionals",
        pattern="a?b?c?d?e?f?g?h?xyz",
        tier="adversarial",
        positive=(
            b"xyz",
            b"axyz",
            b"abcxyz",
            b"abcdefghxyz",
            b"  fghxyz",
        ),
        negative=(b"xy", b"abcxy", b"zxy"),
        notes=("Eight independent optional bytes guarding a fixed "
               "suffix; exponential path count, but TDFA must merge."),
    ),
)


# ---------------------------------------------------------------------------
# Catalogue aggregation
# ---------------------------------------------------------------------------

_ALL_CASES: Tuple[RegexCase, ...] = (
    _TRIVIAL_CASES
    + _BASIC_CASES
    + _COMPLEX_CASES
    + _ADVERSARIAL_CASES
)

_CASES_BY_NAME: Dict[str, RegexCase] = {c.name: c for c in _ALL_CASES}


def all_cases() -> Tuple[RegexCase, ...]:
    """Return every registered regex case in declaration order."""
    return _ALL_CASES


def cases_by_tier(tier: str) -> Tuple[RegexCase, ...]:
    """Return all cases at the given tier."""
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}; expected one of {sorted(_VALID_TIERS)}"
        )
    return tuple(c for c in _ALL_CASES if c.tier == tier)


def portable_cases() -> Tuple[RegexCase, ...]:
    """Return cases safe to cross-check against Python's ``re``."""
    return tuple(c for c in _ALL_CASES if c.portable_re)


def case(name: str) -> RegexCase:
    """Look up a case by name."""
    try:
        return _CASES_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown regex case {name!r}; registered: "
            f"{sorted(_CASES_BY_NAME)}"
        ) from exc


def tiers() -> Tuple[str, ...]:
    """Return the tier names in a stable progression (trivial -> hard)."""
    return ("trivial", "basic", "complex", "adversarial")


def patterns_for_precompile() -> Tuple[str, ...]:
    """Flat tuple of every pattern, useful for bulk pre-compile pushes."""
    return tuple(c.pattern for c in _ALL_CASES)


__all__ = [
    "RegexCase",
    "all_cases",
    "case",
    "cases_by_tier",
    "patterns_for_precompile",
    "portable_cases",
    "tiers",
]
