"""
Shape decomposition for the T-BMA builder.

Decomposes a regex AST (after parse-time normalization) into a
single-pattern shape sequence consumed by the leading-anchor selector:

    shape(Pat) = V_0 F_1 V_1 F_2 ... F_k V_k

with

- ``V_i`` a nullable segment (``V_n(S_i)``); ``V_0({epsilon})`` when
  absent.
- ``F_j`` a fixed-length segment ``F_m(T)`` with ``m >= 1`` and a
  finite string set ``T`` contained in ``Sigma^m``.
- ``k >= 0`` (``k = 0`` means the pattern is entirely one V -- no F
  anchor candidate).
- The sequence always starts and ends with a V (padded with the
  ``V_0({epsilon})`` placeholder if needed).

The decomposition is intentionally conservative:

- If the AST contains a node that is neither F-fit nor V-nullable
  (e.g. an ``a|bc`` alternate with mismatched branch lengths that is
  also not nullable), the whole decomposition degenerates to
  ``V_0({epsilon})`` (``k = 0``). This forces the anchor selector to
  fall through to pure TDFA, which is the correct conservative
  behavior.
- Enumeration of a fixed-length branch's string set ``T`` is capped
  by a compile-time budget. If the budget is exceeded we mark the
  node as "not F-fit" (it then becomes part of a V segment if it is
  also nullable, otherwise the whole decomposition degenerates as
  above).

The module exposes:

- ``VSegment`` / ``FSegment`` dataclasses,
- ``ShapeSeq`` container with a helper ``leading_pair`` for the
  O(1) ``(V_0, F_1)`` anchor rule,
- ``decompose_pattern(ast_node, *, max_f_set_size)`` entry point.

Ordering note: the leading-anchor rule only ever consults
``leading_pair``; the rest of the segments are kept so a future
multi-site anchor can reuse the same ``ShapeSeq`` without
recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from .parse import (
    Alternate,
    Anchor,
    CharClass,
    Concat,
    Dot,
    Group,
    Literal,
    Repeat,
    Tag,
)


# Default cap on the size of the string set ``T`` for an F segment.
# Patterns whose leading factor would expand beyond this fall out of
# "F-fit" classification and the pipeline degenerates to pure TDFA.
# The cap keeps compile-time Cartesian expansion bounded.
DEFAULT_MAX_F_SET_SIZE = 64


# ---------------------------------------------------------------------------
# Internal node classification
# ---------------------------------------------------------------------------

# A node's "F-shape" is either None (not F-fit within budget) or a
# pair ``(m, T)`` with ``m`` the exact byte length and ``T`` the
# finite set of length-``m`` strings accepted by the node.
_FShape = Optional[Tuple[int, FrozenSet[Tuple[int, ...]]]]


def _charclass_bytes(node: CharClass) -> FrozenSet[int]:
    """Enumerate the byte values matched by a CharClass."""
    bytes_in: List[int] = []
    if node.negated:
        allowed = [True] * 256
        for lo, hi in node.ranges:
            for b in range(lo, hi + 1):
                allowed[b] = False
        bytes_in.extend(b for b in range(256) if allowed[b])
    else:
        seen = [False] * 256
        for lo, hi in node.ranges:
            for b in range(lo, hi + 1):
                seen[b] = True
        bytes_in.extend(b for b in range(256) if seen[b])
    return frozenset(bytes_in)


def _singleton_f(byte_val: int) -> _FShape:
    return (1, frozenset({(byte_val,)}))


def _dot_f(max_f_set_size: int) -> _FShape:
    if 256 > max_f_set_size:
        return None
    return (1, frozenset({(b,) for b in range(256)}))


def _charclass_f(node: CharClass, max_f_set_size: int) -> _FShape:
    bs = _charclass_bytes(node)
    if len(bs) == 0 or len(bs) > max_f_set_size:
        return None
    return (1, frozenset({(b,) for b in bs}))


def _concat_fshapes(lhs: _FShape, rhs: _FShape,
                    max_f_set_size: int) -> _FShape:
    if lhs is None or rhs is None:
        return None
    m_l, t_l = lhs
    m_r, t_r = rhs
    if m_l == 0:
        return rhs
    if m_r == 0:
        return lhs
    if len(t_l) * len(t_r) > max_f_set_size:
        return None
    return (m_l + m_r, frozenset(l + r for l in t_l for r in t_r))


def _union_fshapes(options: List[_FShape],
                   max_f_set_size: int) -> _FShape:
    """Union: all branches must have the same m; return merged set."""
    if not options:
        return None
    if any(o is None for o in options):
        return None
    m0 = options[0][0]
    if any(o[0] != m0 for o in options):
        return None
    merged: FrozenSet[Tuple[int, ...]] = frozenset()
    for _, t in options:  # type: ignore[misc]
        merged = merged | t
        if len(merged) > max_f_set_size:
            return None
    return (m0, merged)


def _is_nullable(node) -> bool:
    """Whether the AST node's language contains the empty string."""
    if isinstance(node, (Literal, Dot, CharClass)):
        return False
    if isinstance(node, (Anchor, Tag)):
        return True
    if isinstance(node, Group):
        return _is_nullable(node.child)
    if isinstance(node, Concat):
        return all(_is_nullable(ch) for ch in node.children)
    if isinstance(node, Alternate):
        return any(_is_nullable(ch) for ch in node.children)
    if isinstance(node, Repeat):
        if node.min_count == 0:
            return True
        return _is_nullable(node.child)
    return False


def _f_shape(node, max_f_set_size: int) -> _FShape:
    """Return the F-shape of a node, or None if not F-fit.

    An F-fit node has every accepted string of the same byte length
    ``m >= 1`` and the total finite set fits within ``max_f_set_size``.

    Nodes that can also match the empty string (nullable) are **not**
    F-fit because their language mixes strings of different lengths
    (including 0).
    """
    if isinstance(node, Literal):
        return _singleton_f(node.byte)
    if isinstance(node, Dot):
        return _dot_f(max_f_set_size)
    if isinstance(node, CharClass):
        return _charclass_f(node, max_f_set_size)
    if isinstance(node, Group):
        return _f_shape(node.child, max_f_set_size)
    if isinstance(node, (Anchor, Tag)):
        # zero-width; conceptually F_0 over the empty-string set, we
        # never build an F with m=0.
        return None
    if isinstance(node, Concat):
        acc: _FShape = (0, frozenset({()}))
        for ch in node.children:
            ch_f = _f_shape(ch, max_f_set_size)
            if ch_f is None:
                # child not F-fit: if it is nullable we could try to
                # treat the Concat as F-fit only if *every* child is
                # F-fit (strings of consistent m); nullability mixes
                # lengths so we give up.
                if _is_nullable(ch):
                    return None
                return None
            acc = _concat_fshapes(acc, ch_f, max_f_set_size)
            if acc is None:
                return None
        if acc is None or acc[0] == 0:
            # A concat of only zero-width nodes is not F-fit (m = 0).
            return None
        return acc
    if isinstance(node, Alternate):
        branch_shapes = [_f_shape(ch, max_f_set_size) for ch in node.children]
        return _union_fshapes(branch_shapes, max_f_set_size)
    if isinstance(node, Repeat):
        if node.min_count == node.max_count and node.min_count is not None:
            # fixed count k: repeat child k times
            child_f = _f_shape(node.child, max_f_set_size)
            if child_f is None:
                return None
            acc: _FShape = (0, frozenset({()}))
            for _ in range(node.min_count):
                acc = _concat_fshapes(acc, child_f, max_f_set_size)
                if acc is None:
                    return None
            if acc is None or acc[0] == 0:
                return None
            return acc
        return None
    return None


# ---------------------------------------------------------------------------
# Public segment / sequence dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VSegment:
    """One V segment of a shape decomposition.

    ``ast_nodes`` is the ordered list of original AST children that
    make up this V. ``is_epsilon`` is True iff this is the
    ``V_0`` epsilon placeholder (no ast_nodes, or only zero-width
    ones). The segment is always nullable by construction.
    """

    ast_nodes: Tuple[object, ...]
    is_epsilon: bool

    def __post_init__(self) -> None:
        # Sanity: every non-epsilon V must be nullable.
        assert self.is_epsilon or _is_nullable(Concat(list(self.ast_nodes))), (
            "VSegment marked non-epsilon but its AST is not nullable"
        )

    def as_ast(self) -> object:
        """Return a Concat wrapping this V's AST nodes (may be empty)."""
        return Concat(list(self.ast_nodes))


@dataclass(frozen=True)
class FSegment:
    """One F segment of a shape decomposition."""

    m: int
    T: FrozenSet[Tuple[int, ...]]
    ast_nodes: Tuple[object, ...]

    def __post_init__(self) -> None:
        assert self.m >= 1, f"F segment must have m >= 1, got {self.m}"
        assert len(self.T) >= 1, "F segment must have at least one string"
        for s in self.T:
            assert len(s) == self.m, (
                f"F segment string length mismatch: got {len(s)}, expected {self.m}"
            )


@dataclass(frozen=True)
class ShapeSeq:
    """Normalized ``V_0 F_1 V_1 ... F_k V_k`` sequence for one pattern.

    Always starts and ends with a V (padded with a V epsilon
    placeholder if needed). The invariant ``len(V) == k + 1`` and
    ``len(F) == k`` always holds. A fully-degenerate ``k = 0``
    sequence (one V, no Fs) also satisfies this.
    """

    V: Tuple[VSegment, ...]
    F: Tuple[FSegment, ...]

    def __post_init__(self) -> None:
        assert len(self.V) == len(self.F) + 1, (
            f"shape sequence must satisfy |V| = |F| + 1, "
            f"got |V|={len(self.V)} |F|={len(self.F)}"
        )

    @property
    def k(self) -> int:
        return len(self.F)

    def leading_pair(self) -> Tuple[VSegment, Optional[FSegment]]:
        """Return ``(V_0, F_1 or None)`` for the leading-anchor check."""
        v0 = self.V[0]
        f1 = self.F[0] if self.F else None
        return (v0, f1)


# ---------------------------------------------------------------------------
# Decomposition driver
# ---------------------------------------------------------------------------

@dataclass
class _SegmentBuilder:
    """Accumulator state while walking a Concat's children."""

    max_f_set_size: int
    v_runs: List[List[object]] = field(default_factory=lambda: [[]])
    f_runs: List[Tuple[int, FrozenSet[Tuple[int, ...]], List[object]]] = field(
        default_factory=list)
    # True iff the current accumulator is "inside" an F run (i.e. the
    # last emitted token was F-fit and not nullable).
    in_f: bool = False
    failed: bool = False

    def push_v(self, node: object) -> None:
        if self.failed:
            return
        if self.in_f:
            self.v_runs.append([])
            self.in_f = False
        self.v_runs[-1].append(node)

    def push_f(self, node: object, shape: Tuple[int, FrozenSet[Tuple[int, ...]]]) -> None:
        if self.failed:
            return
        if not self.in_f:
            self.f_runs.append((shape[0], shape[1], [node]))
            self.in_f = True
            return
        m_prev, t_prev, nodes_prev = self.f_runs[-1]
        combined = _concat_fshapes(
            (m_prev, t_prev), shape, self.max_f_set_size)
        if combined is None:
            # Two adjacent F-fit nodes whose Cartesian product
            # overflows the compile-time budget. There is no V
            # separator between them, so we cannot split them into
            # two F segments. Fail conservatively; the pipeline
            # then falls through to pure TDFA.
            self.failed = True
            return
        nodes_prev.append(node)
        self.f_runs[-1] = (combined[0], combined[1], nodes_prev)

    def fail(self) -> None:
        self.failed = True

    def build(self) -> Optional[ShapeSeq]:
        if self.failed:
            return None
        v_segments = [
            VSegment(
                ast_nodes=tuple(run),
                is_epsilon=all(_is_zero_width_and_byte_free(n) for n in run),
            )
            for run in self.v_runs
        ]
        f_segments = [
            FSegment(m=m, T=t, ast_nodes=tuple(ns))
            for (m, t, ns) in self.f_runs
        ]
        if len(v_segments) == len(f_segments):
            # Last element was an F; pad with an empty tail V.
            v_segments.append(VSegment(ast_nodes=(), is_epsilon=True))
        if len(v_segments) != len(f_segments) + 1:
            return None
        return ShapeSeq(V=tuple(v_segments), F=tuple(f_segments))


def _is_zero_width(node) -> bool:
    """True iff the node never consumes bytes (purely positional)."""
    if isinstance(node, (Anchor, Tag)):
        return True
    if isinstance(node, Group):
        return _is_zero_width(node.child)
    if isinstance(node, Concat):
        return all(_is_zero_width(ch) for ch in node.children)
    if isinstance(node, Alternate):
        return all(_is_zero_width(ch) for ch in node.children)
    if isinstance(node, Repeat):
        # A repeat of a zero-width body is zero-width; so is a max=0
        # repeat.
        if node.max_count == 0:
            return True
        return _is_zero_width(node.child)
    return False


def _consumes_bytes(node) -> bool:
    """True iff the node can consume at least one byte on some branch."""
    if isinstance(node, (Literal, Dot, CharClass)):
        return True
    if isinstance(node, (Anchor, Tag)):
        return False
    if isinstance(node, Group):
        return _consumes_bytes(node.child)
    if isinstance(node, Concat):
        return any(_consumes_bytes(ch) for ch in node.children)
    if isinstance(node, Alternate):
        return any(_consumes_bytes(ch) for ch in node.children)
    if isinstance(node, Repeat):
        if node.max_count == 0:
            return False
        return _consumes_bytes(node.child)
    return False


def _is_zero_width_and_byte_free(node) -> bool:
    return _is_zero_width(node) and not _consumes_bytes(node)


def _flatten_concat(node) -> List[object]:
    """Flatten a Group / Concat tree into a flat children list.

    Top-level non-Concat nodes become single-element lists. The
    resulting list is what the shape builder walks left-to-right.
    """
    if isinstance(node, Group):
        return _flatten_concat(node.child)
    if isinstance(node, Concat):
        out: List[object] = []
        for ch in node.children:
            out.extend(_flatten_concat(ch))
        return out
    return [node]


def _walk(node: object,
          builder: _SegmentBuilder) -> None:
    """Walk a single AST node and dispatch into the builder."""
    if builder.failed:
        return

    if isinstance(node, (Anchor, Tag)):
        # Zero-width: always V-fit.
        builder.push_v(node)
        return

    f_shape = _f_shape(node, builder.max_f_set_size)
    nullable = _is_nullable(node)

    if f_shape is not None and not nullable:
        # Pure F-fit: push into current F run.
        builder.push_f(node, f_shape)
        return

    if nullable:
        # Nullable (and possibly non-F-fit): it is a V component.
        builder.push_v(node)
        return

    # Special case: min >= 1 unbounded/variable repeat of an F-fit
    # child: split as child^min ++ child*. The child* part is V-fit.
    if isinstance(node, Repeat):
        child_f = _f_shape(node.child, builder.max_f_set_size)
        if (child_f is not None
                and node.min_count >= 1
                and (node.max_count is None or node.max_count != node.min_count)):
            # Push fixed copies as F.
            for _ in range(node.min_count):
                builder.push_f(node.child, child_f)
                if builder.failed:
                    return
            # Push the remaining ``child*`` / ``child{0, max-min}`` as V.
            new_max = (None if node.max_count is None
                       else node.max_count - node.min_count)
            tail = Repeat(child=node.child, min_count=0,
                          max_count=new_max, lazy=node.lazy)
            builder.push_v(tail)
            return

    # Neither F-fit nor V-nullable and we cannot usefully split it
    # (e.g. ``(a|bc)`` with mismatched branch lengths). Conservatively
    # fail: the caller degrades to a single V epsilon shape, which
    # fails anchor selection and routes to pure TDFA.
    builder.fail()


def decompose_pattern(ast_node: object,
                      *,
                      max_f_set_size: int = DEFAULT_MAX_F_SET_SIZE
                      ) -> ShapeSeq:
    """Decompose a pattern AST into its ``V F V F ... V`` shape.

    If decomposition fails (see module docstring), returns the
    degenerate shape with just a V epsilon placeholder and
    ``k = 0``, which causes the leading-anchor rule to fall through
    to pure TDFA.
    """
    builder = _SegmentBuilder(max_f_set_size=max_f_set_size)
    children = _flatten_concat(ast_node)
    for child in children:
        _walk(child, builder)
        if builder.failed:
            break

    seq = builder.build() if not builder.failed else None
    if seq is None:
        return ShapeSeq(
            V=(VSegment(ast_nodes=(), is_epsilon=True),),
            F=(),
        )
    return seq
