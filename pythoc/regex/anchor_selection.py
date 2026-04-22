"""
Leading-anchor selection for the T-BMA builder.

Implements the "leading-factor anchor" rule used by the current
T-BMA construction: inspect **only** the leading pair ``(V_0, F_1)``
of the pattern's shape sequence. If all shape gates pass, return a
``LeadingAnchor`` describing the single gadget site rooted at the
TDFA start state; otherwise return ``None`` and let ``tbma`` emit a
pure-TDFA artifact.

The check is O(1) in the number of shape segments: no loop over
``F_2..F_k``, no heavy DFA walks except the single
``build_residual_subdfa(V_0)``. Multi-site / phase-aware BM anchoring
is a planned follow-up; keeping ``anchor_state`` parameterized here
keeps the selector future-proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .analysis import ResidualSubDFA, build_residual_subdfa
from .shape import FSegment, ShapeSeq, VSegment


# Recommended defaults for the T-BMA builder. Kept as named constants so
# the pipeline and unit tests can reference them.
DEFAULT_MAX_W = 16
DEFAULT_MAX_Q_V = 32
DEFAULT_MAX_BLOCK_M = 64


@dataclass(frozen=True)
class TBMAParams:
    """Compile-time tuning knobs for the T-BMA builder."""

    max_w: int = DEFAULT_MAX_W
    max_q_v: int = DEFAULT_MAX_Q_V
    max_block_m: int = DEFAULT_MAX_BLOCK_M

    def validate(self) -> None:
        if self.max_w < 1:
            raise ValueError(f"max_w must be >= 1, got {self.max_w}")
        if self.max_q_v < 1:
            raise ValueError(f"max_q_v must be >= 1, got {self.max_q_v}")
        if self.max_block_m < 2:
            raise ValueError(
                f"max_block_m must be >= 2, got {self.max_block_m}")


@dataclass(frozen=True)
class LeadingAnchor:
    """Result of the leading-anchor rule when it succeeds.

    Fields:
        anchor_state: the TDFA state the gadget is rooted at. Always
            the TDFA's effective start control state.
        F:            the chosen anchor factor (always ``F_1``).
        V:            the preceding V segment (always ``V_0``).
        M_V:          ``V_0``'s residual sub-automaton.
        w_g:          probe offset ``min(|F_1| - 1, max_w)``.
    """

    anchor_state: int
    F: FSegment
    V: VSegment
    M_V: ResidualSubDFA
    w_g: int


def is_skip_enabled(v: VSegment) -> bool:
    """Is the V segment a "proper V" (non-epsilon)?

    Equivalent to "V has at least one non-epsilon string". For a
    VSegment produced by ``shape.decompose_pattern`` this is exactly
    ``not v.is_epsilon``.
    """
    return not v.is_epsilon


def select_leading_anchor(
        shape_seq: ShapeSeq,
        anchor_state: int,
        params: TBMAParams = TBMAParams(),
        ) -> Optional[LeadingAnchor]:
    """Run the leading-anchor rule on ``shape_seq``.

    Returns ``None`` when any gate fails. ``anchor_state`` is the TDFA
    start state the anchor should be rooted at; it is plumbed through
    so a future multi-site selector can reuse this function without
    refactoring.
    """
    params.validate()

    if shape_seq.k < 1:
        return None  # no F in the pattern

    v0, f1 = shape_seq.leading_pair()
    assert f1 is not None  # guaranteed by shape_seq.k >= 1

    # Gate 1: V_0 must be skip-enabled.
    if not is_skip_enabled(v0):
        return None

    # Gate 2: F_1 must have length >= 2.
    if f1.m < 2:
        return None

    # Gate 3: F_1's block width must not exceed the compile cap.
    if f1.m > params.max_block_m:
        return None

    # Build M_V_0.
    m_v = build_residual_subdfa(v0.ast_nodes)

    # Gate 4: V_0 state-count budget.
    if m_v.num_states > params.max_q_v:
        return None

    # Gate 5: Sigma_V_0 must cover all 256 bytes. This guarantees that
    # any byte the gadget skips during a BM shift is absorbable by
    # V_0, which makes the standard set-BM safe-shift over F_1.T
    # sound without an explicit phase-legal-shift table.
    # Restricted-Sigma_V patterns (match-mode ``(ab)*cde`` etc.) need
    # phase-aware BM and are deferred to a follow-up milestone. For
    # search-mode patterns the implicit ``.*?`` prefix unions
    # ``Sigma_V`` up to ``Sigma`` so this gate is essentially a no-op
    # there.
    if m_v.sigma != frozenset(range(256)):
        return None

    w_g = min(f1.m - 1, params.max_w)
    # Defensive: w_g must be >= 1 since |F_1| >= 2 and max_w >= 1.
    assert w_g >= 1

    return LeadingAnchor(
        anchor_state=anchor_state,
        F=f1,
        V=v0,
        M_V=m_v,
        w_g=w_g,
    )
