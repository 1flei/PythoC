"""
Compile-time control analysis for the T-BMA builder.

Provides per-control-state analysis on the TDFA control automaton:

  * ``compute_shortest_success_lengths(tdfa)`` -- minimum remaining
    real-byte length needed to reach acceptance from each state,
  * ``choose_block_len(...)`` -- simple block length heuristic,
  * ``ResidualSubDFA`` / ``build_residual_subdfa`` -- V-segment
    residual sub-automaton used by the leading-anchor selector,
  * ``SetBMOverlap`` / ``compute_set_bm_overlap`` -- set-BM overlap
    table used by F-segment safe-shift computations.

Everything here is TDFA-only; there is no NFA/DFA fallback.  The
residual sub-DFA builder reuses the TDFA frontend so V segments share
the same tagged control automaton as the rest of the pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .parse import Concat
from .tdfa import TDFA, build_tdfa
from .tnfa import build_tnfa


DEFAULT_OPCS_MAX_W = 8


def _terminal_control_states(dfa: TDFA) -> FrozenSet[int]:
    """Control states that can accept without consuming another real byte."""
    terminals: Set[int] = set(dfa.accept_states)
    if dfa.sentinel_257_class >= 0:
        for state in range(dfa.num_states):
            if dfa.transitions[state][dfa.sentinel_257_class] in dfa.accept_states:
                terminals.add(state)
    return frozenset(terminals)


def _real_successors(dfa: TDFA, state: int) -> FrozenSet[int]:
    """Unique real-byte successors from one control state."""
    targets: Set[int] = set()
    for byte_val in range(256):
        targets.add(dfa.transitions[state][dfa.class_map[byte_val]])
    return frozenset(targets)


def compute_shortest_success_lengths(dfa: TDFA) -> Dict[int, Optional[int]]:
    """Minimum remaining real-byte length needed to reach acceptance.

    A value of ``None`` means the control state cannot reach acceptance.
    End-of-input acceptance via sentinel ``$`` counts as length ``0``.
    """
    terminals = _terminal_control_states(dfa)
    reverse_edges: Dict[int, Set[int]] = {
        state: set()
        for state in range(dfa.num_states)
    }

    for state in range(dfa.num_states):
        for target in _real_successors(dfa, state):
            reverse_edges[target].add(state)

    dist: Dict[int, int] = {
        state: 0
        for state in terminals
    }
    queue = deque(terminals)

    while queue:
        cur = queue.popleft()
        next_dist = dist[cur] + 1
        for prev in reverse_edges[cur]:
            old = dist.get(prev)
            if old is None or next_dist < old:
                dist[prev] = next_dist
                queue.append(prev)

    return {
        state: dist.get(state)
        for state in range(dfa.num_states)
    }


def choose_block_len(shortest_success_len: Optional[int],
                     max_w: int = DEFAULT_OPCS_MAX_W) -> int:
    """Choose the local OPCS block length from a shortest-success estimate."""
    if shortest_success_len is None or shortest_success_len <= 1:
        return 0
    return min(shortest_success_len, max_w + 1)


# ---------------------------------------------------------------------------
# Residual sub-DFA for a V segment (T-BMA MVP, doc section on nullable V)
# ---------------------------------------------------------------------------

# A sentinel for "no legal transition" in a byte-indexed residual
# transition table. Any byte whose residual transition is DEAD
# represents "V cannot legally consume this byte from this phase".
_RESIDUAL_DEAD = -1


@dataclass(frozen=True)
class ResidualSubDFA:
    """Residual sub-automaton ``M_V`` for a V segment.

    Fields:
        num_states:  number of live (non-dead) phases in ``Q_V``. The
            phases are labeled ``0..num_states-1``; the start phase is
            always ``0`` by construction. A dead phase (byte not in
            ``Sigma_V``) is represented as ``-1`` in ``transitions``.
        start:       the start phase (always ``0``).
        accepting:   V-accepting phases ``A_V`` (phases where V's
            residual language accepts empty and the engine may close V).
        sigma:       global ``Sigma_V`` -- bytes that have at least one
            live outgoing transition from some phase.
        transitions: ``transitions[phase][byte] -> phase | -1``;
            ``-1`` means "byte kills V from this phase".
        is_trivial:  ``num_states == 1``. Lets the gadget builder
            collapse the V-phase product dimension.
    """

    num_states: int
    start: int
    accepting: FrozenSet[int]
    sigma: FrozenSet[int]
    transitions: Tuple[Tuple[int, ...], ...]
    is_trivial: bool

    def step(self, phase: int, byte_val: int) -> int:
        """One byte step from ``phase``. Returns ``-1`` if dead."""
        return self.transitions[phase][byte_val]

    @classmethod
    def epsilon(cls) -> "ResidualSubDFA":
        """The placeholder ``V_0({epsilon})`` residual: one trivial phase."""
        return cls(
            num_states=1,
            start=0,
            accepting=frozenset({0}),
            sigma=frozenset(),
            transitions=(tuple([_RESIDUAL_DEAD] * 256),),
            is_trivial=True,
        )


def _minimize_byte_dfa(
        num_states: int,
        start: int,
        accepting: FrozenSet[int],
        transitions: Sequence[Sequence[int]],  # [state][byte] -> state | DEAD
        ) -> Tuple[int, int, FrozenSet[int], Tuple[Tuple[int, ...], ...]]:
    """Minimize a byte-indexed DFA (values in [-1, num_states-1]).

    Uses straightforward partition refinement. For the residual sub-DFA
    this runs on DFAs with ``num_states`` bounded by the AST size of a V
    segment (typically < 10), so the naive O(n^2 * |Sigma|) behavior is
    fine.

    The dead sink is encoded as ``-1``; it participates in the partition
    refinement as a virtual extra class so equivalence classes correctly
    distinguish "goes to dead on byte b" vs "goes live".

    Returns ``(new_num_states, new_start, new_accepting, new_transitions)``
    renumbered so that ``new_start == 0`` (if the input start maps to a
    non-dead class).
    """
    DEAD = _RESIDUAL_DEAD
    # Initial partition: accepting vs non-accepting (ignoring dead).
    partition: Dict[int, int] = {}
    for s in range(num_states):
        partition[s] = 0 if s in accepting else 1

    def target_class(state: int, byte_val: int) -> int:
        t = transitions[state][byte_val]
        if t == DEAD:
            return -1
        return partition[t]

    while True:
        changed = False
        new_partition: Dict[int, int] = {}
        next_id = 0
        signature_to_id: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        for s in range(num_states):
            sig = (partition[s],
                   tuple(target_class(s, b) for b in range(256)))
            pid = signature_to_id.get(sig)
            if pid is None:
                pid = next_id
                next_id += 1
                signature_to_id[sig] = pid
            new_partition[s] = pid
            if pid != partition[s]:
                changed = True
        if not changed:
            break
        partition = new_partition

    # Build minimized DFA. Canonicalize numbering so the start's class
    # becomes 0.
    class_ids = sorted(set(partition.values()))
    start_class = partition[start]
    ordered = [start_class] + [c for c in class_ids if c != start_class]
    remap: Dict[int, int] = {cid: new_id for new_id, cid in enumerate(ordered)}

    new_num = len(ordered)
    rep_state: Dict[int, int] = {}
    for s, cid in partition.items():
        rep_state.setdefault(cid, s)

    new_transitions = [[DEAD] * 256 for _ in range(new_num)]
    for new_id, cid in enumerate(ordered):
        rep = rep_state[cid]
        for b in range(256):
            t = transitions[rep][b]
            if t == DEAD:
                continue
            new_transitions[new_id][b] = remap[partition[t]]

    new_accepting = frozenset(
        remap[partition[rep_state[cid]]]
        for cid in ordered
        if rep_state[cid] in accepting
    )
    return (
        new_num,
        0,
        new_accepting,
        tuple(tuple(row) for row in new_transitions),
    )


def _project_tdfa_to_byte_dfa(
        dfa: TDFA,
        ) -> Tuple[int, int, FrozenSet[int], List[List[int]]]:
    """Project a TDFA onto a live-only, byte-indexed DFA.

    Returns ``(num_live, start_live_id, accepting, byte_transitions)``
    where sentinel classes are stripped and the dead control state is
    collapsed into the ``_RESIDUAL_DEAD`` sink.
    """
    # Walk from the TDFA's effective start over real-byte classes only.
    if dfa.sentinel_256_class >= 0:
        start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]
    else:
        start = dfa.start_state

    live_order: List[int] = []
    seen: Set[int] = set()
    queue: deque = deque([start])
    seen.add(start)
    while queue:
        s = queue.popleft()
        if s == dfa.dead_state:
            continue
        live_order.append(s)
        for b in range(256):
            cls = dfa.class_map[b]
            t = dfa.transitions[s][cls]
            if t != dfa.dead_state and t not in seen:
                seen.add(t)
                queue.append(t)

    order: Dict[int, int] = {s: i for i, s in enumerate(live_order)}
    num_live = len(live_order)

    raw_transitions: List[List[int]] = [
        [_RESIDUAL_DEAD] * 256 for _ in range(num_live)
    ]
    for new_id, old_id in enumerate(live_order):
        for b in range(256):
            cls = dfa.class_map[b]
            t_old = dfa.transitions[old_id][cls]
            if t_old == dfa.dead_state:
                continue
            raw_transitions[new_id][b] = order[t_old]

    raw_accepting = frozenset(
        order[s] for s in dfa.accept_states if s in order
    )

    return num_live, 0, raw_accepting, raw_transitions


def build_residual_subdfa(v_ast_nodes: Sequence[object]) -> ResidualSubDFA:
    """Build the residual sub-automaton ``M_V`` for one V segment.

    The input is the list of AST nodes that make up the V segment (per
    ``shape.VSegment.ast_nodes``). The resulting M_V is an independent
    mini-DFA over bytes, with phases labeled from ``0`` (start) and a
    dead sentinel for out-of-V bytes. The DFA is minimized before being
    returned so that the ``is_trivial`` fast-path catches ``.*`` /
    ``[a-z]*`` etc.

    Any V whose language is exactly ``{epsilon}`` (no byte-consuming
    node, or no nodes at all) returns the trivial one-phase residual
    via ``ResidualSubDFA.epsilon()``.
    """
    if not v_ast_nodes:
        return ResidualSubDFA.epsilon()

    v_concat = Concat(list(v_ast_nodes))
    v_tdfa = build_tdfa(build_tnfa(v_concat))

    num_live, start, raw_accepting, raw_transitions = (
        _project_tdfa_to_byte_dfa(v_tdfa)
    )
    if num_live == 0:
        # Degenerate: V's start is already the dead control state.  Fall
        # back to the epsilon placeholder so anchor selection is safe.
        return ResidualSubDFA.epsilon()

    (num_states, start,
     accepting, transitions) = _minimize_byte_dfa(
        num_states=num_live,
        start=start,
        accepting=raw_accepting,
        transitions=raw_transitions,
    )

    sigma_bytes: Set[int] = set()
    for row in transitions:
        for b in range(256):
            if row[b] != _RESIDUAL_DEAD:
                sigma_bytes.add(b)

    return ResidualSubDFA(
        num_states=num_states,
        start=start,
        accepting=accepting,
        sigma=frozenset(sigma_bytes),
        transitions=transitions,
        is_trivial=(num_states == 1),
    )


# ---------------------------------------------------------------------------
# Set-BM overlap for an F segment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetBMOverlap:
    """Pre-computed helpers for set-BM safe shifts over ``F_m(T)``.

    The one-probe-per-byte invariant of T-BMA requires that on a
    mismatch inside a block, we shift by the smallest amount delta >= 1
    such that **every** byte we have already probed and kept inside the
    post-shift window remains consistent with at least one candidate
    ``u in T`` at the shifted alignment. This helper packages the two
    ingredients needed to compute that shift per gadget edge:

    - ``hit_at(p, a)`` / ``refine_set(T', p, a)``: detect whether a
      probed byte is still consistent at position ``p`` given the
      current candidate set ``T'``, and narrow ``T'`` on a hit.
    - ``safe_shift(known)``: smallest delta such that after shifting,
      every known byte that remains inside the post-shift block is
      consistent with some ``u in T`` at the new alignment; or ``m``
      (full block advance) if no smaller delta works.

    Instantiating this dataclass once per F is the set-BM overlap table
    step of the T-BMA MVP. Per-gadget-edge queries then go through
    ``safe_shift``.
    """

    m: int
    T: FrozenSet[Tuple[int, ...]]
    # columns[p] = { u[p] : u in T } -- set of bytes legal at offset p.
    columns: Tuple[FrozenSet[int], ...]

    def hit_at(self, p: int, byte_val: int) -> bool:
        """Is byte ``byte_val`` consistent with ``T`` at offset ``p``?"""
        return 0 <= p < self.m and byte_val in self.columns[p]

    def refine_set(self,
                   T_current: FrozenSet[Tuple[int, ...]],
                   p: int,
                   byte_val: int) -> FrozenSet[Tuple[int, ...]]:
        """``T' = { u in T_current : u[p] == byte_val }``."""
        return frozenset(u for u in T_current if u[p] == byte_val)

    def safe_shift(self, known: Dict[int, int]) -> int:
        """Smallest delta >= 1 that preserves the overlap invariant.

        ``known`` maps offset ``p in [0..m-1]`` -> byte ``a`` for each
        byte we have probed at absolute position ``cursor + p``.
        Returns the smallest delta in [1..m] such that for every
        ``(p, a) in known`` either ``p - delta < 0`` (probed byte falls
        before the new cursor and is irrelevant) or
        ``a in columns[p-delta]`` (probed byte still consistent at the
        shifted alignment).  delta = m (full block advance) is always a
        safe default.

        O(m * |known|) per query. With m <= max_w <= 16 this is well
        within the compile budget.
        """
        if self.m == 0:
            return 1
        if not known:
            return self.m
        for delta in range(1, self.m + 1):
            ok = True
            for p, a in known.items():
                new_p = p - delta
                if new_p < 0:
                    continue
                if new_p >= self.m or a not in self.columns[new_p]:
                    ok = False
                    break
            if ok:
                return delta
        return self.m


def compute_set_bm_overlap(T: FrozenSet[Tuple[int, ...]]) -> SetBMOverlap:
    """Compile-time set-BM overlap-table construction for ``F_m(T)``."""
    if not T:
        raise ValueError("compute_set_bm_overlap requires |T| >= 1")
    m = len(next(iter(T)))
    for u in T:
        if len(u) != m:
            raise ValueError(
                f"compute_set_bm_overlap: inconsistent lengths in T "
                f"(expected {m}, saw {len(u)})"
            )
    columns: List[Set[int]] = [set() for _ in range(m)]
    for u in T:
        for p, a in enumerate(u):
            columns[p].add(a)
    return SetBMOverlap(
        m=m,
        T=T,
        columns=tuple(frozenset(c) for c in columns),
    )
