"""
T-BMA (Tagged Boyer-Moore Automaton) builder.

Implements the leading-anchor T-BMA on top of the TDFA control
automaton.

Pipeline
--------
1. ``shape.decompose_pattern(ast)`` produces a ``ShapeSeq``.
2. ``anchor_selection.select_leading_anchor(shape_seq, q_0, params)``
   returns a ``LeadingAnchor`` or ``None``. The selection enforces
   four shape gates plus a ``Sigma_V_0 == Sigma`` gate (so naive
   set-BM shift is sound; restricted-Sigma_V patterns such as
   match-mode ``(ab)*cde`` need phase-aware BM and are deferred).
3. When the rule fires, this module installs **one** block-local
   gadget rooted at ``q_0``. Cover paths come directly from ``F_1.T``
   (one path per byte string in the leading F segment), not from
   TDFA-path enumeration. The block length is ``|F_1|``, the root
   probe offset is ``min(|F_1| - 1, max_w)``.
4. When the rule does not fire, the builder emits a pure TDFA T-BMA
   (consume-1 at every state, no gadget).

The artifact produced is the ``opcs.TaggedOPCSBMA`` consumed by
codegen / runtime. Control input is always a :class:`TDFA` -- tag
commands live directly on its edges, so the builder does not need a
separate tag-runtime side-table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .analysis import (
    SetBMOverlap,
    compute_set_bm_overlap,
    compute_shortest_success_lengths,
)
from .anchor_selection import LeadingAnchor, TBMAParams, select_leading_anchor
from .opcs import (
    OPCSCommand,
    OPCSEdge,
    OPCSState,
    TaggedOPCSBMA,
    _artifact_layout,
    _commands_to_phi_transform,
    _compose_phi_transforms,
    _control_accept_commands,
    _control_eof_commands,
    _control_path_commands,
    _control_step_commands,
    _effective_start_control,
    _group_control_edges,
    _materialize_known_bytes,
    _phi_transform_to_commands,
    _real_class_layout,
    _shift_commands,
    _state_name,
)
from .shape import FSegment, ShapeSeq, decompose_pattern
from .tdfa import TDFA


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------

# Empty interval sentinel (matches the convention used in ``opcs.py``).
_EMPTY_INTERVAL: Tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class _Cover:
    """Multi-path cover for a leading-anchor gadget.

    ``paths[i]`` is the class sequence (one class id per offset) for
    the i-th string in the F segment's ``T``. ``bytes_paths[i]`` is
    the matching tuple of representative bytes — kept around so the
    block-success commands can be derived via the existing
    ``_control_path_commands`` helper. ``classes_at[p]`` is the union
    of allowed class ids at offset ``p`` over **all** paths (used by
    ``SetBMOverlap`` style queries that don't need a per-mask
    refinement).
    """

    block_len: int
    paths: Tuple[Tuple[int, ...], ...]
    bytes_paths: Tuple[Tuple[int, ...], ...]
    classes_at: Tuple[FrozenSet[int], ...]
    overlap: SetBMOverlap

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    def full_mask(self) -> int:
        return (1 << self.num_paths) - 1

    def mask_classes_at(self, mask: int, pos: int) -> FrozenSet[int]:
        if mask == self.full_mask():
            return self.classes_at[pos]
        out: List[int] = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                out.append(self.paths[idx][pos])
            m >>= 1
            idx += 1
        return frozenset(out)

    def known_class_at(self, mask: int, pos: int) -> Optional[int]:
        """If every path in ``mask`` agrees at ``pos``, return that class."""
        classes = self.mask_classes_at(mask, pos)
        if len(classes) == 1:
            return next(iter(classes))
        return None

    def refine_mask(self, mask: int, pos: int, class_id: int) -> int:
        out = 0
        m = mask
        idx = 0
        while m:
            if m & 1 and self.paths[idx][pos] == class_id:
                out |= 1 << idx
            m >>= 1
            idx += 1
        return out

    def single_path_index(self, mask: int) -> Optional[int]:
        if mask == 0 or (mask & (mask - 1)) != 0:
            return None
        return mask.bit_length() - 1

    def known_bytes_for_interval(self,
                                 interval: Tuple[int, int],
                                 mask: int) -> Tuple[int, ...]:
        """Materialize ``known_bytes`` for a gadget state.

        Slot ``-1`` means "not yet probed at this offset"; otherwise
        the slot holds the representative byte for the agreed-upon
        class (single representative byte per path is fine because
        all paths in the mask share the same class at each known
        offset within ``interval``).
        """
        out = [-1] * self.block_len
        if interval[0] < 0:
            return tuple(out)
        # Pick any path in the mask; all paths in the mask agree at
        # every offset inside the interval by construction.
        idx = (mask & -mask).bit_length() - 1
        path = self.bytes_paths[idx]
        for p in range(interval[0], interval[1] + 1):
            out[p] = path[p]
        return tuple(out)


@dataclass
class _GadgetBranch:
    """One branch of a gadget probe state."""

    class_ids: Tuple[int, ...]
    target_state_id: int
    shift: int
    commands: Tuple[OPCSCommand, ...]


@dataclass
class _GadgetState:
    state_id: int
    interval: Tuple[int, int]
    path_mask: int
    probe_offset: int
    branches: List[_GadgetBranch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cover construction
# ---------------------------------------------------------------------------

def _build_cover(dfa: TDFA,
                 anchor: LeadingAnchor) -> _Cover:
    """Build the leading-anchor cover from ``F_1.T``.

    Each byte string ``u in T`` becomes one cover path. The class id
    used by the OPCS runtime is ``dfa.class_map[byte]`` for each
    byte. Two distinct byte strings that map to the same class
    sequence collapse to one path here -- at runtime they are
    indistinguishable anyway, and the block-success commands are
    derived from the class sequence (which is shared).
    """

    f: FSegment = anchor.F
    seen_class_paths: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for byte_seq in sorted(f.T):
        class_seq = tuple(dfa.class_map[b] for b in byte_seq)
        if class_seq in seen_class_paths:
            continue
        seen_class_paths[class_seq] = byte_seq

    paths = tuple(seen_class_paths.keys())
    bytes_paths = tuple(seen_class_paths[k] for k in paths)
    classes_at = tuple(
        frozenset(p[pos] for p in paths)
        for pos in range(f.m)
    )
    overlap = compute_set_bm_overlap(f.T)
    return _Cover(
        block_len=f.m,
        paths=paths,
        bytes_paths=bytes_paths,
        classes_at=classes_at,
        overlap=overlap,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_tagged_tbma(dfa: TDFA,
                      shape_seq: Optional[ShapeSeq] = None,
                      params: TBMAParams = TBMAParams(),
                      ) -> TaggedOPCSBMA:
    """Build a T-BMA artifact from one TDFA control automaton.

    When ``shape_seq`` is supplied and the leading-anchor rule fires,
    a single gadget is installed at ``dfa.start_state`` (= q_0).
    Otherwise the artifact is a pure-TDFA T-BMA (one shadow + one
    runtime state per control state, ``w = 0`` consume-1
    transitions, no gadget states), which is what the existing
    runtime expects when no acceleration is available.
    """

    effective_start = _effective_start_control(dfa)
    anchor: Optional[LeadingAnchor] = None
    if shape_seq is not None:
        # Anchor is rooted at the **runtime** entry point -- i.e.
        # ``effective_start_control``, which is what the runtime
        # actually jumps to first. ``dfa.start_state`` may be a
        # sentinel-256 prelude state that doesn't see real bytes
        # (search-rewrite TDFAs are like this).
        anchor = select_leading_anchor(
            shape_seq, anchor_state=effective_start, params=params)

    return _emit(dfa, anchor, effective_start)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _emit(dfa: TDFA,
          anchor: Optional[LeadingAnchor],
          effective_start: int) -> TaggedOPCSBMA:
    """Common emission path for both the gadget and pure-TDFA cases."""

    shortest_success_len = compute_shortest_success_lengths(dfa)
    real_class_ids, class_representatives, class_bytes = _real_class_layout(dfa)
    (tag_names,
     public_tag_names,
     tag_slots,
     register_count,
     output_registers) = _artifact_layout(dfa)

    # Reserve state ids: 0..N-1 = shadow, N..2N-1 = runtime, then gadget.
    n = dfa.num_states
    shadow_ids = {s: s for s in range(n)}
    runtime_ids = {s: n + s for s in range(n)}

    # Build gadget (if any) before deciding control_entry_states, so
    # we know whether to splice q_0's entry to the gadget root.
    gadget_states: List[_GadgetState] = []
    gadget_root_id: Optional[int] = None
    if anchor is not None:
        gadget_states, gadget_root_id = _build_gadget(
            dfa=dfa,
            anchor=anchor,
            real_class_ids=tuple(real_class_ids),
            runtime_ids=runtime_ids,
            base_state_id=2 * n,
        )

    control_entry_states: Dict[int, int] = dict(runtime_ids)
    if anchor is not None and gadget_root_id is not None:
        control_entry_states[anchor.anchor_state] = gadget_root_id

    total_states = 2 * n + len(gadget_states)
    states: List[Optional[OPCSState]] = [None] * total_states

    for control_state in range(n):
        is_accept = control_state in dfa.accept_states
        eof_accept = (
            dfa.sentinel_257_class >= 0 and
            dfa.transitions[control_state][dfa.sentinel_257_class]
            in dfa.accept_states
        )
        accept_commands = _control_accept_commands(dfa, control_state)
        eof_commands = _control_eof_commands(dfa, control_state)

        shadow_edges = _group_control_edges(
            dfa,
            control_state,
            lambda next_control: shadow_ids[next_control],
        )
        states[shadow_ids[control_state]] = OPCSState(
            id=shadow_ids[control_state],
            name=f"shadow_q{control_state}",
            kind="control_shadow",
            control_state=control_state,
            probe_offset=0,
            block_len=1,
            known_bytes=(),
            possible_targets=(control_state,),
            edges=shadow_edges,
            accepting=is_accept,
            accept_commands=(),
            eof_accept=eof_accept,
            eof_commands=(),
        )

        runtime_edges = _group_control_edges(
            dfa,
            control_state,
            lambda next_control: control_entry_states[next_control],
        )
        states[runtime_ids[control_state]] = OPCSState(
            id=runtime_ids[control_state],
            name=f"q{control_state}",
            kind="control",
            control_state=control_state,
            probe_offset=0,
            block_len=1,
            known_bytes=(),
            possible_targets=(control_state,),
            edges=runtime_edges,
            accepting=is_accept,
            accept_commands=accept_commands,
            eof_accept=eof_accept,
            eof_commands=eof_commands,
        )

    if anchor is not None:
        cover = _build_cover(dfa, anchor)
        for g in gadget_states:
            edges = _materialize_gadget_edges(g, class_bytes)
            known_bytes = cover.known_bytes_for_interval(
                g.interval, g.path_mask)
            states[g.state_id] = OPCSState(
                id=g.state_id,
                name=_state_name(anchor.anchor_state, known_bytes),
                kind="gadget",
                control_state=anchor.anchor_state,
                probe_offset=g.probe_offset,
                block_len=cover.block_len,
                known_bytes=known_bytes,
                possible_targets=(anchor.anchor_state,),
                edges=edges,
                accepting=False,
                accept_commands=(),
                eof_accept=False,
                eof_commands=(),
            )

    assert all(state is not None for state in states), (
        "internal error: not all T-BMA states were materialized"
    )

    initial_commands = _control_step_commands(
        dfa,
        dfa.start_state,
        dfa.sentinel_256_class,
    ) if dfa.sentinel_256_class >= 0 else ()
    initial_accept_commands = _control_accept_commands(
        dfa,
        effective_start,
    ) if effective_start in dfa.accept_states else ()

    accept_states = frozenset(
        runtime_ids[state] for state in dfa.accept_states
    )

    return TaggedOPCSBMA(
        states=tuple(states),  # type: ignore[arg-type]
        start_state=control_entry_states[effective_start],
        dead_state=runtime_ids[dfa.dead_state],
        shadow_dead_state=shadow_ids[dfa.dead_state],
        effective_start_control=effective_start,
        runtime_control_states=runtime_ids,
        shadow_control_states=shadow_ids,
        control_entry_states=control_entry_states,
        accept_states=accept_states,
        tag_names=tag_names,
        public_tag_names=public_tag_names,
        tag_slots=tag_slots,
        register_count=register_count,
        output_registers=output_registers,
        shortest_success_len=shortest_success_len,
        initial_commands=initial_commands,
        initial_accepting=effective_start in dfa.accept_states,
        initial_accept_commands=initial_accept_commands,
    )


# ---------------------------------------------------------------------------
# Gadget construction
# ---------------------------------------------------------------------------

def _build_gadget(*,
                  dfa: TDFA,
                  anchor: LeadingAnchor,
                  real_class_ids: Tuple[int, ...],
                  runtime_ids: Dict[int, int],
                  base_state_id: int,
                  ) -> Tuple[List[_GadgetState], Optional[int]]:
    """Build the leading-anchor gadget rooted at ``q_0``.

    Returns ``(gadget_states, root_state_id)``. ``root_state_id`` is
    the state id used as the q_0 entry point (``Rep(q_0)`` in the
    doc's notation).
    """

    cover = _build_cover(dfa, anchor)
    block_len = cover.block_len
    if cover.num_paths == 0 or block_len <= 1:
        return [], None

    q0 = anchor.anchor_state
    full_mask = cover.full_mask()
    w_g = anchor.w_g

    states_by_key: Dict[Tuple[Tuple[int, int], int], _GadgetState] = {}
    expanded: set = set()
    pending: List[Tuple[Tuple[int, int], int]] = []
    next_id = base_state_id

    def ensure_state(interval: Tuple[int, int], mask: int) -> int:
        nonlocal next_id
        key = (interval, mask)
        gs = states_by_key.get(key)
        if gs is None:
            probe = _choose_probe_offset(interval, block_len, w_g)
            gs = _GadgetState(
                state_id=next_id,
                interval=interval,
                path_mask=mask,
                probe_offset=probe if probe is not None else 0,
            )
            states_by_key[key] = gs
            next_id += 1
            pending.append(key)
        return gs.state_id

    root_key = (_EMPTY_INTERVAL, full_mask)
    root_id = ensure_state(*root_key)

    while pending:
        key = pending.pop()
        if key in expanded:
            continue
        expanded.add(key)
        gs = states_by_key[key]
        interval, mask = gs.interval, gs.path_mask
        probe = _choose_probe_offset(interval, block_len, w_g)
        if probe is None:
            # Should never happen -- caller never enqueues full
            # intervals (block-success is emitted as an edge, not as
            # a state). Defensive no-op so a buggy enqueue doesn't
            # corrupt the artifact.
            continue
        gs.probe_offset = probe

        new_branches: Dict[
            Tuple[int, int, Tuple[OPCSCommand, ...]],
            List[int],
        ] = {}

        for class_id in real_class_ids:
            target_state_id, shift, commands, child_key = _resolve_branch(
                dfa=dfa,
                anchor=anchor,
                cover=cover,
                interval=interval,
                mask=mask,
                probe=probe,
                class_id=class_id,
                real_class_ids=real_class_ids,
                runtime_ids=runtime_ids,
            )
            if child_key is not None:
                target_state_id = ensure_state(*child_key)
            new_branches.setdefault(
                (target_state_id, shift, commands), []
            ).append(class_id)

        gs.branches = [
            _GadgetBranch(
                class_ids=tuple(sorted(class_ids)),
                target_state_id=target_state_id,
                shift=shift,
                commands=commands,
            )
            for (target_state_id, shift, commands), class_ids
            in sorted(new_branches.items(), key=lambda item: item[1][0])
        ]

    # Materialize in id order so the final ``states[]`` slot
    # assignment in ``_emit`` is deterministic.
    gadget_states = sorted(states_by_key.values(), key=lambda s: s.state_id)
    return gadget_states, root_id


def _choose_probe_offset(interval: Tuple[int, int],
                         block_len: int,
                         w_g: int) -> Optional[int]:
    """Pick the next probe offset given the current ``interval``.

    Empty interval → root probe at ``w_g``. Otherwise, adjacent
    extension on the side closest to the block's center (mirrors the
    legacy ``_choose_probe_offset`` heuristic so gadget shapes are
    comparable).
    """
    if interval[0] < 0:
        return w_g
    left, right = interval
    candidates: List[int] = []
    if left > 0:
        candidates.append(left - 1)
    if right + 1 < block_len:
        candidates.append(right + 1)
    if not candidates:
        return None
    center = (block_len - 1) / 2.0
    return min(candidates, key=lambda pos: (abs(pos - center), pos))


def _resolve_branch(*,
                    dfa: TDFA,
                    anchor: LeadingAnchor,
                    cover: _Cover,
                    interval: Tuple[int, int],
                    mask: int,
                    probe: int,
                    class_id: int,
                    real_class_ids: Tuple[int, ...],
                    runtime_ids: Dict[int, int],
                    ) -> Tuple[int, int, Tuple[OPCSCommand, ...],
                               Optional[Tuple[Tuple[int, int], int]]]:
    """Compute the branch outcome for ``class_id`` probed at ``probe``.

    Returns ``(target_state_id, shift, commands, child_gadget_key)``.
    ``child_gadget_key == (interval, mask)`` if the branch lands on
    another gadget state (so the caller can allocate it lazily);
    ``None`` if the branch lands on a runtime control state.
    """

    block_len = cover.block_len
    q0 = anchor.anchor_state

    # ----- match path -------------------------------------------------------
    new_mask = cover.refine_mask(mask, probe, class_id)
    if new_mask != 0:
        new_interval = _expand_interval(interval, probe)
        if new_interval[0] == 0 and new_interval[1] == block_len - 1:
            # block-success -- the surviving mask is necessarily a
            # singleton (every offset has been probed and pinned to
            # the path's class).
            path_idx = cover.single_path_index(new_mask)
            assert path_idx is not None, (
                "block-success mask must be a singleton path; "
                f"got mask={new_mask:b}"
            )
            class_seq = cover.paths[path_idx]
            target_control = _simulate_path_target(dfa, q0, class_seq)
            commands = _control_path_commands(dfa, q0, class_seq)
            return (
                runtime_ids[target_control],
                block_len,
                commands,
                None,
            )
        # adjacent-extension: stay in gadget, shift 0
        return (
            -1,  # placeholder; caller allocates the child state id
            0,
            (),
            (new_interval, new_mask),
        )

    # ----- mismatch path ----------------------------------------------------
    return _resolve_bm_shift(
        dfa=dfa,
        anchor=anchor,
        cover=cover,
        interval=interval,
        mask=mask,
        probe=probe,
        class_id=class_id,
        real_class_ids=real_class_ids,
        runtime_ids=runtime_ids,
    )


def _resolve_bm_shift(*,
                      dfa: TDFA,
                      anchor: LeadingAnchor,
                      cover: _Cover,
                      interval: Tuple[int, int],
                      mask: int,
                      probe: int,
                      class_id: int,
                      real_class_ids: Tuple[int, ...],
                      runtime_ids: Dict[int, int],
                      ) -> Tuple[int, int, Tuple[OPCSCommand, ...],
                                 Optional[Tuple[Tuple[int, int], int]]]:
    """Compute the BM shift for a mismatch.

    For every candidate ``delta`` in ``[safe_shift, block_len]`` we
    both (a) compute the retained interval/mask after the shift and
    (b) walk the TDFA ``delta`` steps from ``q_0`` with positional
    class constraints to collect the unambiguous tag-write transform.
    We then keep the (target, retained, commands) triple with the
    best ``(retained_count, delta)`` score for which both checks
    succeeded.

    If no ``delta`` admits an unambiguous prefix walk -- typically
    because the V_0 sub-DFA has rich internal structure that makes
    the multi-byte skip non-deterministic w.r.t. tag writes -- we
    fall back to ``shift = 0`` routed at the q_0 runtime state. The
    runtime then consumes one byte normally with full TDFA tag
    semantics. Functionally this collapses to pure-TDFA on hot
    bytes, which is the same trade-off the legacy OPCS-BMA makes
    when its prefix walk is ambiguous.
    """

    block_len = cover.block_len
    q0 = anchor.anchor_state
    full_mask = cover.full_mask()

    # Class-domain known map for SetBMOverlap and the prefix walk.
    known_classes: Dict[int, int] = {}
    if interval[0] >= 0:
        for p in range(interval[0], interval[1] + 1):
            cls = cover.known_class_at(mask, p)
            assert cls is not None, (
                "in-interval position must have an agreed class"
            )
            known_classes[p] = cls
    known_classes[probe] = class_id

    byte_known: Dict[int, int] = {}
    if interval[0] >= 0:
        path_idx_any = (mask & -mask).bit_length() - 1
        anchor_path_bytes = cover.bytes_paths[path_idx_any]
        for p in range(interval[0], interval[1] + 1):
            byte_known[p] = anchor_path_bytes[p]
    byte_known[probe] = _representative_byte_for_class(
        dfa, class_id, cover, probe)

    min_delta = cover.overlap.safe_shift(byte_known)
    if min_delta < 1:
        min_delta = 1
    if min_delta > block_len:
        min_delta = block_len

    best: Optional[Tuple[int, Tuple[int, int], int,
                         Tuple[OPCSCommand, ...]]] = None
    best_score: Optional[Tuple[int, int]] = None
    for delta in range(min_delta, block_len + 1):
        outcome = _compute_unique_prefix_outcome(
            dfa=dfa,
            start_state=q0,
            real_class_ids=real_class_ids,
            cover=cover,
            interval=interval,
            mask=mask,
            probe=probe,
            probe_class=class_id,
            delta=delta,
        )
        if outcome is None:
            continue
        target_control, commands = outcome

        retained_interval, retained_mask = _retained_after_shift(
            cover=cover,
            interval=interval,
            mask=mask,
            probe=probe,
            probe_class=class_id,
            delta=delta,
        )
        # Only retain when we'll re-enter the gadget root (q_0). Any
        # other landing point is incompatible with the gadget's
        # invariant that the cover applies at the new cursor.
        if target_control != q0:
            retained_interval = _EMPTY_INTERVAL
            retained_mask = full_mask
        retained_count = (
            0 if retained_interval[0] < 0
            else retained_interval[1] - retained_interval[0] + 1
        )
        score = (retained_count, delta)
        if best_score is None or score > best_score:
            best = (target_control, retained_interval, retained_mask,
                    delta, commands)
            best_score = score

    if best is None:
        # Conservative fallback: rejoin runtime q_0 without advancing
        # the cursor; let the DFA consume one byte with full tag
        # semantics. ``shift == 0`` + control runtime target is the
        # same idiom the legacy OPCS-BMA uses for ambiguous prefixes.
        return (
            runtime_ids[q0],
            0,
            (),
            None,
        )

    target_control, retained_interval, retained_mask, delta, commands = best

    if target_control != q0:
        # Walk lands somewhere other than q_0 -- exit the gadget into
        # that runtime control state. No retained interval applies.
        return (
            runtime_ids[target_control],
            delta,
            commands,
            None,
        )

    if retained_interval[0] < 0 or retained_mask == 0:
        return (
            -1,
            delta,
            commands,
            (_EMPTY_INTERVAL, full_mask),
        )
    return (
        -1,
        delta,
        commands,
        (retained_interval, retained_mask),
    )


_COMMAND_CONFLICT = object()


def _compute_unique_prefix_outcome(*,
                                   dfa: TDFA,
                                   start_state: int,
                                   real_class_ids: Tuple[int, ...],
                                   cover: _Cover,
                                   interval: Tuple[int, int],
                                   mask: int,
                                   probe: int,
                                   probe_class: int,
                                   delta: int,
                                   ) -> Optional[Tuple[int, Tuple[OPCSCommand, ...]]]:
    """Walk ``dfa`` ``delta`` steps from ``start_state`` with constraints.

    At source-coordinate position ``pos`` the byte's class is forced
    when ``pos`` lies inside ``interval`` (the agreed class within
    ``mask``) or when ``pos == probe`` (the probed class). Otherwise
    every real class is explored. The walk's frontier is a map
    ``state -> phi-transform`` accumulated via the OPCS command
    composition helpers; if the final frontier collapses to a single
    state with a single transform we return ``(target, commands)``,
    otherwise ``None``.
    """

    if delta <= 0:
        return (start_state, ())

    frontier: Dict[int, object] = {start_state: ()}
    for pos in range(delta):
        forced_class: Optional[int] = None
        if interval[0] >= 0 and interval[0] <= pos <= interval[1]:
            forced_class = cover.known_class_at(mask, pos)
        elif pos == probe:
            forced_class = probe_class
        class_ids: Tuple[int, ...] = (
            (forced_class,) if forced_class is not None else real_class_ids
        )

        next_frontier: Dict[int, object] = {}
        for state, prefix_commands in frontier.items():
            for class_id in class_ids:
                target = dfa.transitions[state][class_id]
                if prefix_commands is _COMMAND_CONFLICT:
                    next_commands: object = _COMMAND_CONFLICT
                else:
                    step_cmds = _control_step_commands(dfa, state, class_id)
                    shifted = _shift_commands(step_cmds, pos)
                    step_transform = _commands_to_phi_transform(shifted)
                    if step_transform is None:
                        next_commands = _COMMAND_CONFLICT
                    else:
                        next_commands = _compose_phi_transforms(
                            prefix_commands, step_transform)
                prev = next_frontier.get(target)
                if prev is None:
                    next_frontier[target] = next_commands
                elif prev != next_commands:
                    next_frontier[target] = _COMMAND_CONFLICT
        frontier = next_frontier

    if len(frontier) != 1:
        return None
    target_control, commands = next(iter(frontier.items()))
    if commands is _COMMAND_CONFLICT:
        return None
    normalized = _phi_transform_to_commands(commands)
    if normalized is None:
        return None
    return (target_control, normalized)


def _retained_after_shift(*,
                          cover: _Cover,
                          interval: Tuple[int, int],
                          mask: int,
                          probe: int,
                          probe_class: int,
                          delta: int,
                          ) -> Tuple[Tuple[int, int], int]:
    """Compute ``(retained_interval, retained_mask)`` after shift δ.

    The retained interval is the contiguous run of new-cursor
    offsets in ``[0, block_len-1]`` whose class is fixed by the old
    known map AND for which **at least one** cover path agrees. The
    returned mask is the union of cover-path indices consistent with
    every retained offset's pinned class.
    """

    block_len = cover.block_len
    # Old known classes (offset → class). Same composition as in
    # ``_resolve_bm_shift``.
    old_known: Dict[int, int] = {}
    if interval[0] >= 0:
        for p in range(interval[0], interval[1] + 1):
            cls = cover.known_class_at(mask, p)
            assert cls is not None
            old_known[p] = cls
    old_known[probe] = probe_class

    # Map old offsets → new offsets via shift.
    new_known: Dict[int, int] = {}
    for p, cls in old_known.items():
        new_p = p - delta
        if 0 <= new_p < block_len:
            new_known[new_p] = cls

    if not new_known:
        return (_EMPTY_INTERVAL, cover.full_mask())

    # Find the contiguous run with the highest retention count;
    # break ties by the leftmost run.
    new_offsets = sorted(new_known.keys())
    runs: List[Tuple[int, int]] = []
    cur_lo = new_offsets[0]
    cur_hi = new_offsets[0]
    for p in new_offsets[1:]:
        if p == cur_hi + 1:
            cur_hi = p
        else:
            runs.append((cur_lo, cur_hi))
            cur_lo = p
            cur_hi = p
    runs.append((cur_lo, cur_hi))

    best_run: Optional[Tuple[int, int]] = None
    best_mask: int = 0
    full_mask = cover.full_mask()
    for run in runs:
        lo, hi = run
        # Refine over ALL cover paths (full_mask) to compute new mask.
        run_mask = full_mask
        for p in range(lo, hi + 1):
            run_mask = cover.refine_mask(run_mask, p, new_known[p])
            if run_mask == 0:
                break
        if run_mask == 0:
            continue
        run_len = hi - lo + 1
        best_len = 0 if best_run is None else best_run[1] - best_run[0] + 1
        if run_len > best_len:
            best_run = run
            best_mask = run_mask

    if best_run is None or best_mask == 0:
        return (_EMPTY_INTERVAL, full_mask)
    return (best_run, best_mask)


def _expand_interval(interval: Tuple[int, int], pos: int) -> Tuple[int, int]:
    if interval[0] < 0:
        return (pos, pos)
    return (min(interval[0], pos), max(interval[1], pos))


def _simulate_path_target(dfa: TDFA,
                          start_state: int,
                          class_seq: Tuple[int, ...]) -> int:
    state = start_state
    for class_id in class_seq:
        state = dfa.transitions[state][class_id]
    return state


def _representative_byte_for_class(dfa: TDFA,
                                   class_id: int,
                                   cover: _Cover,
                                   probe: int) -> int:
    """Pick a representative byte for ``class_id`` at probe offset ``probe``.

    Prefer a byte that actually appears in some F-segment string at
    that offset (so ``SetBMOverlap`` queries don't false-negative on
    columns it knows nothing about). Fall back to the cover's
    canonical class byte.
    """
    column = cover.overlap.columns[probe] if 0 <= probe < cover.block_len \
        else frozenset()
    for byte_val in column:
        if dfa.class_map[byte_val] == class_id:
            return byte_val
    # No byte from the F set matches this class at this offset (the
    # standard mismatch case). Pick the smallest byte in the class --
    # any byte is fine for safe_shift since the column won't contain
    # it either way.
    for byte_val in range(256):
        if dfa.class_map[byte_val] == class_id:
            return byte_val
    raise RuntimeError(
        f"no byte representative found for class id {class_id}"
    )


# ---------------------------------------------------------------------------
# Edge materialization
# ---------------------------------------------------------------------------

def _materialize_gadget_edges(state: _GadgetState,
                              class_bytes: Dict[int, Tuple[int, ...]],
                              ) -> Tuple[OPCSEdge, ...]:
    """Convert ``_GadgetBranch`` records into deterministic ``OPCSEdge``s."""
    edge_groups: Dict[
        Tuple[int, int, Tuple[OPCSCommand, ...]],
        List[int],
    ] = {}
    for branch in state.branches:
        key = (branch.target_state_id, branch.shift, branch.commands)
        bv = edge_groups.setdefault(key, [])
        for class_id in branch.class_ids:
            bv.extend(class_bytes[class_id])

    return tuple(
        OPCSEdge(
            byte_values=tuple(sorted(bv)),
            target=target_state,
            shift=shift,
            commands=commands,
        )
        for (target_state, shift, commands), bv
        in sorted(edge_groups.items(), key=lambda item: item[1][0])
    )
