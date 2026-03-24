"""
Tagged OPCS-BMA artifacts built on top of the control DFA.

This module keeps two layers:
  - an unreachable shadow control skeleton that preserves the original DFA
  - runtime entry/control/gadget states used by compiled execution

The current builder is intentionally conservative: it grows right-to-left
block-local gadgets only when a fixed-width block can be resolved
deterministically by contiguous suffix summaries. When that fails, the
artifact gracefully falls back to plain ``w = 0, s = 1`` control states.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .analysis import (
    DEFAULT_OPCS_MAX_W,
    choose_block_len,
    compute_shortest_success_lengths,
)
from .dfa import DFA
from .nfa import NFA


INTERNAL_TAG_PREFIX = "__pythoc_internal_"


@dataclass(frozen=True)
class DFATagRuntime:
    """Compile-time tag actions for DFA/BMA execution."""

    tag_names: Tuple[str, ...]
    public_tag_names: Tuple[str, ...]
    tag_slots: Dict[str, int]
    transition_tag_slots: Dict[Tuple[int, int], Tuple[int, ...]]
    accept_tag_slots: Dict[int, Tuple[int, ...]]


@dataclass(frozen=True)
class TagWrite:
    """One ordered write of ``base + delta`` into one tag slot."""

    slot: int
    delta: int


@dataclass(frozen=True)
class OPCSEdge:
    """Grouped deterministic edge in the final runtime BMA."""

    byte_values: Tuple[int, ...]
    target: int
    shift: int
    tag_writes: Tuple[TagWrite, ...] = ()


@dataclass(frozen=True)
class OPCSState:
    """One runtime/shadow state in the final tagged OPCS-BMA."""

    id: int
    name: str
    kind: str
    control_state: int
    probe_offset: int
    block_len: int
    known_bytes: Tuple[int, ...]
    possible_targets: Tuple[int, ...]
    edges: Tuple[OPCSEdge, ...]
    accepting: bool = False
    eof_accept: bool = False
    eof_tag_writes: Tuple[TagWrite, ...] = ()


@dataclass(frozen=True)
class TaggedOPCSBMA:
    """Explicit compile-time BMA artifact used by codegen."""

    states: Tuple[OPCSState, ...]
    start_state: int
    dead_state: int
    shadow_dead_state: int
    effective_start_control: int
    runtime_control_states: Dict[int, int]
    shadow_control_states: Dict[int, int]
    control_entry_states: Dict[int, int]
    accept_states: FrozenSet[int]
    tag_names: Tuple[str, ...]
    public_tag_names: Tuple[str, ...]
    tag_slots: Dict[str, int]
    shortest_success_len: Dict[int, Optional[int]]
    initial_tag_writes: Tuple[TagWrite, ...]
    initial_accepting: bool
    initial_accept_tag_writes: Tuple[TagWrite, ...]


@dataclass(frozen=True)
class _TempBranch:
    class_ids: Tuple[int, ...]
    next_key: Optional[Tuple[int, ...]]
    final_control: Optional[int]
    tag_writes: Tuple[TagWrite, ...]


@dataclass
class _TempState:
    control_state: int
    known_classes: Tuple[int, ...]
    possible_targets: Tuple[int, ...]
    probe_offset: int
    branches: Tuple[_TempBranch, ...] = ()


def _collapse_tag_writes(tag_writes: Sequence[TagWrite]) -> Tuple[TagWrite, ...]:
    """Keep only the final effect for each slot."""
    if not tag_writes:
        return ()

    final_by_slot: Dict[int, int] = {}
    for write in tag_writes:
        final_by_slot[write.slot] = write.delta
    return tuple(
        TagWrite(slot=slot, delta=delta)
        for slot, delta in sorted(final_by_slot.items())
    )


def _shift_tag_writes(tag_writes: Sequence[TagWrite],
                      delta: int) -> Tuple[TagWrite, ...]:
    """Shift all write offsets by ``delta``."""
    if not tag_writes or delta == 0:
        return tuple(tag_writes)
    return tuple(
        TagWrite(slot=write.slot, delta=write.delta + delta)
        for write in tag_writes
    )


def _reverse_epsilon_graph(nfa: NFA) -> Dict[int, Set[int]]:
    rev: Dict[int, Set[int]] = {s.id: set() for s in nfa.states}
    for state in nfa.states:
        for target in state.epsilon:
            rev[target].add(state.id)
    return rev


def _class_representatives(dfa: DFA) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    for byte_val in range(256):
        reps.setdefault(dfa.class_map[byte_val], byte_val)
    if dfa.sentinel_256_class >= 0:
        reps[dfa.sentinel_256_class] = 256
    if dfa.sentinel_257_class >= 0:
        reps[dfa.sentinel_257_class] = 257
    return reps


def _real_class_layout(dfa: DFA) -> Tuple[Tuple[int, ...], Dict[int, int], Dict[int, Tuple[int, ...]]]:
    class_representatives: Dict[int, int] = {}
    class_bytes: Dict[int, List[int]] = {}
    for byte_val in range(256):
        class_id = dfa.class_map[byte_val]
        class_representatives.setdefault(class_id, byte_val)
        class_bytes.setdefault(class_id, []).append(byte_val)
    ordered_class_ids = tuple(sorted(class_representatives))
    return (
        ordered_class_ids,
        class_representatives,
        {
            class_id: tuple(class_bytes[class_id])
            for class_id in ordered_class_ids
        },
    )


def _collect_tag_names(nfa: NFA, include_internal: bool) -> Tuple[str, ...]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for state in nfa.states:
        tag = state.tag
        if tag is None or tag in seen:
            continue
        if not include_internal and tag.startswith(INTERNAL_TAG_PREFIX):
            continue
        seen.add(tag)
        ordered.append(tag)
    return tuple(ordered)


def _collect_reverse_reachable_tags(nfa: NFA, closure, target_states,
                                    reverse_eps: Dict[int, Set[int]],
                                    tag_slots: Dict[str, int]) -> Tuple[int, ...]:
    if not target_states:
        return ()

    allowed = set(closure)
    stack = list(target_states)
    seen = set(target_states)
    tag_slots_seen: List[int] = []
    emitted: Set[int] = set()

    while stack:
        cur = stack.pop()
        state_obj = nfa.states[cur]
        if state_obj.tag is not None and state_obj.tag in tag_slots:
            slot = tag_slots[state_obj.tag]
            if slot not in emitted:
                emitted.add(slot)
                tag_slots_seen.append(slot)
        for prev in reverse_eps[cur]:
            if prev in allowed and prev not in seen:
                seen.add(prev)
                stack.append(prev)

    return tuple(tag_slots_seen)


def compute_dfa_tag_runtime(nfa: NFA,
                            dfa: DFA,
                            include_internal: bool = False) -> DFATagRuntime:
    """Compute dynamic tag writes over deterministic DFA edges."""

    tag_names = _collect_tag_names(nfa, include_internal=include_internal)
    public_tag_names = tuple(
        name for name in tag_names
        if not name.startswith(INTERNAL_TAG_PREFIX)
    )
    tag_slots = {
        name: idx
        for idx, name in enumerate(tag_names)
    }
    reverse_eps = _reverse_epsilon_graph(nfa)
    reps = _class_representatives(dfa)

    transition_tag_slots: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    accept_tag_slots: Dict[int, Tuple[int, ...]] = {}

    for state_id, closure in enumerate(dfa.state_closures):
        if state_id in dfa.accept_states:
            slots = _collect_reverse_reachable_tags(
                nfa, closure, {nfa.accept}, reverse_eps, tag_slots
            )
            if slots:
                accept_tag_slots[state_id] = slots

        for class_id, representative in reps.items():
            if state_id >= len(dfa.transitions):
                continue
            if class_id >= len(dfa.transitions[state_id]):
                continue
            target = dfa.transitions[state_id][class_id]
            if target == dfa.dead_state:
                continue
            byte_sources = {
                nfa_state
                for nfa_state in closure
                if representative in nfa.states[nfa_state].byte_transitions
            }
            slots = _collect_reverse_reachable_tags(
                nfa, closure, byte_sources, reverse_eps, tag_slots
            )
            if slots:
                transition_tag_slots[(state_id, class_id)] = slots

    return DFATagRuntime(
        tag_names=tag_names,
        public_tag_names=public_tag_names,
        tag_slots=tag_slots,
        transition_tag_slots=transition_tag_slots,
        accept_tag_slots=accept_tag_slots,
    )


def _known_probe_offset(known_bytes: Tuple[int, ...]) -> int:
    idx = len(known_bytes) - 1
    while idx >= 0 and known_bytes[idx] >= 0:
        idx -= 1
    return idx


def _byte_name(byte_val: int) -> str:
    if 32 <= byte_val <= 126:
        ch = chr(byte_val)
        if ch.isalnum():
            return ch
    return f"{byte_val:02x}"


def _state_name(control_state: int, known_bytes: Tuple[int, ...]) -> str:
    if all(byte_val < 0 for byte_val in known_bytes):
        return f"q{control_state}_root"
    parts = [
        "x" if byte_val < 0 else _byte_name(byte_val)
        for byte_val in known_bytes
    ]
    return f"q{control_state}_{''.join(parts)}"


def _materialize_known_bytes(known_classes: Tuple[int, ...],
                             class_representatives: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(
        -1 if class_id < 0 else class_representatives[class_id]
        for class_id in known_classes
    )


def _simulate_control_target(dfa: DFA,
                             start_state: int,
                             block_bytes: Sequence[int]) -> int:
    state = start_state
    for byte_val in block_bytes:
        state = dfa.transitions[state][dfa.class_map[byte_val]]
    return state


def _control_path_tag_writes(dfa: DFA,
                             tag_runtime: Optional[DFATagRuntime],
                             start_state: int,
                             block_bytes: Sequence[int],
                             search_restart_controls: FrozenSet[int] = frozenset(),
                             search_entry_slot: Optional[int] = None) -> Tuple[TagWrite, ...]:
    if tag_runtime is None:
        writes: List[TagWrite] = []
    else:
        writes = []

    state = start_state
    for delta, byte_val in enumerate(block_bytes):
        cls = dfa.class_map[byte_val]
        if tag_runtime is not None:
            for slot in tag_runtime.transition_tag_slots.get((state, cls), ()):
                writes.append(TagWrite(slot=slot, delta=delta))
        target_state = dfa.transitions[state][cls]
        if (search_entry_slot is not None and
                state in search_restart_controls and
                target_state not in search_restart_controls):
            writes.append(TagWrite(slot=search_entry_slot, delta=delta))
        state = target_state

    if tag_runtime is not None and state in dfa.accept_states:
        for slot in tag_runtime.accept_tag_slots.get(state, ()):
            writes.append(TagWrite(slot=slot, delta=len(block_bytes)))

    return _collapse_tag_writes(writes)


def _control_edge_tag_writes(dfa: DFA,
                             tag_runtime: Optional[DFATagRuntime],
                             state: int,
                             byte_val: int,
                             search_restart_controls: FrozenSet[int] = frozenset(),
                             search_entry_slot: Optional[int] = None) -> Tuple[TagWrite, ...]:
    if tag_runtime is None and search_entry_slot is None:
        return ()
    target = dfa.transitions[state][dfa.class_map[byte_val]]
    writes = list(_control_path_tag_writes(
        dfa,
        tag_runtime,
        state,
        (byte_val,),
        search_restart_controls=search_restart_controls,
        search_entry_slot=search_entry_slot,
    ))
    if target in dfa.accept_states:
        # _control_path_tag_writes() already emitted the accept writes above.
        return tuple(writes)
    return tuple(writes)


def _control_eof_tag_writes(dfa: DFA,
                            tag_runtime: Optional[DFATagRuntime],
                            state: int) -> Tuple[TagWrite, ...]:
    if tag_runtime is None or dfa.sentinel_257_class < 0:
        return ()
    target = dfa.transitions[state][dfa.sentinel_257_class]
    if target not in dfa.accept_states:
        return ()

    writes: List[TagWrite] = []
    for slot in tag_runtime.transition_tag_slots.get((state, dfa.sentinel_257_class), ()):
        writes.append(TagWrite(slot=slot, delta=0))
    for slot in tag_runtime.accept_tag_slots.get(target, ()):
        writes.append(TagWrite(slot=slot, delta=0))
    return _collapse_tag_writes(writes)


def _group_control_edges(dfa: DFA,
                         state: int,
                         target_lookup,
                         tag_runtime: Optional[DFATagRuntime],
                         search_restart_controls: FrozenSet[int] = frozenset(),
                         search_entry_slot: Optional[int] = None) -> Tuple[OPCSEdge, ...]:
    groups: Dict[Tuple[int, Tuple[TagWrite, ...]], List[int]] = {}
    for byte_val in range(256):
        next_control = dfa.transitions[state][dfa.class_map[byte_val]]
        target = target_lookup(next_control)
        writes = _control_edge_tag_writes(
            dfa,
            tag_runtime,
            state,
            byte_val,
            search_restart_controls=search_restart_controls,
            search_entry_slot=search_entry_slot,
        )
        groups.setdefault((target, writes), []).append(byte_val)
    return tuple(
        OPCSEdge(
            byte_values=tuple(byte_vals),
            target=target,
            shift=1,
            tag_writes=writes,
        )
        for (target, writes), byte_vals in sorted(groups.items(), key=lambda item: item[1][0])
    )


def _effective_start_control(dfa: DFA) -> int:
    effective_start = dfa.start_state
    if dfa.sentinel_256_class >= 0:
        effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]
    return effective_start


def _real_successors(dfa: DFA, state: int) -> FrozenSet[int]:
    seen_classes: Set[int] = set()
    successors: Set[int] = set()
    for byte_val in range(256):
        cls = dfa.class_map[byte_val]
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        successors.add(dfa.transitions[state][cls])
    return frozenset(successors)


def _cover_success_chain(dfa: DFA,
                         shortest_success_len: Dict[int, Optional[int]],
                         start_state: int,
                         block_len: int) -> FrozenSet[int]:
    """One successful control chain used only by gadget scheduling."""
    if block_len <= 0:
        return frozenset()

    chain = [start_state]
    current = start_state

    for _ in range(block_len - 1):
        cur_dist = shortest_success_len.get(current)
        if cur_dist is None or cur_dist <= 1:
            break

        candidates = sorted(
            target for target in _real_successors(dfa, current)
            if target != dfa.dead_state and
            shortest_success_len.get(target) == cur_dist - 1
        )
        if not candidates:
            break

        current = candidates[0]
        chain.append(current)

    return frozenset(chain)


def _unprocessed_control_neighbors(dfa: DFA,
                                   shortest_success_len: Dict[int, Optional[int]],
                                   processed: FrozenSet[int]) -> FrozenSet[int]:
    """Forward control neighbors that advance along one successful chain."""
    neighbors: Set[int] = set()
    for state in processed:
        cur_dist = shortest_success_len.get(state)
        if cur_dist is None or cur_dist <= 0:
            continue
        for target in _real_successors(dfa, state):
            if (target == dfa.dead_state or
                    target in processed or
                    shortest_success_len.get(target) != cur_dist - 1):
                continue
            neighbors.add(target)
    return frozenset(neighbors)


def _build_suffix_gadget(dfa: DFA,
                         control_state: int,
                         block_len: int,
                         tag_runtime: Optional[DFATagRuntime],
                         search_restart_controls: FrozenSet[int],
                         search_entry_slot: Optional[int],
                         real_class_ids: Tuple[int, ...],
                         state_budget: int) -> Optional[Tuple[Dict[Tuple[int, ...], _TempState],
                                                              Tuple[int, ...]]]:
    if block_len <= 1:
        return None

    root_known = tuple([-1] * block_len)
    temp_states: Dict[Tuple[int, ...], _TempState] = {}

    @lru_cache(maxsize=None)
    def suffix_outcomes(state: int,
                        suffix: Tuple[int, ...]) -> FrozenSet[Tuple[int, Tuple[TagWrite, ...]]]:
        if not suffix:
            accept_writes: Tuple[TagWrite, ...] = ()
            if tag_runtime is not None and state in dfa.accept_states:
                accept_writes = _collapse_tag_writes(
                    tuple(
                        TagWrite(slot=slot, delta=0)
                        for slot in tag_runtime.accept_tag_slots.get(state, ())
                    )
                )
            return frozenset({(state, accept_writes)})

        next_byte = suffix[0]
        rest = suffix[1:]
        outcomes: Set[Tuple[int, Tuple[TagWrite, ...]]] = set()
        candidates = (next_byte,) if next_byte >= 0 else real_class_ids

        for cls in candidates:
            edge_writes: Tuple[TagWrite, ...] = ()
            if tag_runtime is not None:
                edge_writes = _collapse_tag_writes(
                    tuple(
                        TagWrite(slot=slot, delta=0)
                        for slot in tag_runtime.transition_tag_slots.get((state, cls), ())
                    )
                )

            target = dfa.transitions[state][cls]
            for final_control, suffix_writes in suffix_outcomes(target, rest):
                outcomes.add((
                    final_control,
                    _collapse_tag_writes(
                        edge_writes + _shift_tag_writes(suffix_writes, 1)
                    ),
                ))

        return frozenset(outcomes)

    @lru_cache(maxsize=None)
    def entry_outcomes(state: int,
                       suffix: Tuple[int, ...]) -> FrozenSet[Tuple[TagWrite, ...]]:
        if search_entry_slot is None:
            return frozenset({()})
        if not suffix:
            return frozenset({()})

        next_byte = suffix[0]
        rest = suffix[1:]
        outcomes: Set[Tuple[TagWrite, ...]] = set()
        candidates = (next_byte,) if next_byte >= 0 else real_class_ids

        for cls in candidates:
            target = dfa.transitions[state][cls]
            edge_writes: Tuple[TagWrite, ...] = ()
            if (state in search_restart_controls and
                    target not in search_restart_controls):
                edge_writes = (TagWrite(slot=search_entry_slot, delta=0),)

            for suffix_writes in entry_outcomes(target, rest):
                outcomes.add(_collapse_tag_writes(
                    edge_writes + _shift_tag_writes(suffix_writes, 1)
                ))

        return frozenset(outcomes)

    def common_entry_writes(outcomes: FrozenSet[Tuple[TagWrite, ...]]) -> Tuple[TagWrite, ...]:
        if not outcomes:
            return ()
        if len(outcomes) == 1:
            return next(iter(outcomes))
        return ()

    restart_or_dead = search_restart_controls | {dfa.dead_state}

    def _try_overlap_finalize(child_outcomes, entry_writes):
        """Try to finalize when all outcomes land in restart/dead states.

        When every concrete completion of the remaining unknown positions
        leads to a restart or dead control state, the block cannot produce
        a match.  Tag writes from the failed attempt are irrelevant since
        they will be overwritten when a new match begins, so we can
        collapse all outcomes into a single restart target and skip the
        remaining probes.
        """
        target_controls = {tc for tc, _tw in child_outcomes}
        if not target_controls.issubset(restart_or_dead):
            return None
        preferred = next(
            (t for t in target_controls if t in search_restart_controls),
            min(target_controls),
        )
        return ("final", preferred, tuple(entry_writes))

    def build_state(known_bytes: Tuple[int, ...]) -> None:
        if known_bytes in temp_states:
            return
        if len(temp_states) >= state_budget:
            raise RuntimeError("gadget budget exceeded")

        possible_targets = tuple(sorted(
            target_control
            for target_control, _tag_effect in suffix_outcomes(control_state, known_bytes)
        ))
        probe_offset = _known_probe_offset(known_bytes)
        if probe_offset < 0:
            raise RuntimeError("full concrete block cannot remain unresolved")

        temp_states[known_bytes] = _TempState(
            control_state=control_state,
            known_classes=known_bytes,
            possible_targets=possible_targets,
            probe_offset=probe_offset,
        )

        outcomes: Dict[Tuple[str, object, Tuple[TagWrite, ...]], List[int]] = {}
        for class_id in real_class_ids:
            child = list(known_bytes)
            child[probe_offset] = class_id
            child_key = tuple(child)
            child_outcomes = tuple(sorted(
                suffix_outcomes(control_state, child_key),
                key=lambda item: (
                    item[0],
                    tuple((write.slot, write.delta) for write in item[1]),
                ),
            ))
            child_entry_outcomes = entry_outcomes(control_state, child_key)
            entry_writes = common_entry_writes(child_entry_outcomes)

            if len(child_outcomes) == 1:
                target_control, topo_writes = child_outcomes[0]
                can_finish = (
                    target_control in search_restart_controls or
                    len(child_entry_outcomes) == 1
                )
                if can_finish:
                    outcome = (
                        "final",
                        target_control,
                        _collapse_tag_writes(topo_writes + entry_writes),
                    )
                else:
                    build_state(child_key)
                    outcome = ("next", child_key, entry_writes)
            else:
                overlap = _try_overlap_finalize(child_outcomes, entry_writes)
                if overlap is not None:
                    outcome = overlap
                elif _known_probe_offset(child_key) < 0:
                    raise RuntimeError("concrete block stayed ambiguous")
                else:
                    build_state(child_key)
                    outcome = ("next", child_key, entry_writes)

            outcomes.setdefault(outcome, []).append(class_id)

        temp_states[known_bytes].branches = tuple(
            _TempBranch(
                class_ids=tuple(class_ids),
                next_key=(payload if kind == "next" else None),
                final_control=(payload if kind == "final" else None),
                tag_writes=tag_writes,
            )
            for (kind, payload, tag_writes), class_ids in sorted(
                outcomes.items(),
                key=lambda item: item[1][0],
            )
        )

    try:
        build_state(root_known)
    except RuntimeError:
        return None

    root_state = temp_states[root_known]
    if len(root_state.branches) <= 1 and all(
        branch.next_key is None
        for branch in root_state.branches
    ):
        return None

    return temp_states, root_known


def build_tagged_opcs_bma(dfa: DFA,
                          tag_runtime: Optional[DFATagRuntime] = None,
                          search_restart_controls: FrozenSet[int] = frozenset(),
                          search_entry_slot: Optional[int] = None,
                          max_w: int = DEFAULT_OPCS_MAX_W,
                          state_budget: int = 128) -> TaggedOPCSBMA:
    """Build one deterministic tagged OPCS-BMA from a control DFA."""

    effective_start = _effective_start_control(dfa)
    shortest_success_len = compute_shortest_success_lengths(dfa)
    real_class_ids, class_representatives, class_bytes = _real_class_layout(dfa)

    shadow_ids = {
        state: state
        for state in range(dfa.num_states)
    }
    runtime_ids = {
        state: dfa.num_states + state
        for state in range(dfa.num_states)
    }

    gadget_defs: Dict[int, Tuple[Dict[Tuple[int, ...], _TempState], Tuple[int, ...], int]] = {}
    processed: Set[int] = set()
    frontier: Set[int] = {effective_start}

    while frontier:
        state = min(frontier)
        frontier.remove(state)

        if state in processed or state == dfa.dead_state:
            continue

        block_len = choose_block_len(shortest_success_len[state], max_w=max_w)
        if state in dfa.accept_states or block_len <= 1:
            processed.add(state)
            frontier.update(_unprocessed_control_neighbors(
                dfa,
                shortest_success_len,
                frozenset(processed),
            ))
            continue

        built = _build_suffix_gadget(
            dfa,
            state,
            block_len,
            tag_runtime,
            search_restart_controls,
            search_entry_slot,
            real_class_ids,
            state_budget=state_budget,
        )
        if built is not None:
            temp_states, root_key = built
            gadget_defs[state] = (temp_states, root_key, block_len)
            processed.update(_cover_success_chain(
                dfa,
                shortest_success_len,
                state,
                block_len,
            ))
        else:
            processed.add(state)

        frontier.update(_unprocessed_control_neighbors(
            dfa,
            shortest_success_len,
            frozenset(processed),
        ))

    next_state_id = 2 * dfa.num_states
    gadget_state_ids: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    control_entry_states = dict(runtime_ids)
    for control_state, (temp_states, root_key, _block_len) in gadget_defs.items():
        ordered_keys = sorted(
            temp_states.keys(),
            key=lambda key: (sum(1 for b in key if b < 0), key),
            reverse=True,
        )
        for key in ordered_keys:
            gadget_state_ids[(control_state, key)] = next_state_id
            next_state_id += 1
        control_entry_states[control_state] = gadget_state_ids[(control_state, root_key)]

    states: List[Optional[OPCSState]] = [None] * next_state_id

    for control_state in range(dfa.num_states):
        is_accept = control_state in dfa.accept_states
        eof_accept = (
            dfa.sentinel_257_class >= 0 and
            dfa.transitions[control_state][dfa.sentinel_257_class] in dfa.accept_states
        )
        eof_writes = _control_eof_tag_writes(dfa, tag_runtime, control_state)

        shadow_edges = _group_control_edges(
            dfa,
            control_state,
            lambda next_control: shadow_ids[next_control],
            None,
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
            eof_accept=eof_accept,
            eof_tag_writes=(),
        )

        runtime_edges = _group_control_edges(
            dfa,
            control_state,
            lambda next_control: control_entry_states[next_control],
            tag_runtime,
            search_restart_controls=search_restart_controls,
            search_entry_slot=search_entry_slot,
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
            eof_accept=eof_accept,
            eof_tag_writes=eof_writes,
        )

    for control_state, (temp_states, _root_key, block_len) in gadget_defs.items():
        for known_classes, temp_state in temp_states.items():
            bma_state_id = gadget_state_ids[(control_state, known_classes)]
            edge_groups: Dict[Tuple[int, int, Tuple[TagWrite, ...]], List[int]] = {}
            for branch in temp_state.branches:
                if branch.next_key is not None:
                    target_state = gadget_state_ids[(control_state, branch.next_key)]
                    shift = 0
                    tw = ()
                else:
                    assert branch.final_control is not None
                    target_state = control_entry_states[branch.final_control]
                    shift = block_len
                    tw = branch.tag_writes
                key = (target_state, shift, tw)
                bv = edge_groups.setdefault(key, [])
                for class_id in branch.class_ids:
                    bv.extend(class_bytes[class_id])
            edges: List[OPCSEdge] = [
                OPCSEdge(
                    byte_values=tuple(sorted(bv)),
                    target=target_state,
                    shift=shift,
                    tag_writes=tw,
                )
                for (target_state, shift, tw), bv
                in sorted(edge_groups.items(), key=lambda item: item[1][0])
            ]

            known_bytes = _materialize_known_bytes(
                known_classes,
                class_representatives,
            )
            states[bma_state_id] = OPCSState(
                id=bma_state_id,
                name=_state_name(control_state, known_bytes),
                kind="gadget",
                control_state=control_state,
                probe_offset=temp_state.probe_offset,
                block_len=block_len,
                known_bytes=known_bytes,
                possible_targets=temp_state.possible_targets,
                edges=tuple(edges),
                accepting=False,
                eof_accept=False,
                eof_tag_writes=(),
            )

    assert all(state is not None for state in states)

    initial_tag_writes: List[TagWrite] = []
    initial_accept_tag_writes: Tuple[TagWrite, ...] = ()
    if (search_entry_slot is not None and (
            effective_start not in search_restart_controls or
            effective_start in dfa.accept_states)):
        initial_tag_writes.append(TagWrite(slot=search_entry_slot, delta=0))
    if tag_runtime is not None and dfa.sentinel_256_class >= 0:
        for slot in tag_runtime.transition_tag_slots.get(
                (dfa.start_state, dfa.sentinel_256_class), ()):
            initial_tag_writes.append(TagWrite(slot=slot, delta=0))
    if tag_runtime is not None and effective_start in dfa.accept_states:
        initial_accept_tag_writes = tuple(
            TagWrite(slot=slot, delta=0)
            for slot in tag_runtime.accept_tag_slots.get(effective_start, ())
        )
        initial_tag_writes.extend(initial_accept_tag_writes)

    accept_states = frozenset(
        runtime_ids[state]
        for state in dfa.accept_states
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
        tag_names=() if tag_runtime is None else tag_runtime.tag_names,
        public_tag_names=() if tag_runtime is None else tag_runtime.public_tag_names,
        tag_slots={} if tag_runtime is None else tag_runtime.tag_slots,
        shortest_success_len=shortest_success_len,
        initial_tag_writes=tuple(initial_tag_writes),
        initial_accepting=effective_start in dfa.accept_states,
        initial_accept_tag_writes=initial_accept_tag_writes,
    )
