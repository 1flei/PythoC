"""
Tagged OPCS-BMA artifacts built on top of the control DFA.

This module keeps two layers:
  - an unreachable shadow control skeleton that preserves the original DFA
  - runtime entry/control/gadget states used by compiled execution

The live builder uses block-local contiguous interval summaries rather than
suffix-only states. Gadget states choose probe offsets per compiled state,
resolve blocks by adjacent interval growth, and conservatively deopt to raw
control states on mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .analysis import (
    DEFAULT_OPCS_MAX_W,
    choose_block_len,
    compute_shortest_success_lengths,
)
from .dfa import DFA
from .nfa import NFA
from .tdfa import TDFA, TDFACommand


INTERNAL_TAG_PREFIX = "__pythoc_internal_"


@dataclass(frozen=True)
class DFATagRuntime:
    """Compile-time tag actions for DFA/BMA execution."""

    tag_names: Tuple[str, ...]
    public_tag_names: Tuple[str, ...]
    tag_slots: Dict[str, int]
    transition_tag_slots: Dict[Tuple[int, int], Tuple[int, ...]]
    accept_tag_slots: Dict[int, Tuple[int, ...]]
    multi_write_slots: FrozenSet[int] = frozenset()


@dataclass(frozen=True)
class TagWrite:
    """One ordered write of ``base + delta`` into one tag slot."""

    slot: int
    delta: int


@dataclass(frozen=True)
class OPCSCommand:
    """One runtime register command executed inside a BMA edge/state."""

    kind: str  # "copy", "set", or "add"
    lhs: int
    rhs: int = 0
    delta: int = 0
    history: Tuple[int, ...] = ()


@dataclass(frozen=True)
class OPCSEdge:
    """Grouped deterministic edge in the final runtime BMA."""

    byte_values: Tuple[int, ...]
    target: int
    shift: int
    commands: Tuple[OPCSCommand, ...] = ()


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
    accept_commands: Tuple[OPCSCommand, ...] = ()
    eof_accept: bool = False
    eof_commands: Tuple[OPCSCommand, ...] = ()


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
    register_count: int
    output_registers: Tuple[int, ...]
    shortest_success_len: Dict[int, Optional[int]]
    initial_commands: Tuple[OPCSCommand, ...]
    initial_accepting: bool
    initial_accept_commands: Tuple[OPCSCommand, ...]


@dataclass(frozen=True)
class _TempBranch:
    class_ids: Tuple[int, ...]
    target_control: int
    target_interval: Optional[Tuple[int, int]]
    shift: int
    commands: Tuple[OPCSCommand, ...]


@dataclass
class _TempState:
    control_state: int
    interval: Tuple[int, int]
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


def _commands_from_tag_writes(tag_writes: Sequence[TagWrite],
                              register_base: int = 1) -> Tuple[OPCSCommand, ...]:
    return tuple(
        OPCSCommand(kind="set", lhs=register_base + write.slot, delta=write.delta)
        for write in _collapse_tag_writes(tag_writes)
    )


def _commands_from_tdfa_commands(commands: Sequence[TDFACommand],
                                 delta: int = 0) -> Tuple[OPCSCommand, ...]:
    return tuple(
        OPCSCommand(
            kind=command.kind,
            lhs=command.lhs,
            rhs=command.rhs,
            delta=(command.delta if hasattr(command, "delta") else 0) + delta
            if command.kind == "set" else 0,
            history=command.history,
        )
        for command in commands
    )


def _shift_commands(commands: Sequence[OPCSCommand],
                    delta: int) -> Tuple[OPCSCommand, ...]:
    if not commands or delta == 0:
        return tuple(commands)
    shifted: List[OPCSCommand] = []
    for command in commands:
        if command.kind == "set":
            shifted.append(OPCSCommand(
                kind=command.kind,
                lhs=command.lhs,
                rhs=command.rhs,
                delta=command.delta + delta,
                history=command.history,
            ))
        else:
            shifted.append(command)
    return tuple(shifted)


_PHI_EXPR_REG = "reg"
_PHI_EXPR_BASE = "base"
_PhiExpr = Tuple[str, int]
_PhiTransform = Tuple[Tuple[int, _PhiExpr], ...]


def _phi_identity_expr(reg_id: int) -> _PhiExpr:
    return (_PHI_EXPR_REG, reg_id)


def _phi_substitute_expr(expr: _PhiExpr,
                         transform: _PhiTransform) -> _PhiExpr:
    if expr[0] != _PHI_EXPR_REG:
        return expr
    return dict(transform).get(expr[1], expr)


def _commands_to_phi_transform(commands: Sequence[OPCSCommand]) -> Optional[_PhiTransform]:
    """Normalize ordered set/copy commands to one functional register transform."""
    env: Dict[int, _PhiExpr] = {}
    for command in commands:
        if command.kind == "set":
            env[command.lhs] = (_PHI_EXPR_BASE, command.delta)
        elif command.kind == "copy":
            env[command.lhs] = env.get(command.rhs, _phi_identity_expr(command.rhs))
        else:
            return None
    return tuple(sorted(
        (lhs, expr)
        for lhs, expr in env.items()
        if expr != _phi_identity_expr(lhs)
    ))


def _compose_phi_transforms(prefix: _PhiTransform,
                            suffix: _PhiTransform) -> _PhiTransform:
    """Compose two transforms: first ``prefix``, then ``suffix``."""
    env: Dict[int, _PhiExpr] = dict(prefix)
    for lhs, expr in suffix:
        env[lhs] = _phi_substitute_expr(expr, prefix)
    return tuple(sorted(
        (lhs, expr)
        for lhs, expr in env.items()
        if expr != _phi_identity_expr(lhs)
    ))


def _topsort_opcs_copy_commands(commands: Sequence[OPCSCommand]) -> Tuple[Tuple[OPCSCommand, ...], bool]:
    if not commands:
        return (), False

    indegree: Dict[int, int] = {}
    for command in commands:
        indegree.setdefault(command.lhs, 0)
        indegree.setdefault(command.rhs, 0)
    for command in commands:
        indegree[command.rhs] += 1

    ordered: List[OPCSCommand] = []
    pending = list(commands)
    while pending:
        progressed = False
        remaining: List[OPCSCommand] = []
        for command in pending:
            if indegree[command.lhs] == 0:
                indegree[command.rhs] -= 1
                ordered.append(command)
                progressed = True
            else:
                remaining.append(command)
        if not progressed:
            non_trivial_cycle = any(command.lhs != command.rhs for command in remaining)
            return tuple(ordered + remaining), non_trivial_cycle
        pending = remaining

    return tuple(ordered), False


def _phi_transform_to_commands(transform: _PhiTransform) -> Optional[Tuple[OPCSCommand, ...]]:
    copy_commands: List[OPCSCommand] = []
    set_commands: List[OPCSCommand] = []
    for lhs, expr in transform:
        kind, value = expr
        if kind == _PHI_EXPR_BASE:
            set_commands.append(OPCSCommand(kind="set", lhs=lhs, delta=value))
        elif kind == _PHI_EXPR_REG:
            if value != lhs:
                copy_commands.append(OPCSCommand(kind="copy", lhs=lhs, rhs=value))
        else:
            return None

    ordered_copies, has_cycle = _topsort_opcs_copy_commands(copy_commands)
    if has_cycle:
        return None
    return ordered_copies + tuple(sorted(set_commands, key=lambda command: command.lhs))


def _normalize_opcs_commands(commands: Sequence[OPCSCommand]) -> Tuple[OPCSCommand, ...]:
    transform = _commands_to_phi_transform(commands)
    if transform is None:
        return tuple(commands)
    normalized = _phi_transform_to_commands(transform)
    return tuple(commands) if normalized is None else normalized


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
    transition_classes_by_slot: Dict[int, Set[int]] = {}
    accept_states_by_slot: Dict[int, Set[int]] = {}

    for state_id, closure in enumerate(dfa.state_closures):
        if state_id in dfa.accept_states:
            slots = _collect_reverse_reachable_tags(
                nfa, closure, {nfa.accept}, reverse_eps, tag_slots
            )
            if slots:
                accept_tag_slots[state_id] = slots
                for slot in slots:
                    accept_states_by_slot.setdefault(slot, set()).add(state_id)

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
                for slot in slots:
                    transition_classes_by_slot.setdefault(slot, set()).add(class_id)

    multi_write_slots = frozenset(
        slot for slot in tag_slots.values()
        if len(transition_classes_by_slot.get(slot, set())) > 1 or
        (transition_classes_by_slot.get(slot) and accept_states_by_slot.get(slot)) or
        len(accept_states_by_slot.get(slot, set())) > 1
    )

    return DFATagRuntime(
        tag_names=tag_names,
        public_tag_names=public_tag_names,
        tag_slots=tag_slots,
        transition_tag_slots=transition_tag_slots,
        accept_tag_slots=accept_tag_slots,
        multi_write_slots=multi_write_slots,
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


def _simulate_control_target(control,
                             start_state: int,
                             class_path: Sequence[int]) -> int:
    state = start_state
    for class_id in class_path:
        state = control.transitions[state][class_id]
    return state


def _control_accept_commands(control,
                             tag_runtime: Optional[DFATagRuntime],
                             state: int) -> Tuple[OPCSCommand, ...]:
    if isinstance(control, TDFA):
        return _commands_from_tdfa_commands(control.accept_commands.get(state, ()))
    if tag_runtime is None or state not in control.accept_states:
        return ()
    return _commands_from_tag_writes(
        tuple(
            TagWrite(slot=slot, delta=0)
            for slot in tag_runtime.accept_tag_slots.get(state, ())
        )
    )


def _control_step_commands(control,
                           tag_runtime: Optional[DFATagRuntime],
                           state: int,
                           class_id: int) -> Tuple[OPCSCommand, ...]:
    if isinstance(control, TDFA):
        return _commands_from_tdfa_commands(
            control.transition_commands.get((state, class_id), ())
        )
    if tag_runtime is None:
        return ()
    return _commands_from_tag_writes(
        tuple(
            TagWrite(slot=slot, delta=0)
            for slot in tag_runtime.transition_tag_slots.get((state, class_id), ())
        )
    )


def _control_path_commands(control,
                           tag_runtime: Optional[DFATagRuntime],
                           start_state: int,
                           class_path: Sequence[int]) -> Tuple[OPCSCommand, ...]:
    commands: List[OPCSCommand] = []
    state = start_state
    for delta, class_id in enumerate(class_path):
        commands.extend(_shift_commands(
            _control_step_commands(
                control,
                tag_runtime,
                state,
                class_id,
            ),
            delta,
        ))
        state = control.transitions[state][class_id]
    return _normalize_opcs_commands(commands)


def _control_edge_commands(control,
                           tag_runtime: Optional[DFATagRuntime],
                           state: int,
                           byte_val: int) -> Tuple[OPCSCommand, ...]:
    return _control_path_commands(
        control,
        tag_runtime,
        state,
        (control.class_map[byte_val],),
    )


def _control_eof_commands(control,
                          tag_runtime: Optional[DFATagRuntime],
                          state: int) -> Tuple[OPCSCommand, ...]:
    if control.sentinel_257_class < 0:
        return ()
    target = control.transitions[state][control.sentinel_257_class]
    if target not in control.accept_states:
        return ()

    return (
        _control_step_commands(control, tag_runtime, state, control.sentinel_257_class) +
        _control_accept_commands(control, tag_runtime, target)
    )


def _group_control_edges(control,
                         state: int,
                         target_lookup,
                         tag_runtime: Optional[DFATagRuntime]) -> Tuple[OPCSEdge, ...]:
    groups: Dict[Tuple[int, Tuple[OPCSCommand, ...]], List[int]] = {}
    for byte_val in range(256):
        next_control = control.transitions[state][control.class_map[byte_val]]
        target = target_lookup(next_control)
        commands = _control_edge_commands(
            control,
            tag_runtime,
            state,
            byte_val,
        )
        groups.setdefault((target, commands), []).append(byte_val)
    return tuple(
        OPCSEdge(
            byte_values=tuple(byte_vals),
            target=target,
            shift=1,
            commands=commands,
        )
        for (target, commands), byte_vals in sorted(groups.items(), key=lambda item: item[1][0])
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
                         real_class_ids: Tuple[int, ...],
                         state_budget: int) -> Optional[Tuple[Dict[Tuple[int, ...], _TempState],
                                                              Tuple[int, ...]]]:
    raise RuntimeError(
        "Legacy suffix-only builder retired; use build_tagged_opcs_bma() interval-summary construction."
    )


def _artifact_layout(control,
                     tag_runtime: Optional[DFATagRuntime]) -> Tuple[
                         Tuple[str, ...],
                         Tuple[str, ...],
                         Dict[str, int],
                         int,
                         Tuple[int, ...],
                     ]:
    if isinstance(control, TDFA):
        return (
            control.tag_names,
            control.public_tag_names,
            control.tag_slots,
            control.register_count,
            control.output_registers,
        )
    if tag_runtime is None:
        return (), (), {}, 0, ()
    num_tags = len(tag_runtime.tag_names)
    return (
        tag_runtime.tag_names,
        tag_runtime.public_tag_names,
        tag_runtime.tag_slots,
        num_tags,
        tuple(range(1, num_tags + 1)),
    )


def build_tagged_opcs_bma(dfa: DFA | TDFA,
                          tag_runtime: Optional[DFATagRuntime] = None,
                          max_w: int = DEFAULT_OPCS_MAX_W,
                          state_budget: int = 128) -> TaggedOPCSBMA:
    """Build one deterministic tagged OPCS-BMA from a control DFA."""

    effective_start = _effective_start_control(dfa)
    shortest_success_len = compute_shortest_success_lengths(dfa)
    real_class_ids, class_representatives, class_bytes = _real_class_layout(dfa)
    (tag_names,
     public_tag_names,
     tag_slots,
     register_count,
     output_registers) = _artifact_layout(dfa, tag_runtime)
    block_len_by_control = {
        state: choose_block_len(shortest_success_len[state], max_w=max_w)
        for state in range(dfa.num_states)
    }
    EMPTY_INTERVAL = (-1, -1)
    state_bits = tuple(1 << state for state in range(dfa.num_states))

    def _state_mask(states: Iterable[int]) -> int:
        mask = 0
        for state in states:
            mask |= state_bits[state]
        return mask

    real_successor_mask = tuple(
        _state_mask(
            {dfa.transitions[state][class_id] for class_id in real_class_ids}
        )
        for state in range(dfa.num_states)
    )

    def _iter_states(mask: int):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    @lru_cache(maxsize=None)
    def _step_any(mask: int) -> int:
        out = 0
        for state in _iter_states(mask):
            out |= real_successor_mask[state]
        return out

    @lru_cache(maxsize=None)
    def _step_class(mask: int, class_id: int) -> int:
        out = 0
        for state in _iter_states(mask):
            out |= state_bits[dfa.transitions[state][class_id]]
        return out

    @lru_cache(maxsize=None)
    def _advance_any(mask: int, count: int) -> int:
        for _ in range(count):
            mask = _step_any(mask)
        return mask

    def _mask_to_states(mask: int) -> Tuple[int, ...]:
        return tuple(_iter_states(mask))

    def _interval_is_empty(interval: Tuple[int, int]) -> bool:
        return interval[0] < 0

    def _interval_known_count(interval: Tuple[int, int]) -> int:
        if _interval_is_empty(interval):
            return 0
        return interval[1] - interval[0] + 1

    def _interval_is_full(interval: Tuple[int, int], block_len: int) -> bool:
        return not _interval_is_empty(interval) and interval[0] == 0 and interval[1] == block_len - 1

    def _expand_interval(interval: Tuple[int, int], pos: int) -> Tuple[int, int]:
        if _interval_is_empty(interval):
            return (pos, pos)
        return (min(interval[0], pos), max(interval[1], pos))

    @lru_cache(maxsize=None)
    def _single_accepting_cover_path(state: int,
                                     remaining: int) -> Optional[Tuple[int, ...]]:
        if remaining == 0:
            return () if state in dfa.accept_states else None

        winner: Optional[Tuple[int, ...]] = None
        for class_id in real_class_ids:
            target = dfa.transitions[state][class_id]
            if target == dfa.dead_state:
                continue
            suffix = _single_accepting_cover_path(target, remaining - 1)
            if suffix is None:
                continue
            candidate = (class_id,) + suffix
            if winner is None:
                winner = candidate
            elif winner != candidate:
                return None
        return winner

    @lru_cache(maxsize=None)
    def _has_live_nonaccept_after_depth(state: int, remaining: int) -> bool:
        if remaining == 0:
            return (
                state != dfa.dead_state and
                state not in dfa.accept_states and
                shortest_success_len[state] is not None
            )

        for class_id in real_class_ids:
            target = dfa.transitions[state][class_id]
            if target == dfa.dead_state:
                continue
            if _has_live_nonaccept_after_depth(target, remaining - 1):
                return True
        return False

    cover_classes_by_control: Dict[int, Tuple[int, ...]] = {}
    for control_state in range(dfa.num_states):
        if (control_state == dfa.dead_state or
                control_state in dfa.accept_states or
                block_len_by_control[control_state] <= 1):
            continue
        block_len = block_len_by_control[control_state]
        cover_path = _single_accepting_cover_path(control_state, block_len)
        if (
            cover_path is not None and
            not _has_live_nonaccept_after_depth(control_state, block_len)
        ):
            cover_classes_by_control[control_state] = cover_path

    @lru_cache(maxsize=None)
    def _known_classes_for_interval(control_state: int,
                                    interval: Tuple[int, int]) -> Tuple[int, ...]:
        block_len = block_len_by_control[control_state]
        known = [-1] * block_len
        if not _interval_is_empty(interval):
            cover = cover_classes_by_control[control_state]
            for pos in range(interval[0], interval[1] + 1):
                known[pos] = cover[pos]
        return tuple(known)

    @lru_cache(maxsize=None)
    def _interval_target_mask(control_state: int,
                              interval: Tuple[int, int]) -> int:
        block_len = block_len_by_control[control_state]
        mask = state_bits[control_state]
        if _interval_is_empty(interval):
            return _advance_any(mask, block_len)

        cover = cover_classes_by_control[control_state]
        mask = _advance_any(mask, interval[0])
        for pos in range(interval[0], interval[1] + 1):
            mask = _step_class(mask, cover[pos])
        return _advance_any(mask, block_len - interval[1] - 1)

    @lru_cache(maxsize=None)
    def _cover_path_target(control_state: int) -> int:
        return _simulate_control_target(
            dfa,
            control_state,
            cover_classes_by_control[control_state],
        )

    @lru_cache(maxsize=None)
    def _cover_path_commands(control_state: int) -> Tuple[OPCSCommand, ...]:
        return _control_path_commands(
            dfa,
            tag_runtime,
            control_state,
            cover_classes_by_control[control_state],
        )

    def _probe_candidates(control_state: int,
                          interval: Tuple[int, int]) -> Tuple[int, ...]:
        block_len = block_len_by_control[control_state]
        if _interval_is_empty(interval):
            return (block_len - 1,)

        left, right = interval
        candidates: List[int] = []
        if left > 0:
            candidates.append(left - 1)
        if right + 1 < block_len:
            candidates.append(right + 1)
        return tuple(candidates)

    def _choose_probe_offset(control_state: int,
                             interval: Tuple[int, int]) -> Optional[int]:
        candidates = _probe_candidates(control_state, interval)
        if not candidates:
            return None
        center = (block_len_by_control[control_state] - 1) / 2.0
        return min(candidates, key=lambda pos: (abs(pos - center), pos))

    @lru_cache(maxsize=None)
    def _shifted_step_commands(state: int,
                               class_id: int,
                               delta: int) -> Tuple[OPCSCommand, ...]:
        return _shift_commands(
            _control_step_commands(dfa, tag_runtime, state, class_id),
            delta,
        )

    def _known_class_at(control_state: int,
                        interval: Tuple[int, int],
                        extra_pos: Optional[int],
                        extra_class: int,
                        pos: int) -> Optional[int]:
        if not _interval_is_empty(interval) and interval[0] <= pos <= interval[1]:
            return cover_classes_by_control[control_state][pos]
        if extra_pos is not None and pos == extra_pos:
            return extra_class
        return None

    def _known_span(interval: Tuple[int, int],
                    extra_pos: Optional[int]) -> Optional[Tuple[int, int]]:
        if extra_pos is None:
            if _interval_is_empty(interval):
                return None
            return interval
        if _interval_is_empty(interval):
            return (extra_pos, extra_pos)
        return (min(interval[0], extra_pos), max(interval[1], extra_pos))

    _COMMAND_CONFLICT = object()

    @lru_cache(maxsize=None)
    def _unique_prefix_outcome(control_state: int,
                               interval: Tuple[int, int],
                               extra_pos: int,
                               extra_class: int,
                               delta: int) -> Optional[Tuple[int, Tuple[OPCSCommand, ...]]]:
        frontier: Dict[int, object] = {control_state: ()}
        for pos in range(delta):
            forced_class = _known_class_at(
                control_state, interval, extra_pos, extra_class, pos)
            class_ids = (forced_class,) if forced_class is not None else real_class_ids
            next_frontier: Dict[int, object] = {}
            for state, prefix_commands in frontier.items():
                for class_id in class_ids:
                    target = dfa.transitions[state][class_id]
                    if prefix_commands is _COMMAND_CONFLICT:
                        next_commands = _COMMAND_CONFLICT
                    else:
                        step_transform = _commands_to_phi_transform(
                            _shifted_step_commands(state, class_id, pos))
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
        return (target_control, normalized)  # type: ignore[return-value]

    def _normalize_retained_interval(control_state: int,
                                     interval: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if interval is None:
            return None
        block_len = block_len_by_control[control_state]
        if _interval_is_empty(interval):
            return None
        if not _interval_is_full(interval, block_len):
            return interval
        if block_len <= 1:
            return None
        return (interval[0], interval[1] - 1)

    def _retained_interval_after_shift(source_control: int,
                                       source_interval: Tuple[int, int],
                                       extra_pos: int,
                                       extra_class: int,
                                       shift: int,
                                       target_control: int) -> Optional[Tuple[int, int]]:
        if target_control not in cover_classes_by_control:
            return None

        known_span = _known_span(source_interval, extra_pos)
        if known_span is None:
            return None

        target_block_len = block_len_by_control[target_control]
        target_cover = cover_classes_by_control[target_control]
        new_lo = max(0, known_span[0] - shift)
        new_hi = min(target_block_len - 1, known_span[1] - shift)
        if new_lo > new_hi:
            return None

        best: Optional[Tuple[int, int]] = None
        cur_start: Optional[int] = None
        for new_pos in range(new_lo, new_hi + 1):
            old_pos = new_pos + shift
            old_class = _known_class_at(
                source_control, source_interval, extra_pos, extra_class, old_pos)
            if old_class is not None and old_class == target_cover[new_pos]:
                if cur_start is None:
                    cur_start = new_pos
            else:
                if cur_start is not None:
                    candidate = (cur_start, new_pos - 1)
                    if (best is None or
                            _interval_known_count(candidate) > _interval_known_count(best)):
                        best = candidate
                    cur_start = None
        if cur_start is not None:
            candidate = (cur_start, new_hi)
            if (best is None or
                    _interval_known_count(candidate) > _interval_known_count(best)):
                best = candidate

        return _normalize_retained_interval(target_control, best)

    def _best_overlap_branch(control_state: int,
                             interval: Tuple[int, int],
                             probe_offset: int,
                             probe_class: int) -> Optional[Tuple[int, Optional[Tuple[int, int]], int, Tuple[OPCSCommand, ...]]]:
        block_len = block_len_by_control[control_state]
        best: Optional[Tuple[int, Optional[Tuple[int, int]], int, Tuple[OPCSCommand, ...]]] = None
        best_score: Optional[Tuple[int, int]] = None

        for shift in range(1, block_len + 1):
            outcome = _unique_prefix_outcome(
                control_state, interval, probe_offset, probe_class, shift)
            if outcome is None:
                continue

            target_control, prefix_commands = outcome
            if target_control == dfa.dead_state and shift < block_len:
                continue

            target_interval = _retained_interval_after_shift(
                control_state,
                interval,
                probe_offset,
                probe_class,
                shift,
                target_control,
            )
            retained = 0 if target_interval is None else _interval_known_count(target_interval)
            score = (retained, shift)
            if best_score is None or score > best_score:
                best = (target_control, target_interval, shift, prefix_commands)
                best_score = score

        return best

    def root_is_useful(temp_state: _TempState) -> bool:
        if temp_state.probe_offset != 0:
            return True
        return any(
            branch.shift != 1 or branch.target_interval is not None
            for branch in temp_state.branches
        )

    shadow_ids = {
        state: state
        for state in range(dfa.num_states)
    }
    runtime_ids = {
        state: dfa.num_states + state
        for state in range(dfa.num_states)
    }
    temp_states: Dict[Tuple[int, Tuple[int, int]], _TempState] = {}
    root_intervals: Dict[int, Tuple[int, int]] = {}

    def ensure_state(control_state: int,
                     interval: Tuple[int, int]) -> None:
        key = (control_state, interval)
        if key in temp_states:
            return
        if control_state not in cover_classes_by_control:
            return
        if sum(1 for st, _interval in temp_states if st == control_state) >= state_budget:
            raise RuntimeError("gadget budget exceeded")

        probe_offset = _choose_probe_offset(control_state, interval)
        if probe_offset is None:
            return

        possible_targets = _mask_to_states(_interval_target_mask(control_state, interval))
        temp_states[key] = _TempState(
            control_state=control_state,
            interval=interval,
            possible_targets=possible_targets,
            probe_offset=probe_offset,
        )

        cover = cover_classes_by_control[control_state]
        block_len = block_len_by_control[control_state]
        expected_class = cover[probe_offset]
        branch_groups: Dict[
            Tuple[int, Optional[Tuple[int, int]], int, Tuple[OPCSCommand, ...]],
            List[int]
        ] = {}

        for class_id in real_class_ids:
            if class_id == expected_class:
                next_interval = _expand_interval(interval, probe_offset)
                if _interval_is_full(next_interval, block_len):
                    target_control = _cover_path_target(control_state)
                    target_interval = None
                    shift = block_len
                    commands = _cover_path_commands(control_state)
                else:
                    target_control = control_state
                    target_interval = next_interval
                    shift = 0
                    commands = ()
                    ensure_state(control_state, next_interval)
            else:
                overlap = _best_overlap_branch(
                    control_state,
                    interval,
                    probe_offset,
                    class_id,
                )
                if overlap is not None:
                    target_control, target_interval, shift, commands = overlap
                    if target_interval is not None:
                        ensure_state(target_control, target_interval)
                else:
                    target_control = control_state
                    target_interval = None
                    shift = 0
                    commands = ()

            branch_groups.setdefault(
                (target_control, target_interval, shift, commands), [],
            ).append(class_id)

        temp_states[key].branches = tuple(
            _TempBranch(
                class_ids=tuple(sorted(class_ids)),
                target_control=target_control,
                target_interval=target_interval,
                shift=shift,
                commands=commands,
            )
            for (target_control, target_interval, shift, commands), class_ids
            in sorted(branch_groups.items(), key=lambda item: item[1][0])
        )

    for control_state in range(dfa.num_states):
        block_len = block_len_by_control[control_state]
        if control_state == dfa.dead_state or control_state in dfa.accept_states or block_len <= 1:
            continue
        root_intervals[control_state] = EMPTY_INTERVAL
        ensure_state(control_state, EMPTY_INTERVAL)

    next_state_id = 2 * dfa.num_states
    gadget_state_ids: Dict[Tuple[int, Tuple[int, int]], int] = {}
    for control_state, interval in sorted(
            temp_states.keys(),
            key=lambda item: (
                item[0],
                -_interval_known_count(item[1]),
                item[1][0],
                item[1][1],
            )):
        gadget_state_ids[(control_state, interval)] = next_state_id
        next_state_id += 1

    control_entry_states = dict(runtime_ids)
    for control_state, root_interval in root_intervals.items():
        root_state = temp_states.get((control_state, root_interval))
        if root_state is not None and root_is_useful(root_state):
            control_entry_states[control_state] = gadget_state_ids[(control_state, root_interval)]

    states: List[Optional[OPCSState]] = [None] * next_state_id

    for control_state in range(dfa.num_states):
        is_accept = control_state in dfa.accept_states
        eof_accept = (
            dfa.sentinel_257_class >= 0 and
            dfa.transitions[control_state][dfa.sentinel_257_class] in dfa.accept_states
        )
        accept_commands = _control_accept_commands(dfa, tag_runtime, control_state)
        eof_commands = _control_eof_commands(dfa, tag_runtime, control_state)

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
            accept_commands=(),
            eof_accept=eof_accept,
            eof_commands=(),
        )

        runtime_edges = _group_control_edges(
            dfa,
            control_state,
            lambda next_control: control_entry_states[next_control],
            tag_runtime,
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

    for (control_state, interval), temp_state in temp_states.items():
        bma_state_id = gadget_state_ids[(control_state, interval)]
        edge_groups: Dict[Tuple[int, int, Tuple[OPCSCommand, ...]], List[int]] = {}
        for branch in temp_state.branches:
            if branch.target_interval is not None:
                target_state = gadget_state_ids[(branch.target_control, branch.target_interval)]
            elif branch.shift == 0:
                target_state = runtime_ids[branch.target_control]
            else:
                target_state = control_entry_states[branch.target_control]
            key = (target_state, branch.shift, branch.commands)
            bv = edge_groups.setdefault(key, [])
            for class_id in branch.class_ids:
                bv.extend(class_bytes[class_id])

        edges = tuple(
            OPCSEdge(
                byte_values=tuple(sorted(bv)),
                target=target_state,
                shift=shift,
                commands=commands,
            )
            for (target_state, shift, commands), bv
            in sorted(edge_groups.items(), key=lambda item: item[1][0])
        )

        known_bytes = _materialize_known_bytes(
            _known_classes_for_interval(control_state, interval),
            class_representatives,
        )
        states[bma_state_id] = OPCSState(
            id=bma_state_id,
            name=_state_name(control_state, known_bytes),
            kind="gadget",
            control_state=control_state,
            probe_offset=temp_state.probe_offset,
            block_len=block_len_by_control[control_state],
            known_bytes=known_bytes,
            possible_targets=temp_state.possible_targets,
            edges=edges,
            accepting=False,
            accept_commands=(),
            eof_accept=False,
            eof_commands=(),
        )

    assert all(state is not None for state in states)

    initial_commands = _control_step_commands(
        dfa,
        tag_runtime,
        dfa.start_state,
        dfa.sentinel_256_class,
    ) if dfa.sentinel_256_class >= 0 else ()
    initial_accept_commands = _control_accept_commands(
        dfa,
        tag_runtime,
        effective_start,
    ) if effective_start in dfa.accept_states else ()

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
