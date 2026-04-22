"""
Tagged OPCS-BMA artifacts on top of the TDFA control automaton.

This module provides the compile-time data types and low-level helpers
that ``tbma.build_tagged_tbma`` and the codegen backend consume. The
control automaton is always a :class:`pythoc.regex.tdfa.TDFA`: tag
commands live on TDFA edges directly, and this module simply lifts
them into the T-BMA ``OPCSCommand`` vocabulary.

The T-BMA runtime keeps two layers:

  * an unreachable shadow control skeleton that preserves the original
    control DFA one-to-one (used by codegen layout checks);
  * runtime entry / control / gadget states used by compiled execution,
    with per-state probe offsets, per-edge shifts, and per-edge tag
    commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .tdfa import TDFA, TDFACommand


INTERNAL_TAG_PREFIX = "__pythoc_internal_"


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


# ---------------------------------------------------------------------------
# TDFA command -> OPCS command lifting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phi-transform normalization for ordered register commands
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TDFA class-layout helpers
# ---------------------------------------------------------------------------

def _real_class_layout(dfa: TDFA) -> Tuple[Tuple[int, ...],
                                           Dict[int, int],
                                           Dict[int, Tuple[int, ...]]]:
    """Return (real_class_ids, class_representatives, class_bytes) for a TDFA.

    Only real byte classes are returned; sentinel classes live in their
    own buckets inside the TDFA and are handled separately by the
    gadget builder / codegen.
    """
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


# ---------------------------------------------------------------------------
# TDFA control-path command extraction
# ---------------------------------------------------------------------------

def _control_accept_commands(control: TDFA,
                             state: int) -> Tuple[OPCSCommand, ...]:
    return _commands_from_tdfa_commands(control.accept_commands.get(state, ()))


def _control_step_commands(control: TDFA,
                           state: int,
                           class_id: int) -> Tuple[OPCSCommand, ...]:
    return _commands_from_tdfa_commands(
        control.transition_commands.get((state, class_id), ())
    )


def _control_path_commands(control: TDFA,
                           start_state: int,
                           class_path: Sequence[int]) -> Tuple[OPCSCommand, ...]:
    commands: List[OPCSCommand] = []
    state = start_state
    for delta, class_id in enumerate(class_path):
        commands.extend(_shift_commands(
            _control_step_commands(control, state, class_id),
            delta,
        ))
        state = control.transitions[state][class_id]
    return _normalize_opcs_commands(commands)


def _control_edge_commands(control: TDFA,
                           state: int,
                           byte_val: int) -> Tuple[OPCSCommand, ...]:
    return _control_path_commands(
        control,
        state,
        (control.class_map[byte_val],),
    )


def _control_eof_commands(control: TDFA,
                          state: int) -> Tuple[OPCSCommand, ...]:
    if control.sentinel_257_class < 0:
        return ()
    target = control.transitions[state][control.sentinel_257_class]
    if target not in control.accept_states:
        return ()

    return (
        _control_step_commands(control, state, control.sentinel_257_class) +
        _control_accept_commands(control, target)
    )


def _group_control_edges(control: TDFA,
                         state: int,
                         target_lookup) -> Tuple[OPCSEdge, ...]:
    groups: Dict[Tuple[int, Tuple[OPCSCommand, ...]], List[int]] = {}
    for byte_val in range(256):
        next_control = control.transitions[state][control.class_map[byte_val]]
        target = target_lookup(next_control)
        commands = _control_edge_commands(control, state, byte_val)
        groups.setdefault((target, commands), []).append(byte_val)
    return tuple(
        OPCSEdge(
            byte_values=tuple(byte_vals),
            target=target,
            shift=1,
            commands=commands,
        )
        for (target, commands), byte_vals in sorted(
            groups.items(), key=lambda item: item[1][0])
    )


def _effective_start_control(dfa: TDFA) -> int:
    effective_start = dfa.start_state
    if dfa.sentinel_256_class >= 0:
        effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]
    return effective_start


def _artifact_layout(control: TDFA) -> Tuple[
        Tuple[str, ...],
        Tuple[str, ...],
        Dict[str, int],
        int,
        Tuple[int, ...],
        ]:
    return (
        control.tag_names,
        control.public_tag_names,
        control.tag_slots,
        control.register_count,
        control.output_registers,
    )
