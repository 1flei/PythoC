"""
NFA-to-DFA conversion via subset construction.

Includes byte-class compression and dead-state elimination.
Produces a transition table suitable for codegen.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .nfa import NFA, NFAState, epsilon_closure


# ---------------------------------------------------------------------------
# Byte-class compression
# ---------------------------------------------------------------------------

def _compute_byte_classes(nfa: NFA) -> Tuple[List[int], int]:
    """Compute byte equivalence classes from an NFA.

    Two bytes are equivalent if they appear in exactly the same set of
    transitions across all NFA states.  Returns (class_map, num_classes)
    where class_map[byte] -> class_id.

    Sentinel bytes (256=^, 257=$) are ALWAYS placed in their own
    dedicated classes, even if their NFA transition signature matches
    a real byte.  This is required because sentinels use transparent
    semantics during subset construction (states without the sentinel
    transition stay in place), which differs from real byte semantics.
    """
    # For each byte value (0-255), compute a signature: the set of
    # (state_id, target_state_id) pairs.
    signatures: Dict[int, FrozenSet] = {}
    for state in nfa.states:
        for bval, targets in state.byte_transitions.items():
            if bval > 257:
                continue
            key = (state.id, frozenset(targets))
            signatures.setdefault(bval, set()).add(key)

    # Convert mutable sets to frozen
    for bval in signatures:
        if isinstance(signatures[bval], set):
            signatures[bval] = frozenset(signatures[bval])

    # Group real bytes (0-255) with the same signature into classes
    sig_to_class: Dict[FrozenSet, int] = {}
    class_map = [0] * 258  # indices 0-255 for bytes, 256 for ^, 257 for $
    next_class = 0

    # Assign class for bytes with no transitions first (the "dead" class)
    dead_sig = frozenset()
    sig_to_class[dead_sig] = next_class
    next_class += 1

    # Assign classes for real bytes only (0-255)
    for bval in range(256):
        sig = signatures.get(bval, frozenset())
        if sig not in sig_to_class:
            sig_to_class[sig] = next_class
            next_class += 1
        class_map[bval] = sig_to_class[sig]

    # Force sentinel bytes into their own dedicated classes
    # Sentinel 256 (^) always gets its own class
    class_map[256] = next_class
    next_class += 1
    # Sentinel 257 ($) always gets its own class
    class_map[257] = next_class
    next_class += 1

    return class_map, next_class


# ---------------------------------------------------------------------------
# DFA representation
# ---------------------------------------------------------------------------

@dataclass
class DFA:
    """Deterministic finite automaton."""
    num_states: int
    num_classes: int
    start_state: int
    accept_states: Set[int]
    # transition[state_id][class_id] -> next_state_id  (-1 = dead)
    transitions: List[List[int]]
    # byte -> equivalence class
    class_map: List[int]
    # Dead state id (if one exists)
    dead_state: int = -1
    # Equivalence class for sentinel byte 256 (^) — feed once at position 0
    sentinel_256_class: int = -1
    # Equivalence class for sentinel byte 257 ($) — feed once at end-of-input
    sentinel_257_class: int = -1
    # Tag map: DFA state -> set of tag names from constituent NFA states
    tag_map: Dict[int, Set[str]] = field(default_factory=dict)
    # NFA epsilon-closures for each DFA state, kept for compile-time analyses
    state_closures: List[FrozenSet[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------

def _move(nfa: NFA, state_set: FrozenSet[int], byte_val: int) -> Set[int]:
    """Compute the set of states reachable from state_set on byte_val."""
    result: Set[int] = set()
    for sid in state_set:
        targets = nfa.states[sid].byte_transitions.get(byte_val, set())
        result.update(targets)
    return result


def _move_class(nfa: NFA, state_set: FrozenSet[int],
                class_map: List[int], class_id: int,
                all_bytes_for_class: Dict[int, List[int]]) -> Set[int]:
    """Compute the set of NFA states reachable from state_set on any byte
    in the given equivalence class."""
    result: Set[int] = set()
    for bval in all_bytes_for_class.get(class_id, []):
        result.update(_move(nfa, state_set, bval))
    return result


def _move_sentinel(nfa: NFA, state_set: FrozenSet[int],
                   sentinel_byte: int) -> Set[int]:
    """Move on a sentinel byte with transparency.

    States with the sentinel transition advance to targets.
    States without it stay (transparent / implicit self-loop).
    This is the core of the per-branch anchor fix: states that
    don't use the anchor remain in place, allowing non-anchored
    branches to coexist with anchored ones.
    """
    result: Set[int] = set()
    for sid in state_set:
        targets = nfa.states[sid].byte_transitions.get(sentinel_byte, None)
        if targets is not None:
            result.update(targets)
        else:
            result.add(sid)  # transparent
    return result


def build_dfa(nfa: NFA) -> DFA:
    """Convert an NFA to a DFA via subset construction.

    Handles anchor sentinels (byte 256 = ^, byte 257 = $) via
    transparent sentinel transitions: states without a sentinel
    transition stay in place (transparent), while states with one
    advance to their targets.  This makes anchors per-branch rather
    than global, fixing patterns like ``^a|bc`` and ``ab|c$``.

    The sentinel byte classes are included in the DFA transition
    table so codegen can feed them at position 0 (^) and end-of-input ($).
    """
    class_map, num_classes = _compute_byte_classes(nfa)

    # Build reverse map: class_id -> list of byte values
    class_to_bytes: Dict[int, List[int]] = {}
    for bval in range(258):
        cid = class_map[bval]
        class_to_bytes.setdefault(cid, []).append(bval)

    # Start state: epsilon-closure of NFA start
    start_closure = epsilon_closure(nfa, {nfa.start})

    # Identify sentinel byte classes (now guaranteed to be their own classes)
    sentinel_256_class = class_map[256]  # ^ sentinel
    sentinel_257_class = class_map[257]  # $ sentinel
    sentinel_classes = {sentinel_256_class, sentinel_257_class}

    # Real byte classes (0-255 only, excluding sentinel classes)
    real_byte_classes: Set[int] = set()
    for bval in range(256):
        real_byte_classes.add(class_map[bval])

    # All classes to process in worklist: real bytes + sentinel classes
    all_classes = real_byte_classes | sentinel_classes

    # -- Subset construction --
    dfa_states: List[FrozenSet[int]] = []
    state_map: Dict[FrozenSet[int], int] = {}
    transitions: List[List[int]] = []
    accept_states: Set[int] = set()
    worklist: List[FrozenSet[int]] = []

    def _get_or_create(closure: FrozenSet[int]) -> int:
        if closure in state_map:
            return state_map[closure]
        sid = len(dfa_states)
        dfa_states.append(closure)
        state_map[closure] = sid
        transitions.append([-1] * num_classes)
        worklist.append(closure)
        # Check if this DFA state is accepting
        if nfa.accept in closure:
            accept_states.add(sid)
        return sid

    start_id = _get_or_create(start_closure)

    # Tag map: DFA state -> set of tag names from constituent NFA states
    tag_map: Dict[int, Set[str]] = {}

    def _collect_tags(closure: FrozenSet[int], dfa_state: int):
        """Collect tag names from NFA states in a DFA state's closure."""
        tags = set()
        for nfa_sid in closure:
            state_obj = nfa.states[nfa_sid]
            if hasattr(state_obj, 'tag') and state_obj.tag is not None:
                tags.add(state_obj.tag)
        if tags:
            tag_map[dfa_state] = tags

    _collect_tags(start_closure, start_id)

    while worklist:
        current_closure = worklist.pop()
        current_id = state_map[current_closure]

        for cid in all_classes:
            if cid in sentinel_classes:
                # Sentinel class: use transparent move
                # Determine which sentinel byte is in this class
                if cid == sentinel_256_class:
                    sentinel_byte = 256
                else:
                    sentinel_byte = 257
                targets = _move_sentinel(nfa, current_closure, sentinel_byte)
                target_closure = epsilon_closure(nfa, targets)
            else:
                # Pure real byte class
                targets = _move_class(nfa, current_closure, class_map,
                                      cid, class_to_bytes)
                if not targets:
                    continue
                target_closure = epsilon_closure(nfa, targets)

            if not target_closure:
                continue
            target_id = _get_or_create(target_closure)
            _collect_tags(target_closure, target_id)
            transitions[current_id][cid] = target_id

    # Identify dead state: a non-accepting state with all transitions -> itself or -1
    dead_state = -1
    for sid in range(len(dfa_states)):
        if sid in accept_states:
            continue
        all_dead = True
        for cid in range(num_classes):
            t = transitions[sid][cid]
            if t != -1 and t != sid:
                all_dead = False
                break
        if all_dead:
            dead_state = sid
            break

    # Replace -1 with dead_state in transitions if we found one
    if dead_state == -1:
        # Create an explicit dead state
        dead_state = len(dfa_states)
        dfa_states.append(frozenset())
        transitions.append([-1] * num_classes)
        # Dead state transitions all go to itself
        for cid in range(num_classes):
            transitions[dead_state][cid] = dead_state

    # Replace all -1 with dead_state
    for sid in range(len(transitions)):
        for cid in range(num_classes):
            if transitions[sid][cid] == -1:
                transitions[sid][cid] = dead_state

    return DFA(
        num_states=len(dfa_states),
        num_classes=num_classes,
        start_state=start_id,
        accept_states=accept_states,
        transitions=transitions,
        class_map=class_map[:256],  # only real bytes for data lookup
        dead_state=dead_state,
        sentinel_256_class=sentinel_256_class,
        sentinel_257_class=sentinel_257_class,
        tag_map=tag_map,
        state_closures=dfa_states,
    )
