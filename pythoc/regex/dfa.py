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
    """
    # For each byte value (0–255, plus 256=^ and 257=$), compute a
    # signature: the set of (state_id, target_state_id) pairs.
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

    # Group bytes with the same signature into classes
    sig_to_class: Dict[FrozenSet, int] = {}
    class_map = [0] * 258  # indices 0–255 for bytes, 256 for ^, 257 for $
    next_class = 0

    # Assign class for bytes with no transitions first (the "dead" class)
    dead_sig = frozenset()
    sig_to_class[dead_sig] = next_class
    next_class += 1

    for bval in range(258):
        sig = signatures.get(bval, frozenset())
        if sig not in sig_to_class:
            sig_to_class[sig] = next_class
            next_class += 1
        class_map[bval] = sig_to_class[sig]

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
    # Whether the pattern is anchored at start (^)
    anchored_start: bool = False
    # Whether the pattern is anchored at end ($)
    anchored_end: bool = False
    # Dead state id (if one exists)
    dead_state: int = -1


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


def build_dfa(nfa: NFA) -> DFA:
    """Convert an NFA to a DFA via subset construction.

    Handles anchor sentinels (byte 256 = ^, byte 257 = $) by detecting
    them and setting anchored_start / anchored_end flags on the DFA.
    """
    class_map, num_classes = _compute_byte_classes(nfa)

    # Build reverse map: class_id -> list of byte values
    class_to_bytes: Dict[int, List[int]] = {}
    for bval in range(258):
        cid = class_map[bval]
        class_to_bytes.setdefault(cid, []).append(bval)

    # Start state: epsilon-closure of NFA start
    start_closure = epsilon_closure(nfa, {nfa.start})

    # Check for anchors: if start closure can transition on sentinel 256,
    # the pattern starts with ^.
    anchored_start = False
    anchored_end = False

    # Handle start anchor: try to move through the ^ sentinel
    anchor_start_targets = _move(nfa, start_closure, 256)
    if anchor_start_targets:
        anchored_start = True
        # The real start state is after consuming ^
        start_closure = epsilon_closure(nfa, anchor_start_targets)

    # We'll detect $ anchoring during DFA construction by checking if
    # accept is only reachable after consuming the $ sentinel.

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

    # We need to only iterate over real byte classes (0–255), not anchor sentinels
    real_byte_classes: Set[int] = set()
    for bval in range(256):
        real_byte_classes.add(class_map[bval])
    # Also include the $ sentinel class for end-anchor detection
    dollar_class = class_map[257]

    while worklist:
        current_closure = worklist.pop()
        current_id = state_map[current_closure]

        # Check for $ transition
        dollar_targets = _move(nfa, current_closure, 257)
        if dollar_targets:
            dollar_closure = epsilon_closure(nfa, dollar_targets)
            if nfa.accept in dollar_closure:
                anchored_end = True
                # This state is accepting only if at end of input
                # We mark it as accepting and set the flag
                accept_states.add(current_id)

        for cid in real_byte_classes:
            targets = _move_class(nfa, current_closure, class_map,
                                  cid, class_to_bytes)
            if not targets:
                continue
            target_closure = epsilon_closure(nfa, targets)
            if not target_closure:
                continue
            target_id = _get_or_create(target_closure)
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
        class_map=class_map[:256],  # only real bytes
        anchored_start=anchored_start,
        anchored_end=anchored_end,
        dead_state=dead_state,
    )
