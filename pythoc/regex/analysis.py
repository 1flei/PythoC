"""
Compile-time control analysis for OPCS BMA construction.

Provides per-control-state analysis: shortest success length, block
length selection, and block target exploration used by the OPCS builder.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, Optional, Sequence, Set

from .dfa import DFA


DEFAULT_OPCS_MAX_W = 8


def _terminal_control_states(dfa: DFA) -> FrozenSet[int]:
    """Control states that can accept without consuming another real byte."""
    terminals: Set[int] = set(dfa.accept_states)
    if dfa.sentinel_257_class >= 0:
        for state in range(dfa.num_states):
            if dfa.transitions[state][dfa.sentinel_257_class] in dfa.accept_states:
                terminals.add(state)
    return frozenset(terminals)


def _real_successors(dfa: DFA, state: int) -> FrozenSet[int]:
    """Unique real-byte successors from one DFA state."""
    targets: Set[int] = set()
    for byte_val in range(256):
        targets.add(dfa.transitions[state][dfa.class_map[byte_val]])
    return frozenset(targets)


def compute_shortest_success_lengths(dfa: DFA) -> Dict[int, Optional[int]]:
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


def possible_block_targets(dfa: DFA,
                           start_state: int,
                           known_bytes: Sequence[int]) -> FrozenSet[int]:
    """Final control targets for all concrete blocks matching ``known_bytes``.

    ``known_bytes`` is a fixed-length block description where ``-1`` means
    "unknown byte here".  Every concrete byte assignment is explored through
    the deterministic DFA, including paths that fall into the dead state.
    """
    current: Set[int] = {start_state}
    for byte_val in known_bytes:
        next_states: Set[int] = set()
        if byte_val >= 0:
            cls = dfa.class_map[byte_val]
            for state in current:
                next_states.add(dfa.transitions[state][cls])
        else:
            for state in current:
                next_states.update(_real_successors(dfa, state))
        current = next_states
    return frozenset(current)
