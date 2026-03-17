"""
Thompson's NFA construction.

Converts regex AST (from parse.py) into an NFA with epsilon and byte
transitions, using the classic Thompson construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from . import parse as ast


# ---------------------------------------------------------------------------
# NFA representation
# ---------------------------------------------------------------------------

@dataclass
class NFAState:
    """A single NFA state."""
    id: int
    # byte_value -> set of target state ids
    byte_transitions: Dict[int, Set[int]] = field(default_factory=dict)
    # epsilon (empty) transitions
    epsilon: Set[int] = field(default_factory=set)

    def add_byte(self, byte_val: int, target: int):
        self.byte_transitions.setdefault(byte_val, set()).add(target)

    def add_epsilon(self, target: int):
        self.epsilon.add(target)


@dataclass
class NFA:
    """Non-deterministic finite automaton."""
    states: List[NFAState]
    start: int
    accept: int  # single accept state (Thompson guarantee)

    def state_count(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# NFA fragment (used during construction)
# ---------------------------------------------------------------------------

@dataclass
class _Fragment:
    start: int
    accept: int


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class NFABuilder:
    """Builds an NFA from a regex AST via Thompson's construction."""

    def __init__(self):
        self.states: List[NFAState] = []

    def _new_state(self) -> int:
        sid = len(self.states)
        self.states.append(NFAState(id=sid))
        return sid

    def build(self, node) -> NFA:
        frag = self._build_node(node)
        return NFA(states=self.states, start=frag.start, accept=frag.accept)

    def _build_node(self, node) -> _Fragment:
        if isinstance(node, ast.Literal):
            return self._build_literal(node)
        elif isinstance(node, ast.Dot):
            return self._build_dot()
        elif isinstance(node, ast.CharClass):
            return self._build_charclass(node)
        elif isinstance(node, ast.Concat):
            return self._build_concat(node)
        elif isinstance(node, ast.Alternate):
            return self._build_alternate(node)
        elif isinstance(node, ast.Repeat):
            return self._build_repeat(node)
        elif isinstance(node, ast.Group):
            return self._build_node(node.child)
        elif isinstance(node, ast.Anchor):
            return self._build_anchor(node)
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    # -- Thompson constructions --

    def _build_literal(self, node: ast.Literal) -> _Fragment:
        s = self._new_state()
        a = self._new_state()
        self.states[s].add_byte(node.byte, a)
        return _Fragment(s, a)

    def _build_dot(self) -> _Fragment:
        s = self._new_state()
        a = self._new_state()
        for b in range(256):
            self.states[s].add_byte(b, a)
        return _Fragment(s, a)

    def _build_charclass(self, node: ast.CharClass) -> _Fragment:
        s = self._new_state()
        a = self._new_state()
        # Compute byte set
        byte_set: Set[int] = set()
        for lo, hi in node.ranges:
            for b in range(lo, hi + 1):
                byte_set.add(b)
        if node.negated:
            byte_set = set(range(256)) - byte_set
        for b in byte_set:
            self.states[s].add_byte(b, a)
        return _Fragment(s, a)

    def _build_concat(self, node: ast.Concat) -> _Fragment:
        if not node.children:
            # Empty concat – matches empty string
            s = self._new_state()
            a = self._new_state()
            self.states[s].add_epsilon(a)
            return _Fragment(s, a)
        frags = [self._build_node(c) for c in node.children]
        # Chain fragments: frag[i].accept --eps--> frag[i+1].start
        for i in range(len(frags) - 1):
            self.states[frags[i].accept].add_epsilon(frags[i + 1].start)
        return _Fragment(frags[0].start, frags[-1].accept)

    def _build_alternate(self, node: ast.Alternate) -> _Fragment:
        s = self._new_state()
        a = self._new_state()
        for child in node.children:
            frag = self._build_node(child)
            self.states[s].add_epsilon(frag.start)
            self.states[frag.accept].add_epsilon(a)
        return _Fragment(s, a)

    def _build_repeat(self, node: ast.Repeat) -> _Fragment:
        if node.min_count == 0 and node.max_count == 1:
            # ? – zero or one
            return self._build_optional(node.child)
        elif node.min_count == 0 and node.max_count is None:
            # * – zero or more
            return self._build_star(node.child)
        elif node.min_count == 1 and node.max_count is None:
            # + – one or more
            return self._build_plus(node.child)
        else:
            # General {m,n}: concat m copies, then (n-m) optional copies
            return self._build_bounded_repeat(node)

    def _build_optional(self, child_node) -> _Fragment:
        """a? – zero or one"""
        s = self._new_state()
        a = self._new_state()
        frag = self._build_node(child_node)
        self.states[s].add_epsilon(frag.start)
        self.states[s].add_epsilon(a)
        self.states[frag.accept].add_epsilon(a)
        return _Fragment(s, a)

    def _build_star(self, child_node) -> _Fragment:
        """a* – zero or more"""
        s = self._new_state()
        a = self._new_state()
        frag = self._build_node(child_node)
        self.states[s].add_epsilon(frag.start)
        self.states[s].add_epsilon(a)
        self.states[frag.accept].add_epsilon(frag.start)
        self.states[frag.accept].add_epsilon(a)
        return _Fragment(s, a)

    def _build_plus(self, child_node) -> _Fragment:
        """a+ – one or more"""
        s = self._new_state()
        a = self._new_state()
        frag = self._build_node(child_node)
        self.states[s].add_epsilon(frag.start)
        self.states[frag.accept].add_epsilon(frag.start)
        self.states[frag.accept].add_epsilon(a)
        return _Fragment(s, a)

    def _build_bounded_repeat(self, node: ast.Repeat) -> _Fragment:
        """General {m, n} repeat."""
        children = []
        # m mandatory copies
        for _ in range(node.min_count):
            children.append(node.child)
        if node.max_count is not None:
            # (n - m) optional copies
            for _ in range(node.max_count - node.min_count):
                children.append(ast.Repeat(child=node.child,
                                           min_count=0, max_count=1))
        else:
            # Unbounded: m copies + star
            children.append(ast.Repeat(child=node.child,
                                       min_count=0, max_count=None))
        return self._build_node(ast.Concat(children=children))

    def _build_anchor(self, node: ast.Anchor) -> _Fragment:
        """Anchors are encoded as special sentinel byte transitions.

        We use byte values 256 (^) and 257 ($) which are outside the
        normal 0–255 range.  The DFA builder treats them specially.
        """
        s = self._new_state()
        a = self._new_state()
        if node.kind == 'start':
            self.states[s].add_byte(256, a)  # sentinel for ^
        else:
            self.states[s].add_byte(257, a)  # sentinel for $
        return _Fragment(s, a)


# ---------------------------------------------------------------------------
# Epsilon-closure utility
# ---------------------------------------------------------------------------

def epsilon_closure(nfa: NFA, state_ids: Set[int]) -> FrozenSet[int]:
    """Compute the epsilon-closure of a set of NFA states."""
    stack = list(state_ids)
    closure = set(state_ids)
    while stack:
        sid = stack.pop()
        for t in nfa.states[sid].epsilon:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_nfa(ast_node) -> NFA:
    """Build an NFA from a regex AST node."""
    builder = NFABuilder()
    return builder.build(ast_node)


def build_search_nfa(nfa: NFA) -> NFA:
    """Build an unanchored search NFA by prepending .* to the original NFA.

    Adds a new start state with a self-loop on all 256 bytes and an
    epsilon transition to the original start state. This allows the DFA
    built from this NFA to match the pattern at any position in a single
    forward pass (O(n) instead of O(n^2)).
    """
    # Deep-copy existing states with shifted IDs (+1)
    new_states: List[NFAState] = []
    offset = 1
    for old_s in nfa.states:
        ns = NFAState(id=old_s.id + offset)
        for bval, targets in old_s.byte_transitions.items():
            ns.byte_transitions[bval] = {t + offset for t in targets}
        ns.epsilon = {t + offset for t in old_s.epsilon}
        new_states.append(ns)

    # Create new start state (id=0) with self-loop on all bytes
    new_start = NFAState(id=0)
    for b in range(256):
        new_start.add_byte(b, 0)
    new_start.add_epsilon(nfa.start + offset)

    all_states = [new_start] + new_states
    return NFA(states=all_states, start=0, accept=nfa.accept + offset)


def reverse_nfa(nfa: NFA) -> NFA:
    """Reverse an NFA: swap start and accept, reverse all transitions.

    Used to build a reverse DFA for leftmost-start-position recovery.
    The reversed NFA, when run backward on the input from the match end,
    finds the leftmost match start position.

    Anchor sentinel transitions (byte values 256=^, 257=$) are skipped
    and replaced with epsilon transitions.  Anchors express positional
    constraints already enforced by the forward pass; the reverse DFA
    only needs to match actual byte content.

    Note: Thompson NFA has a single accept state. The reversed NFA uses
    the original start as accept, the original accept as start.
    Epsilon transitions and byte transitions are all reversed.
    """
    n_states = len(nfa.states)
    new_states = [NFAState(id=i) for i in range(n_states)]

    for old_s in nfa.states:
        # Reverse byte transitions
        for bval, targets in old_s.byte_transitions.items():
            if bval >= 256:
                # Sentinel: treat as epsilon in the reversed NFA
                for t in targets:
                    new_states[t].add_epsilon(old_s.id)
                continue
            for t in targets:
                new_states[t].add_byte(bval, old_s.id)
        # Reverse epsilon transitions
        for t in old_s.epsilon:
            new_states[t].add_epsilon(old_s.id)

    return NFA(states=new_states, start=nfa.accept, accept=nfa.start)
