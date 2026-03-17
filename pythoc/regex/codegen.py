"""
PythoC code generation for regex DFA matchers.

Takes a DFA (from dfa.py) and generates pattern-specialized @compile functions.
Instead of a generic table-driven DFA loop, each pattern produces bespoke
native code where DFA states become explicit if/elif branches.

Pipeline: pattern -> DFA -> AST -> meta.compile_generated -> LLVM native code.

Uses pythoc.meta quasi-quote templates to generate AST, avoiding raw
ast constructor boilerplate.
"""

from __future__ import annotations
import ast
import hashlib
from typing import Dict, List, Optional, Set, Tuple

from pythoc import meta
from pythoc.builtin_entities.types import i8, i32, i64, u8, u64, ptr

from .dfa import DFA
from .parse import parse
from .nfa import build_nfa
from .dfa import build_dfa


def _pattern_digest(pattern: str) -> str:
    """Short hex digest of pattern for unique naming."""
    h = hashlib.md5(pattern.encode('utf-8')).hexdigest()[:12]
    return h


def compile_pattern(pattern: str):
    """Full pipeline: pattern string -> DFA."""
    re_ast = parse(pattern)
    nfa = build_nfa(re_ast)
    dfa = build_dfa(nfa)
    return dfa


# ---------------------------------------------------------------------------
# Quasi-quote expression templates
# ---------------------------------------------------------------------------

@meta.quote_expr
def _tpl_i32(v):
    return i32(v)


@meta.quote_expr
def _tpl_u8(v):
    return u8(v)


@meta.quote_expr
def _tpl_i64(v):
    return i64(v)


@meta.quote_expr
def _tpl_u64(v):
    return u64(v)


@meta.quote_expr
def _tpl_eq(lhs, rhs):
    return lhs == rhs


@meta.quote_expr
def _tpl_lte(lhs, rhs):
    return lhs <= rhs


@meta.quote_expr
def _tpl_gte(lhs, rhs):
    return lhs >= rhs


@meta.quote_expr
def _tpl_add(lhs, rhs):
    return lhs + rhs


@meta.quote_expr
def _tpl_and(lhs, rhs):
    return lhs and rhs


@meta.quote_expr
def _tpl_or(lhs, rhs):
    return lhs or rhs


# ---------------------------------------------------------------------------
# Quasi-quote structural templates (large skeletons)
# ---------------------------------------------------------------------------

@meta.quote_stmts
def _tpl_state_machine(start_state_val, dead_state_val, i_start,
                        dispatch, accept_check):
    """DFA state machine that breaks on dead state.

    accept_check is spliced between dispatch and dead-state break,
    allowing early return when an accept state is reached mid-scan.
    """
    state: i32 = i32(start_state_val)
    i: u64 = i_start
    while i < n:
        ch: i32 = i32(data[i]) & 0xFF
        dispatch
        accept_check
        if state == i32(dead_state_val):
            break
        i = i + 1


@meta.quote_stmts
def _tpl_unanchored_is_match(inner):
    """Unanchored is_match: try each start position."""
    start: u64 = 0
    while start <= n:
        inner
        start = start + 1
    return u8(0)


@meta.quote_stmts
def _tpl_unanchored_search(inner):
    """Unanchored search: try each start position."""
    start: u64 = 0
    while start <= n:
        inner
        start = start + 1
    return i64(-1)


@meta.quote_stmt
def _tpl_if(test_expr, then_body):
    if test_expr:
        then_body


@meta.quote_stmt
def _tpl_if_else(test_expr, then_body, else_body):
    if test_expr:
        then_body
    else:
        else_body


@meta.quote_stmt
def _tpl_assign(target, value):
    target = value


@meta.quote_stmt
def _tpl_return(val):
    return val


# ---------------------------------------------------------------------------
# Convenience wrappers (template call -> raw AST node)
# ---------------------------------------------------------------------------

def _q(template, *args):
    """Call a template, return the raw AST node."""
    return template(*args).node


def _q_i32(v):
    """i32(v) expression."""
    return _q(_tpl_i32, meta.const(v))


def _q_u8(v):
    """u8(v) expression."""
    return _q(_tpl_u8, meta.const(v))


def _q_i64(v):
    """i64(v) expression."""
    return _q(_tpl_i64, v)


def _q_eq(lhs, rhs):
    """lhs == rhs expression."""
    return _q(_tpl_eq, lhs, rhs)


def _q_lte(lhs, rhs):
    """lhs <= rhs expression."""
    return _q(_tpl_lte, lhs, rhs)


def _q_gte(lhs, rhs):
    """lhs >= rhs expression."""
    return _q(_tpl_gte, lhs, rhs)


def _q_add(lhs, rhs):
    """lhs + rhs expression."""
    return _q(_tpl_add, lhs, rhs)


def _q_and(lhs, rhs):
    """lhs and rhs expression."""
    return _q(_tpl_and, lhs, rhs)


def _q_or_chain(parts):
    """Chain parts with 'or'. Returns single part if length 1."""
    result = parts[0]
    for p in parts[1:]:
        result = _q(_tpl_or, result, p)
    return result


def _q_return(val):
    """return val statement."""
    return _q(_tpl_return, val)


def _q_assign(target, value):
    """target = value statement."""
    return _q(_tpl_assign, target, value)


def _splice(stmts):
    """Shorthand for meta.splice_stmts."""
    return meta.splice_stmts(stmts)


# ---------------------------------------------------------------------------
# If/elif chain builder (dynamic branching)
# ---------------------------------------------------------------------------

def _if_elif_chain(branches, else_body=None):
    """Build if/elif/else chain from list of (test, body) pairs.

    Uses quasi-quote templates internally. Variable-length branching
    can't be expressed as a single template, so this builds the chain
    iteratively from the tail.
    """
    if not branches:
        raise ValueError("At least one branch required")

    result = None
    for idx in range(len(branches) - 1, -1, -1):
        test, body = branches[idx]
        if result is None and not else_body:
            result = _q(_tpl_if, test, _splice(body))
        elif result is None:
            result = _q(_tpl_if_else, test, _splice(body),
                        _splice(list(else_body)))
        else:
            result = _q(_tpl_if_else, test, _splice(body),
                        _splice([result]))
    return result


# ---------------------------------------------------------------------------
# DFA analysis helpers
# ---------------------------------------------------------------------------

def _byte_transitions_for_state(dfa: DFA, state: int) -> Dict[int, List[int]]:
    """For a DFA state, return {target_state: [byte_values...]} mapping.

    Only includes non-dead transitions. byte_values are the actual byte
    values (0-255) that lead to the target state.
    """
    class_to_target = {}
    for cls in range(dfa.num_classes):
        target = dfa.transitions[state][cls]
        if target != dfa.dead_state:
            class_to_target[cls] = target

    target_to_bytes: Dict[int, List[int]] = {}
    for byte_val in range(256):
        cls = dfa.class_map[byte_val]
        if cls in class_to_target:
            target = class_to_target[cls]
            target_to_bytes.setdefault(target, []).append(byte_val)

    return target_to_bytes


# ---------------------------------------------------------------------------
# Byte-condition builder
# ---------------------------------------------------------------------------

def _byte_condition(byte_values: List[int], ch_name: str = "ch") -> ast.expr:
    """Build condition that checks if ch_name is in byte_values.

    Uses range checks for contiguous ranges, individual comparisons
    for small sets.
    """
    if not byte_values:
        return meta.const(False)

    sorted_vals = sorted(byte_values)

    # Find contiguous ranges
    ranges = []
    start = sorted_vals[0]
    end = sorted_vals[0]
    for v in sorted_vals[1:]:
        if v == end + 1:
            end = v
        else:
            ranges.append((start, end))
            start = v
            end = v
    ranges.append((start, end))

    ch = meta.ref(ch_name)
    parts = []
    for lo, hi in ranges:
        if lo == hi:
            parts.append(_q_eq(ch, meta.const(lo)))
        elif lo == 0:
            parts.append(_q_lte(ch, meta.const(hi)))
        elif hi == 255:
            parts.append(_q_gte(ch, meta.const(lo)))
        else:
            parts.append(_q_and(
                _q_lte(meta.const(lo), ch), _q_lte(ch, meta.const(hi)),
            ))

    return _q_or_chain(parts)


# ---------------------------------------------------------------------------
# State dispatch builder (the dynamic if/elif part of the state machine)
# ---------------------------------------------------------------------------

def _build_dispatch(dfa: DFA, active_states: List[int]) -> ast.stmt:
    """Build the state dispatch if/elif chain for the DFA loop body."""
    state_ref = meta.ref('state')
    state_branches = []
    for s in active_states:
        test = _q_eq(state_ref, _q_i32(s))
        trans = _byte_transitions_for_state(dfa, s)

        if not trans:
            body = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
        else:
            t_branches = []
            for target, byte_vals in trans.items():
                cond = _byte_condition(byte_vals, 'ch')
                t_body = [_q_assign(meta.ref('state'), _q_i32(target))]
                t_branches.append((cond, t_body))
            t_else = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
            body = [_if_elif_chain(t_branches, else_body=t_else)]

        state_branches.append((test, body))

    state_else = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
    return _if_elif_chain(state_branches, else_body=state_else)


# ---------------------------------------------------------------------------
# Accept-check builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# State machine builder
# ---------------------------------------------------------------------------

def _build_state_machine(dfa: DFA, i_start=None,
                         accept_check_stmts=None) -> List[ast.stmt]:
    """Build the full state machine using a quasi-quote skeleton.

    Args:
        dfa: The DFA.
        i_start: AST expression for initial i value (default: u64(0)).
        accept_check_stmts: Optional list of AST stmts inserted between
            dispatch and dead-state break, for early accept detection.
    """
    # Collect active states
    active_states = []
    for s in range(dfa.num_states):
        if s == dfa.dead_state:
            continue
        if _byte_transitions_for_state(dfa, s):
            active_states.append(s)
    for s in dfa.accept_states:
        if s != dfa.dead_state and s not in active_states:
            active_states.append(s)
    active_states.sort()

    if not active_states:
        dispatch = _q_assign(meta.ref('state'), _q_i32(dfa.dead_state))
    else:
        dispatch = _build_dispatch(dfa, active_states)

    if i_start is None:
        i_start = _q(_tpl_u64, meta.const(0))

    if accept_check_stmts is None:
        accept_check_stmts = []

    return _q(_tpl_state_machine,
              meta.const(dfa.start_state),
              meta.const(dfa.dead_state),
              i_start,
              _splice([dispatch]),
              _splice(accept_check_stmts))


# ---------------------------------------------------------------------------
# Inner match (for unanchored patterns)
# ---------------------------------------------------------------------------

def _build_inner_match(dfa: DFA, return_on_match, start_var: str) -> List[ast.stmt]:
    """Build inner match logic: state machine + accept check at end.

    The state machine starts scanning at position 'start_var' and uses
    in-loop accept detection for non-end-anchored patterns.
    """
    stmts = []
    state_ref = meta.ref('state')
    start_ref = meta.ref(start_var)

    # If start state is accepting, handle it
    if dfa.start_state in dfa.accept_states:
        if dfa.anchored_end:
            stmts.append(_q(_tpl_if,
                            _q_eq(start_ref, meta.ref('n')),
                            _splice([_q_return(return_on_match)])))
        else:
            return [_q_return(return_on_match)]

    # Build in-loop accept check for non-end-anchored patterns.
    # After each transition, if we're in an accept state, return early.
    accept_check_stmts = []
    if not dfa.anchored_end:
        accept_list = sorted(dfa.accept_states)
        if accept_list:
            branches = []
            for s in accept_list:
                branches.append((_q_eq(state_ref, _q_i32(s)),
                                 [_q_return(return_on_match)]))
            accept_check_stmts = [_if_elif_chain(branches)]

    # State machine starts at 'start' position
    stmts.extend(_build_state_machine(
        dfa,
        i_start=start_ref,
        accept_check_stmts=accept_check_stmts,
    ))

    # Post-loop accept check (for end-anchored patterns)
    if dfa.anchored_end:
        accept_list = sorted(dfa.accept_states)
        for s in accept_list:
            inner_if = _q(_tpl_if,
                          _q_eq(meta.ref('i'), meta.ref('n')),
                          _splice([_q_return(return_on_match)]))
            stmts.append(_q(_tpl_if,
                            _q_eq(state_ref, _q_i32(s)),
                            _splice([inner_if])))

    return stmts


# ---------------------------------------------------------------------------
# Body builders: is_match
# ---------------------------------------------------------------------------

def _build_is_match_body(dfa: DFA) -> List[ast.stmt]:
    """Build the full body for an is_match function."""
    if dfa.anchored_start:
        return _build_anchored_is_match(dfa)

    inner = _build_inner_match(dfa, _q_u8(1), 'start')
    return _q(_tpl_unanchored_is_match, _splice(inner))


def _build_anchored_is_match(dfa: DFA) -> List[ast.stmt]:
    """Build is_match body for anchored-start pattern."""
    stmts = []

    if dfa.start_state in dfa.accept_states and not dfa.anchored_end:
        return [_q_return(_q_u8(1))]

    if dfa.start_state in dfa.accept_states and dfa.anchored_end:
        stmts.append(_q(_tpl_if,
                        _q_eq(meta.ref('n'), meta.const(0)),
                        _splice([_q_return(_q_u8(1))])))

    # Build in-loop accept check for non-end-anchored patterns
    accept_check_stmts = []
    state_ref = meta.ref('state')
    if not dfa.anchored_end:
        accept_list = sorted(dfa.accept_states)
        if accept_list:
            branches = []
            for s in accept_list:
                branches.append((_q_eq(state_ref, _q_i32(s)),
                                 [_q_return(_q_u8(1))]))
            accept_check_stmts = [_if_elif_chain(branches)]

    stmts.extend(_build_state_machine(
        dfa, accept_check_stmts=accept_check_stmts))

    if dfa.anchored_end:
        # Only accept at end of input
        accept_list = sorted(dfa.accept_states)
        branches = []
        for s in accept_list:
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_u8(1))]))
        stmts.append(_if_elif_chain(branches))
    stmts.append(_q_return(_q_u8(0)))
    return stmts


# ---------------------------------------------------------------------------
# Body builders: search
# ---------------------------------------------------------------------------

def _build_search_body(dfa: DFA) -> List[ast.stmt]:
    """Build the full body for a search function."""
    if dfa.anchored_start:
        return _build_anchored_search(dfa)

    inner = _build_inner_match(dfa, _q_i64(meta.ref('start')), 'start')
    return _q(_tpl_unanchored_search, _splice(inner))


def _build_anchored_search(dfa: DFA) -> List[ast.stmt]:
    """Build search body for anchored-start pattern."""
    stmts = []

    if dfa.start_state in dfa.accept_states and not dfa.anchored_end:
        return [_q_return(_q_i64(meta.const(0)))]

    if dfa.start_state in dfa.accept_states and dfa.anchored_end:
        stmts.append(_q(_tpl_if,
                        _q_eq(meta.ref('n'), meta.const(0)),
                        _splice([_q_return(_q_i64(meta.const(0)))])))

    # Build in-loop accept check for non-end-anchored patterns
    accept_check_stmts = []
    state_ref = meta.ref('state')
    if not dfa.anchored_end:
        accept_list = sorted(dfa.accept_states)
        if accept_list:
            branches = []
            for s in accept_list:
                branches.append((_q_eq(state_ref, _q_i32(s)),
                                 [_q_return(_q_i64(meta.const(0)))]))
            accept_check_stmts = [_if_elif_chain(branches)]

    stmts.extend(_build_state_machine(
        dfa, accept_check_stmts=accept_check_stmts))

    if dfa.anchored_end:
        branches = []
        for s in sorted(dfa.accept_states):
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_i64(meta.const(0)))]))
        stmts.append(_if_elif_chain(branches))
    stmts.append(_q_return(_q_i64(meta.const(-1))))
    return stmts


# ---------------------------------------------------------------------------
# Compiled function generation via pythoc.meta
# ---------------------------------------------------------------------------

def _build_compiled_fn(dfa: DFA, digest: str, kind: str):
    """Generate a @compile function using pythoc.meta.

    Args:
        dfa: The DFA.
        digest: Pattern digest for unique naming.
        kind: "is_match" or "search".

    Returns:
        The compiled function.
    """
    if kind == "is_match":
        func_name = "regex_is_match"
        suffix = digest
        return_type = u8
        body_stmts = _build_is_match_body(dfa)
    elif kind == "search":
        func_name = "regex_search"
        suffix = digest + "_search"
        return_type = i64
        body_stmts = _build_search_body(dfa)
    else:
        raise ValueError("Unknown kind: {}".format(kind))

    for node in body_stmts:
        ast.fix_missing_locations(node)

    gf = meta.func(
        name=func_name,
        params=[("n", u64), ("data", ptr[i8])],
        return_type=return_type,
        body=body_stmts,
        required_globals={
            'i8': i8, 'i32': i32, 'i64': i64,
            'u8': u8, 'u64': u64, 'ptr': ptr,
        },
        source_file=__file__,
    )

    return meta.compile_generated(gf, suffix=suffix)


# ---------------------------------------------------------------------------
# CompiledRegex
# ---------------------------------------------------------------------------

class CompiledRegex:
    """Holds a compiled regex pattern and provides match/search operations.

    This object performs matching using a Python-level DFA simulation.
    It can also generate @compile functions for native PythoC execution.
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.dfa = compile_pattern(pattern)
        self._digest = _pattern_digest(pattern)

    def _run_dfa(self, data: bytes, start_pos: int = 0) -> bool:
        """Run the DFA on data starting from start_pos.

        Returns True if the DFA accepts.
        For anchored_end patterns, the DFA only accepts if we consume
        all remaining bytes.
        """
        dfa = self.dfa
        state = dfa.start_state
        for i in range(start_pos, len(data)):
            byte_val = data[i]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                return False
        return state in dfa.accept_states

    def _run_dfa_find_end(self, data: bytes, start_pos: int = 0) -> int:
        """Run the DFA from start_pos and find the end of the earliest match.

        Returns the index past the last matched byte, or -1 if no match.
        For anchored_end patterns, only the end of data counts as a match.
        """
        dfa = self.dfa
        state = dfa.start_state
        last_match = -1

        if state in dfa.accept_states and not dfa.anchored_end:
            last_match = start_pos

        for i in range(start_pos, len(data)):
            byte_val = data[i]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                break
            if state in dfa.accept_states and not dfa.anchored_end:
                last_match = i + 1

        # For $ anchor: accept only if at end of data
        if dfa.anchored_end and state in dfa.accept_states:
            last_match = len(data)

        return last_match

    def is_match(self, data: bytes) -> bool:
        """Test if data matches (full match if anchored, partial otherwise).

        For anchored patterns (^...$), tests full match.
        For unanchored patterns, tests if data matches from the start.
        """
        if self.dfa.anchored_start:
            end = self._run_dfa_find_end(data, 0)
            if self.dfa.anchored_end:
                return end == len(data)
            return end >= 0
        else:
            for start in range(len(data) + 1):
                end = self._run_dfa_find_end(data, start)
                if end >= 0:
                    return True
            return False

    def fullmatch(self, data: bytes) -> bool:
        """Test if the entire data matches the pattern."""
        return self._run_dfa(data, 0)

    def search(self, data: bytes) -> int:
        """Search for pattern in data.

        Returns the start position of the first match, or -1 if not found.
        """
        if self.dfa.anchored_start:
            end = self._run_dfa_find_end(data, 0)
            if self.dfa.anchored_end:
                return 0 if end == len(data) else -1
            return 0 if end >= 0 else -1
        else:
            for start in range(len(data)):
                end = self._run_dfa_find_end(data, start)
                if end >= 0:
                    return start
            return -1

    def find_span(self, data: bytes):
        """Find the span (start, end) of the first match.

        Returns (start, end) tuple or None if no match.
        end is exclusive.
        """
        if self.dfa.anchored_start:
            end = self._run_dfa_find_end(data, 0)
            if self.dfa.anchored_end:
                return (0, len(data)) if end == len(data) else None
            return (0, end) if end >= 0 else None
        else:
            for start in range(len(data)):
                end = self._run_dfa_find_end(data, start)
                if end >= 0:
                    return (start, end)
            return None

    # -----------------------------------------------------------------
    # @compile function generation
    # -----------------------------------------------------------------

    def generate_is_match_fn(self):
        """Generate a @compile function for is_match.

        Returns a PythoC compiled function:
            is_match(n: u64, data: ptr[i8]) -> u8
        """
        return _build_compiled_fn(self.dfa, self._digest, "is_match")

    def generate_search_fn(self):
        """Generate a @compile function for search.

        Returns a PythoC compiled function:
            search(n: u64, data: ptr[i8]) -> i64
        """
        return _build_compiled_fn(self.dfa, self._digest, "search")
