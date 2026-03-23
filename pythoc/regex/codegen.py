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
import ctypes
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pythoc import meta
from pythoc.builtin_entities.types import i8, i32, i64, u8, u64, ptr
from pythoc.builtin_entities.scoped_label import label, goto, goto_end

from .dfa import DFA
from .parse import (
    parse, Dot, Concat, Repeat, Anchor, Tag,
)
from .nfa import build_nfa, NFA
from .dfa import build_dfa
from .analysis import (analyze_pattern, compute_accept_depths,
                       find_linear_chains, compute_skip_table,
                       compute_state_skips,
                       PatternInfo, AcceptDepthInfo, LinearChain, SkipInfo,
                       StateSkipInfo)


INTERNAL_TAG_PREFIX = "__pythoc_internal_"
INTERNAL_BEG_TAG = INTERNAL_TAG_PREFIX + "beg"
INTERNAL_END_TAG = INTERNAL_TAG_PREFIX + "end"


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


def compile_pattern_from_ast(re_ast):
    """Full pipeline: AST node -> DFA."""
    nfa = build_nfa(re_ast)
    return build_dfa(nfa)


def rewrite_for_search(ast_node) -> object:
    """Prepend lazy .* to pattern AST for unanchored search.

    Produces the same DFA as the old NFA-level build_search_nfa().

    Injects internal begin/end tags around the pattern body so the
    search entry mode shares the same tagged core shape as ordinary
    matching.
    """
    prefix = Repeat(child=Dot(), min_count=0, max_count=None, lazy=True)
    beg = Tag(name=INTERNAL_BEG_TAG)
    end = Tag(name=INTERNAL_END_TAG)
    if isinstance(ast_node, Concat):
        return Concat(children=[prefix, beg] + ast_node.children + [end])
    return Concat(children=[prefix, beg, ast_node, end])


def compile_search_dfa(pattern: str):
    """Build a search DFA (prepend .*?) for unanchored O(n) matching.

    Uses AST-level rewrite instead of NFA-level build_search_nfa().
    """
    re_ast = parse(pattern)
    search_ast = rewrite_for_search(re_ast)
    nfa = build_nfa(search_ast)
    return build_dfa(nfa)


def compile_search_dfa_from_ast(re_ast):
    """Build a search DFA from AST (prepend .*?)."""
    search_ast = rewrite_for_search(re_ast)
    nfa = build_nfa(search_ast)
    return build_dfa(nfa)


def compile_reverse_dfa(pattern: str):
    """Build a reverse DFA for start-position recovery.

    The reverse DFA matches pattern content backward over real bytes.
    Anchor sentinels (256=^, 257=$) are positional constraints already
    enforced by the forward search DFA, so reverse_nfa skips them.
    """
    re_ast = parse(pattern)
    nfa = build_nfa(re_ast)
    rev_nfa = reverse_nfa(nfa)
    return build_dfa(rev_nfa)


def compile_reverse_dfa_from_ast(re_ast):
    """Build a reverse DFA from AST."""
    nfa = build_nfa(re_ast)
    rev_nfa = reverse_nfa(nfa)
    return build_dfa(rev_nfa)


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


@meta.quote_expr
def _tpl_sub(lhs, rhs):
    return lhs - rhs


@meta.quote_expr
def _tpl_lt(lhs, rhs):
    return lhs < rhs


@meta.quote_expr
def _tpl_data_byte_at(idx_expr):
    return i32(data[idx_expr]) & 0xFF


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
def _tpl_existing_state_machine(dead_state_val, i_start, dispatch, accept_check):
    """State machine loop when `state` is initialized externally."""
    i: u64 = i_start
    while i < n:
        ch: i32 = i32(data[i]) & 0xFF
        dispatch
        accept_check
        if state == i32(dead_state_val):
            return u8(0)
        i = i + 1


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
def _tpl_assign_typed(target, type_ann, value):
    target: type_ann = value


@meta.quote_stmt
def _tpl_return(val):
    return val


@meta.quote_stmt
def _tpl_with_label(label_name, body):
    with label(label_name):
        body


@meta.quote_stmt
def _tpl_goto(label_name):
    goto(label_name)


@meta.quote_stmts
def _tpl_goto_state_preamble(done_label_name):
    """Bounds check + byte load for a goto FSM state block."""
    if i >= n:
        goto(done_label_name)
    ch: i32 = i32(data[i]) & 0xFF


@meta.quote_stmts
def _tpl_advance_goto(label_name):
    """Advance i by 1 and jump to label_name."""
    i = i + u64(1)
    goto(label_name)


@meta.quote_stmts
def _tpl_accept_1pass(neg_depth):
    """1-pass search accept: advance i, return start = i - depth."""
    i = i + u64(1)
    return i64(i + neg_depth)


@meta.quote_stmts
def _tpl_accept_return_match_start():
    """1-pass search accept: return tracked match_start."""
    return match_start


@meta.quote_stmts
def _tpl_set_match_start():
    """Set match_start = i64(i) at restart-to-pattern transition."""
    match_start = i64(i)


@meta.quote_stmts
def _tpl_skip_advance(skip_dist_val, label_name):
    """Advance i by skip distance and jump to label_name."""
    i = i + u64(skip_dist_val)
    goto(label_name)


@meta.quote_stmt
def _tpl_pass():
    pass


@meta.quote_stmts
def _tpl_goto_search_program(init_match_start, goto_start, label_blocks, miss_return):
    """Top-level skeleton for goto-based search codegen."""
    i: u64 = u64(0)
    match_start: i64 = i64(-1)
    init_match_start
    goto_start
    label_blocks
    miss_return


@meta.quote_stmts
def _tpl_goto_state_program(skip_probe, eof_guard, load_byte, transitions):
    """Skeleton for a single goto-FSM state block."""
    skip_probe
    eof_guard
    load_byte
    transitions


@meta.quote_stmt
def _tpl_load_byte_as(var_name, idx_expr):
    var_name: i32 = i32(data[idx_expr]) & 0xFF


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


def _q_sub(lhs, rhs):
    """lhs - rhs expression."""
    return _q(_tpl_sub, lhs, rhs)


def _q_lt(lhs, rhs):
    """lhs < rhs expression."""
    return _q(_tpl_lt, lhs, rhs)


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


def _q_label_block(label_name: str, body: List[ast.stmt]):
    """with label(label_name): <body>"""
    return _q(_tpl_with_label, meta.const(label_name), _splice(body))


def _q_goto(label_name: str):
    """goto(label_name) statement."""
    return _q(_tpl_goto, meta.const(label_name))


def _q_advance_goto(label_name: str):
    """i = i + u64(1); goto(label_name)"""
    return _q(_tpl_advance_goto, meta.const(label_name))


def _q_accept_1pass(depth: int):
    """i = i + u64(1); return i64(i + (-depth))"""
    return _q(_tpl_accept_1pass, meta.const(-depth))


def _q_accept_return_match_start():
    """return match_start"""
    return _q(_tpl_accept_return_match_start)


def _q_set_match_start():
    """match_start = i64(i)"""
    return _q(_tpl_set_match_start)


def _q_skip_advance(skip_dist: int, label_name: str):
    """i = i + u64(skip_dist); goto(label_name)"""
    return _q(_tpl_skip_advance, meta.const(skip_dist), meta.const(label_name))


def _q_pass():
    """pass statement."""
    return _q(_tpl_pass)


def _stmt_list(node_or_nodes) -> List[ast.stmt]:
    """Normalize a stmt or stmt-list into a list."""
    if node_or_nodes is None:
        return []
    if isinstance(node_or_nodes, list):
        return list(node_or_nodes)
    return [node_or_nodes]


def _body_or_pass(node_or_nodes) -> List[ast.stmt]:
    """Ensure a non-empty statement body for splicing into templates."""
    stmts = _stmt_list(node_or_nodes)
    if stmts:
        return stmts
    return [_q_pass()]


def _q_compile_time_if(flag: bool, body) -> ast.stmt:
    """Emit a compile-time-constant if statement."""
    return _q(_tpl_if, meta.const(bool(flag)), _splice(_body_or_pass(body)))


def _q_compile_time_if_else(flag: bool, then_body, else_body) -> ast.stmt:
    """Emit a compile-time-constant if/else statement."""
    return _q(
        _tpl_if_else,
        meta.const(bool(flag)),
        _splice(_body_or_pass(then_body)),
        _splice(_body_or_pass(else_body)),
    )


def _raw_name(name: str, ctx) -> ast.Name:
    return ast.Name(id=name, ctx=ctx)


def _raw_subscript(name: str, idx_expr: ast.expr, ctx) -> ast.Subscript:
    return ast.Subscript(
        value=ast.Name(id=name, ctx=ast.Load()),
        slice=idx_expr,
        ctx=ctx,
    )


def _q_ptr_assign(ptr_name: str, idx_expr: ast.expr, value_expr: ast.expr) -> ast.Assign:
    return ast.Assign(
        targets=[_raw_subscript(ptr_name, idx_expr, ast.Store())],
        value=value_expr,
    )


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
# Label/goto helpers (for goto FSM state machine)
# ---------------------------------------------------------------------------

def _state_label(state: int) -> str:
    """Label name for a DFA state."""
    return "s{}".format(state)


# ---------------------------------------------------------------------------
# Goto FSM builder for forward search
# ---------------------------------------------------------------------------

def _build_goto_forward_search(search_dfa: DFA,
                                accept_depths: AcceptDepthInfo = None,
                                skip_info: SkipInfo = None,
                                state_skips: Dict[int, StateSkipInfo] = None,
                                restart_states: Set[int] = None,
                                ) -> List[ast.stmt]:
    """Build goto-based forward search FSM for search DFA.

    Each DFA state becomes a labeled block with direct goto transitions.
    This compiles to LLVM basic blocks with br instructions -- no state
    variable, no if/elif dispatch overhead.

    Uses restart-state classification to track match_start at runtime.
    Restart states are search DFA states with self-loops on >= 128 bytes
    (the .*? prefix). When transitioning from restart to non-restart,
    match_start = i64(i).

    When skip_info is available (W >= 2), state 0 gets a skip-probe
    prefix that enables O(n/W) scanning.
    """
    if restart_states is None:
        restart_states = set()

    # Compute effective start after sentinel 256 (compile-time)
    effective_start = search_dfa.transitions[search_dfa.start_state][search_dfa.sentinel_256_class]

    if effective_start in search_dfa.accept_states:
        return [_q_return(_q_i64(meta.const(0)))]
    if effective_start == search_dfa.dead_state:
        return [_q_return(_q_i64(meta.const(-1)))]

    active_states = _get_active_states(search_dfa)
    done_label = "rx_done"

    # Ensure effective_start is in active_states
    if effective_start not in active_states and effective_start != search_dfa.dead_state:
        active_states.append(effective_start)
        active_states.sort()

    init_match_start = _q_compile_time_if(
        effective_start not in restart_states,
        [_q_assign(meta.ref('match_start'), _q_i64(meta.const(0)))],
    )

    goto_start = _q_goto(_state_label(effective_start))

    label_blocks = []
    for s in active_states:
        label_name = _state_label(s)
        body = _build_goto_state_body(
            search_dfa, s, active_states, done_label,
            skip_info=skip_info if s == search_dfa.start_state else None,
            state_skip=state_skips.get(s) if state_skips else None,
            restart_states=restart_states,
        )
        label_blocks.append(_q_label_block(label_name, body))

    label_blocks.append(_q_label_block(done_label, [_q_pass()]))

    return _q(
        _tpl_goto_search_program,
        _splice([init_match_start]),
        _splice([goto_start]),
        _splice(label_blocks),
        _splice([_q_return(_q_i64(meta.const(-1)))]),
    )


def _build_goto_state_body(dfa: DFA, state: int,
                            active_states: List[int],
                            done_label: str,
                            skip_info: SkipInfo = None,
                            state_skip: StateSkipInfo = None,
                            restart_states: Set[int] = None,
                            ) -> List[ast.stmt]:
    """Build the body of a single state's label block in the goto FSM.

    The body: bounds check (with sentinel 257 check), optional skip probe,
    byte load, transitions with match_start tracking, accept handling.

    Restart-state tracking: when transitioning from a restart state to a
    non-restart state, emits match_start = i64(i) before the goto.
    On accept: returns match_start immediately.

    skip_info: global skip table (Boyer-Moore, start state only).
    state_skip: per-state skip probe from compute_state_skips().
    """
    if restart_states is None:
        restart_states = set()

    state_is_restart = state in restart_states
    skip_probe_stmt = _q_compile_time_if_else(
        skip_info is not None and skip_info.window >= 2,
        _build_skip_probe(dfa, state, skip_info) if skip_info is not None else [],
        _build_state_skip_probe(dfa, state, state_skip) if state_skip is not None else [],
    )

    # --- Bounds check: when i >= n, check sentinel 257 then done ---
    end_target = dfa.transitions[state][dfa.sentinel_257_class]
    eof_guard_stmt = _q_compile_time_if_else(
        end_target in dfa.accept_states,
        [_q(_tpl_if, _q_gte(meta.ref('i'), meta.ref('n')),
            _splice([_q_return(meta.ref('match_start'))]))],
        [_q(_tpl_if, _q_gte(meta.ref('i'), meta.ref('n')),
            _splice([_q_goto(done_label)]))],
    )

    load_byte_stmt = _q(_tpl_load_byte_as, meta.ident('ch'), meta.ref('i'))

    # --- Byte transitions ---
    trans = _byte_transitions_for_state(dfa, state)
    t_branches = []
    for target, byte_vals in trans.items():
        cond = _byte_condition(byte_vals, 'ch')
        target_label = _state_label(target)
        target_is_restart = target in restart_states
        set_match_start = _q_compile_time_if(
            state_is_restart and not target_is_restart,
            _q_set_match_start(),
        )

        if target in dfa.accept_states:
            # Accept state reached
            # If transitioning from restart to non-restart (accept),
            # set match_start first
            t_body = [set_match_start]
            t_body.extend(_stmt_list(_q_accept_return_match_start()))
        elif target == dfa.dead_state:
            # Dead state: partial match failed
            t_body = [_q_goto(done_label)]
        elif target in active_states:
            # Normal transition: track match_start, advance i, goto target
            t_body = [set_match_start]
            t_body.extend(_stmt_list(_q_advance_goto(target_label)))
        else:
            # Target not in active states -> dead
            t_body = [_q_goto(done_label)]

        t_branches.append((cond, t_body))

    default_transition_body = _stmt_list(_q_compile_time_if_else(
        state == dfa.start_state,
        _q_advance_goto(_state_label(dfa.start_state)),
        [_q_goto(_state_label(dfa.start_state))],
    ))
    transition_chain_body = (
        [_if_elif_chain(t_branches, else_body=default_transition_body)]
        if t_branches else
        [_q_pass()]
    )

    transition_stmt = _q_compile_time_if_else(
        bool(t_branches),
        transition_chain_body,
        [_q_goto(done_label)],
    )

    return _q(
        _tpl_goto_state_program,
        _splice([skip_probe_stmt]),
        _splice([eof_guard_stmt]),
        _splice([load_byte_stmt]),
        _splice([transition_stmt]),
    )


def _build_skip_probe(dfa: DFA, state: int,
                       skip_info: SkipInfo) -> List[ast.stmt]:
    """Build skip-table probe code for state 0 of the goto FSM.

    Probes data[i + W-1] and uses the skip table to jump ahead when
    the probe byte cannot appear at the last position of any match window.

    The probe is wrapped in a bounds check: if i + W <= n.
    """
    W = skip_info.window
    skip_table = skip_info.skip_table
    state_label = _state_label(state)

    # Group bytes by skip distance
    skip_groups: Dict[int, List[int]] = {}
    for byte_val in range(256):
        skip_dist = skip_table[byte_val]
        skip_groups.setdefault(skip_dist, []).append(byte_val)

    # If no bytes have skip > 0, skip probe is useless
    has_useful_skip = any(d > 0 for d in skip_groups)
    if not has_useful_skip:
        return []

    # Find the most common skip distance (for the else branch)
    max_count_dist = max(skip_groups, key=lambda d: len(skip_groups[d]))

    # Build probe body
    probe_body = []

    # probe: i32 = i32(data[i + u64(W-1)]) & 0xFF
    probe_idx = _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(W - 1)))
    probe_body.append(_q(_tpl_load_byte_as, meta.ident('probe'), probe_idx))

    # Build if/elif chain on probe byte, grouped by skip distance
    # Put skip==0 as fall-through (pass), non-zero skips branch back to s0
    branches = []
    for skip_dist in sorted(skip_groups.keys()):
        if skip_dist == max_count_dist:
            continue  # this goes in the else branch
        byte_vals = skip_groups[skip_dist]
        cond = _byte_condition(byte_vals, 'probe')
        if skip_dist == 0:
            # Fall through to normal processing
            branch_body = [ast.Pass()]
        else:
            branch_body = _q_skip_advance(skip_dist, state_label)
        branches.append((cond, branch_body))

    # Else branch: the most common skip distance
    if max_count_dist == 0:
        else_body = [ast.Pass()]
    else:
        else_body = _q_skip_advance(max_count_dist, state_label)

    if branches:
        probe_body.append(_if_elif_chain(branches, else_body=else_body))
    else:
        # Only the default skip distance exists
        probe_body.extend(else_body)

    # Wrap in bounds check: if i + u64(W) <= n:
    bounds = _q_lte(
        _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(W))),
        meta.ref('n'))

    return [_q(_tpl_if, bounds, _splice(probe_body))]


def _build_state_skip_probe(dfa: DFA, state: int,
                             state_skip: StateSkipInfo) -> List[ast.stmt]:
    """Build per-state skip probe code for the goto FSM.

    Similar to _build_skip_probe but uses StateSkipInfo from
    compute_state_skips() instead of the global SkipInfo.

    Probes data[i + offset] and checks if the byte is in live_bytes.
    If not, advances i by shift and gotos back to the same state label.
    """
    offset = state_skip.offset
    shift = state_skip.shift
    live_bytes = state_skip.live_bytes
    state_lbl = _state_label(state)

    if shift <= 0 or len(live_bytes) >= 256:
        return []

    # Build probe body
    probe_body = []

    # probe: i32 = i32(data[i + u64(offset)]) & 0xFF
    probe_idx = _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(offset)))
    probe_body.append(_q(_tpl_load_byte_as, meta.ident('probe'), probe_idx))

    # Build condition: probe NOT in live_bytes -> skip ahead
    # We check the complement: if probe is in live_bytes, fall through.
    # Otherwise, advance by shift and goto back to this state.
    dead_bytes = sorted(set(range(256)) - live_bytes)
    if not dead_bytes:
        return []

    dead_cond = _byte_condition(dead_bytes, 'probe')
    skip_body = _q_skip_advance(shift, state_lbl)
    probe_body.append(_q(_tpl_if, dead_cond, _splice(skip_body)))

    # Wrap in bounds check: if i + u64(offset + 1) <= n:
    bounds = _q_lte(
        _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(offset + 1))),
        meta.ref('n'))

    return [_q(_tpl_if, bounds, _splice(probe_body))]


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

def _build_dispatch(dfa: DFA, active_states: List[int],
                    state_var: str = 'state',
                    ch_var: str = 'ch',
                    chains: List[LinearChain] = None) -> ast.stmt:
    """Build the state dispatch if/elif chain for the DFA loop body.

    If chains is provided, chain start states emit memcmp-style
    multi-byte comparisons instead of single-byte dispatch.
    """
    # Build chain lookup: start_state -> chain
    chain_map: Dict[int, LinearChain] = {}
    chain_internal: Set[int] = set()
    if chains:
        # First, compute internal states and chain membership for each chain
        chain_internals_map: Dict[int, Set[int]] = {}  # chain_start -> internal states
        chain_members_map: Dict[int, Set[int]] = {}    # chain_start -> all member states
        for c in chains:
            internals = set()
            members = {c.start_state}
            current = c.start_state
            for b_val in c.byte_sequence[:-1]:
                target = dfa.transitions[current][dfa.class_map[b_val]]
                if target != dfa.dead_state and target != current:
                    internals.add(target)
                    members.add(target)
                    current = target
            chain_internals_map[c.start_state] = internals
            chain_members_map[c.start_state] = members

        # Check each chain: if ANY internal state has incoming edges from
        # outside the chain, the entire chain must be invalidated (not used
        # for memcmp optimization).
        incoming: Dict[int, Set[int]] = {}
        all_internals = set()
        for internals in chain_internals_map.values():
            all_internals |= internals
        for s in all_internals:
            incoming[s] = set()
        for src in range(dfa.num_states):
            if src == dfa.dead_state:
                continue
            seen_targets: Set[int] = set()
            for cls in range(dfa.num_classes):
                target = dfa.transitions[src][cls]
                if target in all_internals and target not in seen_targets:
                    seen_targets.add(target)
                    incoming[target].add(src)

        valid_chains = []
        for c in chains:
            internals = chain_internals_map[c.start_state]
            members = chain_members_map[c.start_state]
            chain_valid = True
            for s in internals:
                if incoming[s] - members:
                    chain_valid = False
                    break
            if chain_valid:
                chain_map[c.start_state] = c
                chain_internal |= internals
                valid_chains.append(c)
        chains = valid_chains

    state_ref = meta.ref(state_var)
    state_branches = []
    for s in active_states:
        # Skip chain-internal states
        if s in chain_internal:
            continue

        test = _q_eq(state_ref, _q_i32(s))

        if s in chain_map:
            # Emit memcmp-style comparison for this chain
            chain = chain_map[s]
            body = _build_chain_compare(chain, dfa, state_var, ch_var)
        else:
            trans = _byte_transitions_for_state(dfa, s)
            if not trans:
                body = [_q_assign(meta.ref(state_var),
                                  _q_i32(dfa.dead_state))]
            else:
                t_branches = []
                for target, byte_vals in trans.items():
                    cond = _byte_condition(byte_vals, ch_var)
                    t_body = [_q_assign(meta.ref(state_var), _q_i32(target))]
                    t_branches.append((cond, t_body))
                t_else = [_q_assign(meta.ref(state_var),
                                    _q_i32(dfa.dead_state))]
                body = [_if_elif_chain(t_branches, else_body=t_else)]

        state_branches.append((test, body))

    state_else = [_q_assign(meta.ref(state_var), _q_i32(dfa.dead_state))]
    return _if_elif_chain(state_branches, else_body=state_else)


def _build_chain_compare(chain: LinearChain, dfa: DFA,
                          state_var: str, ch_var: str) -> List[ast.stmt]:
    """Build memcmp-style comparison for a linear chain.

    Generates code like:
        if ch == b0 and i + chain_len <= n:
            if i32(data[i+1]) & 0xFF == b1 and
               i32(data[i+2]) & 0xFF == b2 ...:
                state = end_state
                i = i + (chain_len - 1)
            else:
                state = dead
        else:
            state = dead
    """
    byte_seq = chain.byte_sequence
    chain_len = len(byte_seq)

    # First byte check (using ch which is already loaded)
    first_byte_check = _q_eq(meta.ref(ch_var), meta.const(byte_seq[0]))

    # Bounds check: i + chain_len <= n
    bounds_check = _q_lte(
        _q_add(meta.ref('i'), meta.const(chain_len)),
        meta.ref('n'))

    # Combined first-byte + bounds check
    outer_test = _q_and(first_byte_check, bounds_check)

    dead_body = [_q_assign(meta.ref(state_var), _q_i32(dfa.dead_state))]

    if chain_len == 1:
        # Single byte chain (shouldn't happen, chains are >= 2, but handle)
        match_body = [
            _q_assign(meta.ref(state_var), _q_i32(chain.end_state)),
        ]
        return [_q(_tpl_if_else, outer_test,
                    _splice(match_body), _splice(dead_body))]

    # Build remaining byte checks: data[i+1] == b1, data[i+2] == b2, ...
    remaining_checks = []
    for offset in range(1, chain_len):
        idx_expr = _q_add(meta.ref('i'), meta.const(offset))
        data_byte = _q(_tpl_data_byte_at, idx_expr)
        remaining_checks.append(
            _q_eq(data_byte, meta.const(byte_seq[offset])))

    inner_test = remaining_checks[0]
    for check in remaining_checks[1:]:
        inner_test = _q_and(inner_test, check)

    # On full match: state = end_state, i += chain_len - 1
    match_body = [
        _q_assign(meta.ref(state_var), _q_i32(chain.end_state)),
        _q_assign(meta.ref('i'),
                  _q_add(meta.ref('i'), meta.const(chain_len - 1))),
    ]

    inner_stmt = _q(_tpl_if_else, inner_test,
                     _splice(match_body), _splice(dead_body))

    return [_q(_tpl_if_else, outer_test,
               _splice([inner_stmt]), _splice(dead_body))]


# ---------------------------------------------------------------------------
# Accept-check builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# State machine builder
# ---------------------------------------------------------------------------

def _build_state_machine(dfa: DFA, i_start=None,
                         accept_check_stmts=None,
                         chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build the full state machine using a quasi-quote skeleton.

    Args:
        dfa: The DFA.
        i_start: AST expression for initial i value (default: u64(0)).
        accept_check_stmts: Optional list of AST stmts inserted between
            dispatch and dead-state break, for early accept detection.
        chains: Optional list of LinearChains for memcmp optimization.
    """
    active_states = _get_active_states(dfa)

    if not active_states:
        dispatch = _q_assign(meta.ref('state'), _q_i32(dfa.dead_state))
    else:
        dispatch = _build_dispatch(dfa, active_states, chains=chains)

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


def _get_active_states(dfa: DFA) -> List[int]:
    """Get sorted list of active (non-dead) DFA states."""
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
    return active_states


# ---------------------------------------------------------------------------
# Body builders: is_match

# ---------------------------------------------------------------------------
# Body builders: is_match
# ---------------------------------------------------------------------------

def _build_is_match_body(dfa: DFA,
                          chains: List[LinearChain] = None,
                          info: PatternInfo = None) -> List[ast.stmt]:
    """Build the full body for an is_match function.

    Uses sentinel protocol:
    1. Compute effective_start = transitions[start][sentinel_256_class] (compile-time)
    2. Run state machine from effective_start
    3. After loop, check sentinel 257 transition for $ accept
    """
    stmts = []

    # Compute effective start state after sentinel 256 (compile-time constant)
    effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]

    # If effective start is dead, pattern can never match
    if effective_start == dfa.dead_state:
        return [_q_return(_q_u8(0))]

    # If effective start is an accept state, matches empty string
    if effective_start in dfa.accept_states:
        return [_q_return(_q_u8(1))]

    # Build in-loop accept check: if state reaches an accept state mid-scan
    accept_check_stmts = []
    state_ref = meta.ref('state')
    accept_list = sorted(dfa.accept_states)
    if accept_list:
        branches = []
        for s in accept_list:
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_u8(1))]))
        accept_check_stmts = [_if_elif_chain(branches)]

    # Build state machine starting from effective_start
    active_states = _get_active_states(dfa)
    dispatch = _build_dispatch(dfa, active_states, chains=chains) if active_states else \
        _q_assign(meta.ref('state'), _q_i32(dfa.dead_state))

    stmts.extend(_q(_tpl_state_machine,
                     meta.const(effective_start),
                     meta.const(dfa.dead_state),
                     _q(_tpl_u64, meta.const(0)),
                     _splice([dispatch]),
                     _splice(accept_check_stmts)))

    # After loop: check sentinel 257 transition per-state
    # Must check ALL non-dead states, not just active_states, because
    # some states (like the final state before $) may have no real byte
    # transitions but do have a sentinel 257 transition to accept.
    end_accept_branches = []
    for s in range(dfa.num_states):
        if s == dfa.dead_state:
            continue
        end_target = dfa.transitions[s][dfa.sentinel_257_class]
        if end_target in dfa.accept_states:
            end_accept_branches.append(
                (_q_eq(state_ref, _q_i32(s)),
                 [_q_return(_q_u8(1))]))
    if end_accept_branches:
        stmts.append(_if_elif_chain(end_accept_branches))

    stmts.append(_q_return(_q_u8(0)))
    return stmts


def _build_fullmatch_body(dfa: DFA,
                          chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build the full body for a fullmatch function."""
    stmts = []
    effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]

    if effective_start == dfa.dead_state:
        return [_q_return(_q_u8(0))]

    active_states = _get_active_states(dfa)
    dispatch = _build_dispatch(dfa, active_states, chains=chains) if active_states else \
        _q_assign(meta.ref('state'), _q_i32(dfa.dead_state))

    stmts.extend(_q(_tpl_state_machine,
                     meta.const(effective_start),
                     meta.const(dfa.dead_state),
                     _q(_tpl_u64, meta.const(0)),
                     _splice([dispatch]),
                     _splice([])))

    end_accept_branches = []
    state_ref = meta.ref('state')
    for s in range(dfa.num_states):
        if s == dfa.dead_state:
            continue
        end_target = dfa.transitions[s][dfa.sentinel_257_class]
        if end_target in dfa.accept_states:
            end_accept_branches.append(
                (_q_eq(state_ref, _q_i32(s)),
                 [_q_return(_q_u8(1))]))
    if end_accept_branches:
        stmts.append(_if_elif_chain(end_accept_branches))

    stmts.append(_q_return(_q_u8(0)))
    return stmts


def _emit_tag_slot_writes(tag_slots: Tuple[int, ...], pos_expr: ast.expr) -> List[ast.stmt]:
    """Emit writes of i64(pos_expr) into out[1 + slot]."""
    stmts: List[ast.stmt] = []
    value_expr = _q_i64(pos_expr)
    for slot in tag_slots:
        stmts.append(_q_ptr_assign(
            'out',
            ast.Constant(1 + slot),
            value_expr,
        ))
    return stmts


def _emit_match_success(tag_runtime: DFATagRuntime, accept_state: int,
                        end_pos_expr: ast.expr) -> List[ast.stmt]:
    stmts: List[ast.stmt] = []
    stmts.extend(_emit_tag_slot_writes(
        tag_runtime.accept_tag_slots.get(accept_state, ()),
        end_pos_expr,
    ))
    stmts.append(_q_ptr_assign('out', ast.Constant(0), _q_i64(end_pos_expr)))
    stmts.append(_q_return(_q_u8(1)))
    return stmts


def _tagged_transition_groups_for_state(dfa: DFA, state: int,
                                        tag_runtime: DFATagRuntime
                                        ) -> Dict[Tuple[int, Tuple[int, ...]], List[int]]:
    """Group bytes by (target_state, transition_tag_slots)."""
    groups: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for byte_val in range(256):
        cls = dfa.class_map[byte_val]
        target = dfa.transitions[state][cls]
        if target == dfa.dead_state:
            continue
        slots = tag_runtime.transition_tag_slots.get((state, cls), ())
        groups.setdefault((target, slots), []).append(byte_val)
    return groups


def _build_tagged_dispatch(dfa: DFA, tag_runtime: DFATagRuntime,
                           active_states: List[int]) -> ast.stmt:
    """Build state dispatch that also writes dynamic tag positions."""
    state_ref = meta.ref('state')
    state_branches = []

    for s in active_states:
        test = _q_eq(state_ref, _q_i32(s))
        trans = _tagged_transition_groups_for_state(dfa, s, tag_runtime)
        if not trans:
            body = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
        else:
            t_branches = []
            for (target, slots), byte_vals in trans.items():
                cond = _byte_condition(byte_vals, 'ch')
                t_body = []
                t_body.extend(_emit_tag_slot_writes(slots, meta.ref('i')))
                t_body.append(_q_assign(meta.ref('state'), _q_i32(target)))
                t_branches.append((cond, t_body))
            t_else = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
            body = [_if_elif_chain(t_branches, else_body=t_else)]
        state_branches.append((test, body))

    state_else = [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))]
    return _if_elif_chain(state_branches, else_body=state_else)


def _build_match_info_body(dfa: DFA, tag_runtime: DFATagRuntime) -> List[ast.stmt]:
    """Build compiled span/tag recovery from a known start position."""
    effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]

    # Initialize output: out[0] = end, out[1:] = user-visible tag positions.
    output_init = [
        _q_ptr_assign('out', ast.Constant(idx), _q_i64(meta.const(-1)))
        for idx in range(1 + len(tag_runtime.tag_names))
    ]

    state_init = _q(_tpl_assign_typed, meta.ident('state'),
                    meta.type_expr(i32), _q_i32(dfa.start_state))

    start_guards = [
        _q(_tpl_if, _q_lt(meta.ref('n'), meta.ref('start')),
           _splice([_q_return(_q_u8(0))])),
        _q(_tpl_if, _q_eq(meta.ref('start'), _q(_tpl_u64, meta.const(0))),
           _splice([_q_assign(meta.ref('state'), _q_i32(effective_start))])),
        _q(_tpl_if, _q_eq(meta.ref('state'), _q_i32(dfa.dead_state)),
           _splice([_q_return(_q_u8(0))])),
    ]

    initial_accept_branches = []
    for s in sorted(dfa.accept_states):
        initial_accept_branches.append(
            (_q_eq(meta.ref('state'), _q_i32(s)),
             _emit_match_success(tag_runtime, s, meta.ref('start'))))
    initial_accept_stmt = _q_compile_time_if(
        bool(initial_accept_branches),
        [_if_elif_chain(initial_accept_branches)] if initial_accept_branches else [],
    )

    active_states = _get_active_states(dfa)
    dispatch = _q_compile_time_if_else(
        bool(active_states),
        [_build_tagged_dispatch(dfa, tag_runtime, active_states)] if active_states else [],
        [_q_assign(meta.ref('state'), _q_i32(dfa.dead_state))],
    )

    consumed_pos = _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(1)))
    in_loop_accept_branches = []
    for s in sorted(dfa.accept_states):
        in_loop_accept_branches.append(
            (_q_eq(meta.ref('state'), _q_i32(s)),
             _emit_match_success(tag_runtime, s, consumed_pos)))
    loop_accept_stmt = _q_compile_time_if(
        bool(in_loop_accept_branches),
        [_if_elif_chain(in_loop_accept_branches)] if in_loop_accept_branches else [],
    )

    loop_body = _q(
        _tpl_existing_state_machine,
        meta.const(dfa.dead_state),
        meta.ref('start'),
        _splice([dispatch]),
        _splice([loop_accept_stmt]),
    )

    eof_branches = []
    if dfa.sentinel_257_class >= 0:
        for s in range(dfa.num_states):
            if s == dfa.dead_state:
                continue
            end_target = dfa.transitions[s][dfa.sentinel_257_class]
            if end_target not in dfa.accept_states:
                continue
            branch_body = []
            branch_body.extend(_emit_tag_slot_writes(
                tag_runtime.transition_tag_slots.get((s, dfa.sentinel_257_class), ()),
                meta.ref('i'),
            ))
            branch_body.extend(_emit_match_success(tag_runtime, end_target, meta.ref('i')))
            eof_branches.append((_q_eq(meta.ref('state'), _q_i32(s)), branch_body))
    eof_accept_stmt = _q_compile_time_if(
        bool(eof_branches),
        [_if_elif_chain(eof_branches)] if eof_branches else [],
    )

    return (
        output_init +
        [state_init] +
        start_guards +
        [initial_accept_stmt] +
        _stmt_list(loop_body) +
        [eof_accept_stmt, _q_return(_q_u8(0))]
    )


def _build_search_body(dfa: DFA,
                       search_dfa: DFA = None,
                       accept_depths: AcceptDepthInfo = None,
                       chains: List[LinearChain] = None,
                       search_chains: List[LinearChain] = None,
                       info: PatternInfo = None,
                       skip_info: SkipInfo = None,
                       has_start_anchor: bool = False,
                       state_skips: Dict[int, StateSkipInfo] = None,
                       restart_states: Set[int] = None,
                       ) -> List[ast.stmt]:
    """Build the full body for a search function.

    Uses 1-pass search with restart-state match_start tracking.

    dfa: original DFA (for anchor checks and anchored search).
    skip_info: optional skip table for O(n/W) scanning.
    has_start_anchor: True if all branches start with ^.
    state_skips: optional per-state skip probes for non-start states.
    restart_states: set of search DFA states that are restart (.*? prefix).
    """
    search_impl_dfa = search_dfa if search_dfa is not None else dfa
    return [_q_compile_time_if_else(
        has_start_anchor,
        _build_anchored_search(dfa, chains=chains),
        _build_goto_forward_search(
            search_impl_dfa, accept_depths=accept_depths,
            skip_info=skip_info, state_skips=state_skips,
            restart_states=restart_states),
    )]


def _build_anchored_search(dfa: DFA,
                            chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build search body for all-start-anchored pattern.

    Uses sentinel protocol: compute effective_start from sentinel 256,
    run DFA, check sentinel 257 at end.
    """
    stmts = []

    # Compute effective start after sentinel 256
    effective_start = dfa.transitions[dfa.start_state][dfa.sentinel_256_class]

    if effective_start == dfa.dead_state:
        return [_q_return(_q_i64(meta.const(-1)))]

    if effective_start in dfa.accept_states:
        return [_q_return(_q_i64(meta.const(0)))]

    # Build in-loop accept check
    accept_check_stmts = []
    state_ref = meta.ref('state')
    accept_list = sorted(dfa.accept_states)
    if accept_list:
        branches = []
        for s in accept_list:
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_i64(meta.const(0)))]))
        accept_check_stmts = [_if_elif_chain(branches)]

    # Build state machine from effective_start
    active_states = _get_active_states(dfa)
    dispatch = _build_dispatch(dfa, active_states, chains=chains) if active_states else \
        _q_assign(meta.ref('state'), _q_i32(dfa.dead_state))

    stmts.extend(_q(_tpl_state_machine,
                     meta.const(effective_start),
                     meta.const(dfa.dead_state),
                     _q(_tpl_u64, meta.const(0)),
                     _splice([dispatch]),
                     _splice(accept_check_stmts)))

    # After loop: check sentinel 257 per-state (all non-dead states)
    end_accept_branches = []
    for s in range(dfa.num_states):
        if s == dfa.dead_state:
            continue
        end_target = dfa.transitions[s][dfa.sentinel_257_class]
        if end_target in dfa.accept_states:
            end_accept_branches.append(
                (_q_eq(state_ref, _q_i32(s)),
                 [_q_return(_q_i64(meta.const(0)))]))
    if end_accept_branches:
        stmts.append(_if_elif_chain(end_accept_branches))

    stmts.append(_q_return(_q_i64(meta.const(-1))))
    return stmts


def _build_compiled_fn(dfa: DFA, digest: str, kind: str,
                       search_dfa: DFA = None,
                       accept_depths: AcceptDepthInfo = None,
                       chains: List[LinearChain] = None,
                       search_chains: List[LinearChain] = None,
                       info: PatternInfo = None,
                       skip_info: SkipInfo = None,
                       has_start_anchor: bool = False,
                       state_skips: Dict[int, StateSkipInfo] = None,
                       restart_states: Set[int] = None):
    """Generate a @compile function using pythoc.meta.

    Args:
        dfa: The original DFA.
        digest: Pattern digest for unique naming.
        kind: "is_match", "fullmatch", or "search".
        search_dfa: Optional search DFA for O(n) unanchored search.
        accept_depths: Optional accept depth info for 1-pass search.
        chains: Optional linear chains for the DFA.
        search_chains: Optional linear chains for the search DFA.
        info: Optional pattern info.
        skip_info: Optional skip table for O(n/W) scanning.
        has_start_anchor: True if all branches start with ^.
        state_skips: Optional per-state skip probes.
        restart_states: Set of search DFA restart states.

    Returns:
        The compiled function.
    """
    # label/goto/goto_end needed in required_globals for goto FSM
    if kind == "is_match":
        func_name = "regex_is_match"
        suffix = digest
        return_type = u8
        body_stmts = _build_is_match_body(dfa, chains=chains,
                                           info=info)
    elif kind == "fullmatch":
        func_name = "regex_fullmatch"
        suffix = digest + "_fullmatch"
        return_type = u8
        body_stmts = _build_fullmatch_body(dfa, chains=chains)
    elif kind == "search":
        func_name = "regex_search"
        suffix = digest + "_search"
        return_type = i64
        body_stmts = _build_search_body(dfa,
                                         search_dfa=search_dfa,
                                         accept_depths=accept_depths,
                                         chains=chains,
                                         search_chains=search_chains,
                                         info=info,
                                         skip_info=skip_info,
                                         has_start_anchor=has_start_anchor,
                                         state_skips=state_skips,
                                        restart_states=restart_states)
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
            'label': label, 'goto': goto, 'goto_end': goto_end,
        },
        source_file=__file__,
    )

    return meta.compile_generated(gf, suffix=suffix)


def _build_match_info_fn(dfa: DFA, digest: str,
                         tag_runtime: DFATagRuntime):
    """Build a compiled tagged-DFA helper from a known match start."""
    body_stmts = _build_match_info_body(dfa, tag_runtime)
    for node in body_stmts:
        ast.fix_missing_locations(node)

    gf = meta.func(
        name="regex_match_info",
        params=[("n", u64), ("data", ptr[i8]), ("start", u64), ("out", ptr[i64])],
        return_type=u8,
        body=body_stmts,
        required_globals={
            'i8': i8, 'i32': i32, 'i64': i64,
            'u8': u8, 'u64': u64, 'ptr': ptr,
            'label': label, 'goto': goto, 'goto_end': goto_end,
        },
        source_file=__file__,
    )

    return meta.compile_generated(gf, suffix=digest + "_match_info")


# ---------------------------------------------------------------------------
# Anchor analysis helpers
# ---------------------------------------------------------------------------

def _all_branches_start_anchored(node) -> bool:
    """Check if every branch of the pattern starts with ^.

    Returns True only if ALL top-level alternatives begin with ^.
    Used to decide if search needs a search DFA or can just run
    the match DFA at position 0.
    """
    from .parse import Anchor, Concat, Alternate, Group
    if isinstance(node, Anchor):
        return node.kind == 'start'
    if isinstance(node, Concat):
        if not node.children:
            return False
        return _all_branches_start_anchored(node.children[0])
    if isinstance(node, Alternate):
        return all(_all_branches_start_anchored(c) for c in node.children)
    if isinstance(node, Group):
        return _all_branches_start_anchored(node.child)
    return False


def _any_branch_end_anchored(node) -> bool:
    """Check if any branch of the pattern ends with $.

    Used to decide end-of-input checking behavior.
    """
    from .parse import Anchor, Concat, Alternate, Group
    if isinstance(node, Anchor):
        return node.kind == 'end'
    if isinstance(node, Concat):
        if not node.children:
            return False
        return _any_branch_end_anchored(node.children[-1])
    if isinstance(node, Alternate):
        return any(_any_branch_end_anchored(c) for c in node.children)
    if isinstance(node, Group):
        return _any_branch_end_anchored(node.child)
    return False


# ---------------------------------------------------------------------------
# DFA tag runtime helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DFATagRuntime:
    """Compile-time tag actions for a DFA."""
    tag_names: Tuple[str, ...]
    tag_slots: Dict[str, int]
    transition_tag_slots: Dict[Tuple[int, int], Tuple[int, ...]]
    accept_tag_slots: Dict[int, Tuple[int, ...]]


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
    if dfa.sentinel_257_class >= 0:
        reps[dfa.sentinel_257_class] = 257
    return reps


def _collect_user_tag_names(nfa: NFA) -> Tuple[str, ...]:
    seen = set()
    ordered: List[str] = []
    for state in nfa.states:
        tag = state.tag
        if tag is None or tag.startswith(INTERNAL_TAG_PREFIX) or tag in seen:
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
    tag_slots_seen = set()

    while stack:
        cur = stack.pop()
        state_obj = nfa.states[cur]
        if state_obj.tag is not None and state_obj.tag in tag_slots:
            tag_slots_seen.add(tag_slots[state_obj.tag])
        for prev in reverse_eps[cur]:
            if prev in allowed and prev not in seen:
                seen.add(prev)
                stack.append(prev)

    return tuple(sorted(tag_slots_seen))


def _compute_dfa_tag_runtime(nfa: NFA, dfa: DFA) -> DFATagRuntime:
    """Compute dynamic DFA tag actions for user-visible tags."""
    tag_names = _collect_user_tag_names(nfa)
    tag_slots = {name: idx for idx, name in enumerate(tag_names)}
    reverse_eps = _reverse_epsilon_graph(nfa)
    reps = _class_representatives(dfa)

    transition_tag_slots: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    accept_tag_slots: Dict[int, Tuple[int, ...]] = {}

    for state_id, closure in enumerate(dfa.state_closures):
        if state_id in dfa.accept_states:
            slots = _collect_reverse_reachable_tags(
                nfa, closure, {nfa.accept}, reverse_eps, tag_slots)
            if slots:
                accept_tag_slots[state_id] = slots

        for class_id, representative in reps.items():
            if state_id >= len(dfa.transitions):
                continue
            target = dfa.transitions[state_id][class_id]
            if target == dfa.dead_state:
                continue
            byte_sources = {
                nfa_state for nfa_state in closure
                if representative in nfa.states[nfa_state].byte_transitions
            }
            slots = _collect_reverse_reachable_tags(
                nfa, closure, byte_sources, reverse_eps, tag_slots)
            if slots:
                transition_tag_slots[(state_id, class_id)] = slots

    return DFATagRuntime(
        tag_names=tag_names,
        tag_slots=tag_slots,
        transition_tag_slots=transition_tag_slots,
        accept_tag_slots=accept_tag_slots,
    )


# ---------------------------------------------------------------------------
# Native execution helpers
# ---------------------------------------------------------------------------

def _bytes_to_native_args(data: bytes):
    """Convert bytes to (n, ptr_val, buf) for native regex fns.

    Returns (n, ptr_val, buf).  buf must be kept alive for the
    duration of the native call to prevent the GC from freeing
    the underlying memory.
    """
    n = len(data)
    buf = ctypes.create_string_buffer(data, n)
    ptr_val = ctypes.cast(buf, ctypes.c_void_p).value
    return n, ptr_val, buf


# ---------------------------------------------------------------------------
# CompiledRegex
# ---------------------------------------------------------------------------

class CompiledRegex:
    """Holds a compiled regex pattern and provides match/search operations.

    Eagerly compiles native PythoC functions for is_match(), fullmatch(),
    search(), and tagged span recovery.
    Must be created before native execution starts (i.e., at module level
    or during @compile decoration phase).

    Instances are cached by pattern string, so creating CompiledRegex with
    the same pattern returns the same object.  This avoids recompilation
    errors when the same pattern is used in multiple places.

    For unanchored patterns, uses a search DFA (prepending lazy .*) for
    O(n) start-position search. Once the winning start is known, a second
    native DFA helper walks the original DFA and dynamically records tags.
    """

    _cache: Dict[str, 'CompiledRegex'] = {}

    def __new__(cls, pattern: str):
        if pattern in cls._cache:
            return cls._cache[pattern]
        instance = super().__new__(cls)
        cls._cache[pattern] = instance
        return instance

    def __init__(self, pattern: str):
        # Skip re-init if already initialized (returned from cache)
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.pattern = pattern
        self._re_ast = parse(pattern)
        self._nfa = build_nfa(self._re_ast)
        self.dfa = build_dfa(self._nfa)
        self._digest = _pattern_digest(pattern)
        self._info = analyze_pattern(self._re_ast)

        self._accept_depths = compute_accept_depths(self.dfa)
        self._chains = find_linear_chains(self.dfa)
        self._tag_runtime = _compute_dfa_tag_runtime(self._nfa, self.dfa)
        self._match_info_slots = 1 + len(self._tag_runtime.tag_names)

        # Determine if pattern uses anchors (for search behavior)
        self._has_start_anchor = self._pattern_has_start_anchor()

        # Build search DFA for unanchored/mixed patterns.
        if not self._has_start_anchor:
            self._search_dfa = compile_search_dfa(pattern)
            self._search_chains = find_linear_chains(self._search_dfa)
            self._search_depths = compute_accept_depths(self._search_dfa)
            # Compute skip table from original DFA for O(n/W) scanning
            self._skip_info = compute_skip_table(self.dfa)
            # Compute per-state skip probes for non-start states.
            # Identify "restart" states: the .* loop states in the search DFA.
            # These are states that self-loop on >= 128 bytes (more than half).
            # Transitions to restart states represent "restart search" rather
            # than forward progress, so they should not make bytes count as live.
            # Identify "restart" states: the .*? prefix states in the
            # search DFA.  These are the states reachable from the
            # effective start that form the .*? scanning loop BEFORE
            # the actual pattern begins.
            #
            # Criterion: BFS from effective_start, following only
            # transitions to states with broad self-loops (>= 128
            # self-transitions).  States with broad self-loops that
            # appear INSIDE the pattern (e.g., .* in a.*b) are NOT
            # included because they aren't reachable from effective_start
            # without first going through a non-restart state.
            #
            # The effective start itself is also restart if the majority
            # of its transitions go to broad-self-loop states.
            sdfa = self._search_dfa
            eff = sdfa.transitions[sdfa.start_state][sdfa.sentinel_256_class]

            # Step 1: identify broad-self-loop states
            broad_self_loop = set()
            for s in range(sdfa.num_states):
                if s == sdfa.dead_state:
                    continue
                self_count = 0
                for bv in range(256):
                    cls = sdfa.class_map[bv]
                    if sdfa.transitions[s][cls] == s:
                        self_count += 1
                if self_count >= 128:
                    broad_self_loop.add(s)

            # Step 2: BFS from effective_start through broad-self-loop
            # states only.  Include effective_start if it fans out
            # mostly to broad-self-loop states.
            self._restart_states = set()
            frontier = set()

            # Check if effective start should be restart
            if eff in broad_self_loop:
                self._restart_states.add(eff)
                frontier.add(eff)
            elif eff != sdfa.dead_state:
                to_bsl = 0
                for bv in range(256):
                    cls = sdfa.class_map[bv]
                    t = sdfa.transitions[eff][cls]
                    if t in broad_self_loop:
                        to_bsl += 1
                if to_bsl >= 128:
                    self._restart_states.add(eff)
                    frontier.add(eff)

            # BFS: from restart states, add broad-self-loop neighbors
            visited = set(frontier)
            while frontier:
                next_frontier = set()
                for s in frontier:
                    seen = set()
                    for bv in range(256):
                        cls = sdfa.class_map[bv]
                        t = sdfa.transitions[s][cls]
                        if t in seen or t == sdfa.dead_state:
                            continue
                        seen.add(t)
                        if t in broad_self_loop and t not in visited:
                            self._restart_states.add(t)
                            visited.add(t)
                            next_frontier.add(t)
                frontier = next_frontier
            # Per-state skip probes are computed but currently only used
            # for analysis/debugging. The search goto FSM only uses the
            # global skip table (for the start state), since per-state
            # skips in the search FSM would need to account for potential
            # match restarts at skipped positions.
            self._state_skips = compute_state_skips(
                self._search_dfa,
                restart_states=self._restart_states)
        else:
            self._search_dfa = None
            self._search_chains = []
            self._search_depths = None
            self._skip_info = None
            self._state_skips = {}
            self._restart_states = set()

        # Eagerly compile native functions — these are always used
        # by the public API. Compilation must happen before native
        # execution starts; calling CompiledRegex() after that is an error.
        self._native_is_match_fn = self.generate_is_match_fn()
        self._native_fullmatch_fn = self.generate_fullmatch_fn()
        self._native_search_fn = self.generate_search_fn()
        self._native_match_info_fn = self.generate_match_info_fn()

    def _pattern_has_start_anchor(self) -> bool:
        """Check if all branches of the pattern start with ^.

        Returns True only if EVERY top-level alternative begins with ^,
        meaning the pattern can only match at position 0.
        """
        from .parse import Anchor, Concat, Alternate, Group
        return _all_branches_start_anchored(self._re_ast)

    def is_match(self, data: bytes) -> bool:
        """Test if the pattern matches at the beginning of data.

        Semantics match Python's re.match: anchored at the start,
        but the pattern does not need to consume the entire input
        (unless the pattern contains $ anchor on relevant branches).

        Uses native compiled execution.
        """
        n, ptr_val, buf = _bytes_to_native_args(data)
        return bool(self._native_is_match_fn(n, ptr_val))

    def fullmatch(self, data: bytes) -> bool:
        """Test if the entire data matches the pattern."""
        n, ptr_val, buf = _bytes_to_native_args(data)
        return bool(self._native_fullmatch_fn(n, ptr_val))

    def search(self, data: bytes) -> int:
        """Search for pattern in data.

        Returns the start position of the first match, or -1 if not found.

        Uses native compiled execution.
        """
        n, ptr_val, buf = _bytes_to_native_args(data)
        return int(self._native_search_fn(n, ptr_val))

    def _match_info_at(self, data: bytes, start_pos: int):
        """Run the native tagged-DFA helper from a known start position."""
        n, ptr_val, buf = _bytes_to_native_args(data)
        out = (ctypes.c_int64 * self._match_info_slots)()
        matched = self._native_match_info_fn(n, ptr_val, start_pos, out)
        if not matched:
            return None

        result = {
            'start': start_pos,
            'end': int(out[0]),
        }
        for idx, tag_name in enumerate(self._tag_runtime.tag_names):
            tag_pos = int(out[1 + idx])
            if tag_pos >= 0:
                result[tag_name] = tag_pos
        return result

    def find_span(self, data: bytes):
        """Find the span (start, end) of the first match.

        Returns (start, end) tuple or None if no match.
        The start boundary comes from the native search path; the end
        boundary comes from the native tagged-DFA recovery helper.
        """
        result = self._search_span(data)
        if result is None:
            return None
        return (result['start'], result['end'])

    def find_with_tags(self, data: bytes):
        """Find first match and return tag positions.

        Returns a dict mapping tag names to input positions, plus
        'start' and 'end' keys for the match span.  Returns None
        if no match.

        Example:
            cr = CompiledRegex('a{mid}b')
            cr.find_with_tags(b'xxabxx')
            # -> {'start': 2, 'end': 4, 'mid': 3}
        """
        return self._search_span(data)

    def _search_span(self, data: bytes):
        """Recover full match info from the native search start position."""
        match_start = self.search(data)
        if match_start < 0:
            return None
        return self._match_info_at(data, match_start)

    # -----------------------------------------------------------------
    # @compile function generation
    # -----------------------------------------------------------------

    def generate_is_match_fn(self):
        """Generate a @compile function for is_match.

        Returns a PythoC compiled function:
            is_match(n: u64, data: ptr[i8]) -> u8
        """
        return _build_compiled_fn(self.dfa, self._digest, "is_match",
                                  chains=self._chains,
                                  info=self._info)

    def generate_fullmatch_fn(self):
        """Generate a @compile function for fullmatch.

        Returns a PythoC compiled function:
            fullmatch(n: u64, data: ptr[i8]) -> u8
        """
        return _build_compiled_fn(self.dfa, self._digest, "fullmatch",
                                  chains=self._chains)

    def generate_search_fn(self):
        """Generate a @compile function for search.

        Returns a PythoC compiled function:
            search(n: u64, data: ptr[i8]) -> i64
        """
        return _build_compiled_fn(self.dfa, self._digest, "search",
                                  search_dfa=self._search_dfa,
                                  accept_depths=self._accept_depths,
                                  chains=self._chains,
                                  search_chains=self._search_chains,
                                  info=self._info,
                                  skip_info=self._skip_info,
                                  has_start_anchor=self._has_start_anchor,
                                  restart_states=self._restart_states)

    def generate_match_info_fn(self):
        """Generate a @compile function for native span/tag recovery.

        Returns a PythoC compiled function:
            match_info(n: u64, data: ptr[i8], start: u64, out: ptr[i64]) -> u8
        """
        return _build_match_info_fn(self.dfa, self._digest, self._tag_runtime)
