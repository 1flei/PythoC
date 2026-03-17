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
from pythoc.builtin_entities.scoped_label import label, goto, goto_end

from .dfa import DFA
from .parse import parse
from .nfa import build_nfa, build_search_nfa, reverse_nfa
from .dfa import build_dfa
from .analysis import (analyze_pattern, compute_accept_depths,
                       find_linear_chains, find_guaranteed_accept_states,
                       compute_skip_table,
                       PatternInfo, AcceptDepthInfo, LinearChain, SkipInfo)


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


def compile_search_dfa(pattern: str):
    """Build a search DFA (prepend .*) for unanchored O(n) matching."""
    re_ast = parse(pattern)
    nfa = build_nfa(re_ast)
    search_nfa = build_search_nfa(nfa)
    return build_dfa(search_nfa)


def compile_search_dfa_from_ast(re_ast):
    """Build a search DFA from AST (prepend .*)."""
    nfa = build_nfa(re_ast)
    search_nfa = build_search_nfa(nfa)
    return build_dfa(search_nfa)


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
def _tpl_forward_search_machine(start_state_val, dead_state_val,
                                 dispatch, accept_store):
    """Forward search DFA: single pass to find match end position.

    On accept, accept_store sets match_end and state to dead to break.
    """
    state: i32 = i32(start_state_val)
    match_end: i64 = i64(-1)
    i: u64 = u64(0)
    while i < n:
        ch: i32 = i32(data[i]) & 0xFF
        dispatch
        accept_store
        if state == i32(dead_state_val):
            break
        i = i + 1


@meta.quote_stmts
def _tpl_reverse_search_machine(start_state_val, dead_state_val,
                                 rev_dispatch, rev_accept_store):
    """Reverse search DFA: backward pass from match_end to find match start.

    Uses unsigned wraparound: j starts at match_end-1, decrements, and
    the condition j < match_end detects wraparound past 0.
    """
    rev_state: i32 = i32(start_state_val)
    match_start: i64 = match_end
    j: u64 = u64(match_end) - u64(1)
    while j < u64(match_end):
        rch: i32 = i32(data[j]) & 0xFF
        rev_dispatch
        rev_accept_store
        if rev_state == i32(dead_state_val):
            break
        j = j - u64(1)


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
def _tpl_suffix_scan(min_len, loop_bound, last_byte_val,
                     verify_body, early_ret_body):
    """Backward scan from end for literal suffix byte.

    Scans data backward looking for last_byte_val, runs verify_body
    on hit, breaks on sfx_found.  Returns early via early_ret_body
    if data too short or suffix not found.
    """
    if n < u64(min_len):
        early_ret_body
    sfx_found: u8 = u8(0)
    sfx_k: u64 = n - u64(1)
    while sfx_k >= u64(loop_bound):
        if i32(data[sfx_k]) & 0xFF == last_byte_val:
            verify_body
        if sfx_found == u8(1):
            break
        if sfx_k == u64(0):
            break
        sfx_k = sfx_k - u64(1)
    if sfx_found == u8(0):
        early_ret_body


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
def _tpl_accept_2pass(done_label_name):
    """2-pass search accept: advance i, store match_end, goto done."""
    i = i + u64(1)
    match_end = i64(i)
    goto(done_label_name)


@meta.quote_stmts
def _tpl_skip_advance(skip_dist_val, label_name):
    """Advance i by skip distance and jump to label_name."""
    i = i + u64(skip_dist_val)
    goto(label_name)


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


def _q_accept_2pass(done_label: str):
    """i = i + u64(1); match_end = i64(i); goto(done_label)"""
    return _q(_tpl_accept_2pass, meta.const(done_label))


def _q_skip_advance(skip_dist: int, label_name: str):
    """i = i + u64(skip_dist); goto(label_name)"""
    return _q(_tpl_skip_advance, meta.const(skip_dist), meta.const(label_name))


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
                                ) -> List[ast.stmt]:
    """Build goto-based forward search FSM for search DFA.

    Each DFA state becomes a labeled block with direct goto transitions.
    This compiles to LLVM basic blocks with br instructions -- no state
    variable, no if/elif dispatch overhead.

    When accept_depths is provided and all_unique, each accept state
    returns i64(i + 1 - depth) directly (1-pass search).
    Otherwise, on accept sets match_end = i64(i + 1) and jumps to done.

    When skip_info is available (W >= 2), state 0 gets a skip-probe
    prefix that enables O(n/W) scanning.
    """
    if search_dfa.start_state in search_dfa.accept_states:
        return [_q_return(_q_i64(meta.const(0)))]

    one_pass = accept_depths is not None and accept_depths.all_unique

    stmts = []  # top-level statements before label blocks

    if not one_pass:
        # Need match_end variable for two-pass search
        stmts.append(_q(_tpl_assign_typed, meta.ident('match_end'),
                         meta.type_expr(i64), _q_i64(meta.const(-1))))

    # i: u64 = u64(0)
    stmts.append(_q(_tpl_assign_typed, meta.ident('i'),
                     meta.type_expr(u64), _q(_tpl_u64, meta.const(0))))

    # Precompute search DFA depths for 1-pass accept returns
    search_depths = None
    if one_pass:
        search_depths = compute_accept_depths(search_dfa)

    active_states = _get_active_states(search_dfa)
    done_label = "rx_done"

    # Build label blocks for each state
    label_blocks = []
    for s in active_states:
        label_name = _state_label(s)
        body = _build_goto_state_body(
            search_dfa, s, active_states, done_label,
            one_pass=one_pass, search_depths=search_depths,
            skip_info=skip_info if s == search_dfa.start_state else None,
        )
        label_blocks.append(_q_label_block(label_name, body))

    # Done label
    done_body = [ast.Pass()]
    label_blocks.append(_q_label_block(done_label, done_body))

    stmts.extend(label_blocks)

    if one_pass:
        # No match found
        stmts.append(_q_return(_q_i64(meta.const(-1))))
    else:
        # Return match_end (may be -1 if no match)
        stmts.append(_q_return(meta.ref('match_end')))

    return stmts


def _build_goto_state_body(dfa: DFA, state: int,
                            active_states: List[int],
                            done_label: str,
                            one_pass: bool = False,
                            search_depths: AcceptDepthInfo = None,
                            skip_info: SkipInfo = None,
                            ) -> List[ast.stmt]:
    """Build the body of a single state's label block in the goto FSM.

    The body: bounds check, optional skip probe, byte load, transitions,
    accept handling.
    """
    body = []

    # --- Skip table fast path (only for start state when skip_info available) ---
    if skip_info is not None and skip_info.window >= 2:
        body.extend(_build_skip_probe(dfa, state, skip_info))

    # --- Bounds check + byte load ---
    body.extend(_q(_tpl_goto_state_preamble, meta.const(done_label)))

    # --- Byte transitions ---
    trans = _byte_transitions_for_state(dfa, state)

    # Separate accept targets for special handling
    t_branches = []
    for target, byte_vals in trans.items():
        cond = _byte_condition(byte_vals, 'ch')
        target_label = _state_label(target)

        if target in dfa.accept_states:
            # Accept state reached
            if one_pass and search_depths is not None:
                depth = search_depths.depths.get(target, 0)
                t_body = _q_accept_1pass(depth)
            else:
                t_body = _q_accept_2pass(done_label)
        elif target == dfa.dead_state:
            # Dead state: skip to done
            t_body = [_q_goto(done_label)]
        elif target in active_states:
            # Normal transition: advance i, goto target
            t_body = _q_advance_goto(target_label)
        else:
            # Target not in active states -> dead
            t_body = [_q_goto(done_label)]

        t_branches.append((cond, t_body))

    if t_branches:
        # Default: go back to start state (the .* loop for search DFA state 0)
        # or goto done for non-start states
        if state == dfa.start_state:
            default_body = _q_advance_goto(_state_label(dfa.start_state))
        else:
            # For non-start states in a search DFA, unmatched byte means
            # the partial match failed. Go back to start state to continue
            # searching (the search DFA's .* self-loop handles this).
            # We need to NOT advance i here, since the current byte might
            # start a new match. Go to start state without advancing.
            default_body = [_q_goto(_state_label(dfa.start_state))]
        body.append(_if_elif_chain(t_branches, else_body=default_body))
    else:
        # No transitions from this state -> goto done
        body.append(_q_goto(done_label))

    return body


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
        for c in chains:
            chain_map[c.start_state] = c
            # Internal states: all states between start and end
            # Walk the chain to find internals
            current = c.start_state
            for b_val in c.byte_sequence[:-1]:
                # Find the target from this state on this byte
                for byte_v in range(256):
                    if dfa.class_map[byte_v] == dfa.class_map[b_val]:
                        cls = dfa.class_map[byte_v]
                        break
                else:
                    cls = dfa.class_map[b_val]
                target = dfa.transitions[current][dfa.class_map[b_val]]
                if target != dfa.dead_state and target != current:
                    chain_internal.add(target)
                    current = target

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
# Suffix guard builder (Opt #3)
# ---------------------------------------------------------------------------

def _build_suffix_guard(info: PatternInfo, kind: str) -> List[ast.stmt]:
    """Build compiled AST for literal suffix guard.

    Scans data **backward from the end** for the last byte of the suffix,
    then verifies remaining suffix bytes. Returns early if suffix not found.

    Scanning from the end is critical for patterns like ``a.*z`` where the
    suffix is near the end of the data: the guard finds ``z`` in O(1)
    instead of O(n).

    kind: 'is_match' -> return u8(0), 'search' -> return i64(-1)
    """
    suffix = info.literal_suffix
    if not suffix or len(suffix) < 1:
        return []

    suffix_len = len(suffix)
    last_byte = suffix[-1]

    if kind == 'is_match':
        early_ret = _q_return(_q_u8(0))
    else:
        early_ret = _q_return(_q_i64(meta.const(-1)))

    # Build the verify body: check data[k-1] == suffix[-2], ...
    # If all match, set sfx_found = 1
    verify_checks = []
    for offset in range(1, suffix_len):
        idx_expr = _q_sub(meta.ref('sfx_k'), meta.const(offset))
        data_byte = _q(_tpl_data_byte_at, idx_expr)
        verify_checks.append(
            _q_eq(data_byte, meta.const(suffix[suffix_len - 1 - offset])))

    if verify_checks:
        inner_test = verify_checks[0]
        for check in verify_checks[1:]:
            inner_test = _q_and(inner_test, check)
        set_found = [_q_assign(meta.ref('sfx_found'), _q_u8(1))]
        verify_stmt = _q(_tpl_if, inner_test, _splice(set_found))
    else:
        verify_stmt = _q_assign(meta.ref('sfx_found'), _q_u8(1))

    # Use suffix scan template: handles length guard, backward scan, and
    # early return in a single quasi-quote skeleton.
    return _q(_tpl_suffix_scan,
              meta.const(suffix_len),
              meta.const(suffix_len - 1),
              meta.const(last_byte),
              _splice([verify_stmt]),
              _splice([early_ret]))


# ---------------------------------------------------------------------------
# Body builders: is_match
# ---------------------------------------------------------------------------

def _build_is_match_body(dfa: DFA,
                          chains: List[LinearChain] = None,
                          info: PatternInfo = None) -> List[ast.stmt]:
    """Build the full body for an is_match function.

    Semantics: re.match — anchored at the start, pattern does not need
    to consume the entire input (unless $ anchor).
    Always runs the DFA from position 0.
    """
    stmts = []
    stmts.extend(_build_anchored_is_match(dfa, chains=chains))
    return stmts


def _build_anchored_is_match(dfa: DFA,
                              chains: List[LinearChain] = None) -> List[ast.stmt]:
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
        dfa, accept_check_stmts=accept_check_stmts, chains=chains))

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


def _build_search_dfa_is_match(search_dfa: DFA,
                                chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build is_match body using search DFA for O(n) unanchored matching.

    Single forward pass: run the search DFA (which has .* prepended).
    If any accept state is reached, return 1 immediately.
    """
    stmts = []
    state_ref = meta.ref('state')

    # If start state is accepting, pattern matches empty string
    if search_dfa.start_state in search_dfa.accept_states:
        return [_q_return(_q_u8(1))]

    # Build accept check: on accept, return 1
    accept_list = sorted(search_dfa.accept_states)
    accept_check_stmts = []
    if accept_list:
        branches = []
        for s in accept_list:
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_u8(1))]))
        accept_check_stmts = [_if_elif_chain(branches)]

    stmts.extend(_build_state_machine(
        search_dfa, accept_check_stmts=accept_check_stmts, chains=chains))
    stmts.append(_q_return(_q_u8(0)))
    return stmts

def _build_search_body(dfa: DFA,
                       search_dfa: DFA = None,
                       rev_dfa: DFA = None,
                       accept_depths: AcceptDepthInfo = None,
                       chains: List[LinearChain] = None,
                       search_chains: List[LinearChain] = None,
                       info: PatternInfo = None,
                       skip_info: SkipInfo = None) -> List[ast.stmt]:
    """Build the full body for a search function.

    dfa: original DFA (for anchor checks and anchored search).
    skip_info: optional skip table for O(n/W) scanning.
    """
    stmts = []

    if dfa.anchored_start:
        stmts.extend(_build_anchored_search(dfa, chains=chains))
    elif accept_depths and accept_depths.all_unique:
        # 1-pass search: goto FSM with depth-based start recovery
        stmts.extend(_build_goto_forward_search(
            search_dfa, accept_depths=accept_depths,
            skip_info=skip_info))
    else:
        # Non-unique depths: goto FSM forward to find match_end,
        # then old-style reverse DFA to find match_start.
        stmts.extend(_build_goto_forward_search(
            search_dfa, accept_depths=None, skip_info=skip_info))
        # _build_goto_forward_search with accept_depths=None emits:
        #   match_end: i64 = i64(-1); i: u64 = ...; <goto FSM>; return match_end
        # Replace the trailing return with reverse pass + return match_start.
        if stmts and isinstance(stmts[-1], ast.Return):
            stmts.pop()
        stmts.append(_q(_tpl_if,
                         _q_eq(meta.ref('match_end'), _q_i64(meta.const(-1))),
                         _splice([_q_return(_q_i64(meta.const(-1)))])))
        stmts.extend(_build_reverse_pass(rev_dfa))
    return stmts


def _build_anchored_search(dfa: DFA,
                            chains: List[LinearChain] = None) -> List[ast.stmt]:
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
        dfa, accept_check_stmts=accept_check_stmts, chains=chains))

    if dfa.anchored_end:
        branches = []
        for s in sorted(dfa.accept_states):
            branches.append((_q_eq(state_ref, _q_i32(s)),
                             [_q_return(_q_i64(meta.const(0)))]))
        stmts.append(_if_elif_chain(branches))
    stmts.append(_q_return(_q_i64(meta.const(-1))))
    return stmts


def _build_search_dfa_search(search_dfa: DFA,
                              rev_dfa: DFA,
                              chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build search body using search DFA + reverse DFA for O(n).

    Two-pass approach:
    1. Forward pass with search DFA to find match_end
    2. Backward pass with reverse DFA to find match_start
    """
    stmts = []

    # --- Forward pass: find match_end ---
    fwd_active = _get_active_states(search_dfa)
    if not fwd_active:
        fwd_dispatch = _q_assign(meta.ref('state'),
                                 _q_i32(search_dfa.dead_state))
    else:
        fwd_dispatch = _build_dispatch(search_dfa, fwd_active,
                                       'state', 'ch', chains=chains)

    # Build accept store: on accept, set match_end = i64(i + 1) and
    # set state to dead to break out of the loop.
    fwd_accept_list = sorted(search_dfa.accept_states)
    fwd_accept_stmts = []
    if fwd_accept_list:
        branches = []
        for s in fwd_accept_list:
            body = [
                _q_assign(meta.ref('match_end'),
                          _q_i64(_q_add(meta.ref('i'), meta.const(1)))),
                _q_assign(meta.ref('state'),
                          _q_i32(search_dfa.dead_state)),
            ]
            branches.append((_q_eq(meta.ref('state'), _q_i32(s)), body))
        fwd_accept_stmts = [_if_elif_chain(branches)]

    # Handle case where search DFA start state is accepting
    if search_dfa.start_state in search_dfa.accept_states:
        return [_q_return(_q_i64(meta.const(0)))]

    stmts.extend(_q(_tpl_forward_search_machine,
                     meta.const(search_dfa.start_state),
                     meta.const(search_dfa.dead_state),
                     _splice([fwd_dispatch]),
                     _splice(fwd_accept_stmts)))

    # If no match found, return -1
    stmts.append(_q(_tpl_if,
                     _q_eq(meta.ref('match_end'), _q_i64(meta.const(-1))),
                     _splice([_q_return(_q_i64(meta.const(-1)))])))

    # --- Backward pass: find match_start ---
    rev_active = _get_active_states(rev_dfa)
    if not rev_active:
        rev_dispatch = _q_assign(meta.ref('rev_state'),
                                 _q_i32(rev_dfa.dead_state))
    else:
        rev_dispatch = _build_dispatch(rev_dfa, rev_active,
                                       'rev_state', 'rch')

    # Build reverse accept store: on accept, set match_start = i64(j)
    rev_accept_list = sorted(rev_dfa.accept_states)
    rev_accept_stmts = []
    if rev_accept_list:
        branches = []
        for s in rev_accept_list:
            body = [
                _q_assign(meta.ref('match_start'),
                          _q_i64(meta.ref('j'))),
            ]
            branches.append((_q_eq(meta.ref('rev_state'), _q_i32(s)), body))
        rev_accept_stmts = [_if_elif_chain(branches)]

    stmts.extend(_q(_tpl_reverse_search_machine,
                     meta.const(rev_dfa.start_state),
                     meta.const(rev_dfa.dead_state),
                     _splice([rev_dispatch]),
                     _splice(rev_accept_stmts)))

    # Return match_start
    stmts.append(_q_return(meta.ref('match_start')))
    return stmts


def _build_reverse_pass(rev_dfa: DFA) -> List[ast.stmt]:
    """Build reverse DFA pass to find match_start from match_end.

    Assumes match_end is already set by the forward pass.
    Returns stmts that append the reverse machine and return match_start.
    """
    stmts = []
    rev_active = _get_active_states(rev_dfa)
    if not rev_active:
        rev_dispatch = _q_assign(meta.ref('rev_state'),
                                 _q_i32(rev_dfa.dead_state))
    else:
        rev_dispatch = _build_dispatch(rev_dfa, rev_active,
                                       'rev_state', 'rch')

    rev_accept_list = sorted(rev_dfa.accept_states)
    rev_accept_stmts = []
    if rev_accept_list:
        branches = []
        for s in rev_accept_list:
            body = [
                _q_assign(meta.ref('match_start'),
                          _q_i64(meta.ref('j'))),
            ]
            branches.append((_q_eq(meta.ref('rev_state'), _q_i32(s)), body))
        rev_accept_stmts = [_if_elif_chain(branches)]

    stmts.extend(_q(_tpl_reverse_search_machine,
                     meta.const(rev_dfa.start_state),
                     meta.const(rev_dfa.dead_state),
                     _splice([rev_dispatch]),
                     _splice(rev_accept_stmts)))

    stmts.append(_q_return(meta.ref('match_start')))
    return stmts


def _build_depth_search(search_dfa: DFA,
                         accept_depths: AcceptDepthInfo,
                         chains: List[LinearChain] = None) -> List[ast.stmt]:
    """Build search body using 1-pass depth-based start position recovery.

    Each accept state has a known unique depth (number of pattern bytes).
    On accept, return i64(i + 1 - depth) immediately.
    No reverse DFA needed.

    accept_depths comes from the original DFA; we compute search DFA
    accept depths to get the per-state mapping (min BFS distance).
    """
    stmts = []

    # Handle start state accepting
    if search_dfa.start_state in search_dfa.accept_states:
        return [_q_return(_q_i64(meta.const(0)))]

    # Compute depths for search DFA accept states
    search_depths = compute_accept_depths(search_dfa)

    # Forward pass
    fwd_active = _get_active_states(search_dfa)
    if not fwd_active:
        fwd_dispatch = _q_assign(meta.ref('state'),
                                 _q_i32(search_dfa.dead_state))
    else:
        fwd_dispatch = _build_dispatch(search_dfa, fwd_active,
                                       'state', 'ch', chains=chains)

    # Build accept store: per-accept-state return with depth
    fwd_accept_list = sorted(search_dfa.accept_states)
    fwd_accept_stmts = []
    if fwd_accept_list:
        branches = []
        for s in fwd_accept_list:
            depth = search_depths.depths.get(s, 0)
            body = [
                _q_return(_q_i64(
                    _q_add(meta.ref('i'),
                           meta.const(1 - depth)))),
            ]
            branches.append((_q_eq(meta.ref('state'), _q_i32(s)), body))
        fwd_accept_stmts = [_if_elif_chain(branches)]

    stmts.extend(_q(_tpl_forward_search_machine,
                     meta.const(search_dfa.start_state),
                     meta.const(search_dfa.dead_state),
                     _splice([fwd_dispatch]),
                     _splice(fwd_accept_stmts)))

    stmts.append(_q_return(_q_i64(meta.const(-1))))
    return stmts


def _build_compiled_fn(dfa: DFA, digest: str, kind: str,
                       search_dfa: DFA = None, rev_dfa: DFA = None,
                       accept_depths: AcceptDepthInfo = None,
                       chains: List[LinearChain] = None,
                       search_chains: List[LinearChain] = None,
                       info: PatternInfo = None,
                       skip_info: SkipInfo = None):
    """Generate a @compile function using pythoc.meta.

    Args:
        dfa: The original DFA.
        digest: Pattern digest for unique naming.
        kind: "is_match" or "search".
        search_dfa: Optional search DFA for O(n) unanchored search.
        rev_dfa: Optional reverse DFA for O(n) unanchored search.
        accept_depths: Optional accept depth info for 1-pass search.
        chains: Optional linear chains for the DFA.
        search_chains: Optional linear chains for the search DFA.
        info: Optional pattern info.
        skip_info: Optional skip table for O(n/W) scanning.

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
    elif kind == "search":
        func_name = "regex_search"
        suffix = digest + "_search"
        return_type = i64
        body_stmts = _build_search_body(dfa,
                                         search_dfa=search_dfa,
                                         rev_dfa=rev_dfa,
                                         accept_depths=accept_depths,
                                         chains=chains,
                                         search_chains=search_chains,
                                         info=info,
                                         skip_info=skip_info)
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


# ---------------------------------------------------------------------------
# CompiledRegex
# ---------------------------------------------------------------------------

class CompiledRegex:
    """Holds a compiled regex pattern and provides match/search operations.

    This object performs matching using a Python-level DFA simulation.
    It can also generate @compile functions for native PythoC execution.

    For unanchored patterns, uses a search DFA (prepending .*) for O(n)
    matching, and a reverse DFA for O(n) start-position recovery.
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._re_ast = parse(pattern)
        self.dfa = compile_pattern(pattern)
        self._digest = _pattern_digest(pattern)
        self._info = analyze_pattern(self._re_ast)

        self._accept_depths = compute_accept_depths(self.dfa)
        self._chains = find_linear_chains(self.dfa)

        # Build search and reverse DFAs for unanchored patterns
        if not self.dfa.anchored_start:
            self._search_dfa = compile_search_dfa(pattern)
            self._search_chains = find_linear_chains(self._search_dfa)
            self._search_depths = compute_accept_depths(self._search_dfa)
            # Compute skip table from original DFA for O(n/W) scanning
            self._skip_info = compute_skip_table(self.dfa)
            if self._accept_depths.all_unique:
                self._rev_dfa = None  # 1-pass: start = end - depth
            else:
                self._rev_dfa = compile_reverse_dfa(pattern)
        else:
            self._search_dfa = None
            self._search_chains = []
            self._search_depths = None
            self._skip_info = None
            self._rev_dfa = None

    def _run_dfa_accepts(self, dfa, data: bytes, start_pos: int = 0) -> bool:
        """Run a DFA on data starting from start_pos.

        Returns True if the DFA reaches an accept state at any point
        (for non-end-anchored) or at end of input (for end-anchored).
        """
        state = dfa.start_state
        if state in dfa.accept_states and not dfa.anchored_end:
            return True
        for i in range(start_pos, len(data)):
            byte_val = data[i]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                return False
            if state in dfa.accept_states and not dfa.anchored_end:
                return True
        if dfa.anchored_end:
            return state in dfa.accept_states
        return state in dfa.accept_states

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

    def _run_dfa_find_end(self, data: bytes, start_pos: int = 0,
                          dfa: object = None) -> int:
        """Run the DFA from start_pos and find the end of the earliest match.

        Returns the index past the last matched byte, or -1 if no match.
        For anchored_end patterns, only the end of data counts as a match.
        """
        if dfa is None:
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

    def _search_dfa_find_first_end(self, data: bytes) -> int:
        """Run the search DFA and find the end position of the first match.

        Returns index past the last matched byte, or -1 if no match.
        The search DFA has .* prepended, so it finds the earliest match end.

        For end-anchored patterns, only accepts at the end of data.
        """
        dfa = self._search_dfa
        state = dfa.start_state
        anchored_end = self.dfa.anchored_end

        if state in dfa.accept_states:
            if not anchored_end or len(data) == 0:
                return 0

        for i in range(len(data)):
            byte_val = data[i]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                return -1
            if state in dfa.accept_states:
                if not anchored_end:
                    return i + 1
                # For $-anchored: only accept at end of data
                if i + 1 == len(data):
                    return i + 1

        return -1

    def _search_dfa_find_first_end_ex(self, data: bytes):
        """Like _search_dfa_find_first_end but also returns the accept state.

        Returns (match_end, accept_state) or (-1, -1) if no match.
        """
        dfa = self._search_dfa
        state = dfa.start_state
        anchored_end = self.dfa.anchored_end

        if state in dfa.accept_states:
            if not anchored_end or len(data) == 0:
                return (0, state)

        for i in range(len(data)):
            byte_val = data[i]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                return (-1, -1)
            if state in dfa.accept_states:
                if not anchored_end:
                    return (i + 1, state)
                if i + 1 == len(data):
                    return (i + 1, state)

        return (-1, -1)

    def _rev_dfa_find_start(self, data: bytes, match_end: int,
                             scan_start: int = 0) -> int:
        """Run the reverse DFA backward from match_end to find match start.

        Returns the leftmost start position of the match.
        """
        dfa = self._rev_dfa
        state = dfa.start_state
        match_start = match_end  # default if start state is accepting

        if state in dfa.accept_states:
            match_start = match_end

        j = match_end - 1
        while j >= scan_start:
            byte_val = data[j]
            cls = dfa.class_map[byte_val]
            state = dfa.transitions[state][cls]
            if state == dfa.dead_state:
                break
            if state in dfa.accept_states:
                match_start = j
            j -= 1

        return match_start

    def is_match(self, data: bytes) -> bool:
        """Test if the pattern matches at the beginning of data.

        Semantics match Python's re.match: anchored at the start,
        but the pattern does not need to consume the entire input
        (unless the pattern ends with $).
        """
        end = self._run_dfa_find_end(data, 0, self.dfa)
        if self.dfa.anchored_end:
            return end == len(data)
        return end >= 0

    def fullmatch(self, data: bytes) -> bool:
        """Test if the entire data matches the pattern."""
        return self._run_dfa(data, 0)

    def search(self, data: bytes) -> int:
        """Search for pattern in data.

        Returns the start position of the first match, or -1 if not found.
        For unanchored patterns, uses search DFA + reverse DFA for O(n).
        """
        if self.dfa.anchored_start:
            end = self._run_dfa_find_end(data, 0, self.dfa)
            if self.dfa.anchored_end:
                return 0 if end == len(data) else -1
            return 0 if end >= 0 else -1
        else:
            # O(n): forward pass finds match end
            match_end, accept_state = self._search_dfa_find_first_end_ex(data)
            if match_end < 0:
                return -1
            if self._accept_depths.all_unique:
                # 1-pass: compute start from depth
                depth = self._search_depths.depths.get(accept_state, 0)
                return match_end - depth
            # Bounded reverse scan
            scan_start = 0
            if self._accept_depths.max_depth is not None:
                scan_start = max(0, match_end - self._accept_depths.max_depth)
            return self._rev_dfa_find_start(data, match_end, scan_start)

    def find_span(self, data: bytes):
        """Find the span (start, end) of the first match.

        Returns (start, end) tuple or None if no match.
        end is exclusive.
        """
        if self.dfa.anchored_start:
            end = self._run_dfa_find_end(data, 0, self.dfa)
            if self.dfa.anchored_end:
                return (0, len(data)) if end == len(data) else None
            return (0, end) if end >= 0 else None
        else:
            # O(n): forward pass finds match end
            match_end, accept_state = self._search_dfa_find_first_end_ex(data)
            if match_end < 0:
                return None
            if self._accept_depths.all_unique:
                # 1-pass: compute start from depth
                depth = self._search_depths.depths.get(accept_state, 0)
                match_start = match_end - depth
            else:
                # Bounded reverse scan
                scan_start = 0
                if self._accept_depths.max_depth is not None:
                    scan_start = max(0, match_end - self._accept_depths.max_depth)
                match_start = self._rev_dfa_find_start(data, match_end, scan_start)
            return (match_start, match_end)

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

    def generate_search_fn(self):
        """Generate a @compile function for search.

        Returns a PythoC compiled function:
            search(n: u64, data: ptr[i8]) -> i64
        """
        return _build_compiled_fn(self.dfa, self._digest, "search",
                                  search_dfa=self._search_dfa,
                                  rev_dfa=self._rev_dfa,
                                  accept_depths=self._accept_depths,
                                  chains=self._chains,
                                  search_chains=self._search_chains,
                                  info=self._info,
                                  skip_info=self._skip_info)
