"""
PythoC code generation for regex DFA matchers.

Takes a DFA (from dfa.py) and generates pattern-specialized @compile functions.
Instead of a generic table-driven DFA loop, each pattern produces bespoke
native code where DFA states become explicit if/elif branches.

Pipeline: pattern -> DFA -> AST -> meta.compile_generated -> LLVM native code.

Universal ABI:  run(n: u64, data: ptr[i8], out: ptr[i64]) -> u8
Both match and search share the same native calling convention.  The u8
return is 1/0 (matched / not), and tag slot values are written to *out*.
"""

from __future__ import annotations
import ast
import copy
import ctypes
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from pythoc import meta
from pythoc.meta.template import _coerce_to_ast as _meta_coerce_to_ast
from pythoc.builtin_entities.types import i8, i32, i64, u8, u64, ptr
from pythoc.builtin_entities.scoped_label import label, goto, goto_end
from pythoc.builtin_entities.struct import create_struct_type

from .dfa import DFA
from .parse import (
    parse, Dot, Concat, Repeat, Tag,
)
from .nfa import build_nfa
from .dfa import build_dfa
from . import opcs


INTERNAL_TAG_PREFIX = "__pythoc_internal_"
INTERNAL_BEG_TAG = INTERNAL_TAG_PREFIX + "beg"
INTERNAL_END_TAG = INTERNAL_TAG_PREFIX + "end"


def _make_ctypes_tag_struct(tag_names: Tuple[str, ...]) -> type:
    """Build a ctypes.Structure matching the PythoC tag struct layout."""
    fields = [(name, ctypes.c_int64) for name in tag_names]
    return type("RegexTagResult", (ctypes.Structure,), {"_fields_": fields})


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
    """Build a search DFA (prepend .*?) for unanchored O(n) matching."""
    re_ast = parse(pattern)
    search_ast = rewrite_for_search(re_ast)
    nfa = build_nfa(search_ast)
    return build_dfa(nfa)


def compile_search_dfa_from_ast(re_ast):
    """Build a search DFA from AST (prepend .*?)."""
    search_ast = rewrite_for_search(re_ast)
    nfa = build_nfa(search_ast)
    return build_dfa(nfa)


def _filter_tag_runtime_slots(tag_runtime: opcs.DFATagRuntime,
                              excluded_slots: FrozenSet[int]) -> opcs.DFATagRuntime:
    """Keep the slot layout but remove dynamic writes for selected slots."""
    if not excluded_slots:
        return tag_runtime

    return opcs.DFATagRuntime(
        tag_names=tag_runtime.tag_names,
        public_tag_names=tag_runtime.public_tag_names,
        tag_slots=tag_runtime.tag_slots,
        transition_tag_slots={
            key: tuple(slot for slot in slots if slot not in excluded_slots)
            for key, slots in tag_runtime.transition_tag_slots.items()
            if any(slot not in excluded_slots for slot in slots)
        },
        accept_tag_slots={
            state: tuple(slot for slot in slots if slot not in excluded_slots)
            for state, slots in tag_runtime.accept_tag_slots.items()
            if any(slot not in excluded_slots for slot in slots)
        },
    )


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


# ---------------------------------------------------------------------------
# Quasi-quote structural templates (large skeletons)
# ---------------------------------------------------------------------------

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


@meta.quote_stmt
def _tpl_pass():
    pass


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


@meta.quote_stmts
def _tpl_bma_fsm_program(init_body, goto_start, label_blocks, fallback_return):
    init_body
    goto_start
    label_blocks
    fallback_return


for _tpl_name, _tpl in list(globals().items()):
    if _tpl_name.startswith('_tpl_') and hasattr(_tpl, 'debug_source_enabled'):
        _tpl.debug_source_enabled = False
del _tpl_name, _tpl


# ---------------------------------------------------------------------------
# Convenience wrappers (template call -> raw AST node)
# ---------------------------------------------------------------------------

def _q(template, *args):
    """Call a template, return the raw AST node."""
    return template(*args).node


def _coerce_expr(value) -> ast.expr:
    """Coerce a template helper or AST node into a fresh expression node."""
    if isinstance(value, ast.AST):
        return copy.deepcopy(value)
    return _meta_coerce_to_ast(value, {})


def _raw_call(func_name: str, *args) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[_coerce_expr(arg) for arg in args],
        keywords=[],
    )


def _q_i32(v):
    return _raw_call('i32', v)


def _q_u8(v):
    return _raw_call('u8', v)


def _q_i64(v):
    return _raw_call('i64', v)


def _q_eq(lhs, rhs):
    return ast.Compare(
        left=_coerce_expr(lhs),
        ops=[ast.Eq()],
        comparators=[_coerce_expr(rhs)],
    )


def _q_lte(lhs, rhs):
    return ast.Compare(
        left=_coerce_expr(lhs),
        ops=[ast.LtE()],
        comparators=[_coerce_expr(rhs)],
    )


def _q_gte(lhs, rhs):
    return ast.Compare(
        left=_coerce_expr(lhs),
        ops=[ast.GtE()],
        comparators=[_coerce_expr(rhs)],
    )


def _q_add(lhs, rhs):
    return ast.BinOp(
        left=_coerce_expr(lhs),
        op=ast.Add(),
        right=_coerce_expr(rhs),
    )


def _q_and(lhs, rhs):
    return ast.BoolOp(
        op=ast.And(),
        values=[_coerce_expr(lhs), _coerce_expr(rhs)],
    )


def _q_sub(lhs, rhs):
    return ast.BinOp(
        left=_coerce_expr(lhs),
        op=ast.Sub(),
        right=_coerce_expr(rhs),
    )


def _q_lt(lhs, rhs):
    return ast.Compare(
        left=_coerce_expr(lhs),
        ops=[ast.Lt()],
        comparators=[_coerce_expr(rhs)],
    )


def _q_or_chain(parts):
    """Chain parts with 'or'. Returns single part if length 1."""
    result = _coerce_expr(parts[0])
    for p in parts[1:]:
        result = ast.BoolOp(
            op=ast.Or(),
            values=[result, _coerce_expr(p)],
        )
    return result


def _q_return(val):
    return ast.Return(value=_coerce_expr(val))


def _q_assign(target, value):
    target_expr = _coerce_expr(target)
    if hasattr(target_expr, 'ctx'):
        target_expr.ctx = ast.Store()
    return ast.Assign(
        targets=[target_expr],
        value=_coerce_expr(value),
    )


def _splice(stmts):
    return meta.splice_stmts(stmts)


def _q_label_block(label_name: str, body: List[ast.stmt]):
    return _q(_tpl_with_label, meta.const(label_name), _splice(body))


def _q_goto(label_name: str):
    return _q(_tpl_goto, meta.const(label_name))


def _q_pass():
    return _q(_tpl_pass)


def _stmt_list(node_or_nodes) -> List[ast.stmt]:
    if node_or_nodes is None:
        return []
    if isinstance(node_or_nodes, list):
        return list(node_or_nodes)
    return [node_or_nodes]


def _body_or_pass(node_or_nodes) -> List[ast.stmt]:
    stmts = _stmt_list(node_or_nodes)
    if stmts:
        return stmts
    return [_q_pass()]


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


def _raw_if(test_expr: ast.expr, then_body, else_body=None) -> ast.If:
    return ast.If(
        test=test_expr,
        body=_body_or_pass(then_body),
        orelse=_stmt_list(else_body),
    )


# ---------------------------------------------------------------------------
# If/elif chain builder (dynamic branching)
# ---------------------------------------------------------------------------

def _if_elif_chain(branches, else_body=None):
    if not branches:
        raise ValueError("At least one branch required")

    result = None
    for idx in range(len(branches) - 1, -1, -1):
        test, body = branches[idx]
        if result is None and not else_body:
            result = _raw_if(test, body)
        elif result is None:
            result = _raw_if(test, body, list(else_body))
        else:
            result = _raw_if(test, body, [result])
    return result


# ---------------------------------------------------------------------------
# Byte-condition builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _byte_condition_template(byte_values: Tuple[int, ...],
                             ch_name: str = "ch") -> ast.expr:
    if not byte_values:
        return meta.const(False)

    sorted_vals = list(byte_values)
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


def _byte_condition(byte_values: List[int], ch_name: str = "ch") -> ast.expr:
    if not byte_values:
        return meta.const(False)
    return copy.deepcopy(_byte_condition_template(
        tuple(sorted(byte_values)),
        ch_name,
    ))


def _compute_search_restart_states(search_dfa: DFA) -> FrozenSet[int]:
    """Identify the ``.*?``-prefix restart states in the search DFA."""
    if search_dfa.sentinel_256_class < 0:
        return frozenset()

    effective_start = search_dfa.transitions[
        search_dfa.start_state][search_dfa.sentinel_256_class]
    if effective_start == search_dfa.dead_state:
        return frozenset()

    broad_self_loop = set()
    for state in range(search_dfa.num_states):
        if state == search_dfa.dead_state:
            continue
        self_count = 0
        for byte_val in range(256):
            cls = search_dfa.class_map[byte_val]
            if search_dfa.transitions[state][cls] == state:
                self_count += 1
        if self_count >= 128:
            broad_self_loop.add(state)

    restart_states: Set[int] = set()
    frontier: Set[int] = set()
    if effective_start in broad_self_loop:
        restart_states.add(effective_start)
        frontier.add(effective_start)
    else:
        to_bsl = 0
        for byte_val in range(256):
            cls = search_dfa.class_map[byte_val]
            if search_dfa.transitions[effective_start][cls] in broad_self_loop:
                to_bsl += 1
        if to_bsl >= 128:
            restart_states.add(effective_start)
            frontier.add(effective_start)

    visited = set(frontier)
    while frontier:
        next_frontier = set()
        for state in frontier:
            seen = set()
            for byte_val in range(256):
                cls = search_dfa.class_map[byte_val]
                target = search_dfa.transitions[state][cls]
                if target in seen or target == search_dfa.dead_state:
                    continue
                seen.add(target)
                if target in broad_self_loop and target not in visited:
                    restart_states.add(target)
                    visited.add(target)
                    next_frontier.add(target)
        frontier = next_frontier

    return frozenset(restart_states)


# ---------------------------------------------------------------------------
# OPCS-BMA codegen — unified backend
# ---------------------------------------------------------------------------

def _bma_tag_pos_expr(base_expr: ast.expr, delta: int) -> ast.expr:
    if delta == 0:
        return base_expr
    return _q_add(
        base_expr,
        _q(_tpl_u64, meta.const(delta)),
    )


def _emit_bma_tag_writes(tag_writes: Tuple[opcs.TagWrite, ...],
                         base_expr: ast.expr,
                         with_tags: bool) -> List[ast.stmt]:
    """Emit per-slot writes to the ``out`` buffer."""
    if not with_tags:
        return []
    stmts: List[ast.stmt] = []
    for write in opcs._collapse_tag_writes(tag_writes):
        pos_expr = _bma_tag_pos_expr(base_expr, write.delta)
        stmts.append(_q_ptr_assign(
            'out',
            ast.Constant(write.slot),
            _q_i64(pos_expr),
        ))
    return stmts


@dataclass(frozen=True)
class _BMARuntimeState:
    id: int
    label_name: str
    probe_offset: int
    edges: Tuple[opcs.OPCSEdge, ...]
    accepting: bool
    eof_accept: bool
    eof_tag_writes: Tuple[opcs.TagWrite, ...]
    is_dead: bool = False


@dataclass(frozen=True)
class _BMARuntime:
    states: Tuple[_BMARuntimeState, ...]
    state_by_id: Dict[int, _BMARuntimeState]
    start_state: int
    dead_state: int
    start_label: str
    dead_label: str
    initial_tag_writes: Tuple[opcs.TagWrite, ...]
    initial_accepting: bool


def _bma_state_label(state_id: int) -> str:
    return "bma_s{}".format(state_id)


def _normalize_bma_runtime(bma: opcs.TaggedOPCSBMA) -> _BMARuntime:
    accept_states = set(bma.accept_states)
    executable_states: List[_BMARuntimeState] = []
    state_by_id: Dict[int, _BMARuntimeState] = {}

    for state in bma.states:
        if state.kind == "control_shadow":
            continue
        runtime_state = _BMARuntimeState(
            id=state.id,
            label_name=_bma_state_label(state.id),
            probe_offset=state.probe_offset,
            edges=state.edges,
            accepting=state.id in accept_states,
            eof_accept=state.eof_accept,
            eof_tag_writes=state.eof_tag_writes,
            is_dead=state.id == bma.dead_state,
        )
        executable_states.append(runtime_state)
        state_by_id[state.id] = runtime_state

    if bma.start_state not in state_by_id:
        raise ValueError("BMA start_state is not executable")
    if bma.dead_state not in state_by_id:
        raise ValueError("BMA dead_state is not executable")

    for state in executable_states:
        for edge in state.edges:
            if edge.target not in state_by_id:
                raise ValueError(
                    "BMA edge target {} is not executable".format(edge.target)
                )

    return _BMARuntime(
        states=tuple(executable_states),
        state_by_id=state_by_id,
        start_state=bma.start_state,
        dead_state=bma.dead_state,
        start_label=state_by_id[bma.start_state].label_name,
        dead_label=state_by_id[bma.dead_state].label_name,
        initial_tag_writes=bma.initial_tag_writes,
        initial_accepting=bma.initial_accepting,
    )


def _build_bma_fsm_edge_body(runtime: _BMARuntime,
                             edge: opcs.OPCSEdge,
                             with_tags: bool) -> List[ast.stmt]:
    branch_body: List[ast.stmt] = []
    if edge.tag_writes:
        branch_body.extend(_emit_bma_tag_writes(
            edge.tag_writes, meta.ref('i'), with_tags=with_tags))
    if edge.shift:
        branch_body.append(_q_assign(
            meta.ref('i'),
            _q_add(meta.ref('i'), _q(_tpl_u64, meta.const(edge.shift))),
        ))
    branch_body.append(_q_goto(runtime.state_by_id[edge.target].label_name))
    return branch_body


def _build_bma_eof_success_body(state: _BMARuntimeState,
                                with_tags: bool) -> Optional[List[ast.stmt]]:
    """Build the success branch taken when execution reaches EOF."""
    if state.accepting:
        return [_q_return(_q_u8(1))]
    if not state.eof_accept:
        return None

    body: List[ast.stmt] = []
    if state.eof_tag_writes:
        body.extend(_emit_bma_tag_writes(
            state.eof_tag_writes, meta.ref('i'), with_tags=with_tags))
    body.append(_q_return(_q_u8(1)))
    return body


def _build_bma_fsm_state_body(runtime: _BMARuntime,
                              state: _BMARuntimeState,
                              with_tags: bool,
                              eager_accept: bool) -> List[ast.stmt]:
    if state.is_dead:
        return [_q_return(_q_u8(0))]

    if eager_accept and state.accepting:
        return [_q_return(_q_u8(1))]

    eof_success = _build_bma_eof_success_body(state, with_tags=with_tags)
    eof_guard = _raw_if(
        _q_gte(meta.ref('i'), meta.ref('n')),
        eof_success if eof_success is not None else [_q_goto(runtime.dead_label)],
    )

    probe_idx: ast.expr = meta.ref('i')
    probe_guard: List[ast.stmt] = []
    if state.probe_offset:
        probe_idx = _q_add(
            meta.ref('i'),
            _q(_tpl_u64, meta.const(state.probe_offset)),
        )
        probe_guard.append(_raw_if(
            _q_gte(probe_idx, meta.ref('n')),
            [_q_goto(runtime.dead_label)],
        ))

    load_byte_stmt = _q(
        _tpl_load_byte_as,
        meta.ident('ch'),
        probe_idx,
    )

    edge_branches = [
        (
            _byte_condition(list(edge.byte_values), 'ch'),
            _build_bma_fsm_edge_body(runtime, edge, with_tags=with_tags),
        )
        for edge in state.edges
    ]
    transitions_stmt = (
        _if_elif_chain(edge_branches, else_body=[_q_goto(runtime.dead_label)])
        if edge_branches else
        _q_goto(runtime.dead_label)
    )

    return _stmt_list(_q(
        _tpl_goto_state_program,
        _splice(_body_or_pass(probe_guard)),
        _splice([eof_guard]),
        _splice([load_byte_stmt]),
        _splice([transitions_stmt]),
    ))


def _build_bma_fsm_body(bma: opcs.TaggedOPCSBMA,
                        init_body: List[ast.stmt],
                        with_tags: bool,
                        eager_accept: bool) -> List[ast.stmt]:
    """Lower a TaggedOPCSBMA into one u8-returning goto/FSM runner."""
    runtime = _normalize_bma_runtime(bma)
    program_init = list(init_body)
    if eager_accept and runtime.initial_accepting:
        program_init.append(_q_return(_q_u8(1)))

    label_blocks = [
        _q_label_block(
            state.label_name,
            _build_bma_fsm_state_body(
                runtime, state,
                with_tags=with_tags,
                eager_accept=eager_accept,
            ),
        )
        for state in runtime.states
    ]

    return _stmt_list(_q(
        _tpl_bma_fsm_program,
        _splice(_body_or_pass(program_init)),
        _splice([_q_goto(runtime.start_label)]),
        _splice(label_blocks),
        _splice([_q_return(_q_u8(0))]),
    ))


def _build_bma_body(bma: opcs.TaggedOPCSBMA,
                    with_tags: bool,
                    eager_accept: bool = True) -> List[ast.stmt]:
    """Build the unified u8-returning FSM body.

    If ``with_tags`` is True the generated code expects an ``out``
    parameter (``ptr[i64]``) and writes tag slot values into it.
    """
    num_slots = len(bma.tag_names) if with_tags else 0
    init_stmts: List[ast.stmt] = [
        _q(_tpl_assign_typed, meta.ident('i'), meta.type_expr(u64),
           _q(_tpl_u64, meta.const(0))),
    ]
    for slot in range(num_slots):
        init_stmts.append(
            _q_ptr_assign('out', ast.Constant(slot), _q_i64(meta.const(-1))))
    if bma.initial_tag_writes and with_tags:
        init_stmts.extend(_emit_bma_tag_writes(
            bma.initial_tag_writes, meta.ref('i'), with_tags=True))

    return _build_bma_fsm_body(
        bma,
        init_body=init_stmts,
        with_tags=with_tags,
        eager_accept=eager_accept,
    )


def _build_bma_fn(bma: opcs.TaggedOPCSBMA,
                  digest: str,
                  kind: str):
    """Compile one public regex runner.

    ``kind`` values:
        ``"match"``  — no tags, ``(n, data) -> u8``
        ``"match_tagged"`` — match BMA with tags, ``(n, data, out) -> u8``
        ``"search"`` — search BMA with tags, ``(n, data, out) -> u8``

    All return ``u8`` (1 = matched, 0 = not).
    """
    if kind == "match":
        func_name = "regex_match"
        params = [("n", u64), ("data", ptr[i8])]
        body_stmts = _build_bma_body(bma, with_tags=False, eager_accept=True)
    elif kind == "match_tagged":
        func_name = "regex_match_tagged"
        params = [("n", u64), ("data", ptr[i8]), ("out", ptr[i64])]
        body_stmts = _build_bma_body(bma, with_tags=True, eager_accept=True)
    elif kind == "search":
        func_name = "regex_search"
        params = [("n", u64), ("data", ptr[i8]), ("out", ptr[i64])]
        body_stmts = _build_bma_body(bma, with_tags=True, eager_accept=True)
    else:
        raise ValueError("Unknown BMA kind: {}".format(kind))

    for node in body_stmts:
        ast.fix_missing_locations(node)

    gf = meta.func(
        name=func_name,
        params=params,
        return_type=u8,
        body=body_stmts,
        required_globals={
            'i8': i8, 'i32': i32, 'i64': i64,
            'u8': u8, 'u64': u64, 'ptr': ptr,
            'label': label, 'goto': goto, 'goto_end': goto_end,
        },
        source_file=__file__,
        debug_source=f"# regex {kind} {digest}",
    )
    return meta.compile_generated(gf, suffix=digest)


# ---------------------------------------------------------------------------
# Native execution helpers
# ---------------------------------------------------------------------------

def _bytes_to_native_args(data: bytes):
    n = len(data)
    buf = ctypes.create_string_buffer(data, n)
    ptr_val = ctypes.cast(buf, ctypes.c_void_p).value
    return n, ptr_val, buf


# ---------------------------------------------------------------------------
# CompiledRegex
# ---------------------------------------------------------------------------

class CompiledRegex:
    """Holds a compiled regex pattern and provides ``match``/``search``.

    Both methods share the same universal ABI:
        ``run(n: u64, data: ptr[i8], out: ptr[i64]) -> u8``

    ``match`` runs the BMA on the original pattern (anchored at start).
    ``search`` runs the BMA on ``.*?{_beg}pattern{_end}`` (unanchored).

    Both return ``(bool, dict)`` where the dict maps tag names to
    byte-offset positions.  For ``search``, the dict always contains
    ``'start'`` and ``'end'`` keys derived from internal tags.
    """

    VALID_MODES = frozenset({"match", "search", "both"})

    _cache: Dict[Tuple[str, str], 'CompiledRegex'] = {}

    def __new__(cls, pattern: str, mode: str = "both"):
        key = (pattern, mode)
        if key in cls._cache:
            return cls._cache[key]
        instance = super().__new__(cls)
        cls._cache[key] = instance
        return instance

    def __init__(self, pattern: str, mode: str = "both"):
        if hasattr(self, '_initialized'):
            return
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'; expected one of {sorted(self.VALID_MODES)}")
        self._initialized = True
        self.pattern = pattern
        self.mode = mode
        self._re_ast = parse(pattern)
        self._digest = _pattern_digest(pattern)

        self._match_bma: Optional[opcs.TaggedOPCSBMA] = None
        self._search_bma: Optional[opcs.TaggedOPCSBMA] = None
        self._native_match_fn = None
        self._native_search_fn = None
        self._match_ctypes_struct = None
        self._search_ctypes_struct = None

        need_match = mode in ("match", "both")
        need_search = mode in ("search", "both")

        if need_match:
            match_nfa = build_nfa(self._re_ast)
            self.dfa = build_dfa(match_nfa)
            match_tag_runtime = opcs.compute_dfa_tag_runtime(
                match_nfa, self.dfa, include_internal=False)
            has_match_tags = bool(match_tag_runtime.tag_names)
            if has_match_tags:
                self._match_bma = opcs.build_tagged_opcs_bma(
                    self.dfa, match_tag_runtime)
                self._match_ctypes_struct = _make_ctypes_tag_struct(
                    self._match_bma.tag_names)
            else:
                self._match_bma = opcs.build_tagged_opcs_bma(self.dfa)
            self._native_match_fn = self._compile_match_fn()

        if need_search:
            search_ast = rewrite_for_search(self._re_ast)
            search_nfa = build_nfa(search_ast)
            search_dfa = build_dfa(search_nfa)
            search_tag_runtime = opcs.compute_dfa_tag_runtime(
                search_nfa, search_dfa, include_internal=True)
            beg_slot = search_tag_runtime.tag_slots[INTERNAL_BEG_TAG]
            search_tag_runtime = _filter_tag_runtime_slots(
                search_tag_runtime, frozenset({beg_slot}))
            search_restart_states = _compute_search_restart_states(search_dfa)
            self._search_bma = opcs.build_tagged_opcs_bma(
                search_dfa, search_tag_runtime,
                search_restart_controls=search_restart_states,
                search_entry_slot=beg_slot)
            self._search_ctypes_struct = _make_ctypes_tag_struct(
                self._search_bma.tag_names)
            self._native_search_fn = self._compile_search_fn()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def match(self, data: bytes) -> Tuple[bool, dict]:
        """Anchored match at position 0.

        Returns ``(matched, tags)`` where *tags* is a dict mapping
        user-defined tag names to byte positions.  When the pattern
        has no user tags the dict is empty.
        """
        if self._native_match_fn is None:
            raise RuntimeError(
                f"match unavailable: compiled with mode='{self.mode}'")
        n, ptr_val, buf = _bytes_to_native_args(data)
        if self._match_ctypes_struct is not None:
            out = self._match_ctypes_struct()
            matched = bool(self._native_match_fn(n, ptr_val, ctypes.pointer(out)))
            if not matched:
                return (False, {})
            tags: dict = {}
            for tag_name in self._match_bma.tag_names:
                val = int(getattr(out, tag_name))
                if val >= 0:
                    tags[tag_name] = val
            return (True, tags)
        else:
            matched = bool(self._native_match_fn(n, ptr_val))
            return (matched, {})

    def search(self, data: bytes) -> Tuple[bool, dict]:
        """Unanchored search — find the first occurrence anywhere.

        Returns ``(found, info)`` where *info* contains ``'start'``
        and ``'end'`` keys plus any user-defined tags.
        """
        if self._native_search_fn is None:
            raise RuntimeError(
                f"search unavailable: compiled with mode='{self.mode}'")
        n, ptr_val, buf = _bytes_to_native_args(data)
        out = self._search_ctypes_struct()
        matched = bool(self._native_search_fn(n, ptr_val, ctypes.pointer(out)))
        if not matched:
            return (False, {})
        beg_slot = self._search_bma.tag_slots[INTERNAL_BEG_TAG]
        end_slot = self._search_bma.tag_slots[INTERNAL_END_TAG]
        beg_name = self._search_bma.tag_names[beg_slot]
        end_name = self._search_bma.tag_names[end_slot]
        result: dict = {
            'start': int(getattr(out, beg_name)),
            'end': int(getattr(out, end_name)),
        }
        for tag_name in self._search_bma.tag_names:
            if tag_name.startswith(INTERNAL_TAG_PREFIX):
                continue
            val = int(getattr(out, tag_name))
            if val >= 0:
                result[tag_name] = val
        return (True, result)

    # -----------------------------------------------------------------
    # Native function compilation
    # -----------------------------------------------------------------

    def _compile_match_fn(self):
        kind = "match_tagged" if self._match_ctypes_struct else "match"
        return _build_bma_fn(self._match_bma, self._digest, kind)

    def _compile_search_fn(self):
        return _build_bma_fn(self._search_bma, self._digest, "search")

    def generate_match_fn(self):
        """Return the compiled match native function.

        Signature depends on whether the pattern contains user tags:
            no tags:  ``(n: u64, data: ptr[i8]) -> u8``
            tags:     ``(n: u64, data: ptr[i8], out: ptr[i64]) -> u8``

        Intended for embedding inside ``@compile`` functions.
        """
        if self._native_match_fn is None:
            raise RuntimeError(
                f"match unavailable: compiled with mode='{self.mode}'")
        return self._native_match_fn

    def generate_search_fn(self):
        """Return the compiled search native function.

        Signature: ``(n: u64, data: ptr[i8], out: ptr[i64]) -> u8``

        The ``out`` buffer must have room for all tag slots (including
        internal ``_beg``/``_end``).  Use ``search_num_slots`` for the
        required size.

        Intended for embedding inside ``@compile`` functions.
        """
        if self._native_search_fn is None:
            raise RuntimeError(
                f"search unavailable: compiled with mode='{self.mode}'")
        return self._native_search_fn

    @property
    def search_num_slots(self) -> int:
        """Number of i64 slots needed for the search output buffer."""
        if self._search_bma is None:
            return 0
        return len(self._search_bma.tag_names)
