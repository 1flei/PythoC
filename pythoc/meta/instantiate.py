"""
State-machine lowering for ``instantiate()``.

Supports four cases:
  Case 1a: bare yield function (no args)      -> instantiate(yield_fn)
  Case 1b: yield function call (with args)     -> instantiate(yield_fn(args))
  Case 2:  generator expression                -> instantiate(genexpr)
  Case 3:  constant iterable                   -> instantiate([1,2,3])

Architecture:
  1. All four cases normalise to a ``FunctionDef`` AST containing
     ``yield`` statements.
  2. ``_compile_yield_fn_pipeline`` does the API compilation:
     - scope analysis (``ScopeAnalyzer``)
     - state-machine lowering (``inline.yield_state_machine``)
     - struct compilation
     - function compilation
  3. The state-machine lowerer produces **structured** label/goto AST
     (``__pc_intrinsics.label/goto_begin/goto_end``) and lets the
     PythoC compiler build the actual jump table.
     No hand-rolled if-chain dispatch, no fragile AST scanning.
"""
from __future__ import annotations
import ast
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..meta.template import quote
from ..meta.compile_api import compile_ast as meta_compile_ast
from ..builtin_entities.types import i32
from ..builtin_entities.types import bool as pc_bool
from ..builtin_entities import ptr as pc_ptr

from ..inline.scope_analyzer import ScopeAnalyzer, ScopeContext
from ..inline._intrinsics import _PC_INTRINSICS
from ..inline.genexpr_builder import build_genexpr_yield_function_ast
from ..inline.yield_state_machine import (
    YieldStateMachineRequest, lower_yield_state_machine,
)

_SOURCE_FILE = os.path.abspath(__file__)

_INTRINSICS = frozenset({
    '_pc', '_yield_value',
    'goto', 'label',
    'True', 'False', 'None',
    'i32', 'i64', 'i8', 'i16',
    'f32', 'f64',
    'bool', 'ptr',
    'range', 'len', 'list', 'tuple',
})


@dataclass(frozen=True)
class _InstantiateSource:
    """Normalized input to the instantiate lowering pipeline."""

    source_kind: str
    func_ast: Optional[ast.FunctionDef] = None
    call_args: Tuple[Any, ...] = ()
    capture_bindings: Optional[Dict[str, Any]] = None
    callee_globals: Optional[Dict[str, Any]] = None
    source_object_id: int = 0
    func_name_hint: str = "instantiate"
    capture_mode: str = "yield"


def _compile_normalized_source(source: _InstantiateSource) -> Any:
    """Dispatch a normalized instantiate source to the right lowerer."""
    if source.source_kind == "closure":
        raise NotImplementedError(
            "instantiate: closure sources are reserved but not supported yet")

    if source.func_ast is None:
        raise TypeError("instantiate: normalized source has no function AST")

    return _compile_yield_fn_pipeline(
        source.func_ast,
        capture_bindings=source.capture_bindings,
        source_object_id=source.source_object_id,
        func_name_hint=source.func_name_hint,
        callee_globals=source.callee_globals,
    )


# ---------------------------------------------------------------------------
# Public helpers (used by builtin_entities/instantiate.py)
# ---------------------------------------------------------------------------

def _instantiate_yield(source) -> Any:
    """Case 1b: yield function CALL with arguments."""
    info = getattr(source, '_yield_inline_info', None)
    if not info:
        raise TypeError("instantiate: expected yield call placeholder")

    fa = info.get('original_ast')
    if fa is None:
        raise TypeError("instantiate: no original_ast in _yield_inline_info")

    call_args = info.get('call_args', [])
    arg_names = [a.arg for a in fa.args.args]
    if len(call_args) != len(arg_names):
        raise TypeError(
            f"instantiate: argument count mismatch: expected {len(arg_names)}, "
            f"got {len(call_args)}")

    # Extract compile-time constants from ValueRefs and build capture bindings
    capture_bindings: Dict[str, Any] = {}
    for name, vref in zip(arg_names, call_args):
        val = _extract_value_from_vref(vref)
        # ValueRef carries type_hint; we forward both value + type
        # to _compile_yield_fn_pipeline so the struct field gets
        # the right annotation.
        capture_bindings[name] = val

    return _compile_normalized_source(_InstantiateSource(
        source_kind="yield_call",
        func_ast=fa,
        call_args=tuple(call_args),
        capture_bindings=capture_bindings,
        source_object_id=id(info.get('placeholder', source)),
        func_name_hint=fa.name,
        callee_globals=info.get('callee_globals', {}),
    ))


def _instantiate_genexpr(source) -> Any:
    """Case 2: generator expression."""
    info = getattr(source, '_pc_generator_expr_info', None)
    if not info or info.get('kind') != 'generator_expression':
        raise TypeError("instantiate: expected generator expression")

    genexp_ast = info.get('ast')
    if genexp_ast is None:
        raise TypeError("instantiate: no ast in _pc_generator_expr_info")

    fa = build_genexpr_yield_function_ast(genexp_ast, "__pc_genexp")

    if isinstance(genexp_ast.elt, ast.Constant):
        fa.returns = _type_to_annotation_ast(
            _python_constant_pc_type(genexp_ast.elt.value))

    return _compile_normalized_source(_InstantiateSource(
        source_kind="generator_expression",
        func_ast=fa,
        capture_bindings=None,          # genexpr captures handled below
        source_object_id=id(source),
        func_name_hint="genexpr",
    ))


def _instantiate_closure(source) -> Any:
    """Reserve the architecture slot for closure-backed instantiate."""
    func_ast = getattr(source, "func_ast", None)
    return _compile_normalized_source(_InstantiateSource(
        source_kind="closure",
        func_ast=func_ast,
        capture_bindings=None,
        source_object_id=id(source),
        func_name_hint=getattr(func_ast, "name", "closure"),
        callee_globals=getattr(source, "func_globals", None) or {},
        capture_mode="closure",
    ))


class _ExternalNameCollector(ast.NodeVisitor):
    """Collect all Name references that are not assignments."""
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Don't recurse into nested function bodies
        pass


def _find_external_names(
    func_ast: ast.FunctionDef,
    locals_set: Set[str],
    protect_set: Set[str],
) -> List[str]:
    """Find names referenced in func_ast that need external resolution."""
    collector = _ExternalNameCollector()
    for stmt in func_ast.body:
        collector.visit(stmt)
    # Exclude annotations on parameters (not Load context, so collector
    # ignores them anyway), but filter against locals and intrinsics.
    return sorted(
        n for n in collector.names
        if n not in locals_set and n not in protect_set
    )


def _instantiate_const_iterable(values: List[Any]) -> Any:
    """Case 3: compile-time constant iterable (e.g. ``[1, 2, 3]``)."""
    elem_type = _python_constant_pc_type(values[0]) if values else i32
    for v in values:
        vt = _python_constant_pc_type(v)
        if vt is not elem_type:
            raise TypeError(
                f"instantiate: heterogeneous element types: "
                f"{elem_type.__name__} vs {vt.__name__}")

    suffix = ("_c", "const", id(values) % 10007)
    _Iter = _build_state_struct({"_pc": i32}, suffix)
    ip = pc_ptr[_Iter]
    sfx = "_".join(str(p) for p in suffix)
    gv = {"i32": i32, "bool": pc_bool, "ptr": pc_ptr, "_Iter": _Iter}
    elem_type_name = getattr(elem_type, "__name__", None)
    if elem_type_name:
        gv[elem_type_name] = elem_type

    nf = _cq(
        _next_tmpl.instantiate(N=len(values)),
        sfx + "_n", {"s": ip}, pc_bool, gv)
    inf = _cq(
        _init_tmpl.instantiate(),
        sfx + "_i", {"s": ip}, None, gv)

    # Build value-case statements via template
    elem_converter = _type_to_annotation_ast(elem_type)
    case_stmts = [
        _value_case.instantiate(i=i, converter=elem_converter, v=v).stmts[0]
        for i, v in enumerate(values)
    ]
    vf = _cq(
        _value_tmpl.instantiate(
            cases=case_stmts,
            default_value=_typed_zero(elem_type),
        ),
        sfx + "_v", {"s": ip}, elem_type, gv)

    class _Api:
        Iter = _Iter
        next = nf
        value = vf
        init = inf
    return _Api()


# ---------------------------------------------------------------------------
# Core pipeline:  AST -> state struct -> compiled API
# ---------------------------------------------------------------------------

def _compile_yield_fn_pipeline(
    func_ast: ast.FunctionDef,
    *,
    capture_bindings: Optional[Dict[str, Any]],
    source_object_id: int,
    func_name_hint: str,
    callee_globals: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Shared pipeline.

    Args:
        func_ast: Yield function AST (may contain parameters).
        capture_bindings: Mapping ``name -> compile_time_value`` for
            arguments that need to be bound at instantiate-time.
        source_object_id: Used for suffix uniqueness.
        func_name_hint: Human-readable name fragment for suffix.
        callee_globals: Mapping ``name -> compile_time_value`` for
            external symbols that need to be bound at instantiate-time.
    """
    fa = copy.deepcopy(func_ast)

    # -- 1. splice capture initialisers at body top --
    if capture_bindings:
        pre = [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_python_value_to_ast(val))
            for name, val in capture_bindings.items()
        ]
        fa.body = pre + fa.body

    # -- 2. scope analysis --
    analyzer = ScopeAnalyzer(caller_context=ScopeContext.empty())
    captured_vars, local_vars, param_vars = analyzer.analyze(fa.body, fa.args.args)
    # Parameters need to be stored in the state struct too, because label/goto
    # control flow breaks normal lexical scoping – a variable assigned in one
    # state block is not visible in another unless it lives on the state object.
    locals_set: Set[str] = local_vars | captured_vars | param_vars
    # Intrinsics + struct param ('s') should never be rewritten
    protect_set: Set[str] = set(_INTRINSICS) | {'s'}
    type_resolver = _build_type_resolver(callee_globals)

    # -- 3. infer yield value type from function annotation --
    yield_type = (
        _annotation_ast_to_type(fa.returns, type_resolver)
        if fa.returns is not None
        else None
    )
    if yield_type is None:
        yield_type = i32  # default for backward compat

    # -- 4. build state machine (structured label/goto) --
    compiletime_globals = dict(callee_globals or {})
    if capture_bindings:
        compiletime_globals.update(capture_bindings)
    sm_ast = lower_yield_state_machine(YieldStateMachineRequest(
        func_ast=fa,
        locals_set=locals_set,
        protect_set=protect_set,
        compiletime_globals=compiletime_globals,
    ))

    # -- 5. suffix / naming --
    # Include capture bindings in suffix so different parameter instantiations
    # of the same yield function get unique compiled artefacts.
    if capture_bindings:
        binding_hash = hash(
            tuple(sorted(capture_bindings.items(), key=lambda kv: kv[0]))
        ) % 1000003
        suffix = ("_y", func_name_hint, source_object_id % 10007, binding_hash)
    else:
        suffix = ("_y", func_name_hint, source_object_id % 10007)
    sfx = "_".join(str(p) for p in suffix)

    # -- 6. state struct fields --
    fields = {"_pc": i32, "_yield_value": yield_type}
    for name in locals_set:
        # Determine type from local annotations when available
        lt = _local_type_hint(name, fa, type_resolver)
        fields[name] = lt if lt is not None else i32

    _Iter = _build_state_struct(fields, suffix)
    ip = pc_ptr[_Iter]

    # -- 7. compile next / init / value --
    gv = {
        "i32": i32,
        "bool": pc_bool,
        "ptr": pc_ptr,
        "_Iter": _Iter,
        "__pc_intrinsics": _PC_INTRINSICS,
    }

    # -- 7b. resolve external references from callee globals --
    for ext_name in _find_external_names(fa, locals_set, protect_set):
        val = callee_globals.get(ext_name) if callee_globals else None
        if val is not None:
            protect_set.add(ext_name)
            gv[ext_name] = val

    nf = meta_compile_ast(
        sm_ast,
        param_types={"s": ip},
        return_type=pc_bool,
        suffix=sfx + "_n",
        user_globals=gv,
        source_file=_SOURCE_FILE,
    )
    inf = _cq(_init_tmpl.instantiate(), sfx + "_i", {"s": ip}, None, gv)
    vf = _cq(_yld_value.instantiate(), sfx + "_v", {"s": ip}, yield_type, gv)

    class _Api:
        Iter = _Iter
        next = nf
        value = vf
        init = inf
    return _Api()

# ---------------------------------------------------------------------------
# Templates (quote)
# ---------------------------------------------------------------------------

@quote
def _init_tmpl():
    def _init(s):
        s._pc = i32(0)


@quote
def _yld_value():
    def _value(s):
        return s._yield_value


@quote
def _next_tmpl(N):
    def _next(s):
        if s._pc >= i32(N):
            return bool(0)
        s._pc = s._pc + i32(1)
        return bool(1)


@quote
def _value_case(i, converter, v):
    if idx == i32(i):
        return converter(v)


@quote
def _value_tmpl(cases, default_value):
    def _value(s):
        idx: i32 = s._pc - i32(1)
        cases
        return default_value


# ---------------------------------------------------------------------------
# Type utilities
# ---------------------------------------------------------------------------

def _python_constant_pc_type(value: Any) -> type:
    """Infer the PC type using the compiler's Python-constant promotion rule."""
    from ..type_converter import TypeConverter
    return TypeConverter.infer_default_pc_type_from_python(value)


def _extract_value_from_vref(vref) -> Any:
    """Extract a Python compile-time constant from a ValueRef."""
    th = getattr(vref, 'type_hint', None)
    if th is not None:
        if hasattr(th, 'is_constant') and th.is_constant():
            return th.get_constant_value()
        if hasattr(th, 'get_python_object'):
            return th.get_python_object()
    return vref.get_python_value()


def _build_type_resolver(callee_globals: Optional[Dict[str, Any]]):
    from ..type_resolver import TypeResolver
    return TypeResolver(user_globals=callee_globals or {})


def _annotation_ast_to_type(node, type_resolver) -> Optional[type]:
    """Resolve a PythoC annotation using the shared type resolver."""
    if node is None:
        return None
    return type_resolver.parse_annotation(node)


def _type_to_annotation_ast(ty: type) -> ast.expr:
    """Reverse of _annotation_ast_to_type."""
    name = getattr(ty, '__name__', None)
    if name is None:
        return ast.Name(id="i32", ctx=ast.Load())
    return ast.Name(id=name, ctx=ast.Load())


def _typed_zero(ty: type) -> ast.expr:
    return ast.Call(
        func=_type_to_annotation_ast(ty),
        args=[ast.Constant(value=0)],
        keywords=[],
    )


def _local_type_hint(
    name: str,
    fa: ast.FunctionDef,
    type_resolver,
) -> Optional[type]:
    """Search function parameters and body for a type annotation of ``name``."""
    # Search parameter annotations
    for arg in fa.args.args:
        if arg.arg == name and arg.annotation is not None:
            return _annotation_ast_to_type(arg.annotation, type_resolver)
    for arg in fa.args.posonlyargs:
        if arg.arg == name and arg.annotation is not None:
            return _annotation_ast_to_type(arg.annotation, type_resolver)
    for arg in fa.args.kwonlyargs:
        if arg.arg == name and arg.annotation is not None:
            return _annotation_ast_to_type(arg.annotation, type_resolver)
    if fa.args.vararg and fa.args.vararg.arg == name and fa.args.vararg.annotation is not None:
        return _annotation_ast_to_type(fa.args.vararg.annotation, type_resolver)
    if fa.args.kwarg and fa.args.kwarg.arg == name and fa.args.kwarg.annotation is not None:
        return _annotation_ast_to_type(fa.args.kwarg.annotation, type_resolver)
    # Search body annotated assignments
    for stmt in fa.body:
        if (isinstance(stmt, ast.AnnAssign) and
                isinstance(stmt.target, ast.Name) and
                stmt.target.id == name):
            return _annotation_ast_to_type(stmt.annotation, type_resolver)
    return None


def _python_value_to_ast(v: Any) -> ast.expr:
    """Convert a Python compile-time value into an AST expression node."""
    if isinstance(v, ast.AST):
        return v
    if isinstance(v, (int, bool, float, str, type(None))):
        return ast.Constant(value=v)
    # ValueRef – try to unwrap
    th = getattr(v, 'type_hint', None)
    if th is not None and hasattr(th, 'is_constant') and th.is_constant():
        return ast.Constant(value=th.get_constant_value())
    raise TypeError(
        f"instantiate: cannot convert {type(v).__name__} to AST expression")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _build_state_struct(fields_dict: Dict[str, type], suffix: Tuple) -> type:
    from ..decorators.structs import compile_dynamic_class
    class _Iter:
        __annotations__ = fields_dict
    _Iter.__name__ = 'Iter'
    return compile_dynamic_class(_Iter, suffix=suffix)


def _cq(frag, sfx, pt, rt, gv):
    if not frag.stmts or not isinstance(frag.stmts[0], ast.FunctionDef):
        raise TypeError("instantiate: expected function template fragment")
    f = frag.with_func_name(frag.stmts[0].name + '_' + sfx).stmts[0]
    return meta_compile_ast(f, param_types=pt, return_type=rt,
                            suffix=sfx, user_globals=gv, source_file=_SOURCE_FILE)