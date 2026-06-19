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
from ..builtin_entities.types import i32, void
from ..builtin_entities.types import bool as pc_bool
from ..builtin_entities import ptr as pc_ptr

from ..inline.scope_analyzer import ScopeAnalyzer, ScopeContext, analyze_function_scope
from ..inline._intrinsics import _PC_INTRINSICS
from ..inline.genexpr_builder import build_genexpr_yield_function_ast
from ..inline.closure_capture import (
    RuntimeCapture,
    extract_value_from_vref,
    normalize_runtime_captures,
)
from ..inline.state_field_rewriter import (
    StateFieldRewriter,
    StateFieldRewritePolicy,
)
from ..inline.yield_state_machine import (
    YieldStateMachineRequest, lower_yield_state_machine,
    inline_yield_function_iterators,
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
    'array', 'struct', 'union', 'enum',
})

_RETURN_SENTINEL_PC = -1


@dataclass
class _InstantiateSource:
    """Normalized source descriptor for all instantiate() cases."""
    source_kind: str  # "yield_call", "generator_expression", "closure", "const_iterable"
    func_ast: Optional[ast.FunctionDef]
    capture_bindings: Optional[Dict[str, Any]]
    source_object_id: int
    func_name_hint: str
    callee_globals: Optional[Dict[str, Any]] = None
    call_args: Tuple[Any, ...] = ()
    capture_mode: Optional[str] = None
    capture_runtime: Optional[List[Tuple[str, type]]] = None


class _InitDispatcher:
    def __init__(
        self,
        init_into_fn=None,
        bound_args=None,
        required_bound_arg_count: Optional[int] = None,
        state_type: Optional[type] = None,
        runtime_captures: Optional[List[RuntimeCapture]] = None,
        set_pc: bool = False,
    ):
        self._init_into_fn = init_into_fn
        self._bound_args = list(bound_args or [])
        self._required_bound_arg_count = (
            len(self._bound_args)
            if required_bound_arg_count is None
            else required_bound_arg_count
        )
        self._state_type = state_type
        self._runtime_captures = list(runtime_captures or [])
        self._set_pc = set_pc

    def handle_call(self, visitor, func_ref, args, node):
        if not args:
            if len(self._bound_args) != self._required_bound_arg_count:
                from ..logger import logger
                logger.error(
                    "instantiate init cannot bind runtime captures at this "
                    "call site",
                    node=node,
                    exc_type=TypeError,
                )
            bound_args = [
                _capture_value_as_rvalue(visitor, arg)
                for arg in self._bound_args
            ]
            return _materialize_state_value(
                visitor,
                self._state_type,
                self._runtime_captures,
                bound_args,
                set_pc=self._set_pc,
            )
        if self._init_into_fn is None:
            from ..logger import logger
            logger.error(
                "instantiate init does not accept explicit state arguments",
                node=node,
                exc_type=TypeError,
            )
        return self._init_into_fn.handle_call(visitor, func_ref, args, node)


def _compile_normalized_source(source: _InstantiateSource) -> Any:
    """Dispatch a normalized instantiate source to the right lowerer."""
    if source.func_ast is None:
        raise TypeError("instantiate: normalized source has no function AST")

    if source.source_kind == "closure":
        return _compile_closure_fn_pipeline(
            source.func_ast,
            capture_bindings=source.capture_bindings,
            source_object_id=source.source_object_id,
            func_name_hint=source.func_name_hint,
            callee_globals=source.callee_globals,
            capture_runtime=source.capture_runtime,
        )

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
        callee_globals=info.get("globals", {}),
    ))


def _instantiate_closure(source) -> Any:
    """Reserve the architecture slot for closure-backed instantiate."""
    func_ast = getattr(source, "func_ast", None)
    callee_globals = dict(getattr(source, "func_globals", None) or {})
    # Ensure builtin PC types are available for type annotation resolution
    # ( FakeClosure tests or closures defined with empty globals need this )
    from ..builtin_entities import (
        i8, i16, i32, i64, u8, u16, u32, u64, f32, f64,
        bool as pc_bool_b, ptr as pc_ptr_b, array, struct, union, enum,
    )
    _builtins = {
        "i8": i8, "i16": i16, "i32": i32, "i64": i64,
        "u8": u8, "u16": u16, "u32": u32, "u64": u64,
        "f32": f32, "f64": f64, "bool": pc_bool_b, "ptr": pc_ptr_b,
        "array": array, "struct": struct, "union": union, "enum": enum,
    }
    for _name, _val in _builtins.items():
        if _name not in callee_globals:
            callee_globals[_name] = _val
    return _compile_normalized_source(_InstantiateSource(
        source_kind="closure",
        func_ast=func_ast,
        capture_bindings=getattr(source, "_capture_bindings", None),
        source_object_id=id(source),
        func_name_hint=getattr(func_ast, "name", "closure"),
        callee_globals=callee_globals,
        capture_mode="closure",
        capture_runtime=getattr(source, "_capture_runtime", None),
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
    _State = _build_state_struct({"_pc": i32}, suffix)
    ip = pc_ptr[_State]
    sfx = "_".join(str(p) for p in suffix)
    gv = {"i32": i32, "bool": pc_bool, "ptr": pc_ptr, "_State": _State}
    elem_type_name = getattr(elem_type, "__name__", None)
    if elem_type_name:
        gv[elem_type_name] = elem_type

    nf = _cq(
        _next_tmpl.instantiate(N=len(values)),
        sfx + "_n", {"s": ip}, pc_bool, gv)
    init_into_fn = _cq(
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
        State = _State
        Iter = _State
        next = nf
        value = vf
        init = _InitDispatcher(
            init_into_fn,
            state_type=_State,
            set_pc=True,
        )
    return _Api()


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------

def _splice_capture_bindings(
    fa: ast.FunctionDef,
    capture_bindings: Optional[Dict[str, Any]],
) -> None:
    """Splice compile-time capture initialisers at the top of ``fa.body``."""
    if not capture_bindings:
        return
    pre = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=_python_value_to_ast(val))
        for name, val in capture_bindings.items()
    ]
    fa.body = pre + fa.body


def _infer_return_type(
    fa: ast.FunctionDef,
    type_resolver,
    default: type = i32,
) -> type:
    """Infer the return/yield value type from annotations, falling back to ``default``."""
    ret = (
        _annotation_ast_to_type(fa.returns, type_resolver)
        if fa.returns is not None
        else None
    )
    return ret if ret is not None else default


def _compute_instantiate_suffix(
    prefix: str,
    func_name_hint: str,
    source_object_id: int,
    *binding_parts,
) -> Tuple:
    """Build a deterministic suffix tuple for compiled artefacts."""
    parts = [prefix, func_name_hint, source_object_id % 10007]
    parts.extend(binding_parts)
    return tuple(parts)


def _build_callee_globals(
    callee_globals: Optional[Dict[str, Any]],
    locals_set: Set[str],
    protect_set: Set[str],
    fa: ast.FunctionDef,
    *,
    extra_globals: Optional[Dict[str, Any]] = None,
    include_intrinsics: bool = True,
) -> Dict[str, Any]:
    """Merge callee globals and resolve external names referenced by ``fa``.

    Exposes builtin PC types, resolves names that appear in ``callee_globals``,
    and forwards the callee's own globals so annotations and builtin names are
    available to the compiled functions.
    """
    gv = {
        "i32": i32,
        "bool": pc_bool,
        "ptr": pc_ptr,
    }
    if include_intrinsics:
        gv["__pc_intrinsics"] = _PC_INTRINSICS
    if extra_globals:
        gv.update(extra_globals)

    for ext_name in _find_external_names(fa, locals_set, protect_set):
        val = callee_globals.get(ext_name) if callee_globals else None
        if val is not None:
            protect_set.add(ext_name)
            gv[ext_name] = val

    if callee_globals:
        for cg_name, cg_val in callee_globals.items():
            if cg_name not in gv and not cg_name.startswith("_"):
                gv[cg_name] = cg_val

    return gv


def _compile_ast_function(
    func_ast: ast.FunctionDef,
    *,
    param_types: Dict[str, type],
    return_type: type,
    suffix: str,
    gv: Dict[str, Any],
) -> Any:
    """Compile a single AST function via the meta API."""
    return meta_compile_ast(
        func_ast,
        param_types=param_types,
        return_type=return_type,
        suffix=suffix,
        user_globals=gv,
        source_file=_SOURCE_FILE,
    )


def _build_init_into_fn(
    runtime_captures: List[RuntimeCapture],
    ip: type,
    suffix: str,
    gv: Dict[str, Any],
    *,
    set_pc: bool = False,
) -> Optional[Any]:
    """Build and compile the state materialisation function.

    The generated function takes the state pointer ``s`` followed by one
    argument per runtime capture, sets ``s._pc`` when ``set_pc`` is true,
    and stores each capture in its state field.
    """
    if any(_is_array_type(capture.pc_type) for capture in runtime_captures):
        return None

    init_body: List[ast.stmt] = []
    init_args_list = [ast.arg(arg="s", annotation=None)]
    init_param_types: Dict[str, type] = {"s": ip}

    if set_pc:
        init_body.append(ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="s", ctx=ast.Load()),
                attr="_pc",
                ctx=ast.Store(),
            )],
            value=ast.Call(
                func=ast.Name(id="i32", ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[],
            ),
        ))

    for capture in runtime_captures:
        init_arg_name = f"_{capture.name}_in"
        init_args_list.append(ast.arg(arg=init_arg_name, annotation=None))
        init_param_types[init_arg_name] = capture.pc_type
        assign = _closure_capture_assign(init_arg_name).stmts[0]
        assign.targets[0].attr = capture.name
        init_body.append(assign)

    if not init_body:
        init_body.append(ast.Pass())

    init_ast = _closure_init_template(init_body).stmts[0]
    init_ast.args = ast.arguments(
        posonlyargs=[],
        args=init_args_list,
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
    )
    ast.fix_missing_locations(init_ast)

    return _compile_ast_function(
        init_ast,
        param_types=init_param_types,
        return_type=void,
        suffix=suffix,
        gv=gv,
    )


def _build_yield_api(next_fn, value_fn, init_dispatcher, state_type) -> Any:
    """Build the public API object for a compiled yield iterator."""
    class _Api:
        State = state_type
        Iter = state_type
        next = next_fn
        value = value_fn
        init = init_dispatcher
    return _Api()


def _build_closure_api(call_fn, init_dispatcher, state_type) -> Any:
    """Build the public API object for a compiled closure."""
    class _Api:
        State = state_type
        Obj = state_type
        call = call_fn
        init = init_dispatcher
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
    capture_runtime: Optional[List[Tuple[str, type]]] = None,
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
        capture_runtime: Optional list of ``(name, pc_type)`` runtime captures
            that must be stored in the state struct and passed to ``init``.
    """
    fa = copy.deepcopy(func_ast)
    runtime_captures = normalize_runtime_captures(capture_runtime)

    _splice_capture_bindings(fa, capture_bindings)

    captured_vars, local_vars, param_vars = analyze_function_scope(fa)
    # Parameters need to be stored in the state struct too, because label/goto
    # control flow breaks normal lexical scoping – a variable assigned in one
    # state block is not visible in another unless it lives on the state object.
    locals_set: Set[str] = local_vars | captured_vars | param_vars
    # Runtime captures are also stored on the state struct and must be rewritten
    # to state-field accesses inside the state-machine function.
    locals_set.update(capture.name for capture in runtime_captures)
    # Intrinsics + struct param ('s') should never be rewritten
    protect_set: Set[str] = set(_INTRINSICS) | {'s'}
    type_resolver = _build_type_resolver(callee_globals)

    yield_type = _infer_return_type(fa, type_resolver)

    compiletime_globals = dict(callee_globals or {})
    if capture_bindings:
        compiletime_globals.update(capture_bindings)

    fa = inline_yield_function_iterators(fa, compiletime_globals)

    sm_ast = lower_yield_state_machine(YieldStateMachineRequest(
        func_ast=fa,
        locals_set=locals_set,
        protect_set=protect_set,
        compiletime_globals=compiletime_globals,
    ))

    binding_parts = []
    if capture_bindings:
        binding_parts.append(
            hash(
                tuple(sorted(capture_bindings.items(), key=lambda kv: kv[0]))
            ) % 1000003
        )
    if runtime_captures:
        binding_parts.append(
            hash(tuple(capture.name for capture in runtime_captures)) % 1000003
        )
    suffix = _compute_instantiate_suffix(
        "_y", func_name_hint, source_object_id, *binding_parts
    )
    sfx = "_".join(str(p) for p in suffix)

    fields = {"_pc": i32, "_yield_value": yield_type}
    for name in locals_set:
        lt = _local_type_hint(name, fa, type_resolver)
        fields[name] = lt if lt is not None else i32
    for capture in runtime_captures:
        fields[capture.name] = capture.pc_type

    _State = _build_state_struct(fields, suffix)
    ip = pc_ptr[_State]

    from pythoc import move
    gv = _build_callee_globals(
        callee_globals,
        locals_set,
        protect_set,
        fa,
        extra_globals={"_State": _State, "move": move},
        include_intrinsics=True,
    )

    nf = _compile_ast_function(
        sm_ast,
        param_types={"s": ip},
        return_type=pc_bool,
        suffix=sfx + "_n",
        gv=gv,
    )
    init_into_fn = _build_init_into_fn(
        runtime_captures, ip, sfx + "_i", gv, set_pc=True
    )
    vf = _cq(_yld_value.instantiate(), sfx + "_v", {"s": ip}, yield_type, gv)

    return _build_yield_api(
        nf, vf,
        _InitDispatcher(
            init_into_fn,
            _runtime_capture_values(runtime_captures),
            len(runtime_captures),
            state_type=_State,
            runtime_captures=runtime_captures,
            set_pc=True,
        ),
        _State,
    )


def _compile_closure_fn_pipeline(
    func_ast: ast.FunctionDef,
    *,
    capture_bindings: Optional[Dict[str, Any]],
    source_object_id: int,
    func_name_hint: str,
    callee_globals: Optional[Dict[str, Any]] = None,
    capture_runtime: Optional[List[Tuple[str, type]]] = None,
) -> Any:
    """Compile a closure into a callable first-class object.

    Supports both compile-time constant capture (via capture_bindings)
    and runtime capture (via capture_runtime + api.init()).

    The compiled API exposes::

        api.State – state struct type (holds runtime capture variables)
        api.init  – init function ``init() -> State``
        api.call  – compiled wrapper function ``call(s, *args) -> ret``
    """
    fa = copy.deepcopy(func_ast)
    runtime_captures = normalize_runtime_captures(capture_runtime)
    if _contains_yield(fa):
        from ..logger import logger
        logger.error(
            "instantiate: closure sources containing yield are not supported",
            node=func_ast,
            exc_type=TypeError,
        )

    _splice_capture_bindings(fa, capture_bindings)

    captured_vars, local_vars, param_vars = analyze_function_scope(fa)
    # For closures:
    #   - captured_vars -> state struct fields (persist across calls)
    #   - local_vars    -> ordinary stack locals (reset each call)
    #   - param_vars    -> real function parameters (passed each call)
    # Only captured variables need to be rewritten to s.xxx.
    state_vars: Set[str] = set(captured_vars)
    protect_set: Set[str] = set(_INTRINSICS) | {"s"}
    type_resolver = _build_type_resolver(callee_globals)

    for capture in runtime_captures:
        state_vars.add(capture.name)

    return_type = _infer_return_type(fa, type_resolver)

    rewriter = StateFieldRewriter(StateFieldRewritePolicy(
        field_names=state_vars,
        protect_names=protect_set,
        state_arg="s",
        strip_yield_expr=False,
        preserve_nested_functions=True,
    ))
    rewritten_body: List[ast.stmt] = []
    for stmt in fa.body:
        rewritten = rewriter.visit(stmt)
        if rewritten is not None:
            if isinstance(rewritten, (list, tuple)):
                rewritten_body.extend(rewritten)
            else:
                rewritten_body.append(rewritten)

    # Ensure the wrapper body always ends with a return (compile_ast needs
    # a concrete return type; if the original closure lacked one we emit
    # a zero of the inferred return type).
    if not rewritten_body or not isinstance(rewritten_body[-1], ast.Return):
        rewritten_body.append(ast.Return(value=_typed_zero(return_type)))

    wrapper_args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="s", annotation=None)] + list(fa.args.args),
        kwonlyargs=[],
        kw_defaults=[],
        defaults=list(fa.args.defaults),
        vararg=fa.args.vararg,
        kwarg=fa.args.kwarg,
    )
    wrapper_ast = _closure_call_template(rewritten_body).stmts[0]
    wrapper_ast.args = wrapper_args
    ast.fix_missing_locations(wrapper_ast)

    binding_parts = []
    if capture_bindings:
        binding_parts.append(
            hash(tuple(sorted(capture_bindings.items(), key=lambda kv: kv[0]))) % 1000003
        )
    if runtime_captures:
        binding_parts.append(
            hash(tuple(capture.name for capture in runtime_captures)) % 1000003
        )
    suffix = _compute_instantiate_suffix(
        "_cl", func_name_hint, source_object_id, *binding_parts
    )
    sfx = "_".join(str(p) for p in suffix)

    fields: Dict[str, type] = {}
    for name in captured_vars:
        lt = _local_type_hint(name, fa, type_resolver)
        fields[name] = lt if lt is not None else i32
    for capture in runtime_captures:
        fields[capture.name] = capture.pc_type

    _State = _build_state_struct(fields, suffix)
    ip = pc_ptr[_State]

    param_types: Dict[str, type] = {"s": ip}
    for arg in fa.args.args:
        at = (
            _annotation_ast_to_type(arg.annotation, type_resolver)
            if arg.annotation
            else None
        )
        if at is not None:
            param_types[arg.arg] = at

    from pythoc import move
    gv = _build_callee_globals(
        callee_globals,
        state_vars | param_vars,
        protect_set,
        wrapper_ast,
        extra_globals={"_State": _State, "_Ret": return_type, "move": move},
        include_intrinsics=False,
    )

    cf = _compile_ast_function(
        wrapper_ast,
        param_types=param_types,
        return_type=return_type,
        suffix=sfx,
        gv=gv,
    )

    init_into_fn = _build_init_into_fn(
        runtime_captures, ip, sfx + "_ii", gv, set_pc=False
    )

    return _build_closure_api(
        cf,
        _InitDispatcher(
            init_into_fn,
            _runtime_capture_values(runtime_captures),
            len(runtime_captures),
            state_type=_State,
            runtime_captures=runtime_captures,
        ),
        _State,
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _runtime_capture_values(captures: List[RuntimeCapture]) -> List[Any]:
    values = []
    for capture in captures:
        if capture.value_ref is not None:
            values.append(capture.value_ref)
    return values


def _capture_value_as_rvalue(visitor, value_ref):
    if not hasattr(value_ref, "has_place") or not value_ref.has_place():
        return value_ref
    from ..valueref import wrap_value
    loaded = visitor.builder.load(value_ref.require_place())
    return wrap_value(loaded, kind="value", type_hint=value_ref.type_hint)


def _materialize_state_value(
    visitor,
    state_type: type,
    runtime_captures: List[RuntimeCapture],
    bound_values: List[Any],
    *,
    set_pc: bool,
):
    from ..valueref import wrap_value

    llvm_type = state_type.get_llvm_type(visitor.module.context)
    storage = visitor._create_alloca_in_entry(llvm_type, "instantiate_state")
    zero_value = visitor.type_converter.create_zero_constant(llvm_type)
    visitor.builder.store(zero_value, storage)

    state_ref = wrap_value(
        visitor.builder.load(storage),
        kind="address",
        type_hint=state_type,
        address=storage,
    )

    if set_pc and state_type.has_field("_pc"):
        from ..valueref import wrap_value
        pc_field = state_type.handle_attribute(
            visitor,
            state_ref,
            "_pc",
            None,
        )
        _store_capture_field(
            visitor,
            pc_field,
            wrap_value(_i32_const_ir(), kind="value", type_hint=i32),
        )

    for capture, value in zip(runtime_captures, bound_values):
        field_ref = state_type.handle_attribute(
            visitor,
            state_ref,
            capture.name,
            None,
        )
        _store_capture_field(visitor, field_ref, value)

    return wrap_value(
        visitor.builder.load(storage),
        kind="address",
        type_hint=state_type,
        address=storage,
    )


def _store_capture_field(visitor, field_ref, value_ref):
    from ..valueref import ensure_ir
    from ..ir_helpers import safe_store

    target_type = field_ref.get_pc_type()
    value_ref = visitor.implicit_coercer.coerce(value_ref, target_type, None)
    safe_store(
        visitor.builder,
        ensure_ir(value_ref),
        field_ref.require_place(),
        target_type,
        node=None,
    )


def _i32_const_ir(value: int = 0):
    from llvmlite import ir
    return ir.Constant(ir.IntType(32), value)


def _contains_yield(func_ast: ast.FunctionDef) -> bool:
    for node in ast.walk(func_ast):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _is_array_type(pc_type: type) -> bool:
    return hasattr(pc_type, "get_decay_pointer_type")


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


@quote
def _closure_call_template(body):
    def _call(s) -> _Ret:
        body


@quote
def _closure_init_template(body):
    def _init_into(s) -> None:
        body


@quote
def _closure_capture_assign(arg_name):
    s._placeholder = move(arg_name)


# ---------------------------------------------------------------------------
# Type utilities
# ---------------------------------------------------------------------------

def _python_constant_pc_type(value: Any) -> type:
    """Infer the PC type using the compiler's Python-constant promotion rule."""
    from ..type_converter import TypeConverter
    return TypeConverter.infer_default_pc_type_from_python(value)


def _extract_value_from_vref(vref) -> Any:
    """Extract a Python compile-time constant from a ValueRef."""
    return extract_value_from_vref(vref)


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
    class _State:
        __annotations__ = fields_dict
    _State.__name__ = 'State'
    return compile_dynamic_class(_State, suffix=suffix)


def _cq(frag, sfx, pt, rt, gv):
    if not frag.stmts or not isinstance(frag.stmts[0], ast.FunctionDef):
        raise TypeError("instantiate: expected function template fragment")
    f = frag.with_func_name(frag.stmts[0].name + '_' + sfx).stmts[0]
    return meta_compile_ast(f, param_types=pt, return_type=rt,
                            suffix=sfx, user_globals=gv, source_file=_SOURCE_FILE)