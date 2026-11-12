"""
State-machine lowering for ``instantiate()``.

Supports four cases:
  Case 1a: bare yield function (no args)      -> instantiate(yield_fn)
  Case 1b: yield function call (with args)     -> instantiate(yield_fn(args))
  Case 2:  generator expression with known yield metadata
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

from .._inline.scope_analyzer import ScopeAnalyzer, ScopeContext, analyze_function_scope
from .._inline._intrinsics import _PC_INTRINSICS
from .._inline.genexpr_builder import build_genexpr_yield_function_ast
from .._inline.names import ExpansionNames
from .._inline.definition_bindings import bind_definition, symbol_expression
from .._inline.closure_capture import (
    RuntimeCapture,
    extract_value_from_vref,
    normalize_runtime_captures,
    snapshot_closure_captures,
)
from .._inline.state_field_rewriter import (
    StateFieldRewriter,
    StateFieldRewritePolicy,
)
from .._inline.yield_state_machine import (
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
    source_kind: str  # "yield_call", "generator_expression"
    func_ast: Optional[ast.FunctionDef]
    capture_bindings: Optional[Dict[str, Any]]
    source_object_id: int
    func_name_hint: str
    callee_globals: Optional[Dict[str, Any]] = None
    call_args: Tuple[Any, ...] = ()


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
            bound_args = []
            for arg in self._bound_args:
                visitor._transfer_linear_ownership(
                    arg, reason="instantiate capture", node=node,
                )
                bound_args.append(_capture_value_as_rvalue(visitor, arg))
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

    return _compile_yield_fn_pipeline(
        source.func_ast,
        capture_bindings=source.capture_bindings,
        source_object_id=source.source_object_id,
        func_name_hint=source.func_name_hint,
        callee_globals=source.callee_globals,
        bound_types={
            arg.arg: value.get_pc_type()
            for arg, value in zip(source.func_ast.args.args, source.call_args)
        },
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
    from ..type_descriptor import _constant_value
    for name, vref in zip(arg_names, call_args):
        known, val = _constant_value(vref)
        if not known:
            raise TypeError(
                "instantiate yield arguments must be compile-time constants")
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
        fa.returns = symbol_expression(
            '_instantiate_yield_type',
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
    """Materialize the same definition bindings used by ordinary inline calls."""
    return _compile_closure_fn_pipeline(
        source.func_ast,
        capture_references=source._capture_references,
        call_signature=source._pc_call_signature,
        source_object_id=id(source),
        func_name_hint=source.func_ast.name,
        callee_globals=source.func_globals,
    )


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
    if not values:
        raise TypeError(
            "instantiate cannot determine the element type of an empty "
            "iterable; use an annotated yield function")
    elem_type = _python_constant_pc_type(values[0])
    for v in values:
        vt = _python_constant_pc_type(v)
        if vt is not elem_type:
            raise TypeError(
                f"instantiate: heterogeneous element types: "
                f"{elem_type.__name__} vs {vt.__name__}")

    suffix = ("_c", "const", elem_type, tuple(values))
    _State = _build_state_struct({"_pc": i32}, suffix)
    ip = pc_ptr[_State]
    gv = {"i32": i32, "bool": pc_bool, "ptr": pc_ptr, "_State": _State}
    elem_type_name = getattr(elem_type, "__name__", None)
    if elem_type_name:
        gv[elem_type_name] = elem_type

    nf = _cq(
        _next_tmpl.instantiate(N=len(values)),
        (suffix, "next"), {"s": ip}, pc_bool, gv)
    init_into_fn = _cq(
        _init_tmpl.instantiate(),
        (suffix, "init"), {"s": ip}, None, gv)

    # Build value-case statements via template
    elem_converter = symbol_expression('_instantiate_element_type', elem_type)
    case_stmts = [
        _value_case.instantiate(i=i, converter=elem_converter, v=v, idx='idx').stmts[0]
        for i, v in enumerate(values)
    ]
    vf = _cq(
        _value_tmpl.instantiate(
            cases=case_stmts,
            default_value=_typed_zero(elem_type),
        ),
        (suffix, "value"), {"s": ip}, elem_type, gv)

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
    callee_globals: Dict[str, Any],
) -> None:
    """Bind captured Python objects through collision-free global aliases."""
    if not capture_bindings:
        return
    names = ExpansionNames()
    names.reserve(callee_globals)
    names.reserve_ast(fa)
    pre = []
    for name, value in capture_bindings.items():
        alias = names.fresh(f'__pc_capture_{name}')
        callee_globals[alias] = value
        pre.append(ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Name(id=alias, ctx=ast.Load()),
        ))
    fa.body = pre + fa.body


def _require_yield_type(fa: ast.FunctionDef, type_resolver) -> type:
    """State storage requires a declared yield type before lowering."""
    if fa.returns is None:
        raise TypeError(
            "instantiate requires a declared yield type; "
            "use a yield function with a return annotation")
    return type_resolver.parse_annotation(fa.returns)


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
    callee_globals = callee_globals or {}
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
        if ext_name in callee_globals:
            protect_set.add(ext_name)
            gv[ext_name] = callee_globals[ext_name]

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
        assign = _closure_capture_assign(init_arg_name, s='s').stmts[0]
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
    bound_types: Optional[Dict[str, Any]] = None,
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
        bound_types: Existing type metadata of evaluated call arguments.
        capture_runtime: Optional list of ``(name, pc_type)`` runtime captures
            that must be stored in the state struct and passed to ``init``.
    """
    fa = copy.deepcopy(func_ast)
    runtime_captures = normalize_runtime_captures(capture_runtime)

    callee_globals = dict(callee_globals or {})
    callee_globals.update(capture_bindings or {})
    _splice_capture_bindings(fa, capture_bindings, callee_globals)

    type_resolver = _build_type_resolver(callee_globals)

    yield_type = _require_yield_type(fa, type_resolver)

    compiletime_globals = dict(callee_globals or {})
    if capture_bindings:
        compiletime_globals.update(capture_bindings)

    fa = inline_yield_function_iterators(fa, compiletime_globals)

    captured_vars, local_vars, param_vars = analyze_function_scope(fa)
    # Read bindings must survive across state blocks. Store-only bindings
    # remain ordinary locals; their statements still evaluate normally.
    read_names = {node.id for node in ast.walk(fa)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    locals_set: Set[str] = (local_vars | captured_vars | param_vars) & read_names
    # Runtime captures are also stored on the state struct and must be rewritten
    # to state-field accesses inside the state-machine function.
    locals_set.update(capture.name for capture in runtime_captures)
    # Intrinsics + struct param ('s') should never be rewritten
    protect_set: Set[str] = set(_INTRINSICS) | {'s'}

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
    annotations = _local_annotations(fa)
    capture_types = {capture.name: capture.pc_type for capture in runtime_captures}
    for name in sorted(locals_set - protect_set):
        hint = _annotation_ast_to_type(annotations.get(name), type_resolver)
        if hint is None:
            hint = capture_types.get(name, (bound_types or {}).get(name))
        if hint is None:
            raise TypeError(
                f"instantiate cannot determine the state type of '{name}'; "
                "add a type annotation")
        fields[name] = hint

    sm_ast = lower_yield_state_machine(YieldStateMachineRequest(
        func_ast=fa,
        locals_set=locals_set,
        protect_set=protect_set,
        compiletime_globals=compiletime_globals,
    ))

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
    source_object_id: int,
    func_name_hint: str,
    callee_globals: Optional[Dict[str, Any]] = None,
    capture_references: Optional[Dict[str, Any]] = None,
    call_signature: Optional[Dict[str, Any]] = None,
) -> Any:
    """Compile a closure into a callable first-class object.

    Resolve lexical bindings through the inline binder, then snapshot Python
    values and materialize runtime captures through api.init().

    The compiled API exposes::

        api.State – state struct type (holds runtime capture variables)
        api.init  – init function ``init() -> State``
        api.call  – compiled wrapper function ``call(s, *args) -> ret``
    """
    if _contains_yield(func_ast):
        from ..logger import logger
        logger.error(
            "instantiate: closure sources containing yield are not supported",
            node=func_ast,
            exc_type=TypeError,
        )

    from ..call_normalization import _build_sig, lower_inline_signature

    if call_signature is None:
        func_ast, call_signature = lower_inline_signature(func_ast, {})

    names = ExpansionNames()
    definition = bind_definition(
        func_ast, callee_globals, capture_references, names=names)
    fa = definition.syntax
    plan = snapshot_closure_captures(
        definition.references, definition.reference_names)
    capture_bindings = plan.bindings
    runtime_captures = plan.runtime
    runtime_names = {capture.name for capture in runtime_captures}
    field_aliases = {alias: name
                     for alias, name in definition.reference_names.items()
                     if name in runtime_names}

    # Default values remain optional arguments. Runtime snapshots live on state.
    python_defaults = {}
    default_fields = {}
    for name, vref in call_signature['defaults'].items():
        if vref.is_python_value():
            python_defaults[name] = vref
        else:
            field = names.fresh('_default_' + name)
            runtime_captures.append(RuntimeCapture(field, vref.get_pc_type(), vref))
            default_fields[name] = field

    callee_globals = definition.globals
    callee_globals.update(capture_bindings)
    state_arg = names.fresh('s')
    type_resolver = _build_type_resolver(callee_globals)

    inline_call = (fa.returns is None
                   or any(arg.annotation is None for arg in fa.args.args))
    return_type = (None if inline_call else
                   _annotation_ast_to_type(fa.returns, type_resolver))

    rewriter = StateFieldRewriter(StateFieldRewritePolicy(
        fields=field_aliases,
        protect_names={state_arg},
        state_arg=state_arg,
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

    wrapper_args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=state_arg, annotation=None)] + list(fa.args.args),
        kwonlyargs=[],
        kw_defaults=[],
        defaults=list(fa.args.defaults),
        vararg=fa.args.vararg,
        kwarg=fa.args.kwarg,
    )
    wrapper_ast = _closure_call_template(rewritten_body).stmts[0]
    wrapper_ast.args = wrapper_args
    wrapper_ast.returns = copy.deepcopy(fa.returns)
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

    fields = {capture.name: capture.pc_type for capture in runtime_captures}
    _State = _build_state_struct(fields, suffix)
    ip = pc_ptr[_State]

    param_types = {}
    if not inline_call:
        param_types = {state_arg: ip}
        for arg in fa.args.args:
            param_types[arg.arg] = _annotation_ast_to_type(
                arg.annotation, type_resolver)

    from pythoc import move
    gv = dict(callee_globals)
    gv.update({"_State": _State, "move": move})

    if inline_call:
        from ..ast_visitor.functions import _ClosureWrapper
        cf = _ClosureWrapper(
            wrapper_ast, visitor=None, func_globals=gv,
            defaults={}, capture_references={})
    else:
        cf = _compile_ast_function(
            wrapper_ast,
            param_types=param_types,
            return_type=return_type,
            suffix=sfx,
            gv=gv,
        )

    cf._pc_call_signature = _build_sig(
        [state_arg, *call_signature['param_names']],
        has_varargs=call_signature['has_varargs'],
        has_kwargs=call_signature['has_kwargs'],
        positional_only=call_signature['positional_only'],
        keyword_only=call_signature['keyword_only'],
        defaults=python_defaults,
        default_resolvers={
            **call_signature.get('default_resolvers', {}),
            **{name: _StateFieldDefault(state_arg, field)
               for name, field in default_fields.items()},
        },
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

@dataclass(frozen=True)
class _StateFieldDefault:
    """Read an omitted argument from the supplied materialized state."""

    state_param: str
    field: str

    def __call__(self, visitor, bound, node):
        field_ref = visitor.value_dispatcher.handle_attribute(
            bound[self.state_param], self.field, node)
        return _capture_value_as_rvalue(visitor, field_ref)


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
def _value_case(i, converter, v, *, idx):
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
    def _call(s):
        body


@quote
def _closure_init_template(body):
    def _init_into(s) -> None:
        body


@quote
def _closure_capture_assign(arg_name, *, s):
    s._placeholder = move(arg_name)


# ---------------------------------------------------------------------------
# Type utilities
# ---------------------------------------------------------------------------

def _python_constant_pc_type(value: Any) -> type:
    """Infer the PC type using the compiler's Python-constant promotion rule."""
    from ..type_converter import TypeConverter
    return TypeConverter.infer_default_pc_type_from_python(value)


def _build_type_resolver(callee_globals: Optional[Dict[str, Any]]):
    from ..type_resolver import TypeResolver
    return TypeResolver(user_globals=callee_globals or {})


def _annotation_ast_to_type(node, type_resolver) -> Optional[type]:
    """Resolve a PythoC annotation using the shared type resolver."""
    if node is None:
        return None
    return type_resolver.parse_annotation(node)


def _typed_zero(ty: type) -> ast.expr:
    return ast.Call(
        func=symbol_expression('_instantiate_type', ty),
        args=[ast.Constant(value=0)],
        keywords=[],
    )


def _local_annotations(fa):
    """Collect explicit local types once, including expanded compound bodies."""
    from .._inline.renaming import _parameters
    annotations = {arg.arg: arg.annotation for arg in _parameters(fa.args)
                   if arg.annotation is not None}

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name):
                annotations.setdefault(node.target.id, node.annotation)

    collector = Collector()
    for statement in fa.body:
        collector.visit(statement)
    return annotations


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
    from ..utils.naming import normalize_suffix

    if not frag.stmts or not isinstance(frag.stmts[0], ast.FunctionDef):
        raise TypeError("instantiate: expected function template fragment")
    f = frag.with_func_name(
        frag.stmts[0].name + '_' + normalize_suffix(sfx)).stmts[0]
    return meta_compile_ast(f, param_types=pt, return_type=rt,
                            suffix=sfx, user_globals=gv, source_file=_SOURCE_FILE)
