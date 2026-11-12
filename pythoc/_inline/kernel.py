"""
Inline expansion kernel

Contains:
- InlineResult: output container for inline expansion
- MetaInlineRequest: structured input for expand_inline()
- expand_inline(): the AST generation engine
- merge_inline_globals / restore_globals: shared globals helpers

The expansion uses the quasi-quote layer throughout. Frame templates
define the complete shape of the inline output, one per exit-rule variant:

  _return_typed_frame  -- return with typed result:
      [bindings] + result_var: result_type + with label(name): [body]

  _return_void_frame   -- return without result (void/untyped):
      [bindings] + with label(name): [body]

  _passthrough_frame   -- yield inlining / no return wrapping:
      [bindings] + [body]

  _yield_frame         -- yield with loop-var declarations:
      [bindings] + [decls] + [body]

Each param binding is generated via _annotated_binding or _plain_binding
templates. The call site picks the right frame and feeds all params.
"""

import ast
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..meta.template import quote, const as meta_const
from .names import ExpansionNames
from .scope_analyzer import ScopeContext


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InlineResult:
    """
    Result of inline execution

    Contains:
    - stmts: Generated AST statements
    - required_globals: Globals that must be merged into caller's user_globals
                        before visiting the statements
    """
    stmts: List[ast.stmt]
    required_globals: Dict[str, Any]
    reference_bindings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaInlineRequest:
    """Encapsulates an inline expansion request.

    This is the structured input to expand_inline(). Adapters build
    this from their specific call-site context.

    Attributes:
        callee_ast: The callee function or lambda AST to inline.
        callee_globals: Merged globals from the callee's definition site.
        call_args: AST expressions for each call argument.
        call_site: The call expression node (for location copying).
        caller_context: Consumer bindings, including existing loop targets.
        capture_context: Lexical bindings available to the callee.
        exit_rule: How to transform exit points (return, yield, etc.).
    """
    callee_ast: Any  # ast.FunctionDef | ast.Lambda
    callee_globals: Dict[str, Any]
    call_args: List[ast.expr]
    call_site: ast.expr
    caller_context: Any  # ScopeContext
    exit_rule: Any  # ExitPointRule
    names: Optional[ExpansionNames] = None
    capture_context: ScopeContext = field(default_factory=ScopeContext.empty)


# ---------------------------------------------------------------------------
# Quasi-quote templates
#
# These define the *shapes* of generated code declaratively via Python
# syntax. No manual ast.Call / ast.AnnAssign / ast.With construction.
# ---------------------------------------------------------------------------

# -- Per-param binding templates --

# Unified: has_annotation controls whether annotation is emitted.
# _fold_const_if eliminates the dead branch at instantiation.
@quote
def _param_binding(has_annotation, target, annotation, value):
    if has_annotation:
        target: annotation = __pc_intrinsics.move(value)  # noqa: F821
    if not has_annotation:
        target = __pc_intrinsics.move(value)  # noqa: F821


class _InlineReturnValue:
    """Carry one evaluated value across inline scope cleanup."""

    _pc_compile_level_callable = True

    @staticmethod
    def create():
        return _InlineReturnValue()

    def handle_call(self, visitor, func_ref, args, node):
        from ..valueref import wrap_python_constant
        self.value = args[0].clone(
            var_name=None, linear_path=None, linear_owner_id=None)
        return wrap_python_constant(None)

    def take(self):
        value = self.value
        del self.value
        return value


@quote
def _terminal_return(label_name, body, has_value, make_capture, capture,
                     value, result_var):
    if has_value:
        capture = make_capture()
    with __pc_intrinsics.label(label_name):  # noqa: F821
        body
        if has_value:
            capture(__pc_intrinsics.move(value))  # noqa: F821
    if has_value:
        result_var = __pc_intrinsics.move(capture.take())  # noqa: F821


# -- Frame templates --
# Unified: all branches encoded as const guards.
# _fold_const_if eliminates dead branches at instantiation.
@quote
def _inline_frame(has_label, has_result_decl, has_decls,
                  bindings, result_var, result_type, label_name, decls, body):
    bindings
    if has_result_decl:
        result_var: result_type
    if has_decls:
        decls
    if has_label:
        with __pc_intrinsics.label(label_name):  # noqa: F821
            body
    if not has_label:
        body


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_callee(callee_ast):
    """Deep-copy callee body and params from a FunctionDef or Lambda.

    Returns (body: list[ast.stmt], params: list[ast.arg]).
    """
    if isinstance(callee_ast, ast.FunctionDef):
        return copy.deepcopy(callee_ast.body), copy.deepcopy(callee_ast.args.args)
    elif isinstance(callee_ast, ast.Lambda):
        return [ast.Return(value=copy.deepcopy(callee_ast.body))], \
               copy.deepcopy(callee_ast.args.args)
    else:
        raise TypeError(
            "Expected FunctionDef or Lambda, got {}".format(
                type(callee_ast).__name__))


def _count_returns(body):
    """Number of Return statements in body.

    Returns inside nested function/lambda definitions do not count.
    """
    class _ReturnCounter(ast.NodeVisitor):
        count = 0

        def visit_Return(self, node):
            self.count += 1

        def visit_FunctionDef(self, node):
            pass

        def visit_AsyncFunctionDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass

    counter = _ReturnCounter()
    for stmt in body:
        counter.visit(stmt)
    return counter.count


def _build_rename_map(local_vars, param_vars, captured_vars, inline_id, names):
    """Build rename map: suffix locals + params, keep captured unchanged."""
    rename_map = {}
    suffix = "_{}".format(inline_id)
    for var in local_vars:
        if var not in captured_vars:
            rename_map[var] = names.fresh("{}{}".format(var, suffix))
    for var in param_vars:
        if var not in captured_vars:
            rename_map[var] = names.fresh("{}{}".format(var, suffix))
    return rename_map


def _build_param_bindings(callee_params, call_args, rename_map):
    """Build parameter binding statements via unified template.

    Each param is a single _param_binding call with has_annotation as
    const guard. Returns list of Fragments (auto-spliced by consumers).
    """
    if len(callee_params) != len(call_args):
        raise TypeError(
            f'AST expansion argument count mismatch: expected {len(callee_params)}, '
            f'got {len(call_args)}')
    from .renaming import ScopedVariableRenamer
    frags = []
    for param, argument in zip(callee_params, call_args):
        param_name = param.arg
        renamed = rename_map.get(param_name, param_name)
        arg_value = copy.deepcopy(argument)

        target = ast.Name(id=renamed, ctx=ast.Store())
        has_annotation = bool(param.annotation)

        frags.append(_param_binding(
            meta_const(has_annotation),
            target,
            ScopedVariableRenamer(rename_map).visit(copy.deepcopy(param.annotation))
            if has_annotation else ast.Constant(value=0),
            arg_value,
        ))
    return frags


def _transform_body(callee_body, exit_rule, rename_map):
    """Apply rename + exit-rule transformation to callee body.

    Returns list[ast.stmt].
    """
    from .transformers import InlineBodyTransformer

    transformer = InlineBodyTransformer(exit_rule, rename_map, flag_var=None)
    return transformer.transform(callee_body)


def _result_storage_type(annotation):
    """Represent a void inline result as a compile-time None value."""
    from ..builtin_entities import pyconst, void
    from ..type_resolver import TypeResolver
    resolved = (void if annotation is None else
                TypeResolver.resolve_type(annotation, freeze=True))
    return pyconst[None] if resolved is void else resolved


def _tuple_element_type(annotation, index):
    from ..schema_protocol import get_schema_field_type
    from ..type_resolver import TypeResolver
    from ..type_converter import get_base_type
    resolved = TypeResolver.resolve_type(annotation, freeze=True)
    return get_schema_field_type(get_base_type(resolved), index)


def _type_expression(function, *arguments):
    from .definition_bindings import symbol_expression
    return ast.Call(func=symbol_expression(function.__name__, function),
                    args=list(arguments), keywords=[])


def _build_loop_var_declarations(
    loop_var,
    return_type_annotation,
    call_site,
    visible_names=frozenset(),
):
    """Build loop-variable declaration AST nodes for yield inlining.

    For simple Name: creates single AnnAssign.
    For Tuple: creates declarations for each element using subscript types.

    Args:
        loop_var: Loop variable target (Name or Tuple).
        return_type_annotation: Return type annotation from the yield function.
        call_site: AST node for copy_location.

    Returns:
        List of AnnAssign statements.
    """
    decls = []

    if isinstance(loop_var, ast.Name) and loop_var.id not in visible_names:
        decl = ast.AnnAssign(
            target=ast.Name(id=loop_var.id, ctx=ast.Store()),
            annotation=copy.deepcopy(return_type_annotation),
            value=None,
            simple=1
        )
        ast.copy_location(decl, call_site)
        decls.append(decl)
    elif isinstance(loop_var, ast.Tuple):
        for i, elt in enumerate(loop_var.elts):
            if isinstance(elt, ast.Name) and elt.id not in visible_names:
                element_type = _type_expression(
                    _tuple_element_type, copy.deepcopy(return_type_annotation),
                    ast.Constant(i))
                decl = ast.AnnAssign(
                    target=ast.Name(id=elt.id, ctx=ast.Store()),
                    annotation=element_type,
                    value=None,
                    simple=1
                )
                ast.copy_location(decl, call_site)
                decls.append(decl)

    return decls


# ---------------------------------------------------------------------------
# Shared globals helpers
# ---------------------------------------------------------------------------

def merge_inline_globals(visitor, inline_result):
    """Merge required_globals into visitor's user_globals.

    Returns old_globals for later restore via restore_globals().
    """
    old = visitor.ctx.user_globals
    merged = dict(old or {})
    merged.update(inline_result.required_globals)
    visitor.ctx.user_globals = merged
    visitor._python_global_cache.clear()
    return old


def restore_globals(visitor, old_globals):
    """Restore previously saved user_globals."""
    visitor.ctx.user_globals = old_globals
    visitor._python_global_cache.clear()


@contextmanager
def inline_globals_scope(visitor, inline_result):
    """Merge inline globals, run the wrapped block, then restore them.

    This is the shared lifecycle for visiting statements produced by
    ``expand_inline``.  Missing source locations are filled in before the
    block runs; adapters and visitors should put the actual
    statement-visiting logic inside the ``with`` block.

    Example::

        with inline_globals_scope(visitor, inline_result):
            for stmt in inline_result.stmts:
                visitor.visit(stmt)
    """
    from .definition_bindings import materialize_definition_symbols
    from .names import visitor_expansion_names
    syntax = ast.Module(body=inline_result.stmts, type_ignores=[])
    syntax, symbols = materialize_definition_symbols(
        syntax, inline_result.required_globals,
        names=visitor_expansion_names(visitor))
    inline_result.stmts = syntax.body
    inline_result.required_globals.update(symbols)
    for stmt in inline_result.stmts:
        ast.fix_missing_locations(stmt)

    old_user_globals = merge_inline_globals(visitor, inline_result)
    try:
        with visitor.scope_manager.binding_aliases(inline_result.reference_bindings):
            yield
    finally:
        restore_globals(visitor, old_user_globals)


def _build_inline_request(
    *,
    callee_ast,
    callee_globals,
    call_args,
    call_site,
    caller_context,
    exit_rule,
    names=None,
    capture_context=None,
) -> MetaInlineRequest:
    """Build a ``MetaInlineRequest`` from the standard adapter inputs."""
    return MetaInlineRequest(
        callee_ast=callee_ast,
        callee_globals=callee_globals or {},
        call_args=call_args,
        call_site=call_site,
        caller_context=caller_context,
        exit_rule=exit_rule,
        names=names,
        capture_context=(capture_context if capture_context is not None
                         else ScopeContext.empty()),
    )


def expand_inline_or_raise(
    *,
    callee_ast,
    callee_globals,
    call_args,
    call_site,
    caller_context,
    exit_rule,
    tag: str = "InlineAdapter",
    names=None,
    capture_context=None,
) -> InlineResult:
    """Build a request, expand it, and raise on failure.

    This is the shared expansion path for adapters that cannot fall back
    (``@inline`` / closure inlining).
    """
    request = _build_inline_request(
        callee_ast=callee_ast,
        callee_globals=callee_globals,
        call_args=call_args,
        call_site=call_site,
        caller_context=caller_context,
        exit_rule=exit_rule,
        names=names,
        capture_context=capture_context,
    )
    try:
        return expand_inline(request)
    except Exception as e:
        from ..logger import logger
        logger.error(f"{tag}: meta inline expansion failed: {e}")
        raise


def try_expand_inline(
    *,
    callee_ast,
    callee_globals,
    call_args,
    call_site,
    caller_context,
    exit_rule,
    tag: str = "YieldInlineAdapter",
    names=None,
    capture_context=None,
) -> Optional[InlineResult]:
    """Build a request, expand it, and return ``None`` on failure.

    This is the shared expansion path for adapters that want to attempt
    inlining but allow the caller to fall back (yield-for-loop inlining).
    """
    request = _build_inline_request(
        callee_ast=callee_ast,
        callee_globals=callee_globals,
        call_args=call_args,
        call_site=call_site,
        caller_context=caller_context,
        exit_rule=exit_rule,
        names=names,
        capture_context=capture_context,
    )
    try:
        return expand_inline(request)
    except Exception as e:
        from ..logger import logger
        logger.debug(f"{tag}: meta inline expansion rejected: {e}")
        return None


# ---------------------------------------------------------------------------
# AST checkers shared by adapters
# ---------------------------------------------------------------------------

class ReturnValueChecker(ast.NodeVisitor):
    """Detect whether a function AST contains a ``return <value>`` statement."""

    def __init__(self):
        self.has_return_value = False

    def visit_Return(self, node: ast.Return):
        if node.value is not None:
            self.has_return_value = True
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_inline(request):
    """Generate inlined AST from a MetaInlineRequest.

    Picks the appropriate unified frame template based on exit rule
    and return type, then feeds all parameters in one shot::

        _return_typed_frame(bindings, result_var, result_type, label, body)
        _return_void_frame(bindings, label, body)
        _yield_frame(bindings, decls, body)
        _passthrough_frame(bindings, body)

    Args:
        request: A MetaInlineRequest.

    Returns:
        InlineResult with .stmts and .required_globals.
    """
    from .scope_analyzer import ScopeAnalyzer
    from .exit_rules import ReturnExitRule, YieldExitRule
    from ._intrinsics import _PC_INTRINSICS
    from ..utils import get_next_id

    callee_ast = request.callee_ast
    inline_id = "inline_{}".format(get_next_id())
    names = request.names if request.names is not None else ExpansionNames()
    names.reserve(request.caller_context.available_vars)
    names.reserve(request.callee_globals or {})
    names.reserve_ast(callee_ast, request.call_site, *request.call_args)
    result_var = getattr(request.exit_rule, 'result_var', None)
    if result_var is not None:
        names.reserve([result_var])
    if isinstance(request.exit_rule, YieldExitRule):
        names.reserve_ast(request.exit_rule.loop_var, *request.exit_rule.loop_body)
        request.exit_rule.names = names

    # --- Extract callee body and params ---
    callee_body, callee_params = _extract_callee(callee_ast)

    # --- Scope analysis ---
    analyzer = ScopeAnalyzer(request.capture_context)
    captured_vars, local_vars, param_vars = analyzer.analyze(
        callee_body, callee_params)

    # --- Rename map ---
    rename_map = _build_rename_map(
        local_vars, param_vars, captured_vars, inline_id, names)

    # --- Build param bindings (list[Fragment]) ---
    binding_frags = _build_param_bindings(
        callee_params, request.call_args, rename_map)

    # --- Set exit label BEFORE body transform so goto_end nodes see it ---
    exit_label = None
    if isinstance(request.exit_rule, ReturnExitRule):
        exit_label = names.fresh("_inline_exit_{}".format(inline_id))
        request.exit_rule.exit_label = exit_label

    # Unannotated terminal returns retain the evaluated value across cleanup.
    # Multiple return points require an explicit type for their shared slot.
    result_var = getattr(request.exit_rule, 'result_var', None)
    terminal_value = None
    has_terminal_return = False
    prefix_label = exit_label
    if (isinstance(request.exit_rule, ReturnExitRule)
            and result_var
            and getattr(callee_ast, 'returns', None) is None):
        n_returns = _count_returns(callee_body)
        if n_returns == 1 and isinstance(callee_body[-1], ast.Return):
            terminal_value = callee_body[-1].value
            has_terminal_return = True
            callee_body = callee_body[:-1]
            exit_label = None
            request.exit_rule.exit_label = None
        elif n_returns > 0:
            raise RuntimeError(
                "Cannot inline an unannotated callee with multiple return "
                "points; add a return type annotation"
            )

    # --- Transform body (rename + exit-rule) ---
    body_stmts = _transform_body(callee_body, request.exit_rule, rename_map)

    if has_terminal_return:
        from .exit_rules import VariableRenamer
        from .definition_bindings import symbol_expression
        renamed_value = (VariableRenamer(rename_map).visit(
            copy.deepcopy(terminal_value))
            if terminal_value is not None else ast.Constant(None))
        body_stmts = _terminal_return(
            ast.Constant(prefix_label), body_stmts,
            meta_const(terminal_value is not None),
            symbol_expression('_inline_return_factory', _InlineReturnValue.create),
            ast.Name(id=names.fresh('_inline_return_value'), ctx=ast.Load()),
            renamed_value, ast.Name(id=result_var, ctx=ast.Store()),
        ).stmts

    # Resolve type expressions through normal annotation evaluation.
    return_annotation = getattr(callee_ast, 'returns', None)
    result_type = None
    if return_annotation is not None:
        from .exit_rules import VariableRenamer
        annotation = VariableRenamer(rename_map).visit(
            copy.deepcopy(return_annotation))
        result_type = _type_expression(_result_storage_type, annotation)

    has_label = exit_label is not None
    has_result_decl = result_type is not None and result_var is not None
    decl_stmts = []
    if isinstance(request.exit_rule, YieldExitRule) and result_type is not None:
        decl_stmts = _build_loop_var_declarations(
            request.exit_rule.loop_var, result_type, request.call_site,
            request.caller_context.available_vars)
    has_decls = bool(decl_stmts)

    # --- Single unified frame template call ---
    frame = _inline_frame(
        meta_const(has_label),
        meta_const(has_result_decl),
        meta_const(has_decls),
        binding_frags,
        ast.Name(id=result_var, ctx=ast.Store()) if has_result_decl else "_unused",
        copy.deepcopy(result_type) if has_result_decl else ast.Constant(value=0),
        ast.Constant(value=exit_label or "_unused"),
        decl_stmts,
        body_stmts,
    )

    result_stmts = frame.stmts

    # --- Fix locations from call site ---
    for stmt in result_stmts:
        ast.copy_location(stmt, request.call_site)
        ast.fix_missing_locations(stmt)

    # --- Debug ---
    from ..utils.ast_debug import ast_debugger
    func_name = callee_ast.name if hasattr(callee_ast, 'name') else 'lambda'
    ast_debugger.capture(
        "after_inline", result_stmts,
        func_name=func_name, inline_id=inline_id,
        param_count=len(callee_params), local_count=len(local_vars))

    # --- Required globals ---
    required_globals = {}
    if request.callee_globals:
        required_globals.update(request.callee_globals)
    required_globals['__pc_intrinsics'] = _PC_INTRINSICS

    return InlineResult(stmts=result_stmts, required_globals=required_globals)
