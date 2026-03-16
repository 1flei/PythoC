"""
Meta-aware inline expansion.

This module implements the meta expansion pipeline for inline scenarios.
It provides MetaInlineRequest as the structured input
and expand_inline() as the AST generation engine.

The expansion uses the quasi-quote layer throughout. Three frame templates
define the complete shape of the inline output, one per exit-rule variant:

  _return_typed_frame  — return with typed result:
      [bindings] + result_var: result_type + with label(name): [body]

  _return_void_frame   — return without result (void/untyped):
      [bindings] + with label(name): [body]

  _passthrough_frame   — yield inlining / no return wrapping:
      [bindings] + [body]

Each param binding is generated via _annotated_binding or _plain_binding
templates. The call site picks the right frame and feeds all params.

The reusable sub-components from pythoc.inline (ScopeAnalyzer,
InlineBodyTransformer, exit rules) are still used for the parts that
are genuinely inline-specific (exit-rule transformation, scope semantics).

NOTE: All imports from pythoc.inline are deferred (lazy) to avoid
circular import issues. pythoc.meta is imported by pythoc.__init__,
and eagerly loading pythoc.inline would shadow the @inline decorator.
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .template import (
    quote_expr, quote_stmt, quote_stmts,
    splice_stmts,
)

if TYPE_CHECKING:
    pass


@dataclass
class MetaInlineRequest:
    """Encapsulates a meta-level inline expansion request.

    This is the structured input to expand_inline(). Adapters build
    this from their specific call-site context.

    Attributes:
        callee_ast: The callee function or lambda AST to inline.
        callee_globals: Merged globals from the callee's definition site.
        call_args: AST expressions for each call argument.
        call_site: The call expression node (for location copying).
        caller_context: Caller's scope context (for capture detection).
        exit_rule: How to transform exit points (return, yield, etc.).
        result_var: Name of the variable to hold the return value.
    """
    callee_ast: Any  # ast.FunctionDef | ast.Lambda
    callee_globals: Dict[str, Any]
    call_args: List[ast.expr]
    call_site: ast.expr
    caller_context: Any  # ScopeContext
    exit_rule: Any  # ExitPointRule
    result_var: Optional[str] = None


# ---------------------------------------------------------------------------
# Quasi-quote templates
#
# These define the *shapes* of generated code declaratively via Python
# syntax. No manual ast.Call / ast.AnnAssign / ast.With construction.
# ---------------------------------------------------------------------------

# -- Per-param binding templates --

# target: annotation = move(value)
@quote_stmt
def _annotated_binding(target, annotation, value):
    target: annotation = __pc_intrinsics.move(value)  # noqa: F821

# target = move(value)
@quote_stmt
def _plain_binding(target, value):
    target = __pc_intrinsics.move(value)  # noqa: F821

# converter(value) — type-convert a constant arg
@quote_expr
def _type_convert(converter, value):
    return converter(value)

# -- Frame templates (one per exit-rule variant) --

# Return with typed result: bindings + result decl + scoped-label body
@quote_stmts
def _return_typed_frame(bindings, result_var, result_type, label_name, body):
    bindings
    result_var: result_type
    with __pc_intrinsics.label(label_name):  # noqa: F821
        body

# Return void/untyped: bindings + scoped-label body (no result decl)
@quote_stmts
def _return_void_frame(bindings, label_name, body):
    bindings
    with __pc_intrinsics.label(label_name):  # noqa: F821
        body

# Passthrough (yield, etc.): bindings + body directly
@quote_stmts
def _passthrough_frame(bindings, body):
    bindings
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


def _build_rename_map(local_vars, param_vars, captured_vars, inline_id):
    """Build rename map: suffix locals + params, keep captured unchanged."""
    rename_map = {}
    suffix = "_{}".format(inline_id)
    for var in local_vars:
        if var not in captured_vars:
            rename_map[var] = "{}{}".format(var, suffix)
    for var in param_vars:
        if var not in captured_vars:
            rename_map[var] = "{}{}".format(var, suffix)
    return rename_map


def _build_param_bindings(callee_params, call_args, rename_map, inline_id):
    """Build parameter binding statements via per-param templates.

    Returns list[ast.stmt].
    """
    stmts = []
    for i, param in enumerate(callee_params):
        param_name = param.arg
        renamed = rename_map.get(param_name, param_name)

        if i < len(call_args):
            arg_value = copy.deepcopy(call_args[i])
        else:
            raise ValueError(
                "Missing argument for parameter '{}' in {}".format(
                    param_name, inline_id))

        # Type-convert constants when param has annotation
        if param.annotation and isinstance(arg_value, ast.Constant):
            arg_value = _type_convert(
                copy.deepcopy(param.annotation), arg_value
            ).as_expr

        target = ast.Name(id=renamed, ctx=ast.Store())

        if param.annotation:
            stmt = _annotated_binding(
                target, copy.deepcopy(param.annotation), arg_value,
            ).as_stmt
        else:
            stmt = _plain_binding(target, arg_value).as_stmt
        stmts.append(stmt)
    return stmts


def _transform_body(callee_body, exit_rule, rename_map):
    """Apply rename + exit-rule transformation to callee body.

    Returns list[ast.stmt].
    """
    from ..inline.transformers import InlineBodyTransformer

    transformer = InlineBodyTransformer(exit_rule, rename_map, flag_var=None)
    return transformer.transform(callee_body)


def _get_result_type(callee_ast):
    """Extract non-void return type from callee, or None."""
    if callee_ast and hasattr(callee_ast, 'returns') and callee_ast.returns:
        ret = callee_ast.returns
        if isinstance(ret, ast.Name) and ret.id == 'void':
            return None
        return ret
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_inline(request):
    """Generate inlined AST from a MetaInlineRequest.

    Picks the appropriate unified frame template based on exit rule
    and return type, then feeds all parameters in one shot::

        _return_typed_frame(bindings, result_var, result_type, label, body)
        _return_void_frame(bindings, label, body)
        _passthrough_frame(bindings, body)

    Args:
        request: A MetaInlineRequest.

    Returns:
        InlineResult with .stmts and .required_globals.
    """
    from ..inline.scope_analyzer import ScopeAnalyzer
    from ..inline.kernel import InlineResult
    from ..inline.exit_rules import ReturnExitRule
    from ..inline._intrinsics import _PC_INTRINSICS
    from ..utils import get_next_id

    callee_ast = request.callee_ast
    inline_id = "inline_{}".format(get_next_id())

    # --- Extract callee body and params ---
    callee_body, callee_params = _extract_callee(callee_ast)

    # --- Scope analysis ---
    analyzer = ScopeAnalyzer(request.caller_context)
    captured_vars, local_vars, param_vars = analyzer.analyze(
        callee_body, callee_params)

    # --- Rename map ---
    rename_map = _build_rename_map(
        local_vars, param_vars, captured_vars, inline_id)

    # --- Build param bindings ---
    binding_stmts = _build_param_bindings(
        callee_params, request.call_args, rename_map, inline_id)

    # --- Set exit label BEFORE body transform so goto_end nodes see it ---
    if isinstance(request.exit_rule, ReturnExitRule):
        exit_label = "_inline_exit_{}".format(inline_id)
        request.exit_rule.exit_label = exit_label

    # --- Transform body (rename + exit-rule) ---
    body_stmts = _transform_body(callee_body, request.exit_rule, rename_map)

    # --- Pick frame template and instantiate in one shot ---
    if isinstance(request.exit_rule, ReturnExitRule):

        result_type = _get_result_type(callee_ast)

        if result_type and request.result_var:
            # Return with typed result
            frame = _return_typed_frame(
                splice_stmts(binding_stmts),
                ast.Name(id=request.result_var, ctx=ast.Store()),
                copy.deepcopy(result_type),
                ast.Constant(value=exit_label),
                splice_stmts(body_stmts),
            )
        else:
            # Return void / untyped
            frame = _return_void_frame(
                splice_stmts(binding_stmts),
                ast.Constant(value=exit_label),
                splice_stmts(body_stmts),
            )
    else:
        # Passthrough (yield, etc.)
        frame = _passthrough_frame(
            splice_stmts(binding_stmts),
            splice_stmts(body_stmts),
        )

    result_stmts = frame.as_stmts

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
