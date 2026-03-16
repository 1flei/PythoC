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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..meta.template import (
    quote_expr, quote_stmt, quote_stmts,
    splice_stmts,
)


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
        caller_context: Caller's scope context (for capture detection).
        exit_rule: How to transform exit points (return, yield, etc.).
    """
    callee_ast: Any  # ast.FunctionDef | ast.Lambda
    callee_globals: Dict[str, Any]
    call_args: List[ast.expr]
    call_site: ast.expr
    caller_context: Any  # ScopeContext
    exit_rule: Any  # ExitPointRule


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

# converter(value) -- type-convert a constant arg
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

# Passthrough (no return wrapping, no loop-var decls)
@quote_stmts
def _passthrough_frame(bindings, body):
    bindings
    body

# Yield with loop-var declarations prepended
@quote_stmts
def _yield_frame(bindings, decls, body):
    bindings
    decls
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
    from .transformers import InlineBodyTransformer

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


def _build_loop_var_declarations(loop_var, return_type_annotation, call_site):
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

    if isinstance(loop_var, ast.Name):
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
            if isinstance(elt, ast.Name):
                element_type = _extract_tuple_element_type(
                    return_type_annotation, i)
                decl = ast.AnnAssign(
                    target=ast.Name(id=elt.id, ctx=ast.Store()),
                    annotation=element_type,
                    value=None,
                    simple=1
                )
                ast.copy_location(decl, call_site)
                decls.append(decl)

    return decls


def _extract_tuple_element_type(type_annotation, index):
    """Extract the type of a tuple element from a struct type annotation.

    For struct[T1, T2, ...], returns T_index.
    If we can't determine the type, returns the full annotation.
    """
    if isinstance(type_annotation, ast.Subscript):
        slice_node = type_annotation.slice
        if isinstance(slice_node, ast.Tuple):
            if index < len(slice_node.elts):
                return copy.deepcopy(slice_node.elts[index])
        elif index == 0:
            return copy.deepcopy(slice_node)
    return copy.deepcopy(type_annotation)


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
    return old


def restore_globals(visitor, old_globals):
    """Restore previously saved user_globals."""
    visitor.ctx.user_globals = old_globals


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
        # Read result_var from the single source of truth: the exit rule
        result_var = request.exit_rule.result_var
        result_type = _get_result_type(callee_ast)

        if result_type and result_var:
            # Return with typed result
            frame = _return_typed_frame(
                splice_stmts(binding_stmts),
                ast.Name(id=result_var, ctx=ast.Store()),
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
    elif isinstance(request.exit_rule, YieldExitRule):
        # Yield: build loop-var declarations if type annotation exists
        exit_rule = request.exit_rule
        decl_stmts = []
        if exit_rule.return_type_annotation:
            decl_stmts = _build_loop_var_declarations(
                exit_rule.loop_var,
                exit_rule.return_type_annotation,
                request.call_site,
            )

        if decl_stmts:
            frame = _yield_frame(
                splice_stmts(binding_stmts),
                splice_stmts(decl_stmts),
                splice_stmts(body_stmts),
            )
        else:
            frame = _passthrough_frame(
                splice_stmts(binding_stmts),
                splice_stmts(body_stmts),
            )
    else:
        # Passthrough (other exit rules)
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
