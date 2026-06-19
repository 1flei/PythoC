"""
State-machine lowering for instantiate-backed yield functions.

This module owns the resumable iterator transform.  The caller is responsible
for source normalization, scope analysis, state struct creation, and compiling
the returned ``_next`` function AST.
"""

import ast
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ..meta.template import quote, const as meta_const
from .state_field_rewriter import StateFieldRewriter, StateFieldRewritePolicy

_DONE_LABEL = "_done"


def inline_yield_function_iterators(
    func_ast: ast.FunctionDef,
    compiletime_globals: Optional[Dict[str, Any]] = None,
) -> ast.FunctionDef:
    """Pre-inline for-loops whose iterator is a call to a yield function.

    Generator expressions such as ``(x for x in view_three())`` produce a
    synthetic yield function containing a ``for x in view_three()`` loop.  The
    state-machine lowerer does not handle arbitrary runtime iterators, so we
    flatten those loops here by inlining the callee yield function using the
    same expansion kernel that the AST visitor uses for yield-for-loop
    inlining.

    Constant iterables are expected to have been expanded already by
    ``_expand_compiletime_fors``.  This pass handles the remaining dynamic
    yield-function iterators.
    """
    from .kernel import try_expand_inline
    from .exit_rules import YieldExitRule
    from .scope_analyzer import ScopeContext
    from ..utils import get_next_id

    compiletime_globals = compiletime_globals or {}

    def _is_yield_function_call(expr: ast.expr) -> Optional[Any]:
        if not isinstance(expr, ast.Call):
            return None
        func_obj = _resolve_expr_object(expr.func, compiletime_globals)
        if func_obj is None:
            return None
        if not hasattr(func_obj, "_original_ast"):
            return None
        fa = getattr(func_obj, "_original_ast")
        if not isinstance(fa, ast.FunctionDef):
            return None
        # Quick check: must contain at least one yield and no return-with-value.
        checker = _YieldInlinabilityChecker()
        checker.visit(fa)
        if not checker.has_yield or checker.has_return_value:
            return None
        return func_obj

    def _transform_body(stmts: List[ast.stmt]) -> List[ast.stmt]:
        changed = False
        result: List[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                callee = _is_yield_function_call(stmt.iter)
                if callee is not None:
                    inlined = _inline_for_loop(stmt, callee)
                    if inlined is not None:
                        result.extend(_transform_body(inlined))
                        changed = True
                        continue
                # Not a yield-function iterator: keep the loop but recurse into
                # its body in case nested loops are yield-function iterators.
                stmt.body = _transform_body(stmt.body)
                stmt.orelse = _transform_body(stmt.orelse)
                result.append(stmt)
                continue
            # Recurse into other compound statements.
            new_stmt = _transform_substatements(stmt)
            result.append(new_stmt)
        return result

    def _transform_substatements(stmt: ast.stmt) -> ast.stmt:
        stmt = copy.deepcopy(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt.body = _transform_body(stmt.body)
        elif isinstance(stmt, ast.If):
            stmt.body = _transform_body(stmt.body)
            stmt.orelse = _transform_body(stmt.orelse)
        elif isinstance(stmt, ast.While):
            stmt.body = _transform_body(stmt.body)
            stmt.orelse = _transform_body(stmt.orelse)
        elif isinstance(stmt, ast.For):
            stmt.body = _transform_body(stmt.body)
            stmt.orelse = _transform_body(stmt.orelse)
        elif isinstance(stmt, ast.With):
            stmt.body = _transform_body(stmt.body)
        elif isinstance(stmt, ast.Try):
            stmt.body = _transform_body(stmt.body)
            stmt.orelse = _transform_body(stmt.orelse)
            stmt.finalbody = _transform_body(stmt.finalbody)
            for handler in stmt.handlers:
                handler.body = _transform_body(handler.body)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = _transform_body(case.body)
        return stmt

    def _inline_for_loop(
        for_node: ast.For,
        callee: Any,
    ) -> Optional[List[ast.stmt]]:
        fa = getattr(callee, "_original_ast")
        callee_globals = dict(getattr(callee, "__globals__", None) or {})
        callee_globals.update(compiletime_globals)

        loop_var = for_node.target
        if not isinstance(loop_var, ast.Name):
            # Tuple unpacking not supported yet.
            return None

        loop_body = copy.deepcopy(for_node.body)
        return_type_annotation = (
            copy.deepcopy(fa.returns) if hasattr(fa, "returns") and fa.returns
            else None
        )

        exit_rule = YieldExitRule(
            loop_var=loop_var,
            loop_body=loop_body,
            return_type_annotation=return_type_annotation,
        )

        caller_context = ScopeContext(available_vars=set())
        result = try_expand_inline(
            callee_ast=fa,
            callee_globals=callee_globals,
            call_args=copy.deepcopy(for_node.iter.args),
            call_site=for_node.iter,
            caller_context=caller_context,
            exit_rule=exit_rule,
            tag="instantiate_yield_iter",
        )
        if result is None:
            return None
        return result.stmts

    transformed = copy.deepcopy(func_ast)
    transformed.body = _transform_body(transformed.body)
    return transformed


class _YieldInlinabilityChecker(ast.NodeVisitor):
    """Simple checker for yield function inlinability."""

    def __init__(self):
        self.has_yield = False
        self.has_return_value = False

    def visit_Yield(self, node):
        self.has_yield = True
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value is not None:
            self.has_return_value = True
        self.generic_visit(node)


def _resolve_expr_object(expr: ast.expr, globs: Dict[str, Any]) -> Optional[Any]:
    """Resolve a simple expression to a Python object using globals."""
    if isinstance(expr, ast.Name):
        return globs.get(expr.id)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        obj = globs.get(expr.value.id)
        if obj is None:
            return None
        return getattr(obj, expr.attr, None)
    return None


@dataclass(frozen=True)
class YieldStateMachineRequest:
    """Input to the yield-to-state-machine transform."""

    func_ast: ast.FunctionDef
    locals_set: Set[str]
    protect_set: Set[str]
    compiletime_globals: Optional[Dict[str, Any]] = None
    state_arg: str = "s"


def lower_yield_state_machine(
    request: YieldStateMachineRequest,
) -> ast.FunctionDef:
    """
    Lower a yield function into a flat label/goto state machine.

    The generated labels are top-level siblings, so the dispatch prologue can
    jump to any resume state without violating scoped-label visibility.
    """
    state_arg = request.state_arg
    locals_set = request.locals_set
    protect_set = request.protect_set | {state_arg}

    states: Dict[str, List[ast.stmt]] = {}
    label_order: List[str] = []
    label_pc: Dict[str, int] = {}

    def fresh(prefix: str = "_st") -> str:
        idx = len(label_order)
        name = f"_{prefix}{idx}"
        label_order.append(name)
        label_pc[name] = idx
        return name

    def add_state(name: str, stmts: List[ast.stmt]):
        if name in states:
            raise RuntimeError(f"state {name!r} already exists")
        states[name] = list(stmts) if stmts else [ast.Pass()]

    rewriter = StateFieldRewriter(StateFieldRewritePolicy(
        field_names=locals_set,
        protect_names=protect_set,
        state_arg=state_arg,
        strip_yield_expr=True,
        preserve_nested_functions=False,
    ))

    def rewrite_expr(expr: ast.expr) -> ast.expr:
        return rewriter.visit(expr)

    def rewrite_stmt(stmt: ast.stmt) -> Optional[ast.stmt]:
        return rewriter.visit(stmt)

    def make_yield_state(value_expr: ast.expr, resume_label: str) -> List[ast.stmt]:
        return _yield_state_tmpl.instantiate(
            value_expr=value_expr,
            resume_pc=_i32_const(label_pc[resume_label]),
        ).stmts

    def lower_stmts(stmts: List[ast.stmt], entry_label: str, next_label: str):
        cur_label = entry_label
        cur_stmts: List[ast.stmt] = []

        def flush(term: List[ast.stmt]):
            nonlocal cur_stmts
            add_state(cur_label, cur_stmts + term)
            cur_stmts = []

        i = 0
        while i < len(stmts):
            stmt = stmts[i]

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                value = rewrite_expr(copy.deepcopy(stmt.value.value))
                yield_label = fresh("_yield")
                resume_label = fresh("_resume")

                flush([_goto_stmt(yield_label)])
                add_state(yield_label, make_yield_state(value, resume_label))

                cur_label = resume_label
                i += 1
                continue

            if isinstance(stmt, ast.If):
                test = rewrite_expr(copy.deepcopy(stmt.test))
                then_label = fresh("_then")
                else_label = fresh("_else")
                after_label = fresh("_after")

                flush([
                    ast.If(
                        test=test,
                        body=[_goto_stmt(then_label)],
                        orelse=[_goto_stmt(else_label)],
                    )
                ])

                lower_stmts(stmt.body, then_label, after_label)
                if stmt.orelse:
                    lower_stmts(stmt.orelse, else_label, after_label)
                else:
                    add_state(else_label, [_goto_stmt(after_label)])

                cur_label = after_label
                i += 1
                continue

            if isinstance(stmt, ast.While):
                header_label = fresh("_while_hdr")
                body_label = fresh("_while_body")
                after_label = fresh("_while_after")

                flush([_goto_stmt(header_label)])

                test = rewrite_expr(copy.deepcopy(stmt.test))
                add_state(header_label, [
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=test),
                        body=[_goto_stmt(after_label)],
                        orelse=[_goto_stmt(body_label)],
                    )
                ])

                lower_stmts(stmt.body, body_label, header_label)

                cur_label = after_label
                i += 1
                continue

            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    ret_expr = _i32_const(0)
                else:
                    ret_expr = rewrite_expr(copy.deepcopy(stmt.value))
                flush([ast.Return(value=ret_expr)])
                break

            rewritten = rewrite_stmt(stmt)
            if rewritten is not None:
                cur_stmts.append(rewritten)
            i += 1
        else:
            flush([_goto_stmt(next_label)])

    constexpr = _ConstexprEvaluator(request.compiletime_globals)
    body_stmts = _expand_compiletime_fors(request.func_ast.body, constexpr)

    entry = fresh("_st")
    lower_stmts(body_stmts, entry, _DONE_LABEL)

    body = _dispatch_cases(label_order, state_arg)
    for label_name in label_order:
        body.append(_label_block(label_name, states[label_name]))
    body.append(_label_block(_DONE_LABEL, [_return_false()]))

    fn_ast = _yield_next_template(body).stmts[0]
    fn_ast.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=state_arg)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
    )
    ast.fix_missing_locations(fn_ast)
    return fn_ast


def _expand_compiletime_fors(
    stmts: List[ast.stmt],
    constexpr: "_ConstexprEvaluator",
) -> List[ast.stmt]:
    result: List[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            result.extend(_expand_compiletime_for(stmt, constexpr))
            continue
        result.append(_expand_nested_compiletime_fors(stmt, constexpr))
    return result


def _expand_nested_compiletime_fors(
    stmt: ast.stmt,
    constexpr: "_ConstexprEvaluator",
) -> ast.stmt:
    stmt = copy.deepcopy(stmt)
    if isinstance(stmt, ast.If):
        stmt.body = _expand_compiletime_fors(stmt.body, constexpr)
        stmt.orelse = _expand_compiletime_fors(stmt.orelse, constexpr)
    elif isinstance(stmt, ast.While):
        stmt.body = _expand_compiletime_fors(stmt.body, constexpr)
        stmt.orelse = _expand_compiletime_fors(stmt.orelse, constexpr)
    elif isinstance(stmt, ast.With):
        stmt.body = _expand_compiletime_fors(stmt.body, constexpr)
    elif isinstance(stmt, ast.Try):
        stmt.body = _expand_compiletime_fors(stmt.body, constexpr)
        stmt.orelse = _expand_compiletime_fors(stmt.orelse, constexpr)
        stmt.finalbody = _expand_compiletime_fors(stmt.finalbody, constexpr)
        for handler in stmt.handlers:
            handler.body = _expand_compiletime_fors(handler.body, constexpr)
    elif isinstance(stmt, ast.Match):
        for case in stmt.cases:
            case.body = _expand_compiletime_fors(case.body, constexpr)
    return stmt


def _expand_compiletime_for(
    stmt: ast.For,
    constexpr: "_ConstexprEvaluator",
) -> List[ast.stmt]:
    elements = _constant_iterable_elements(stmt.iter, constexpr)
    if elements is None:
        # Leave dynamic iterators untouched; a later pass may inline
        # yield-function iterators before the state machine lowerer runs.
        stmt.body = _expand_compiletime_fors(stmt.body, constexpr)
        stmt.orelse = _expand_compiletime_fors(stmt.orelse, constexpr)
        return [stmt]

    expanded: List[ast.stmt] = []
    for element in elements:
        assign = ast.Assign(
            targets=[copy.deepcopy(stmt.target)],
            value=_python_value_to_ast(element),
        )
        ast.copy_location(assign, stmt)
        expanded.append(assign)
        expanded.extend(_expand_compiletime_fors(stmt.body, constexpr))
    expanded.extend(_expand_compiletime_fors(stmt.orelse, constexpr))
    return expanded


def _constant_iterable_elements(
    expr: ast.expr,
    constexpr: "_ConstexprEvaluator",
) -> Optional[List[Any]]:
    try:
        value = constexpr.eval(expr)
        return list(value) if hasattr(value, "__iter__") else None
    except Exception:
        return None


class _ConstexprEvaluator:
    def __init__(self, compiletime_globals: Optional[Dict[str, Any]]):
        import builtins

        from ..backend.constexpr_backend import ConstexprBackend
        from ..ast_visitor import LLVMIRVisitor

        user_globals = dict(compiletime_globals or {})
        user_globals.setdefault("__builtins__", builtins)
        backend = ConstexprBackend(user_globals=user_globals)
        self._visitor = LLVMIRVisitor(
            backend=backend,
            user_globals=user_globals,
        )

    def eval(self, expr: ast.expr) -> Any:
        value_ref = self._visitor.visit_expression(copy.deepcopy(expr))
        if not value_ref.is_python_value():
            raise TypeError("expected compile-time Python value")
        return value_ref.get_python_value()


def _python_value_to_ast(value: Any) -> ast.expr:
    if isinstance(value, ast.AST):
        return value
    if isinstance(value, (int, bool, float, str, type(None))):
        return ast.Constant(value=value)
    raise TypeError(
        f"cannot embed {type(value).__name__} as compile-time for value")


def _dispatch_cases(ordered_labels: List[str], state_arg: str) -> List[ast.stmt]:
    match_cases = [
        ast.match_case(
            pattern=ast.MatchValue(value=ast.Constant(value=idx)),
            guard=None,
            body=[_goto_stmt(label_name)],
        )
        for idx, label_name in enumerate(ordered_labels)
    ]
    match_cases.append(ast.match_case(
        pattern=ast.MatchAs(pattern=None, name=None),
        guard=None,
        body=[ast.Pass()],
    ))
    return [ast.Match(
        subject=ast.Attribute(
            value=ast.Name(id=state_arg, ctx=ast.Load()),
            attr="_pc",
            ctx=ast.Load(),
        ),
        cases=match_cases,
    )]


def _i32_const(value: int) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="i32", ctx=ast.Load()),
        args=[ast.Constant(value=value)],
        keywords=[],
    )


@quote
def _goto_tmpl(label_name):
    __pc_intrinsics.goto_begin(label_name)  # noqa: F821


@quote
def _label_block_tmpl(label_name, body):
    with __pc_intrinsics.label(label_name):  # noqa: F821
        body


@quote
def _yield_state_tmpl(value_expr, resume_pc):
    s._yield_value = value_expr
    s._pc = resume_pc
    return bool(1)


@quote
def _yield_next_template(body):
    def _next(s):
        body


def _goto_stmt(label_name: str) -> ast.stmt:
    return _goto_tmpl.instantiate(
        label_name=meta_const(label_name),
    ).stmts[0]


def _label_block(label_name: str, body: List[ast.stmt]) -> ast.stmt:
    return _label_block_tmpl.instantiate(
        label_name=meta_const(label_name),
        body=body,
    ).stmts[0]


def _return_false() -> ast.Return:
    return ast.Return(value=_bool_const(0))


def _bool_const(value: int) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="bool", ctx=ast.Load()),
        args=[ast.Constant(value=value)],
        keywords=[],
    )
