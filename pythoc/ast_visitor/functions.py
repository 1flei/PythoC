"""
Functions mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
)
from ..builtin_entities import bool as pc_bool
from ..builder import LLVMBuilder
from ..logger import logger


class _ClosureWrapper:
    """Callable protocol object for a compile-level closure.

    Calling the closure means compile-time inline expansion of its body
    (via ClosureAdapter). Copyable defaults are pre-evaluated ValueRefs
    captured eagerly at the closure definition site.

    A closure whose body contains yield is a for-loop producer, not a
    plain call: handle_call returns a ValueRef carrying
    ``_yield_inline_info`` so the yield-inline machinery splices the
    consumer's loop body into the closure body. Captures work by
    reference exactly as for plain closures - which makes a closure
    handler able to write a frame-local channel (e.g. an error slot)
    with zero runtime cost.

    Marked as a compile-level callable so misuse in runtime positions
    produces the uniform boundary error.
    """

    _pc_compile_level_callable = True

    # NOTE: defer_linear_transfer is deliberately NOT a class attribute.
    # _ClosureWrapper also serves PLAIN (non-yield) closures, whose call
    # sites must keep transferring linear argument ownership. Only yield
    # closures go through the for-loop inline path where the param-binding
    # template moves each argument exactly once (see yield_transform
    # placeholder), so the flag is set per-instance in __init__ below.

    def __init__(self, func_ast, visitor, func_globals, capture_bindings,
                 capture_runtime, defaults):
        from ..call_normalization import lower_inline_signature

        self.func_ast, self._pc_call_signature = lower_inline_signature(
            func_ast, defaults)
        self.name = func_ast.name
        self.visitor = visitor
        self.func_globals = func_globals
        self._capture_bindings = capture_bindings
        self._capture_runtime = capture_runtime
        self._has_yield = _contains_yield(func_ast)
        # Yield closures are inlined by the for-loop visitor; the param-
        # binding template transfers argument ownership via move() exactly
        # once. The call site must not transfer as well. Plain closures
        # keep the default (False): their call sites own the transfer.
        self.defer_linear_transfer = self._has_yield

    def handle_call(self, visitor, func_ref, args, call_node):
        """Execute closure inline, or hand it to the yield-inline path."""
        from ..call_normalization import NormalizedCallArgs, bind_call_args

        if not isinstance(args, NormalizedCallArgs):
            args = bind_call_args(
                self._pc_call_signature, args, [],
                visitor.visit_rvalue_expression, call_node)
        if self._has_yield:
            from ..builtin_entities.python_type import PythonType
            result = wrap_value(self, kind='python', type_hint=PythonType(self))
            result._yield_inline_info = {
                'func_obj': None,
                'callee_globals': self.func_globals,
                'placeholder': self,
                'original_ast': self.func_ast,
                'call_node': call_node,
                'evaluated_args': True,
                'call_args': args,
                # Closures capture by reference from ALL enclosing scopes,
                # unlike module-level yield functions (whose free names
                # resolve via callee globals). The yield-inline path must
                # use the matching caller visibility.
                'is_closure': True,
            }
            return result

        from .._inline import ClosureAdapter

        param_bindings = dict(zip(
            self._pc_call_signature['param_names'], args))

        # Use ClosureAdapter to inline the closure with captured globals
        adapter = ClosureAdapter(visitor, param_bindings, func_globals=self.func_globals)
        return adapter.execute_closure(self.func_ast)


def _contains_yield(func_ast: ast.FunctionDef) -> bool:
    """True if the function body contains yield directly (not in nested defs)."""
    class _YieldFinder(ast.NodeVisitor):
        found = False

        def visit_Yield(self, node):
            self.found = True

        def visit_FunctionDef(self, node):
            pass

        def visit_AsyncFunctionDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass

    finder = _YieldFinder()
    for stmt in func_ast.body:
        finder.visit(stmt)
    return finder.found


class FunctionsMixin:
    """Mixin containing functions-related visitor methods"""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definition in AST traversal

        If encountered within a function body (nested), treat as closure.
        Otherwise, treat as top-level function definition.
        """
        if node.decorator_list:
            logger.error(
                f"Decorators are not supported on nested function '{node.name}'; "
                f"nested functions are compile-level closures",
                node=node, exc_type=TypeError
            )
        # This is a closure - register it as a callable
        self._register_closure(node)
        return None

    def visit_Lambda(self, node: ast.Lambda):
        """Lower a lambda to the closure machinery.

        lambda a, b: EXPR  ===  def _anon(a, b): return EXPR

        The synthesized definition goes through the same ClosureAdapter /
        inline kernel as nested def closures, so both forms share one
        semantics. Default arguments follow Python semantics: they are
        evaluated eagerly at the lambda definition site (snapshot as
        rvalues), which is the eager-capture idiom `lambda x=x: ...`.
        """
        args_node = node.args
        if args_node.vararg is not None or args_node.kwarg is not None:
            logger.error(
                "lambda with *args or **kwargs is not supported",
                node=node, exc_type=TypeError
            )
        if args_node.kwonlyargs:
            logger.error(
                "lambda with keyword-only parameters is not supported",
                node=node, exc_type=TypeError
            )

        param_names = [a.arg for a in args_node.posonlyargs + args_node.args]
        n_required = len(param_names) - len(args_node.defaults)

        # Defaults are reusable copyable snapshots of definition-time values.
        default_vrefs = self._evaluate_closure_defaults(args_node.defaults)

        func_def = ast.FunctionDef(
            name=f"_lambda_{node.lineno}_{node.col_offset}",
            args=ast.arguments(
                posonlyargs=args_node.posonlyargs,
                args=args_node.args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[ast.Return(value=node.body)],
            decorator_list=[],
        )
        func_def = ast.fix_missing_locations(ast.copy_location(func_def, node))

        wrapper = self._make_closure_wrapper(
            func_def, dict(zip(param_names[n_required:], default_vrefs)))
        return wrap_value(wrapper, kind='python', type_hint=wrapper)

    def _register_closure(self, node: ast.FunctionDef):
        """Register a closure function for inline execution

        Creates a handle_call wrapper that uses ClosureAdapter to inline
        the closure body when called.
        """
        from ..valueref import ValueRef, wrap_value
        from ..registry import VariableInfo

        positional = node.args.posonlyargs + node.args.args
        default_start = len(positional) - len(node.args.defaults)
        default_names = [arg.arg for arg in positional[default_start:]]
        expressions = list(node.args.defaults)
        keyword_defaults = zip(node.args.kwonlyargs, node.args.kw_defaults)
        for arg, expression in keyword_defaults:
            if expression is not None:
                default_names.append(arg.arg)
                expressions.append(expression)
        defaults = self._evaluate_closure_defaults(expressions)
        wrapper = self._make_closure_wrapper(
            node, dict(zip(default_names, defaults)))

        # Register as a variable in current scope
        var_info = VariableInfo(
            name=node.name,
            value_ref=wrap_value(
                wrapper,
                kind='python',
                type_hint=wrapper,
            ),
            alloca=None,
            source='closure',
            is_mutable=False,
        )

        self.scope_manager.declare_variable(var_info, allow_shadow=True)

    def _evaluate_closure_defaults(self, expressions):
        from ..literal_protocol import iter_value_ref_leaves

        defaults = []
        for expression in expressions:
            value = self.visit_rvalue_expression(expression)
            if any(self._is_linear_type(leaf.type_hint)
                   for leaf in iter_value_ref_leaves(value)):
                logger.error(
                    'linear default arguments are not supported; '
                    'pass owned values explicitly',
                    node=expression, exc_type=TypeError)
            if value.is_python_value():
                defaults.append(value)
            else:
                defaults.append(wrap_value(
                    value.value, kind='value', type_hint=get_type_hint(value)))
        return defaults

    def _make_closure_wrapper(self, node: ast.FunctionDef, defaults):
        """Build the callable wrapper for a closure/lambda AST."""
        from .._inline.scope_analyzer import analyze_function_scope, build_caller_context
        from .._inline.closure_capture import build_closure_capture_plan

        # Capture the current user_globals at closure definition time
        # This is the caller's globals context
        closure_globals = self.ctx.user_globals

        # Analyze captured variables for this closure
        caller_context = build_caller_context(
            self.scope_manager, visibility="all_visible"
        )
        captured_vars, _, _ = analyze_function_scope(
            node, caller_context=caller_context
        )

        visible = self.scope_manager.get_all_visible()
        nested_refs = {
            name_node.id
            for name_node in ast.walk(node)
            if isinstance(name_node, ast.Name) and isinstance(name_node.ctx, ast.Load)
        }
        capture_plan = build_closure_capture_plan(
            captured_vars | nested_refs,
            visible,
        )

        return _ClosureWrapper(
            node,
            self,
            closure_globals,
            capture_plan.bindings,
            capture_plan.runtime,
            defaults,
        )
