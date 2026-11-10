"""
Inline / Closure adapter - bridges @inline decorator and direct closure
calls with the meta expansion pipeline.

This is a pure AST adapter -- it only builds AST and registers temporary
variables in the visitor scope. All IR generation happens in the visitor
when it later processes the returned statements.

Two call modes exist:

- ``kind='inline'`` (@inline decorator): free names use its Python definition
  environment; runtime locals of the calling function are not captures.
- ``kind='closure'``: runtime references use the bindings selected when the
  closure was defined, including through later lexical shadows.

The two modes differ only in:

- the naming prefix used for argument / result temporaries,
- the explicit definition-site references supplied by the closure wrapper.
"""

import ast
from typing import Dict, Optional, Any, TYPE_CHECKING
from .exit_rules import ReturnExitRule
from .scope_analyzer import ScopeContext, build_caller_context
from .kernel import ReturnValueChecker
from ..valueref import ValueRef, wrap_value
from ..logger import logger
from ..utils import get_next_id
from .arguments import bind_argument_values
from .names import visitor_expansion_names

if TYPE_CHECKING:
    from ..ast_visitor.visitor_impl import LLVMIRVisitor


_VALID_KINDS = ("inline", "closure")


class InlineAdapter:
    """Adapter for inline expansion (both @inline and closure) using the
    meta expansion pipeline.

    Strategy:

    1. Create temporary variable names for each argument ValueRef.
    2. Register these temporaries in visitor scope (no alloca -- they are
       pure value references).
    3. Invoke ``meta.expand_inline`` to produce AST that references the
       temporaries.
    4. Merge globals, visit the produced statements, restore globals.
    5. For non-void callees, return the result by looking up the generated
       result variable.
    """

    def __init__(self, parent_visitor: 'LLVMIRVisitor',
                 param_bindings: Dict[str, ValueRef],
                 func_globals: Dict[str, Any] = None,
                 kind: str = "inline",
                 capture_references: Dict[str, Any] = None):
        """
        Args:
            parent_visitor: The visitor that's calling the inline function.
            param_bindings: Mapping of parameter name -> evaluated ValueRef.
            func_globals: The callee's ``__globals__`` (for name
                resolution inside the inlined body).
            kind: Either ``'inline'`` (for @inline) or ``'closure'`` (for
                closures). Controls capture-context visibility and temp
                naming.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"InlineAdapter kind must be one of {_VALID_KINDS}, got {kind!r}"
            )
        self.visitor = parent_visitor
        self.param_bindings = param_bindings
        self.func_globals = func_globals
        self.kind = kind
        self.capture_references = capture_references or {}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def execute_inline(self, func_ast: ast.FunctionDef) -> Optional[ValueRef]:
        """Execute callee inline using the universal kernel via meta pipeline.

        Returns the ValueRef of the return value, or a void ValueRef for
        void callees.
        """
        tag = self._tag()
        logger.debug(f"{tag}: executing inline for {func_ast.name}")

        # Determine result variable name using global ID
        unique_id = get_next_id()
        names = visitor_expansion_names(self.visitor)
        names.reserve_ast(func_ast)
        from .definition_bindings import bind_definition
        definition = bind_definition(
            func_ast, self.func_globals, self.capture_references, names=names)
        result_var = names.fresh(f"_{self.kind}_result_{unique_id}")

        # Exit rule (goto-based approach, no flag_var needed).
        exit_rule = ReturnExitRule(result_var=result_var)

        # Register temporaries for arguments so the generated AST can
        # reference them by name.
        arg_temps = self._create_arg_temps()

        # Build AST argument expressions (Name nodes referencing temps).
        arg_exprs = [ast.Name(id=temp_name, ctx=ast.Load())
                     for temp_name in arg_temps.values()]

        # Consumer names protect generated bindings; they never select captures.
        caller_context = self._build_caller_context()

        # Dummy call site -- only used for location copying.
        call_site = ast.Call(
            func=ast.Name(id=func_ast.name, ctx=ast.Load()),
            args=arg_exprs,
            keywords=[],
        )

        from .kernel import (
            expand_inline_or_raise,
            inline_globals_scope,
        )
        inline_result = expand_inline_or_raise(
            callee_ast=definition.syntax,
            callee_globals=definition.globals,
            call_args=arg_exprs,
            call_site=call_site,
            caller_context=caller_context,
            capture_context=ScopeContext(set(definition.references)),
            exit_rule=exit_rule,
            tag=tag,
            names=names,
        )
        inline_result.reference_bindings = definition.references

        with inline_globals_scope(self.visitor, inline_result):
            for i, stmt in enumerate(inline_result.stmts):
                stmt_str = ast.unparse(stmt) if hasattr(ast, 'unparse') else str(stmt)
                logger.debug(f"{tag}: visiting stmt[{i}]: {stmt_str}")
                self.visitor.visit(stmt)

        if self._has_return_value(func_ast):
            return self._lookup_result_var(result_var)

        # Void callee -- return a void ValueRef.
        from ..builtin_entities import void
        return wrap_value(None, kind='python', type_hint=void)

    # Convenience alias: code in the closure path historically calls
    # ``adapter.execute_closure``. Keep one public name so we don't force
    # an immediate migration at every call site.
    execute_closure = execute_inline

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tag(self) -> str:
        """Logger tag reflecting the current mode."""
        return "InlineAdapter" if self.kind == "inline" else "ClosureAdapter"

    def _create_arg_temps(self) -> Dict[str, str]:
        """Create temporary variables for arguments and register in scope.

        Returns a mapping of parameter name -> temp variable name. The
        registered temps have ``alloca=None`` so that ``visit_Name``
        returns the ValueRef directly.

        Linear references retain their CFG owner identity. The expanded
        parameter-binding move transfers that owner exactly once; these
        Python name aliases do not create or discharge ownership.
        """
        names = bind_argument_values(self.visitor, self.param_bindings.values())
        return dict(zip(self.param_bindings, names))

    def _build_caller_context(self) -> ScopeContext:
        """Expose consumer names for declarations and generated-name avoidance."""
        scope_manager = getattr(self.visitor, 'scope_manager', None)
        return build_caller_context(scope_manager, visibility="all_visible")

    def _has_return_value(self, func_ast: ast.FunctionDef) -> bool:
        """True if ``func_ast`` can return a non-void value."""
        if func_ast.returns:
            from .kernel import _get_result_type
            return _get_result_type(func_ast, self.func_globals) is not None
        checker = ReturnValueChecker()
        checker.visit(func_ast)
        return checker.has_return_value

    def _lookup_result_var(self, var_name: str) -> Optional[ValueRef]:
        """Resolve the generated result variable and load its value.

        For linear types, the result temp is treated like the source of
        a ``move()``: ownership is transferred out of the temp and the
        returned ValueRef carries no variable tracking info.
        """
        var_info = self.visitor.scope_manager.lookup_variable(var_name)
        if not var_info:
            logger.warning(f"Result variable {var_name} not found")
            scopes = getattr(self.visitor.scope_manager, 'scopes', None)
            if scopes is not None:
                logger.warning(f"Current scopes: {len(scopes)}")
                for i, scope in enumerate(scopes):
                    logger.warning(f"  Scope {i}: {list(scope.keys())}")
            return None

        # Python-valued results (void callees, compile-time constants):
        # pass the stored value through instead of loading from an alloca.
        if var_info.alloca is None or var_info.value_ref.is_python_value():
            return var_info.value_ref

        alloca = var_info.alloca
        loaded_value = self.visitor.builder.load(alloca)

        if self.visitor._is_linear_type(var_info.type_hint):
            # Mark the result temp as consumed by emitting the transfer
            # event through a throwaway ValueRef that carries tracking
            # info.
            temp_ref = wrap_value(
                loaded_value,
                kind='value',
                type_hint=var_info.type_hint,
                var_name=var_name,
                linear_path=(),
            )
            self.visitor._transfer_linear_ownership(
                temp_ref,
                reason=f"{self.kind} return",
                node=None,
            )

        # Return a fresh ValueRef without var_name -- caller should see
        # it as a pure value (like the return of move()).
        return wrap_value(
            loaded_value,
            kind='value',
            type_hint=var_info.type_hint,
        )
