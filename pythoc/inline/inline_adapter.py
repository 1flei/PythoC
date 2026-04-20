"""
Inline / Closure adapter - bridges @inline decorator and direct closure
calls with the meta expansion pipeline.

This is a pure AST adapter -- it only builds AST and registers temporary
variables in the visitor scope. All IR generation happens in the visitor
when it later processes the returned statements.

Two call modes exist:

- ``kind='inline'`` (@inline decorator): the callee's free variables are
  looked up against the *current* visitor scope only. The @inline path
  does not capture outer-scope variables.
- ``kind='closure'``: a closure defined inside another compiled function
  must be able to capture variables from *all visible* enclosing scopes.

The two modes differ only in:

- the naming prefix used for argument / result temporaries,
- which scope query is used when building ``ScopeContext`` for capture
  detection.
"""

import ast
from typing import Dict, Optional, Any, TYPE_CHECKING
from .exit_rules import ReturnExitRule
from .scope_analyzer import ScopeContext
from ..valueref import ValueRef, wrap_value
from ..context import VariableInfo
from ..logger import logger
from ..utils import get_next_id

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
                 param_bindings: Dict[str, Any],
                 func_globals: Dict[str, Any] = None,
                 kind: str = "inline"):
        """
        Args:
            parent_visitor: The visitor that's calling the inline function.
            param_bindings: Mapping of parameter name -> ValueRef / Python
                object.
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
        result_var = f"_{self.kind}_result_{unique_id}"

        # Exit rule (goto-based approach, no flag_var needed).
        exit_rule = ReturnExitRule(result_var=result_var)

        # Register temporaries for arguments so the generated AST can
        # reference them by name.
        arg_temps = self._create_arg_temps()

        # Build AST argument expressions (Name nodes referencing temps).
        arg_exprs = [ast.Name(id=temp_name, ctx=ast.Load())
                     for temp_name in arg_temps.values()]

        # Caller context drives capture detection inside expand_inline.
        caller_context = self._build_caller_context()

        # Dummy call site -- only used for location copying.
        call_site = ast.Call(
            func=ast.Name(id=func_ast.name, ctx=ast.Load()),
            args=arg_exprs,
            keywords=[],
        )

        from .kernel import (
            MetaInlineRequest, expand_inline,
            merge_inline_globals, restore_globals,
        )
        request = MetaInlineRequest(
            callee_ast=func_ast,
            callee_globals=self.func_globals or {},
            call_args=arg_exprs,
            call_site=call_site,
            caller_context=caller_context,
            exit_rule=exit_rule,
        )

        try:
            inline_result = expand_inline(request)
        except Exception as e:
            logger.error(f"{tag}: meta inline expansion failed: {e}")
            raise

        old_user_globals = merge_inline_globals(self.visitor, inline_result)

        for stmt in inline_result.stmts:
            ast.fix_missing_locations(stmt)

        # Debug hook - accumulate inlined statements at function level.
        # Useful for both inline and closure expansion.
        if not hasattr(self.visitor, '_all_inlined_stmts'):
            self.visitor._all_inlined_stmts = []
        self.visitor._all_inlined_stmts.extend(inline_result.stmts)

        for i, stmt in enumerate(inline_result.stmts):
            stmt_str = ast.unparse(stmt) if hasattr(ast, 'unparse') else str(stmt)
            logger.debug(f"{tag}: visiting stmt[{i}]: {stmt_str}")
            self.visitor.visit(stmt)

        restore_globals(self.visitor, old_user_globals)

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

        Linear types: we strip ``var_name`` / ``linear_path`` from the
        ValueRef so the temporary is not tracked for consumption; the
        ownership has already been transferred by the caller when the
        arguments were evaluated.
        """
        arg_temps: Dict[str, str] = {}
        call_id = get_next_id()

        for i, (param_name, param_value) in enumerate(self.param_bindings.items()):
            temp_name = f"_arg_{self.kind}_{call_id}_{i}"
            arg_temps[param_name] = temp_name

            if not isinstance(param_value, ValueRef):
                from ..builtin_entities.python_type import PythonType
                param_value = wrap_value(
                    param_value,
                    kind="python",
                    type_hint=PythonType.wrap(param_value, is_constant=True),
                )
            else:
                # Fresh ValueRef without var_name tracking so the temp is
                # not re-consumed in the caller's linear state.
                param_value = param_value.clone(var_name=None, linear_path=None)

            temp_info = VariableInfo(
                name=temp_name,
                value_ref=param_value,
                alloca=None,  # CRITICAL: no alloca for pure value temps
                source=f"{self.kind}_arg_temp",
                is_parameter=False,
            )
            self.visitor.scope_manager.declare_variable(temp_info, allow_shadow=True)

        return arg_temps

    def _build_caller_context(self) -> ScopeContext:
        """Construct caller ScopeContext for kernel capture detection.

        The two modes differ here:

        - ``inline``: only variables declared in the *current* scope are
          in scope for the inlined body.
        - ``closure``: *all visible* variables from every enclosing scope
          are available, which is essential for by-reference capture when
          a closure is defined inside a nested scope (e.g. a loop body).
        """
        available_vars = set()
        scope_manager = getattr(self.visitor, 'scope_manager', None)
        if scope_manager is None:
            return ScopeContext(available_vars=available_vars)

        if self.kind == "closure":
            for var_name in scope_manager.get_all_visible().keys():
                available_vars.add(var_name)
        else:
            for var_info in scope_manager.get_all_in_current_scope():
                available_vars.add(var_info.name)

        return ScopeContext(available_vars=available_vars)

    def _has_return_value(self, func_ast: ast.FunctionDef) -> bool:
        """True if ``func_ast`` can return a non-void value."""
        if func_ast.returns:
            if isinstance(func_ast.returns, ast.Name) and func_ast.returns.id == 'void':
                return False
            return True
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False

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
