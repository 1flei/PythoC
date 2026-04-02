"""
Inline Adapter - Bridges @inline decorator with meta expansion pipeline

Pure AST adapter - generates AST only, no IR/builder operations.
The only interaction with visitor is registering temporary argument variables.
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


class InlineAdapter:
    """
    Adapter for @inline decorator using meta expansion pipeline

    Strategy:
    1. Create temporary variable names for each argument ValueRef
    2. Register these temporary variables in visitor scope
    3. Use meta.expand_inline() to generate AST with references to these temp vars
    4. Return AST for visitor to process

    This adapter only touches visitor.scope_manager (read-only for scope, write for temp vars).
    All IR generation happens in visitor when it processes the returned AST.
    """

    def __init__(self, parent_visitor: 'LLVMIRVisitor', param_bindings: Dict[str, Any],
                 func_globals: Dict[str, Any] = None):
        """
        Initialize adapter

        Args:
            parent_visitor: The visitor that's calling the inline function
            param_bindings: Dict mapping parameter names to ValueRefs or Python objects
            func_globals: The inline function's __globals__ (for name resolution)
        """
        self.visitor = parent_visitor
        self.param_bindings = param_bindings
        self.func_globals = func_globals
    
    def execute_inline(self, func_ast: ast.FunctionDef) -> Optional[ValueRef]:
        """
        Execute function inline using universal kernel via meta pipeline

        Args:
            func_ast: The function AST to execute inline

        Returns:
            ValueRef of the return value, or None if no return
        """
        logger.debug(f"InlineAdapter: executing inline for {func_ast.name}")

        # Determine result variable name using global ID
        result_var = f"_inline_result_{get_next_id()}"

        # Create exit rule
        exit_rule = ReturnExitRule(result_var=result_var)

        # Create temporary variables for arguments and register them
        # This allows kernel-generated AST to reference these temps
        arg_temps = self._create_arg_temps()

        # Create AST argument expressions (Name nodes referencing temp vars)
        arg_exprs = [ast.Name(id=temp_name, ctx=ast.Load()) for temp_name in arg_temps.values()]

        # Build caller context from current scope
        caller_context = self._build_caller_context()

        # Create dummy call site
        call_site = ast.Call(
            func=ast.Name(id=func_ast.name, ctx=ast.Load()),
            args=arg_exprs,
            keywords=[]
        )

        # Build MetaInlineRequest and delegate to expand_inline
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
            logger.error(f"InlineAdapter: meta inline expansion failed: {e}")
            raise

        # Merge globals, visit statements, restore globals
        old_user_globals = merge_inline_globals(self.visitor, inline_result)

        for stmt in inline_result.stmts:
            ast.fix_missing_locations(stmt)

        # Debug hook - accumulate inlined statements at function level
        from ..utils.ast_debug import ast_debugger
        if not hasattr(self.visitor, '_all_inlined_stmts'):
            self.visitor._all_inlined_stmts = []
        self.visitor._all_inlined_stmts.extend(inline_result.stmts)

        import ast as ast_module
        for i, stmt in enumerate(inline_result.stmts):
            stmt_str = ast_module.unparse(stmt) if hasattr(ast_module, 'unparse') else str(stmt)
            logger.debug(f"InlineAdapter: visiting stmt[{i}]: {stmt_str}")
            self.visitor.visit(stmt)

        restore_globals(self.visitor, old_user_globals)
        
        # Return result if function has return value
        has_return = self._has_return_value(func_ast)
        if has_return:
            return self._lookup_result_var(result_var)
        
        # For void functions, return a void ValueRef
        from ..builtin_entities import void
        return wrap_value(None, kind='python', type_hint=void)
    
    def _create_arg_temps(self) -> Dict[str, str]:
        """
        Create temporary variables for arguments and register in visitor scope
        
        Returns:
            Mapping of parameter names to temporary variable names
            
        Example:
            param_bindings = {'a': ValueRef(value_of_x), 'b': ValueRef(value_of_10)}
            Returns: {'a': '_arg_inline_1_0', 'b': '_arg_inline_1_1'}
            And registers these temps in visitor scope
            
        IMPORTANT: These temps have NO alloca - they are pure value references.
        When visitor processes "a = _temp", it will load/use the value directly.
        
        CRITICAL for linear types: We clear var_name from the ValueRef so that
        the temporary variable is not tracked for consumption. The ownership
        was already transferred when the caller evaluated the arguments.
        """
        arg_temps = {}
        inline_id = get_next_id()
        
        for i, (param_name, param_value) in enumerate(self.param_bindings.items()):
            # Create unique temp variable name
            temp_name = f"_arg_inline_{inline_id}_{i}"
            arg_temps[param_name] = temp_name
            
            # Wrap non-ValueRef as python ValueRef
            if not isinstance(param_value, ValueRef):
                from ..builtin_entities.python_type import PythonType
                param_value = wrap_value(param_value, kind="python", 
                                     type_hint=PythonType.wrap(param_value, is_constant=True))
            else:
                # Create a fresh ValueRef without var_name tracking
                # This is critical for linear types: the ownership was already
                # transferred when the caller evaluated the arguments
                param_value = param_value.clone(var_name=None, linear_path=None)
            
            # Register temp variable WITHOUT alloca
            # This ensures visit_Name returns the ValueRef directly
            temp_info = VariableInfo(
                name=temp_name,
                value_ref=param_value,
                alloca=None,  # CRITICAL: No alloca!
                source="inline_arg_temp",
                is_parameter=False
            )
            self.visitor.scope_manager.declare_variable(temp_info, allow_shadow=True)
        
        return arg_temps
    
    def _build_caller_context(self) -> ScopeContext:
        """Build caller scope context from current visitor state"""
        available_vars = set()
        
        if hasattr(self.visitor, 'scope_manager'):
            registry = self.visitor.scope_manager
            for var_info in registry.get_all_in_current_scope():
                available_vars.add(var_info.name)
        
        return ScopeContext(available_vars=available_vars)
    
    def _has_return_value(self, func_ast: ast.FunctionDef) -> bool:
        """Check if function has non-void return value"""
        if func_ast.returns:
            # Check if return type is void
            if isinstance(func_ast.returns, ast.Name) and func_ast.returns.id == 'void':
                return False
            return True
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Return) and node.value:
                return True
        return False
    
    def _lookup_result_var(self, var_name: str) -> Optional[ValueRef]:
        """Look up result variable and load its value
        
        CRITICAL for linear types: 
        1. The inline result variable is a temporary that holds the return value
        2. When we read it, we transfer ownership OUT of the temporary
        3. We must mark the temporary as consumed here
        4. The returned ValueRef should NOT have var_name (like move() returns)
           so that the caller treats it as a fresh value, not a variable reference
        """
        var_info = self.visitor.scope_manager.lookup_variable(var_name)
        if not var_info:
            # Debug: print all variables in all scopes
            logger.warning(f"Result variable {var_name} not found")
            logger.warning(f"Current scopes: {len(self.visitor.scope_manager.scopes)}")
            for i, scope in enumerate(self.visitor.scope_manager.scopes):
                logger.warning(f"  Scope {i}: {list(scope.keys())}")
            return None
        
        # Load the value from the alloca
        alloca = var_info.alloca
        loaded_value = self.visitor.builder.load(alloca)
        
        # CRITICAL: Transfer ownership OUT of the inline result variable
        # This marks the temporary as consumed, similar to what move() does
        if self.visitor._is_linear_type(var_info.type_hint):
            # Create a temporary ValueRef with tracking info for ownership transfer
            temp_ref = wrap_value(
                loaded_value,
                kind='value',
                type_hint=var_info.type_hint,
                var_name=var_name,
                linear_path=()
            )
            # Transfer ownership - marks _inline_result_N as consumed
            # node=None is acceptable, it's only used for error reporting
            self.visitor._transfer_linear_ownership(temp_ref, reason="inline return", node=None)
        
        # Return a NEW ValueRef WITHOUT var_name tracking (like move() does)
        # This ensures the caller treats it as a fresh value, not a variable reference
        return wrap_value(
            loaded_value, 
            kind='value', 
            type_hint=var_info.type_hint
        )

