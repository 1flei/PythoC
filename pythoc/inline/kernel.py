"""
Universal Inline Kernel

The core inlining engine that handles all inlining scenarios.
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any, TYPE_CHECKING

from .scope_analyzer import ScopeAnalyzer, ScopeContext
from .transformers import InlineBodyTransformer
from ..utils import get_next_id

if TYPE_CHECKING:
    from .exit_rules import ExitPointRule


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
class InlineOp:
    """
    Represents a single inlining operation
    
    Contains all information needed to perform inlining:
    - What to inline (callee body/params)
    - Where to inline (call site, caller context)
    - How to inline (exit rule, scope information)
    """
    
    # Source: what to inline
    callee_body: List[ast.stmt]
    callee_params: List[ast.arg]
    callee_func: Optional[ast.FunctionDef | ast.Lambda]  # Keep original function for metadata
    
    # Target: where to inline
    caller_context: ScopeContext
    call_site: ast.expr
    call_args: List[ast.expr]
    
    # Scope analysis results
    captured_vars: Set[str]
    local_vars: Set[str]
    param_vars: Set[str]
    
    # Transformation rule
    exit_rule: 'ExitPointRule'
    
    # Uniqueness
    inline_id: str
    
    # Callee's globals (for name resolution in inlined code)
    callee_globals: Dict[str, Any] = field(default_factory=dict)


class InlineKernel:
    """
    Universal inlining engine - the single source of truth for all inlining
    
    This kernel handles:
    1. Scope analysis (what variables are captured)
    2. Variable renaming (avoid conflicts)
    3. Parameter binding (map args to params)
    4. Body transformation (apply exit rules)
    5. AST insertion (place transformed body at call site)
    
    Usage:
        kernel = InlineKernel()
        op = kernel.create_inline_op(...)
        inlined_stmts = kernel.execute_inline(op)
    """
    
    def create_inline_op(
        self,
        callee_func: ast.FunctionDef | ast.Lambda,
        call_site: ast.expr,
        call_args: List[ast.expr],
        caller_context: ScopeContext,
        exit_rule: 'ExitPointRule',
        callee_globals: Dict[str, Any] = None
    ) -> InlineOp:
        """
        Create an InlineOp by analyzing callee and caller
        
        This is the first phase: analysis
        
        Args:
            callee_func: Function or lambda to inline
            call_site: Call expression or For node
            call_args: Argument expressions from call site
            caller_context: Caller's scope information
            exit_rule: Rule for transforming exit points
            callee_globals: Callee function's __globals__ (for name resolution)
            
        Returns:
            InlineOp ready for execution
            
        Example:
            # For: result = add(x, 10)
            op = kernel.create_inline_op(
                callee_func=add_func_ast,
                call_site=call_node,
                call_args=[Name('x'), Constant(10)],
                caller_context=ScopeContext(['x', 'y']),
                exit_rule=ReturnExitRule('_result'),
                callee_globals=add_func.__globals__
            )
        """
        # Generate unique ID using global generator
        inline_id = f"inline_{get_next_id()}"
        
        # Extract callee body and params
        # CRITICAL: Deep copy to avoid mutating the original function AST!
        # Without this, nested inlining will corrupt the original function.
        if isinstance(callee_func, ast.FunctionDef):
            callee_body = copy.deepcopy(callee_func.body)
            callee_params = copy.deepcopy(callee_func.args.args)
        elif isinstance(callee_func, ast.Lambda):
            # Lambda body is an expression, wrap in Return
            callee_body = [ast.Return(value=copy.deepcopy(callee_func.body))]
            callee_params = copy.deepcopy(callee_func.args.args)
        else:
            raise TypeError(f"Expected FunctionDef or Lambda, got {type(callee_func)}")
        
        # Analyze scope
        scope_analyzer = ScopeAnalyzer(caller_context)
        captured_vars, local_vars, param_vars = scope_analyzer.analyze(
            callee_body, callee_params
        )
        
        return InlineOp(
            callee_body=callee_body,
            callee_params=callee_params,
            callee_func=callee_func,  # Save original function
            caller_context=caller_context,
            call_site=call_site,
            call_args=call_args,
            captured_vars=captured_vars,
            local_vars=local_vars,
            param_vars=param_vars,
            exit_rule=exit_rule,
            inline_id=inline_id,
            callee_globals=callee_globals or {}
        )
    
    def execute_inline(self, op: InlineOp) -> InlineResult:
        """
        Execute inlining operation
        
        This is the second phase: transformation
        
        Args:
            op: InlineOp from create_inline_op
            
        Returns:
            InlineResult containing:
            - stmts: List of statements to replace the call site
            - required_globals: Globals that must be merged into caller's user_globals
            
        Example:
            Input:
                def add(a, b):
                    return a + b
                result = add(x, 10)
                
            Output (for ReturnExitRule with flag variable):
                _result: <type>
                _is_return: bool = False
                while True:
                    a = x
                    b = 10
                    _result = a + b
                    _is_return = True
                    break
                    break
        """
        # Debug hook - capture before
        from ..utils.ast_debug import ast_debugger
        func_name = op.callee_func.name if op.callee_func and hasattr(op.callee_func, 'name') else 'lambda'
        ast_debugger.capture(
            "before_inline",
            op.callee_body,
            func_name=func_name,
            inline_id=op.inline_id,
            call_site=ast.unparse(op.call_site) if hasattr(ast, 'unparse') else str(op.call_site)
        )
        
        # 1. Create variable renaming map
        rename_map = self._create_rename_map(op)
        
        # 2. Create parameter bindings
        param_bindings = self._create_param_bindings(op, rename_map)
        
        # 3. For ReturnExitRule, create flag variable and result variable declarations
        from .exit_rules import ReturnExitRule
        flag_var = None
        prefix_stmts = []
        
        if isinstance(op.exit_rule, ReturnExitRule):
            # Create flag variable name if not already set
            if op.exit_rule.flag_var:
                flag_var = op.exit_rule.flag_var
            else:
                flag_var = f"_is_return_{op.inline_id}"
                op.exit_rule.flag_var = flag_var
            
            # Declare result variable if needed
            if op.exit_rule.result_var:
                # Try to infer type from function return annotation
                result_type = None
                if op.callee_func and hasattr(op.callee_func, 'returns') and op.callee_func.returns:
                    result_type = op.callee_func.returns
                
                # Skip result declaration for void return type
                is_void = False
                if result_type and isinstance(result_type, ast.Name) and result_type.id == 'void':
                    is_void = True
                
                # Create result variable declaration (without initialization)
                # We'll let the first return assign to it
                if result_type and not is_void:
                    result_decl = ast.AnnAssign(
                        target=ast.Name(id=op.exit_rule.result_var, ctx=ast.Store()),
                        annotation=copy.deepcopy(result_type),
                        value=None,
                        simple=1
                    )
                    prefix_stmts.append(result_decl)
            
            # Declare and initialize flag variable: _is_return = False
            flag_decl = ast.AnnAssign(
                target=ast.Name(id=flag_var, ctx=ast.Store()),
                annotation=ast.Name(id='bool', ctx=ast.Load()),
                value=ast.Constant(value=False),
                simple=1
            )
            prefix_stmts.append(flag_decl)
        
        # 4. Transform callee body with flag variable
        transformer = InlineBodyTransformer(op.exit_rule, rename_map, flag_var)
        transformed_body = transformer.transform(op.callee_body)
        
        # 5. Wrap transformed body in while True for ReturnExitRule (Rule 3)
        # This enables break to exit from multiple return points
        if isinstance(op.exit_rule, ReturnExitRule):
            # Prepend parameter bindings to while True body (move inside scope)
            transformed_body = param_bindings + transformed_body
            
            # Append final break to while True wrapper
            transformed_body.append(ast.Break())
            
            wrapped_body = [ast.While(
                test=ast.Constant(value=True),
                body=transformed_body,
                orelse=[]
            )]
            transformed_body = wrapped_body
            
            # Clear param_bindings since they are now inside while True
            param_bindings = []
        
        # 6. Combine: param bindings + prefix (result/flag decls) + body
        result = param_bindings + prefix_stmts + transformed_body
        
        # 7. Fix AST locations (copy from call site)
        for stmt in result:
            ast.copy_location(stmt, op.call_site)
            ast.fix_missing_locations(stmt)
        
        # Debug hook - capture after (after fixing locations)
        ast_debugger.capture(
            "after_inline",
            result,
            func_name=func_name,
            inline_id=op.inline_id,
            param_count=len(op.callee_params),
            local_count=len(op.local_vars)
        )
        
        # Build required_globals: callee's globals + intrinsics needed by inlined code
        required_globals = self._build_required_globals(op)
        
        return InlineResult(stmts=result, required_globals=required_globals)
    
    def _build_required_globals(self, op: InlineOp) -> Dict[str, Any]:
        """
        Build the globals dict that must be merged into caller's user_globals
        
        This includes:
        1. Callee's __globals__ (for name resolution in inlined code)
        2. Intrinsics needed by the transformation (e.g., 'move' for linear types)
        
        IMPORTANT: Intrinsics are stored with a special key prefix '_pc_intrinsic_'
        to avoid being overwritten during merge. The adapter should apply them last.
        
        Args:
            op: InlineOp with callee_globals
            
        Returns:
            Dict of globals to merge
        """
        required = {}
        
        # Start with callee's globals
        if op.callee_globals:
            required.update(op.callee_globals)
        
        # Add 'move' intrinsic for linear type ownership transfer
        # This is needed because yield/inline transformations wrap values in move()
        # CRITICAL: This must take precedence over any user-defined 'move'
        from ..builtin_entities import move, bool as pc_bool
        required['move'] = move
        
        # Add 'bool' type for flag variable declarations
        # This is needed because kernel generates: _is_return: bool = False
        required['bool'] = pc_bool
        
        return required
    
    def _create_rename_map(self, op: InlineOp) -> Dict[str, str]:
        """
        Create variable renaming map to avoid conflicts
        
        Strategy:
        - Captured variables: NOT renamed (they reference outer scope)
        - Parameters and local variables: ALL renamed with unique suffix
        
        CRITICAL: Parameters MUST be renamed to avoid conflicts in nested inline scenarios.
        Even though parameters are bound to arguments, the binding creates a variable
        with the parameter name, which can conflict with outer scope variables.
        
        Args:
            op: InlineOp with scope information
            
        Returns:
            Mapping of old names to new names
            
        Example:
            For inline_3:
                local_vars = {'x', 'y'}
                param_vars = {'a', 'b'}
                captured_vars = {'base'}
                
            Result:
                {'x': 'x_inline_3', 'y': 'y_inline_3', 'a': 'a_inline_3', 'b': 'b_inline_3'}
                # All renamed except base (captured)
        """
        rename_map = {}
        suffix = f"_{op.inline_id}"
        
        # Rename ALL local variables (not captured)
        for var in op.local_vars:
            if var not in op.captured_vars:
                rename_map[var] = f"{var}{suffix}"
        
        # Rename ALL parameters (not captured)
        # This is critical for nested inlining
        for var in op.param_vars:
            if var not in op.captured_vars:
                rename_map[var] = f"{var}{suffix}"
        
        return rename_map
    
    def _create_param_bindings(
        self, 
        op: InlineOp, 
        rename_map: Dict[str, str]
    ) -> List[ast.stmt]:
        """
        Create parameter binding statements
        
        For each parameter, create:
            renamed_param_name = move(arg_value) (with type conversion if needed)
            
        With proper type annotations and automatic type conversion.
        Parameter names are renamed according to rename_map.
        
        CRITICAL: All parameter bindings use move() to handle linear type ownership.
        For non-linear types, move() is a no-op.
        
        Args:
            op: InlineOp with parameter information
            rename_map: Variable rename mapping (applied to parameter names)
            
        Returns:
            List of assignment statements
            
        Example:
            def add(a: i32, b: i32):
                return a + b
                
            Call: add(x, 1)
            rename_map: {'a': 'a_inline_0', 'b': 'b_inline_0'}
            
            Result:
                [
                    AnnAssign(target=Name('a_inline_0'), annotation=Name('i32'), value=Call(move, [Name('x')])),
                    AnnAssign(target=Name('b_inline_0'), annotation=Name('i32'), value=Call(move, [Call(i32, [Constant(1)])]))
                ]
        """
        bindings = []
        
        for i, param in enumerate(op.callee_params):
            param_name = param.arg
            
            # Apply renaming to parameter name
            renamed_param_name = rename_map.get(param_name, param_name)
            
            # Get argument value
            if i < len(op.call_args):
                arg_value = copy.deepcopy(op.call_args[i])
            else:
                # Missing argument - should be caught earlier by type checker
                raise ValueError(
                    f"Missing argument for parameter '{param_name}' "
                    f"in inline operation {op.inline_id}"
                )
            
            # Wrap argument with type conversion if parameter has type annotation
            if param.annotation:
                # Check if arg_value needs type conversion
                # If it's a constant (Constant node), wrap it with type constructor call
                if isinstance(arg_value, ast.Constant):
                    arg_value = self._wrap_with_type_conversion(arg_value, param.annotation)
                
                # Wrap in move() for ownership transfer (essential for linear types)
                moved_value = ast.Call(
                    func=ast.Name(id='move', ctx=ast.Load()),
                    args=[arg_value],
                    keywords=[]
                )
                
                binding = ast.AnnAssign(
                    target=ast.Name(id=renamed_param_name, ctx=ast.Store()),
                    annotation=copy.deepcopy(param.annotation),
                    value=moved_value,
                    simple=1
                )
            else:
                # Wrap in move() for ownership transfer
                moved_value = ast.Call(
                    func=ast.Name(id='move', ctx=ast.Load()),
                    args=[arg_value],
                    keywords=[]
                )
                
                binding = ast.Assign(
                    targets=[ast.Name(id=renamed_param_name, ctx=ast.Store())],
                    value=moved_value
                )
            
            bindings.append(binding)
        
        return bindings
    
    def _wrap_with_type_conversion(self, value: ast.expr, type_annotation: ast.expr) -> ast.expr:
        """
        Wrap a value with type conversion call
        
        Args:
            value: The value expression to wrap
            type_annotation: The target type annotation
            
        Returns:
            Call node: type_annotation(value)
            
        Example:
            value = Constant(1)
            type_annotation = Name('i32')
            
            Result:
                Call(func=Name('i32'), args=[Constant(1)])
        """
        return ast.Call(
            func=copy.deepcopy(type_annotation),
            args=[value],
            keywords=[]
        )
