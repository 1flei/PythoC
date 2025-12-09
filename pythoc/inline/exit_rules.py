"""
Exit point transformation rules

Defines how different exit points (return, yield, etc.) are transformed
during inlining.
"""

import ast
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .transformers import InlineContext


class ExitPointRule(ABC):
    """
    Abstract rule for transforming exit points (return/yield)
    
    Different inlining scenarios require different exit point handling:
    - @inline/closure: return -> assignment
    - yield: yield -> loop_var assignment + loop_body
    - macro: return -> direct AST substitution
    """
    
    @abstractmethod
    def transform_exit(
        self, 
        exit_node: ast.stmt, 
        context: 'InlineContext'
    ) -> List[ast.stmt]:
        """
        Transform a single exit point into target statements
        
        Args:
            exit_node: The exit point node (Return, Yield, etc.)
            context: Inline context with renaming information
            
        Returns:
            List of statements to replace the exit point
        """
        pass
    
    @abstractmethod
    def get_exit_node_types(self) -> Tuple[type, ...]:
        """
        Return tuple of AST node types that are exit points
        
        Used by transformer to identify which nodes to transform
        """
        pass
    
    def _rename(self, node: ast.expr, context: 'InlineContext') -> ast.expr:
        """
        Helper: Apply variable renaming to an expression
        
        Uses context's rename_map
        """
        if context and hasattr(context, 'rename_map'):
            renamer = VariableRenamer(context.rename_map)
            return renamer.visit(copy.deepcopy(node))
        return copy.deepcopy(node)


class ReturnExitRule(ExitPointRule):
    """
    Transform return statements for @inline and closures
    
    Transformation using flag variable approach:
        return expr  -->  result_var = expr; is_return_flag = True; break
        
    Multiple returns are handled by:
    1. Each return sets the flag and breaks
    2. All loops get flag check after them: if is_return_flag: break
    3. Entire body wrapped in while True
    """
    
    def __init__(self, result_var: Optional[str] = None, flag_var: Optional[str] = None):
        """
        Args:
            result_var: Variable name to store return value
                       If None, return value is discarded
            flag_var: Variable name for return flag (is_return)
                     If None, auto-generated
        """
        self.result_var = result_var
        self.flag_var = flag_var
    
    def get_exit_node_types(self) -> Tuple[type, ...]:
        return (ast.Return,)
    
    def transform_exit(
        self, 
        exit_node: ast.Return, 
        context: 'InlineContext'
    ) -> List[ast.stmt]:
        """
        return expr  -->  result_var = expr; is_return_flag = True; break
        """
        stmts = []
        
        if exit_node.value and self.result_var:
            # Assignment: result_var = return_value
            renamed_value = self._rename(exit_node.value, context)
            assign = ast.Assign(
                targets=[ast.Name(id=self.result_var, ctx=ast.Store())],
                value=renamed_value
            )
            stmts.append(assign)
        
        # Set flag: is_return_flag = True (use 1 for PC bool)
        if self.flag_var:
            set_flag = ast.Assign(
                targets=[ast.Name(id=self.flag_var, ctx=ast.Store())],
                value=ast.Constant(value=1)  # 1 will be converted to bool
            )
            stmts.append(set_flag)
        
        # Break to exit current loop
        stmts.append(ast.Break())
        
        return stmts


class YieldExitRule(ExitPointRule):
    """
    Transform yield statements for generators
    
    Transformation:
        yield expr  -->  loop_var = expr; <loop_body>
        
    With type annotation:
        def gen() -> i32:
            yield 1
        
        Becomes:
            loop_var: i32 = i32(1)
            <loop_body>
    
    For tuple unpacking:
        def gen() -> struct[i32, i32]:
            yield a, b
        
        for x, y in gen():
            ...
        
        Becomes:
            _tmp = (a, b)
            x = _tmp[0]
            y = _tmp[1]
            <loop_body>
    """
    
    def __init__(
        self, 
        loop_var: ast.AST,  # Can be Name or Tuple
        loop_body: List[ast.stmt],
        return_type_annotation: Optional[ast.expr] = None
    ):
        """
        Args:
            loop_var: Loop variable target (Name or Tuple AST node)
            loop_body: Statements in the for loop body
            return_type_annotation: Return type annotation from function (optional)
        """
        self.loop_var = loop_var
        self.loop_body = loop_body
        self.return_type_annotation = return_type_annotation
    
    def get_exit_node_types(self) -> Tuple[type, ...]:
        # Only Yield expressions, not all Expr nodes
        # Expr nodes containing Yield are handled specially in visit_Expr
        return (ast.Yield,)
    
    def transform_exit(
        self, 
        exit_node: ast.stmt, 
        context: 'InlineContext'
    ) -> List[ast.stmt]:
        """
        yield expr  -->  loop_var = move(expr); <loop_body>
        
        The move() wrapper is essential for linear types because:
        - yield is semantically a continuation call: yield x <==> continuation(x)
        - Function calls transfer ownership of linear arguments
        - But the AST transformation converts this to an assignment
        - Wrapping in move() restores the ownership transfer semantic
        
        For non-linear types, move() is a no-op.
        
        For tuple unpacking:
            yield a, b  -->  x, y = move((a, b)); <loop_body>
        """
        # Extract yield value
        if isinstance(exit_node, ast.Expr) and isinstance(exit_node.value, ast.Yield):
            yield_val = exit_node.value.value
        elif isinstance(exit_node, ast.Yield):
            yield_val = exit_node.value
        else:
            # Not a yield - return as is
            return [exit_node]
        
        stmts = []
        
        # Assignment: loop_var = yield_value (with type conversion if needed)
        if yield_val:
            renamed_value = self._rename(yield_val, context)
            
            # Apply type conversion if we have type annotation and value is constant
            if self.return_type_annotation and isinstance(renamed_value, ast.Constant):
                renamed_value = self._wrap_with_type_conversion(
                    renamed_value, 
                    self.return_type_annotation
                )
            
            # Wrap in move() for ownership transfer
            # This is essential for linear types: yield is semantically a continuation call,
            # which transfers ownership. The move() wrapper makes this explicit to the compiler.
            # For non-linear types, move() is a no-op.
            moved_value = ast.Call(
                func=ast.Name(id='move', ctx=ast.Load()),
                args=[renamed_value],
                keywords=[]
            )
            
            # Handle tuple unpacking vs simple assignment
            if isinstance(self.loop_var, ast.Tuple):
                # Tuple unpacking: a, b = move((x, y))
                stmts.extend(self._create_tuple_unpack_stmts(moved_value))
            else:
                # Simple assignment (loop variable is pre-declared by yield_adapter)
                loop_var_name = self.loop_var.id if isinstance(self.loop_var, ast.Name) else str(self.loop_var)
                assign = ast.Assign(
                    targets=[ast.Name(id=loop_var_name, ctx=ast.Store())],
                    value=moved_value
                )
                stmts.append(assign)
        
        # Insert loop body (deep copy to avoid mutation)
        for stmt in self.loop_body:
            stmts.append(copy.deepcopy(stmt))
        
        return stmts
    
    def _create_tuple_unpack_stmts(self, value: ast.expr) -> List[ast.stmt]:
        """Create statements for tuple unpacking
        
        For: for a, b in gen(): ...
        Where gen() yields (x, y)
        
        Creates a single tuple unpacking assignment:
            a, b = (x, y)
        
        This uses Python's native tuple unpacking syntax, which pythoc's
        assignment visitor will handle correctly for linear types.
        """
        # Create tuple unpacking assignment: a, b = value
        # The target is already a Tuple AST node from the for loop
        unpack_assign = ast.Assign(
            targets=[copy.deepcopy(self.loop_var)],  # Tuple target
            value=value
        )
        return [unpack_assign]
        
        return stmts
    
    def _wrap_with_type_conversion(self, value: ast.expr, type_annotation: ast.expr) -> ast.expr:
        """
        Wrap a value with type conversion call
        
        Args:
            value: The value expression to wrap
            type_annotation: The target type annotation
            
        Returns:
            Call node: type_annotation(value)
        """
        return ast.Call(
            func=copy.deepcopy(type_annotation),
            args=[value],
            keywords=[]
        )


class MacroExitRule(ExitPointRule):
    """
    Transform for compile-time macro expansion (future)
    
    Transformation:
        return expr  -->  expr (direct AST substitution)
        
    Used for pure compile-time evaluation
    """
    
    def get_exit_node_types(self) -> Tuple[type, ...]:
        return (ast.Return,)
    
    def transform_exit(
        self, 
        exit_node: ast.Return, 
        context: 'InlineContext'
    ) -> List[ast.stmt]:
        """
        return expr  -->  expr (as expression statement)
        """
        if exit_node.value:
            renamed_value = self._rename(exit_node.value, context)
            return [ast.Expr(value=renamed_value)]
        return []


class VariableRenamer(ast.NodeTransformer):
    """
    Helper: Rename variables in AST according to rename_map
    
    Only renames Name nodes, preserves everything else
    """
    
    def __init__(self, rename_map: dict):
        self.rename_map = rename_map
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename if in map, otherwise keep original"""
        if node.id in self.rename_map:
            return ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
        return node
