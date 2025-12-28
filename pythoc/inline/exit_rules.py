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
        return expr  -->  result_var = move(expr); is_return_flag = True; break
        
        Note: We wrap the return value in move() to properly transfer
        ownership of linear types. This is necessary because the generated
        assignment `result_var = prf` would otherwise be rejected by the
        linear type checker as an implicit copy.
        """
        stmts = []
        
        if exit_node.value and self.result_var:
            # Assignment: result_var = move(return_value)
            renamed_value = self._rename(exit_node.value, context)
            # Wrap in move() for linear type ownership transfer
            moved_value = ast.Call(
                func=ast.Name(id='move', ctx=ast.Load()),
                args=[renamed_value],
                keywords=[]
            )
            assign = ast.Assign(
                targets=[ast.Name(id=self.result_var, ctx=ast.Store())],
                value=moved_value
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
    
    When loop_body contains break/continue, each yield is wrapped in a mini-loop:
        while not __yield_break_flag:
            loop_var = move(expr)
            <transformed_loop_body>  # break -> set flag + break, continue -> break
            break  # Always exit after one iteration
    """
    
    def __init__(
        self, 
        loop_var: ast.AST,  # Can be Name or Tuple
        loop_body: List[ast.stmt],
        return_type_annotation: Optional[ast.expr] = None,
        break_flag_var: Optional[str] = None
    ):
        """
        Args:
            loop_var: Loop variable target (Name or Tuple AST node)
            loop_body: Statements in the for loop body
            return_type_annotation: Return type annotation from function (optional)
            break_flag_var: Name of break flag variable (if loop body has break/continue)
        """
        self.loop_var = loop_var
        self.loop_body = loop_body
        self.return_type_annotation = return_type_annotation
        self.break_flag_var = break_flag_var
        
        # Check if loop body has break or continue
        self._body_has_break_or_continue = _has_break_or_continue(loop_body)
    
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
        
        When loop_body has break/continue, wraps in mini-loop for correct semantics.
        """
        # Extract yield value
        if isinstance(exit_node, ast.Expr) and isinstance(exit_node.value, ast.Yield):
            yield_val = exit_node.value.value
        elif isinstance(exit_node, ast.Yield):
            yield_val = exit_node.value
        else:
            # Not a yield - return as is
            return [exit_node]
        
        # Build the body statements (assignment + loop body)
        body_stmts = []
        
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
            moved_value = ast.Call(
                func=ast.Name(id='move', ctx=ast.Load()),
                args=[renamed_value],
                keywords=[]
            )
            
            # Handle tuple unpacking vs simple assignment
            if isinstance(self.loop_var, ast.Tuple):
                body_stmts.extend(self._create_tuple_unpack_stmts(moved_value))
            else:
                loop_var_name = self.loop_var.id if isinstance(self.loop_var, ast.Name) else str(self.loop_var)
                assign = ast.Assign(
                    targets=[ast.Name(id=loop_var_name, ctx=ast.Store())],
                    value=moved_value
                )
                body_stmts.append(assign)
        
        # Insert loop body (deep copy to avoid mutation)
        if self._body_has_break_or_continue and self.break_flag_var:
            # Transform break/continue in loop body
            for stmt in self.loop_body:
                transformed = _transform_break_continue(
                    copy.deepcopy(stmt), 
                    self.break_flag_var
                )
                body_stmts.append(transformed)
        else:
            for stmt in self.loop_body:
                body_stmts.append(copy.deepcopy(stmt))
        
        # If body has break/continue, wrap in mini-loop
        if self._body_has_break_or_continue and self.break_flag_var:
            # Add final break to exit mini-loop after one iteration
            body_stmts.append(ast.Break())
            
            # Create: while not __yield_break_flag: <body>
            while_loop = ast.While(
                test=ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Name(id=self.break_flag_var, ctx=ast.Load())
                ),
                body=body_stmts,
                orelse=[]
            )
            # Copy location from exit_node and fix missing locations
            ast.copy_location(while_loop, exit_node)
            ast.fix_missing_locations(while_loop)
            return [while_loop]
        else:
            # Copy location from exit_node and fix missing locations
            for stmt in body_stmts:
                ast.copy_location(stmt, exit_node)
                ast.fix_missing_locations(stmt)
            return body_stmts
    
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


def _has_break_or_continue(body: List[ast.stmt]) -> bool:
    """Check if body contains any break or continue statement
    
    Only checks at the current loop level - does NOT recurse into nested loops.
    
    Args:
        body: List of AST statements
        
    Returns:
        True if body contains break or continue that would affect current loop
    """
    for stmt in body:
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return True
        # Recursively check control flow statements, but NOT nested loops
        if isinstance(stmt, ast.If):
            if _has_break_or_continue(stmt.body) or _has_break_or_continue(stmt.orelse):
                return True
        elif isinstance(stmt, ast.With):
            if _has_break_or_continue(stmt.body):
                return True
        elif isinstance(stmt, ast.Try):
            if (_has_break_or_continue(stmt.body) or 
                _has_break_or_continue(stmt.orelse) or
                _has_break_or_continue(stmt.finalbody)):
                return True
            for handler in stmt.handlers:
                if _has_break_or_continue(handler.body):
                    return True
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                if _has_break_or_continue(case.body):
                    return True
        # Do NOT recurse into For/While - break/continue inside nested loop
        # doesn't affect the outer loop
    return False


def _transform_break_continue(stmt: ast.stmt, break_flag_var: str) -> ast.stmt:
    """Transform break/continue statements in loop body for yield expansion
    
    Transforms:
        break    -->  __break_flag = True; break
        continue -->  break  (just exit mini-loop, proceed to next yield)
    
    Args:
        stmt: AST statement to transform
        break_flag_var: Name of the break flag variable
        
    Returns:
        Transformed statement
    """
    transformer = _BreakContinueTransformer(break_flag_var)
    return transformer.visit(stmt)


class _BreakContinueTransformer(ast.NodeTransformer):
    """Transform break/continue for yield expansion mini-loops"""
    
    def __init__(self, break_flag_var: str):
        self.break_flag_var = break_flag_var
        self.loop_depth = 0  # Track nested loop depth
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Don't transform break/continue inside nested for loops"""
        self.loop_depth += 1
        result = self.generic_visit(node)
        self.loop_depth -= 1
        return result
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Don't transform break/continue inside nested while loops"""
        self.loop_depth += 1
        result = self.generic_visit(node)
        self.loop_depth -= 1
        return result
    
    def visit_Break(self, node: ast.Break) -> ast.stmt:
        """Transform break: set flag and break mini-loop"""
        if self.loop_depth > 0:
            # Inside nested loop, don't transform
            return node
        
        # Create: if True: __break_flag = True; break
        # We use If with body to create a statement list
        set_flag = ast.Assign(
            targets=[ast.Name(id=self.break_flag_var, ctx=ast.Store())],
            value=ast.Constant(value=True)
        )
        # Return an If that always executes to hold multiple statements
        return ast.If(
            test=ast.Constant(value=True),
            body=[set_flag, ast.Break()],
            orelse=[]
        )
    
    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        """Transform continue: just break mini-loop (proceed to next yield)"""
        if self.loop_depth > 0:
            # Inside nested loop, don't transform
            return node
        
        # continue -> break (exit mini-loop without setting flag)
        return ast.Break()
