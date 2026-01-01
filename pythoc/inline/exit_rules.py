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
    
    Transformation using scoped label approach:
        return expr  -->  result_var = move(expr); goto_end("_inline_exit_{id}")
        
    Multiple returns are handled by:
    1. Each return assigns result and jumps to exit label via goto_end
    2. Inlined body is wrapped in with label("_inline_exit_{id}")
    
    This uses scoped labels for structured control flow:
    - goto_end exits the label scope cleanly
    - Defers are properly executed on exit
    - No unstructured jumps
    """
    
    def __init__(self, result_var: Optional[str] = None, exit_label: Optional[str] = None):
        """
        Args:
            result_var: Variable name to store return value
                       If None, return value is discarded
            exit_label: Label name for exit point (set by kernel)
        """
        self.result_var = result_var
        self.exit_label = exit_label
    
    def get_exit_node_types(self) -> Tuple[type, ...]:
        return (ast.Return,)
    
    def transform_exit(
        self, 
        exit_node: ast.Return, 
        context: 'InlineContext'
    ) -> List[ast.stmt]:
        """
        return expr  -->  result_var = move(expr); goto_end("_inline_exit_{id}")
        
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
        
        # Jump to exit label using scoped goto_end: goto_end("_inline_exit_{id}")
        if self.exit_label:
            goto_call = ast.Expr(value=ast.Call(
                func=ast.Name(id='goto_end', ctx=ast.Load()),
                args=[ast.Constant(value=self.exit_label)],
                keywords=[]
            ))
            stmts.append(goto_call)
        
        return stmts


class YieldExitRule(ExitPointRule):
    """
    Transform yield statements for generators using scoped label approach
    
    Transformation:
        yield expr  -->  with label("_yield_{id}"): loop_var = expr; <loop_body>
        
    With type annotation:
        def gen() -> i32:
            yield 1
        
        Becomes:
            with label("_yield_0"):
                loop_var: i32 = i32(1)
                <loop_body>
    
    For tuple unpacking:
        def gen() -> struct[i32, i32]:
            yield a, b
        
        for x, y in gen():
            ...
        
        Becomes:
            with label("_yield_0"):
                _tmp = (a, b)
                x = _tmp[0]
                y = _tmp[1]
                <loop_body>
    
    When loop_body contains break/continue, transforms them to scoped goto:
        break    --> goto_begin("_for_after_else_{id}")  (skip all remaining yields and else)
        continue --> goto_end("_yield_{id}")             (skip to next yield)
        
    This scoped label approach:
    - Uses structured control flow
    - Properly handles defer execution
    - No unstructured jumps
    """
    
    def __init__(
        self, 
        loop_var: ast.AST,  # Can be Name or Tuple
        loop_body: List[ast.stmt],
        return_type_annotation: Optional[ast.expr] = None,
        after_else_label: Optional[str] = None
    ):
        """
        Args:
            loop_var: Loop variable target (Name or Tuple AST node)
            loop_body: Statements in the for loop body
            return_type_annotation: Return type annotation from function (optional)
            after_else_label: Label name for after-else (if loop body has break)
        """
        self.loop_var = loop_var
        self.loop_body = loop_body
        self.return_type_annotation = return_type_annotation
        self.after_else_label = after_else_label
        
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
        yield expr  -->  with label("_yield_{id}"): loop_var = move(expr); <loop_body>
        
        The move() wrapper is essential for linear types because:
        - yield is semantically a continuation call: yield x <==> continuation(x)
        - Function calls transfer ownership of linear arguments
        - But the AST transformation converts this to an assignment
        - Wrapping in move() restores the ownership transfer semantic
        
        For non-linear types, move() is a no-op.
        
        For tuple unpacking:
            yield a, b  -->  x, y = move((a, b)); <loop_body>
        
        When loop_body has break/continue, transforms them to scoped goto:
            break    --> goto_begin("_for_after_else_{id}")
            continue --> goto_end("_yield_{id}")
        """
        from ..utils import get_next_id
        
        # Extract yield value
        if isinstance(exit_node, ast.Expr) and isinstance(exit_node.value, ast.Yield):
            yield_val = exit_node.value.value
        elif isinstance(exit_node, ast.Yield):
            yield_val = exit_node.value
        else:
            # Not a yield - return as is
            return [exit_node]
        
        # Generate unique label for this yield scope
        yield_label = f"_yield_{get_next_id()}" if self._body_has_break_or_continue else None
        
        # Build the body statements (assignment + loop body)
        inner_stmts = []
        
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
                inner_stmts.extend(self._create_tuple_unpack_stmts(moved_value))
            else:
                loop_var_name = self.loop_var.id if isinstance(self.loop_var, ast.Name) else str(self.loop_var)
                assign = ast.Assign(
                    targets=[ast.Name(id=loop_var_name, ctx=ast.Store())],
                    value=moved_value
                )
                inner_stmts.append(assign)
        
        # Insert loop body (deep copy to avoid mutation)
        if self._body_has_break_or_continue and self.after_else_label:
            # Transform break/continue in loop body to scoped goto
            for stmt in self.loop_body:
                transformed = _transform_break_continue_scoped(
                    copy.deepcopy(stmt), 
                    self.after_else_label,
                    yield_label
                )
                inner_stmts.append(transformed)
        else:
            for stmt in self.loop_body:
                inner_stmts.append(copy.deepcopy(stmt))
        
        # Wrap in scoped label if we have break/continue
        if yield_label:
            # Create: with label("_yield_{id}"): <inner_stmts>
            label_call = ast.Call(
                func=ast.Name(id='label', ctx=ast.Load()),
                args=[ast.Constant(value=yield_label)],
                keywords=[]
            )
            with_stmt = ast.With(
                items=[ast.withitem(context_expr=label_call, optional_vars=None)],
                body=inner_stmts
            )
            body_stmts = [with_stmt]
        else:
            body_stmts = inner_stmts
        
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


def _transform_break_continue_scoped(stmt: ast.stmt, after_else_label: str, yield_label: str) -> ast.stmt:
    """Transform break/continue statements in loop body for yield expansion using scoped labels
    
    Transforms:
        break    -->  goto_begin(after_else_label)  (skip all remaining yields and else)
        continue -->  goto_end(yield_label)         (exit current yield scope)
    
    Args:
        stmt: AST statement to transform
        after_else_label: Label name for after-else (break target)
        yield_label: Label name for current yield scope (continue target)
        
    Returns:
        Transformed statement
    """
    transformer = _BreakContinueTransformer(after_else_label, yield_label)
    return transformer.visit(stmt)


class _BreakContinueTransformer(ast.NodeTransformer):
    """Transform break/continue for yield expansion using scoped labels."""
    
    def __init__(self, after_else_label: str, yield_label: str, use_scoped: bool = True):
        self.after_else_label = after_else_label
        self.yield_label = yield_label
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
        """Transform break to skip all remaining code
        
        Uses goto_begin(after_else_label) - forward sibling jump
        """
        if self.loop_depth > 0:
            # Inside nested loop, don't transform
            return node
        
        # Create: goto_begin("_for_after_else_{id}")
        goto_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='goto_begin', ctx=ast.Load()),
            args=[ast.Constant(value=self.after_else_label)],
            keywords=[]
        ))
        return goto_call
    
    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        """Transform continue to skip to next yield
        
        Uses goto_end(yield_label) - exit current yield scope
        """
        if self.loop_depth > 0:
            # Inside nested loop, don't transform
            return node
        
        # Create: goto_end("_yield_{id}")
        # This exits the current yield scope, moving to the next yield
        goto_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='goto_end', ctx=ast.Load()),
            args=[ast.Constant(value=self.yield_label)],
            keywords=[]
        ))
        return goto_call
