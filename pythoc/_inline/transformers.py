"""
AST transformers for inline operations

Transforms callee body by:
1. Renaming variables according to rename_map
2. Transforming exit points according to exit_rule
3. Preserving control flow structure
"""

import ast
import copy
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from .renaming import ScopedVariableRenamer

if TYPE_CHECKING:
    from .exit_rules import ExitPointRule


@dataclass
class InlineContext:
    """
    Context information passed to transformers
    
    Contains rename_map and other information needed during transformation
    """
    rename_map: Dict[str, str]


class InlineBodyTransformer(ast.NodeTransformer):
    """
    Transform callee body by:
    1. Renaming variables according to rename_map
    2. Transforming exit points according to exit_rule
    3. Preserving control flow structure
    
    Note: With goto-based ReturnExitRule, we no longer need:
    - flag_var parameter
    - flag checks after loops
    
    Usage:
        transformer = InlineBodyTransformer(exit_rule, rename_map)
        new_body = transformer.transform(original_body)
    """
    
    def __init__(self, exit_rule: 'ExitPointRule', rename_map: Dict[str, str], flag_var: Optional[str] = None):
        """
        Args:
            exit_rule: Rule for transforming exit points
            rename_map: Mapping of old variable names to new names
            flag_var: DEPRECATED - kept for API compatibility, ignored
        """
        self.exit_rule = exit_rule
        self.rename_map = rename_map
        self.exit_types = exit_rule.get_exit_node_types()
        self.context = InlineContext(rename_map=rename_map)
        # flag_var is no longer used with goto-based approach
    
    def transform(self, body: List[ast.stmt]) -> List[ast.stmt]:
        """
        Transform entire body
        
        Args:
            body: List of statements to transform
            
        Returns:
            Transformed statements (may be more or fewer than input)
        """
        result = []
        for stmt in body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                result.extend(transformed)
            elif transformed is not None:
                result.append(transformed)
        return result
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variables according to rename_map"""
        if node.id in self.rename_map:
            return ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
        return node
    
    def visit_Return(self, node: ast.Return):
        """Transform return using exit rule"""
        if ast.Return in self.exit_types:
            return self.exit_rule.transform_exit(node, self.context)
        return node
    
    def visit_Expr(self, node: ast.Expr):
        """Transform expression statements (may contain yield)"""
        # Check if this is a yield expression
        if isinstance(node.value, ast.Yield) and ast.Yield in self.exit_types:
            return self.exit_rule.transform_exit(node, self.context)
        elif ast.Expr in self.exit_types:
            return self.exit_rule.transform_exit(node, self.context)
        
        # Otherwise, just rename variables (use generic_visit's return value)
        return self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Transform while loop (recursively transform body)
        
        With goto-based approach, no flag check is needed after the loop.
        """
        new_test = self.visit(node.test)
        new_body = self.transform(node.body)
        new_orelse = self.transform(node.orelse) if node.orelse else []
        
        return ast.While(test=new_test, body=new_body, orelse=new_orelse)
    
    def visit_If(self, node: ast.If) -> ast.If:
        """Transform if statement (recursively transform branches)
        
        For yield inlining: branches are transformed recursively.
        The loop body is only inserted at yield points (via YieldExitRule.transform_exit),
        so non-yield branches naturally don't execute the loop body.
        
        Note: We do NOT add 'continue' to non-yield branches because the inlined
        code is not wrapped in a loop. The yield transformation already handles
        this correctly by only inserting the loop body at yield points.
        """
        new_test = self.visit(node.test)
        new_body = self.transform(node.body)
        new_orelse = self.transform(node.orelse) if node.orelse else []
        
        return ast.If(test=new_test, body=new_body, orelse=new_orelse)
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Transform for loop (recursively transform body)
        
        With goto-based approach, no flag check is needed after the loop.
        """
        new_target = self.visit(node.target)
        new_iter = self.visit(node.iter)
        new_body = self.transform(node.body)
        new_orelse = self.transform(node.orelse) if node.orelse else []
        
        return ast.For(
            target=new_target, 
            iter=new_iter, 
            body=new_body, 
            orelse=new_orelse
        )
    
    def visit_With(self, node: ast.With) -> ast.With:
        """Transform with statement (rename variables in context)"""
        new_items = []
        for item in node.items:
            new_context_expr = self.visit(item.context_expr)
            new_optional_vars = self.visit(item.optional_vars) if item.optional_vars else None
            new_items.append(ast.withitem(
                context_expr=new_context_expr,
                optional_vars=new_optional_vars
            ))
        
        new_body = self.transform(node.body)
        
        return ast.With(items=new_items, body=new_body)
    
    def visit_Try(self, node: ast.Try) -> ast.Try:
        """Transform try statement (recursively transform all blocks)"""
        new_body = self.transform(node.body)
        
        new_handlers = []
        for handler in node.handlers:
            new_handler_type = self.visit(handler.type) if handler.type else None
            new_handler_name = handler.name  # Exception var name - don't rename
            new_handler_body = self.transform(handler.body)
            new_handlers.append(ast.ExceptHandler(
                type=new_handler_type,
                name=new_handler_name,
                body=new_handler_body
            ))
        
        new_orelse = self.transform(node.orelse) if node.orelse else []
        new_finalbody = self.transform(node.finalbody) if node.finalbody else []
        
        return ast.Try(
            body=new_body,
            handlers=new_handlers,
            orelse=new_orelse,
            finalbody=new_finalbody
        )
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename captures without rewriting nested locals or exit points."""
        return ScopedVariableRenamer(self.rename_map).visit(copy.deepcopy(node))

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return ScopedVariableRenamer(self.rename_map).visit(node)
