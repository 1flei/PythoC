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
    3. Adding flag checks after all loops (for ReturnExitRule)
    4. Preserving control flow structure
    
    Usage:
        transformer = InlineBodyTransformer(exit_rule, rename_map, flag_var)
        new_body = transformer.transform(original_body)
    """
    
    def __init__(self, exit_rule: 'ExitPointRule', rename_map: Dict[str, str], flag_var: Optional[str] = None):
        """
        Args:
            exit_rule: Rule for transforming exit points
            rename_map: Mapping of old variable names to new names
            flag_var: Flag variable for return propagation (only for ReturnExitRule)
        """
        self.exit_rule = exit_rule
        self.rename_map = rename_map
        self.exit_types = exit_rule.get_exit_node_types()
        self.context = InlineContext(rename_map=rename_map)
        self.flag_var = flag_var
    
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
    
    def visit_While(self, node: ast.While) -> List[ast.stmt]:
        """Transform while loop (recursively transform body)
        
        Returns a list containing:
        1. The transformed while loop
        2. Flag check: if flag_var: break (for ReturnExitRule)
        """
        new_test = self.visit(node.test)
        new_body = self.transform(node.body)
        new_orelse = self.transform(node.orelse) if node.orelse else []
        
        result = [ast.While(test=new_test, body=new_body, orelse=new_orelse)]
        
        # Add flag check after loop (Rule 2)
        if self.flag_var:
            flag_check = ast.If(
                test=ast.Name(id=self.flag_var, ctx=ast.Load()),
                body=[ast.Break()],
                orelse=[]
            )
            result.append(flag_check)
        
        return result
    
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
    
    def visit_For(self, node: ast.For) -> List[ast.stmt]:
        """Transform for loop (recursively transform body)
        
        Returns a list containing:
        1. The transformed for loop
        2. Flag check: if flag_var: break (for ReturnExitRule)
        """
        new_target = self.visit(node.target)
        new_iter = self.visit(node.iter)
        new_body = self.transform(node.body)
        new_orelse = self.transform(node.orelse) if node.orelse else []
        
        result = [ast.For(
            target=new_target, 
            iter=new_iter, 
            body=new_body, 
            orelse=new_orelse
        )]
        
        # Add flag check after loop (Rule 2)
        if self.flag_var:
            flag_check = ast.If(
                test=ast.Name(id=self.flag_var, ctx=ast.Load()),
                body=[ast.Break()],
                orelse=[]
            )
            result.append(flag_check)
        
        return result
    
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
        """Transform nested function definition (rename the function name if needed)
        
        Nested functions are treated as local variables, so their names get renamed.
        We also need to rename captured variables from outer scope in the function body,
        but we must NOT transform return statements (those belong to the nested function).
        """
        # Rename the function name if it's in the rename map
        new_name = self.rename_map.get(node.name, node.name)
        
        # Create a transformer that only renames variables, without exit transformation
        # We use a simple Name visitor instead of full transformer
        class VariableRenamer(ast.NodeTransformer):
            def __init__(self, rename_map):
                self.rename_map = rename_map
            
            def visit_Name(self, node):
                if node.id in self.rename_map:
                    return ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
                return node
            
            def visit_FunctionDef(self, node):
                # Don't recurse into nested nested functions
                # They will be handled when their closure is called
                return node
        
        # Rename variables in the body but preserve return statements
        renamer = VariableRenamer(self.rename_map)
        new_body = [renamer.visit(stmt) for stmt in node.body]
        
        # Return a new FunctionDef with renamed name and body with renamed variables
        return ast.FunctionDef(
            name=new_name,
            args=node.args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment if hasattr(node, 'type_comment') else None
        )
        
        new_orelse = self.transform(node.orelse) if node.orelse else []
        new_finalbody = self.transform(node.finalbody) if node.finalbody else []
        
        return ast.Try(
            body=new_body,
            handlers=new_handlers,
            orelse=new_orelse,
            finalbody=new_finalbody
        )
