"""
Scope analysis for inline operations

Determines which variables are:
- Parameters (declared in function signature)
- Local variables (assigned within function body)
- Captured variables (referenced but not local/param, from outer scope)
"""

import ast
from typing import Set, Tuple, List
from dataclasses import dataclass


@dataclass
class ScopeContext:
    """
    Represents the caller's scope context
    
    Used to determine if a referenced variable exists in outer scope
    """
    available_vars: Set[str]
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable is available in this scope"""
        return name in self.available_vars
    
    @classmethod
    def from_var_list(cls, vars: List[str]) -> 'ScopeContext':
        """Create context from list of variable names"""
        return cls(available_vars=set(vars))
    
    @classmethod
    def empty(cls) -> 'ScopeContext':
        """Create empty context (no variables available)"""
        return cls(available_vars=set())


class ScopeAnalyzer:
    """
    Analyze scope to determine captured/local/parameter variables
    
    Key classifications:
    - Parameters: Declared in function signature
    - Local variables: Assigned within function body
    - Captured variables: Referenced but not local/param (from outer scope)
    """
    
    def __init__(self, caller_context: ScopeContext):
        self.caller_context = caller_context
    
    def analyze(
        self, 
        body: List[ast.stmt], 
        params: List[ast.arg]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Analyze body to extract scope information
        
        Args:
            body: Function body statements
            params: Function parameters
            
        Returns:
            (captured_vars, local_vars, param_vars)
            
        Example:
            def outer():
                x = 1
                def inner(y):
                    z = 2
                    return x + y + z
                    
            For inner():
                captured_vars = {'x'}     # from outer scope
                local_vars = {'z'}        # assigned in inner
                param_vars = {'y'}        # parameter
        """
        # Extract param names
        param_vars = {p.arg for p in params}
        
        # Find all assigned variables (locals)
        local_vars = self._find_assigned_vars(body)
        
        # Find all referenced variables
        referenced_vars = self._find_referenced_vars(body)
        
        # Captured = referenced but not local/param and exists in outer scope
        captured_vars = set()
        for var in referenced_vars:
            if var not in local_vars and var not in param_vars:
                if self.caller_context.has_variable(var):
                    captured_vars.add(var)
        
        return captured_vars, local_vars, param_vars
    
    def _find_assigned_vars(self, body: List[ast.stmt]) -> Set[str]:
        """
        Find all variables assigned in body
        
        Includes:
        - Assignment targets (x = 1)
        - Annotated assignments (x: int = 1)
        - For loop targets (for x in ...)
        - With statement targets (with f as x)
        - Function/class definitions (def f() / class C)
        """
        visitor = AssignmentCollector()
        for stmt in body:
            visitor.visit(stmt)
        return visitor.assigned
    
    def _find_referenced_vars(self, body: List[ast.stmt]) -> Set[str]:
        """
        Find all variables referenced in body
        
        Includes all Name nodes (Load context)
        """
        visitor = ReferenceCollector()
        for stmt in body:
            visitor.visit(stmt)
        return visitor.referenced


class AssignmentCollector(ast.NodeVisitor):
    """Collect all variables assigned in AST"""
    
    def __init__(self):
        self.assigned = set()
    
    def visit_Assign(self, node: ast.Assign):
        """x = value"""
        for target in node.targets:
            self._collect_names(target)
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """x: type = value"""
        self._collect_names(node.target)
        self.generic_visit(node)
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """x += value"""
        self._collect_names(node.target)
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """for x in iter"""
        self._collect_names(node.target)
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With):
        """with expr as x"""
        for item in node.items:
            if item.optional_vars:
                self._collect_names(item.optional_vars)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """def f(): ..."""
        self.assigned.add(node.name)
        # Don't visit function body - it's a separate scope
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """class C: ..."""
        self.assigned.add(node.name)
        # Don't visit class body - it's a separate scope
    
    def _collect_names(self, node: ast.expr):
        """Recursively collect names from target expression"""
        if isinstance(node, ast.Name):
            self.assigned.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_names(elt)
        elif isinstance(node, ast.Starred):
            self._collect_names(node.value)
        # Ignore Subscript, Attribute (not creating new variables)


class ReferenceCollector(ast.NodeVisitor):
    """Collect all variables referenced in AST"""
    
    def __init__(self):
        self.referenced = set()
    
    def visit_Name(self, node: ast.Name):
        """Any name reference"""
        self.referenced.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Don't visit nested function bodies
        
        Nested functions have their own scope, their references
        don't count as references in the outer function.
        """
        # Visit decorators and annotations (they're in outer scope)
        for dec in node.decorator_list:
            self.visit(dec)
        if node.returns:
            self.visit(node.returns)
        for arg in node.args.args:
            if arg.annotation:
                self.visit(arg.annotation)
        # Don't visit body
    
    def visit_Lambda(self, node: ast.Lambda):
        """
        Don't visit lambda bodies
        
        Same reasoning as FunctionDef
        """
        # Visit default arguments (they're in outer scope)
        for default in node.args.defaults:
            self.visit(default)
        # Don't visit body
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Don't visit class bodies (separate scope)"""
        # Visit decorators and bases
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        # Don't visit body
