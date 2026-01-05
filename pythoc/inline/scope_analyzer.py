"""
Scope analysis for inline operations

Determines which variables are:
- Parameters (declared in function signature)
- Local variables (declared with type annotation in function body)
- Captured variables (referenced or assigned without annotation, from outer scope)

By-ref capture semantics:
- Assignment WITH type annotation (x: i32 = 10) -> new local variable
- Assignment WITHOUT type annotation (x = 10) AND x exists in outer scope -> captured by-ref
- Assignment WITHOUT type annotation AND x not in outer scope -> new local variable
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
    
    Key classifications (with by-ref capture):
    - Parameters: Declared in function signature
    - Local variables: Declared with type annotation, or assigned without annotation
                       when variable doesn't exist in outer scope
    - Captured variables: Referenced or assigned (without annotation) when variable
                          exists in outer scope (by-ref capture)
    """
    
    def __init__(self, caller_context: ScopeContext):
        self.caller_context = caller_context
    
    def analyze(
        self, 
        body: List[ast.stmt], 
        params: List[ast.arg]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Analyze body to extract scope information with by-ref capture semantics
        
        Args:
            body: Function body statements
            params: Function parameters
            
        Returns:
            (captured_vars, local_vars, param_vars)
            
        Example (by-ref capture):
            def outer():
                x = 1
                def inner(y):
                    x = 2      # captured by-ref (no annotation, x exists in outer)
                    z: i32 = 3 # local (has annotation)
                    w = 4      # local (no annotation, w not in outer)
                    return x + y + z + w
                    
            For inner():
                captured_vars = {'x'}     # by-ref from outer scope
                local_vars = {'z', 'w'}   # declared locally
                param_vars = {'y'}        # parameter
        """
        # Extract param names
        param_vars = {p.arg for p in params}
        
        # Find variables with different assignment types
        collector = AssignmentCollector()
        for stmt in body:
            collector.visit(stmt)
        
        # Variables assigned with type annotation are always local
        annotated_vars = collector.annotated_assigned
        
        # Variables assigned without annotation
        unannotated_vars = collector.unannotated_assigned
        
        # Find all referenced variables
        referenced_vars = self._find_referenced_vars(body)
        
        # Determine captured vs local based on by-ref semantics
        captured_vars = set()
        local_vars = set()
        
        # Annotated assignments are always local (new variable declaration)
        for var in annotated_vars:
            if var not in param_vars:
                local_vars.add(var)
        
        # Unannotated assignments: check if variable exists in outer scope
        for var in unannotated_vars:
            if var in param_vars:
                continue  # Parameter, not local or captured
            if var in local_vars:
                continue  # Already declared as local via annotated assignment
            if self.caller_context.has_variable(var):
                # Variable exists in outer scope -> captured by-ref
                captured_vars.add(var)
            else:
                # Variable doesn't exist in outer scope -> new local
                local_vars.add(var)
        
        # Referenced-only variables (not assigned): captured if in outer scope
        for var in referenced_vars:
            if var in param_vars or var in local_vars or var in captured_vars:
                continue
            if self.caller_context.has_variable(var):
                captured_vars.add(var)
        
        return captured_vars, local_vars, param_vars
    
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
    """
    Collect all variables assigned in AST, distinguishing between:
    - annotated_assigned: Variables with type annotation (x: i32 = 10)
    - unannotated_assigned: Variables without type annotation (x = 10)
    
    This distinction is critical for by-ref capture semantics.
    """
    
    def __init__(self):
        self.annotated_assigned = set()    # x: type = value
        self.unannotated_assigned = set()  # x = value (no annotation)
    
    def visit_Assign(self, node: ast.Assign):
        """x = value (unannotated)"""
        for target in node.targets:
            self._collect_names(target, annotated=False)
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """x: type = value (annotated - always creates new local)"""
        self._collect_names(node.target, annotated=True)
        self.generic_visit(node)
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """x += value (unannotated, modifies existing)"""
        self._collect_names(node.target, annotated=False)
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """for x in iter (unannotated)"""
        self._collect_names(node.target, annotated=False)
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With):
        """with expr as x (unannotated)"""
        for item in node.items:
            if item.optional_vars:
                self._collect_names(item.optional_vars, annotated=False)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """def f(): ... (annotated - function name is a new binding)"""
        self.annotated_assigned.add(node.name)
        # Don't visit function body - it's a separate scope
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """class C: ... (annotated - class name is a new binding)"""
        self.annotated_assigned.add(node.name)
        # Don't visit class body - it's a separate scope
    
    def _collect_names(self, node: ast.expr, annotated: bool):
        """Recursively collect names from target expression"""
        target_set = self.annotated_assigned if annotated else self.unannotated_assigned
        if isinstance(node, ast.Name):
            target_set.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_names(elt, annotated)
        elif isinstance(node, ast.Starred):
            self._collect_names(node.value, annotated)
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
