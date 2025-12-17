"""
Yield transformation for generator functions

Transforms yield-based generator functions into inline continuation placeholders.

Design:
- Detects yield statements in function body
- Creates a placeholder that triggers inline expansion at call sites
- Zero runtime overhead: all yields are inlined during compilation
- No vtable generation, no runtime iterator overhead
"""

import ast
from typing import Optional, List


class YieldAnalyzer(ast.NodeVisitor):
    """Analyze function to detect yield patterns"""
    
    def __init__(self):
        self.has_yield = False
        self.yield_nodes: List[ast.Yield] = []
        
    def visit_Yield(self, node: ast.Yield):
        """Record yield statement"""
        self.has_yield = True
        self.yield_nodes.append(node)
        self.generic_visit(node)


def analyze_yield_function(func_ast: ast.FunctionDef) -> Optional[YieldAnalyzer]:
    """
    Analyze a function to determine if it's a yield-based generator
    
    Args:
        func_ast: Function AST node
        
    Returns:
        YieldAnalyzer if function contains yield, None otherwise
    """
    analyzer = YieldAnalyzer()
    analyzer.visit(func_ast)
    
    if not analyzer.has_yield:
        return None
    
    return analyzer


def create_yield_iterator_wrapper(func, func_ast, analyzer, user_globals, source_file, registry):
    """
    Create iterator wrapper that triggers inline expansion at call sites
    
    This creates a placeholder that will force inlining - yield functions
    MUST be inlined and cannot be called at runtime.
    
    The placeholder contains metadata (_yield_inline_info) that triggers
    yield inlining in the AST visitor when used in a for loop.
    """
    # All yield functions must be inlined - create a placeholder
    def placeholder_wrapper(*args, **kwargs):
        raise RuntimeError(
            f"Function '{func.__name__}' with yield requires inlining. "
            f"Cannot be called at runtime without inlining optimization."
        )
    
    placeholder_wrapper._is_yield_generated = True
    placeholder_wrapper.__name__ = func.__name__
    placeholder_wrapper._original_ast = func_ast
    
    # Add handle_call that triggers yield inlining
    def handle_call(visitor, func_ref, args, node):
        from ..valueref import wrap_value
        from ..builtin_entities.python_type import PythonType
        # Create a dummy result that will trigger inlining in for loop
        # The actual value doesn't matter - only _yield_inline_info is used
        result = wrap_value(placeholder_wrapper, kind='python', type_hint=PythonType(placeholder_wrapper))
        result._yield_inline_info = {
            'func_obj': func,  # Use original function to access its __globals__
            'placeholder': placeholder_wrapper,
            'original_ast': func_ast,
            'call_node': node,
            'call_args': args
        }
        return result
    
    placeholder_wrapper.handle_call = handle_call
    return placeholder_wrapper


