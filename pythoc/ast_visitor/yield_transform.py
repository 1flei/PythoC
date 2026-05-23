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


def _make_yield_placeholder(func, func_ast, callee_globals, effect_suffix=None):
    """Create a yield placeholder with explicit callee globals."""
    def placeholder_wrapper(*args, **kwargs):
        raise RuntimeError(
            f"Function '{func.__name__}' with yield requires inlining. "
            f"Cannot be called at runtime without inlining optimization."
        )

    placeholder_wrapper._is_yield_generated = True
    placeholder_wrapper.__name__ = func.__name__
    placeholder_wrapper._original_ast = func_ast
    placeholder_wrapper._yield_func_obj = func
    placeholder_wrapper._yield_callee_globals = dict(callee_globals or {})
    placeholder_wrapper._effect_suffix = effect_suffix
    if effect_suffix:
        placeholder_wrapper._mangled_name = f"{func.__name__}_{effect_suffix}"

    def handle_call(visitor, func_ref, args, node):
        from ..valueref import wrap_value
        from ..builtin_entities.python_type import PythonType
        result = wrap_value(placeholder_wrapper, kind='python', type_hint=PythonType(placeholder_wrapper))
        result._yield_inline_info = {
            'func_obj': func,
            'callee_globals': placeholder_wrapper._yield_callee_globals,
            'placeholder': placeholder_wrapper,
            'original_ast': func_ast,
            'call_node': node,
            'call_args': args
        }
        return result

    placeholder_wrapper.handle_call = handle_call
    return placeholder_wrapper


def _specialize_compiled_value(value, suffix, effect_overrides):
    if not callable(value) or not getattr(value, '_is_compiled', False):
        return value
    if getattr(value, '_pc_effect_impl', False):
        return value
    from ..effect import resolve_effect_wrapper
    specialized = resolve_effect_wrapper(
        value, suffix=suffix, effect_overrides=effect_overrides,
    )
    return specialized


def _copy_namespace_with_specialized_values(value, suffix, effect_overrides):
    value_dict = getattr(value, '__dict__', None)
    if value_dict is None or not hasattr(value_dict, 'items'):
        return value
    if isinstance(value, type) and (
        hasattr(value, 'get_llvm_type')
        or hasattr(value, '_field_types')
        or hasattr(value, '_enum_info')
    ):
        return value

    replacements = {}
    for attr_name in value_dict:
        if attr_name.startswith('__'):
            continue
        try:
            attr_value = getattr(value, attr_name)
        except Exception:
            continue
        specialized = _specialize_compiled_value(attr_value, suffix, effect_overrides)
        if specialized is not attr_value:
            replacements[attr_name] = specialized

    if not replacements:
        return value

    from types import ModuleType, SimpleNamespace
    if isinstance(value, ModuleType):
        return value
    if isinstance(value, type):
        attrs = {}
        for attr_name in value_dict:
            if attr_name.startswith('__'):
                continue
            try:
                attrs[attr_name] = getattr(value, attr_name)
            except Exception:
                pass
        attrs.update(replacements)
        return SimpleNamespace(**attrs)

    import copy
    try:
        cloned = copy.copy(value)
        for attr_name, specialized in replacements.items():
            setattr(cloned, attr_name, specialized)
        return cloned
    except Exception:
        attrs = {}
        for attr_name in value_dict:
            if attr_name.startswith('__'):
                continue
            try:
                attrs[attr_name] = getattr(value, attr_name)
            except Exception:
                pass
        attrs.update(replacements)
        return SimpleNamespace(**attrs)


def specialize_yield_wrapper(placeholder, suffix, effect_overrides):
    """Create an effect-specialized yield placeholder for import-time use."""
    if not suffix or not effect_overrides:
        return placeholder
    cache = getattr(placeholder, '_effect_specialized_cache', None)
    if cache is None:
        cache = {}
        placeholder._effect_specialized_cache = cache
    if suffix in cache:
        return cache[suffix]

    globals_dict = dict(getattr(placeholder, '_yield_callee_globals', {}) or {})
    for name, value in list(globals_dict.items()):
        specialized = _specialize_compiled_value(value, suffix, effect_overrides)
        if specialized is value:
            specialized = _copy_namespace_with_specialized_values(
                value, suffix, effect_overrides,
            )
        globals_dict[name] = specialized

    func = getattr(placeholder, '_yield_func_obj')
    func_ast = getattr(placeholder, '_original_ast')
    specialized_placeholder = _make_yield_placeholder(
        func, func_ast, globals_dict, effect_suffix=suffix,
    )
    cache[suffix] = specialized_placeholder
    return specialized_placeholder


def create_yield_iterator_wrapper(func, func_ast, analyzer, user_globals, source_file, registry):
    """
    Create iterator wrapper that triggers inline expansion at call sites
    
    This creates a placeholder that will force inlining - yield functions
    MUST be inlined and cannot be called at runtime.
    
    The placeholder contains metadata (_yield_inline_info) that triggers
    yield inlining in the AST visitor when used in a for loop.
    """
    return _make_yield_placeholder(func, func_ast, user_globals)


