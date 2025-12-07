"""
Type checking utilities using metaprogramming for easy extension.

This module dynamically generates type checking functions based on a list of type names.
Each function checks if a type has the corresponding method and calls it.
"""

# List of type check methods to generate functions for
# Add new type checks here to automatically generate corresponding functions
TYPE_CHECK_METHODS = [
    "is_struct_type",
    "is_enum_type",
    "is_pc_type",
    "is_python_value",
]


def _make_type_checker(method_name):
    """Factory function to create type checking functions.
    
    Args:
        method_name: Name of the method to check (e.g., "is_struct_type")
    
    Returns:
        A function that checks if a type has the method and calls it
    """
    def checker(t):
        """Check if type t satisfies the type predicate.
        
        Args:
            t: Type to check (should be a BuiltinEntity class or instance)
        
        Returns:
            True if type has the method and it returns True, False otherwise
        """
        if hasattr(t, method_name):
            method = getattr(t, method_name)
            if callable(method):
                return method()
        return False
    
    # Set function name and docstring for better debugging
    checker.__name__ = method_name
    checker.__doc__ = f"""Check if a type is a {method_name.replace('is_', '').replace('_type', '')} type.
    
    Args:
        t: Type to check
    
    Returns:
        True if type.{method_name}() returns True, False otherwise
    """
    
    return checker


# Dynamically generate type checking functions and add to module namespace
import sys
_current_module = sys.modules[__name__]
for method_name in TYPE_CHECK_METHODS:
    checker_func = _make_type_checker(method_name)
    setattr(_current_module, method_name, checker_func)


# Expose all generated functions in __all__
__all__ = TYPE_CHECK_METHODS.copy()