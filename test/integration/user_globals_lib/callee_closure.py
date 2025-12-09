"""
Callee module for closure-returning functions

This module defines functions that return closures using module-level globals.
When these closures are called at the caller site, the caller
does NOT have access to these globals - they must be captured
from this module's __globals__.
"""

from pythoc import compile, i32

# =============================================================================
# Module-level definitions (only visible in THIS module)
# =============================================================================

# Type alias - caller site won't have this
ClosureInt = i32

# Constant - caller site won't have this
CLOSURE_MAGIC = 300

# Compiled helper function - caller site won't have this
@compile
def closure_helper_mul(a: i32, b: i32) -> i32:
    """Compiled helper function for closure tests"""
    return a * b


# =============================================================================
# Functions that return closures using module globals
# =============================================================================

@compile
def func_with_closure_using_type_alias(base: i32) -> i32:
    """
    Function containing closure that uses ClosureInt type alias.
    
    The closure uses ClosureInt which is defined in this module.
    When the closure is inlined, it must capture ClosureInt from here.
    """
    def inner(x: ClosureInt) -> ClosureInt:
        result: ClosureInt = x + base
        return result
    
    return inner(i32(10))


@compile
def func_with_closure_using_constant(x: i32) -> i32:
    """
    Function containing closure that uses CLOSURE_MAGIC constant.
    
    The constant is defined in this module.
    """
    def inner(val: i32) -> i32:
        return val + CLOSURE_MAGIC
    
    return inner(x)


@compile
def func_with_closure_using_helper(a: i32, b: i32) -> i32:
    """
    Function containing closure that uses closure_helper_mul.
    
    The helper is defined in this module.
    """
    def inner(x: i32, y: i32) -> i32:
        return closure_helper_mul(x, y)
    
    return inner(a, b)
