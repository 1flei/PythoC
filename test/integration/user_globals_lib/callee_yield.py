"""
Callee module for yield functions

This module defines yield functions that use module-level globals.
When these generators are iterated at the caller site, the caller
does NOT have access to these globals - they must be captured
from this module's __globals__.
"""

from pythoc import compile, i32

# =============================================================================
# Module-level definitions (only visible in THIS module)
# =============================================================================

# Type alias - caller site won't have this
YieldInt = i32

# Constant - caller site won't have this
YIELD_MAGIC = 200

# Compiled helper function - caller site won't have this
@compile
def yield_helper_add(a: i32, b: i32) -> i32:
    """Compiled helper function for yield tests"""
    return a + b


# =============================================================================
# Yield functions that depend on module globals
# =============================================================================

@compile
def yield_using_type_alias() -> YieldInt:
    """
    Yield function using YieldInt type alias.
    
    When this is iterated at caller site, the caller doesn't have YieldInt
    in its globals. The kernel must capture it from this module's __globals__.
    """
    x: YieldInt = 10
    yield x
    x = 20
    yield x


@compile
def yield_using_constant() -> i32:
    """
    Yield function using YIELD_MAGIC constant.
    
    The constant is defined in this module, not the caller's module.
    """
    yield YIELD_MAGIC
    yield YIELD_MAGIC + 50


@compile
def yield_using_helper() -> i32:
    """
    Yield function using yield_helper_add.
    
    The helper is defined in this module, not the caller's module.
    """
    yield yield_helper_add(10, 5)
    yield yield_helper_add(20, 7)
