"""
Callee module for inline functions

This module defines inline functions that use module-level globals.
When these functions are inlined at the caller site, the caller
does NOT have access to these globals - they must be captured
from this module's __globals__.
"""

from pythoc import i32, void
from pythoc.decorators.inline import inline

# =============================================================================
# Module-level definitions (only visible in THIS module)
# =============================================================================

# Type alias - caller site won't have this
LocalInt = i32

# Constant - caller site won't have this
LOCAL_MAGIC = 100

# Helper function - caller site won't have this
def _local_helper(x: int) -> int:
    """Pure Python helper for compile-time computation"""
    return x * 2


# =============================================================================
# Inline functions that depend on module globals
# =============================================================================

@inline
def inline_using_type_alias(x: LocalInt) -> LocalInt:
    """
    Inline function using LocalInt type alias.
    
    When this is inlined at caller site, the caller doesn't have LocalInt
    in its globals. The kernel must capture it from this module's __globals__.
    """
    result: LocalInt = x + LOCAL_MAGIC
    return result


@inline
def inline_using_constant(x: i32) -> i32:
    """
    Inline function using LOCAL_MAGIC constant.
    
    The constant is defined in this module, not the caller's module.
    """
    return x + LOCAL_MAGIC


@inline
def inline_using_local_alias_only() -> LocalInt:
    """
    Inline function that only uses local type alias.
    
    This tests that even simple type references are captured correctly.
    """
    val: LocalInt = 42
    return val
