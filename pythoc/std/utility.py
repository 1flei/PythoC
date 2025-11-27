"""
Utility functions for PC standard library

Provides basic utility functions like move() for ownership transfer.
"""
from ..decorators import compile
from ..builtin_entities import linear


@compile
def move(x: linear) -> linear:
    """Identity function for transferring linear token ownership
    
    move() is a no-op at runtime (just returns its argument),
    but signals intent to transfer ownership in linear type system.
    
    Args:
        x: Linear token to transfer ownership of
        
    Returns:
        The same token (ownership transferred)
    """
    return x
