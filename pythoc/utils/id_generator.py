"""
Global unique ID generator for the compiler.

This module provides a centralized ID generation mechanism to ensure
uniqueness across different compiler components (inline, labels, temporaries, etc.).
"""


class IDGenerator:
    """Centralized ID generator with thread-safe incremental counter."""
    
    def __init__(self):
        self._counter = 0
    
    def next_id(self) -> int:
        """Get next unique ID.
        
        Returns:
            int: A unique incremental ID
        """
        current_id = self._counter
        self._counter += 1
        return current_id
    
    def reset(self):
        """Reset counter to 0. Use with caution - mainly for testing."""
        self._counter = 0
    
    def peek(self) -> int:
        """Peek at the next ID without incrementing.
        
        Returns:
            int: The next ID that would be returned
        """
        return self._counter


# Global singleton instance
_global_id_generator = IDGenerator()


def get_next_id() -> int:
    """Get next unique ID from global generator.
    
    This is the main function that should be used throughout the compiler
    for any component that needs unique IDs.
    
    Returns:
        int: A unique incremental ID
    """
    return _global_id_generator.next_id()


def reset_id_generator():
    """Reset the global ID generator.
    
    WARNING: This should only be used in testing scenarios.
    In production, IDs should never be reset to maintain uniqueness.
    """
    _global_id_generator.reset()


def peek_next_id() -> int:
    """Peek at the next ID without consuming it.
    
    Returns:
        int: The next ID that would be generated
    """
    return _global_id_generator.peek()
