"""
Global unique ID generator for the compiler.

This module provides a centralized ID generation mechanism to ensure
uniqueness across different compiler components (inline, labels, temporaries, etc.).
"""

import itertools


class IDGenerator:
    """Centralized ID generator with thread-safe incremental counter.

    Backed by itertools.count: next() is atomic under the GIL.
    """

    def __init__(self):
        self._counter = itertools.count()

    def next_id(self) -> int:
        """Get next unique ID.

        Returns:
            int: A unique incremental ID
        """
        return next(self._counter)

    def reset(self):
        """Reset counter to 0. Use with caution - mainly for testing."""
        self._counter = itertools.count()

    def peek(self) -> int:
        """Peek at the next ID without incrementing.

        Returns:
            int: The next ID that would be returned
        """
        # CPython itertools.count reduce state: (current,) for default step
        return self._counter.__reduce__()[1][0]


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
