"""
CFG (Control Flow Graph) module for pythoc compiler

This module provides CFG-based intermediate representation for:
- Control flow analysis
- Linear type checking

Architecture:
    Python Source -> AST -> Visitor + ControlFlowBuilder -> CFG
                                                             |
                                                             +-> LinearChecker (validation)
"""

from .graph import CFGBlock, CFGEdge, CFG
from .linear_checker import (
    LinearSnapshot,
    LinearError,
    LinearChecker,
    capture_linear_snapshot,
    restore_linear_snapshot,
    copy_snapshot,
    snapshots_compatible,
    find_snapshot_diffs,
    check_linear_types_on_cfg,
)

__all__ = [
    # Core data structures
    'CFGBlock',
    'CFGEdge', 
    'CFG',
    # Linear checking
    'LinearSnapshot',
    'LinearError',
    'LinearChecker',
    'capture_linear_snapshot',
    'restore_linear_snapshot',
    'copy_snapshot',
    'snapshots_compatible',
    'find_snapshot_diffs',
    'check_linear_types_on_cfg',
]
