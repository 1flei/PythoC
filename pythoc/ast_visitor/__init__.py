"""
AST Visitor module for LLVM IR generation

This module provides the LLVMIRVisitor class which traverses Python AST
and generates LLVM IR code. The visitor is organized into multiple mixins
for better code organization:

- base.py: Core visitor class with initialization and context management
- expressions_mixin.py: Expression evaluation (BinOp, Compare, etc.)
- calls_mixin.py: Function calls and related operations
- subscripts_mixin.py: Subscript and attribute access
- statements_mixin.py: Control flow statements (if, while, for, etc.)
- assignments_mixin.py: Variable assignments and initialization
- functions_mixin.py: Function definitions
- helpers_mixin.py: Utility methods and type conversions
"""

from .visitor_impl import LLVMIRVisitor

__all__ = ['LLVMIRVisitor']
