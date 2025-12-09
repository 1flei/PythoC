"""
Universal Inline Kernel

A unified framework for all inlining operations in the pc compiler.
Supports @inline, closures, yield, and future extensions.
"""

from .kernel import InlineKernel, InlineOp, InlineResult
from .exit_rules import ExitPointRule, ReturnExitRule, YieldExitRule, MacroExitRule
from .scope_analyzer import ScopeAnalyzer, ScopeContext
from .transformers import InlineBodyTransformer
from .yield_adapter import YieldInlineAdapter
from .inline_adapter import InlineAdapter
from .closure_adapter import ClosureAdapter

__all__ = [
    'InlineKernel',
    'InlineOp',
    'InlineResult',
    'ExitPointRule',
    'ReturnExitRule',
    'YieldExitRule',
    'MacroExitRule',
    'ScopeAnalyzer',
    'ScopeContext',
    'InlineBodyTransformer',
    'YieldInlineAdapter',
    'InlineAdapter',
    'ClosureAdapter',
]
