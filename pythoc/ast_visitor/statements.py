"""
Statements mixin for LLVMIRVisitor - combines all statement handling mixins
"""

from .stmt_if import IfStatementMixin
from .stmt_loops import LoopsMixin
from .stmt_match import MatchStatementMixin
from .stmt_control import ControlFlowMixin


class StatementsMixin(
    IfStatementMixin,
    LoopsMixin,
    MatchStatementMixin,
    ControlFlowMixin
):
    """Combines all statement handling mixins"""
    pass
