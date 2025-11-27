"""
Main visitor implementation - combines all mixin classes
"""

from .base import LLVMIRVisitor as BaseVisitor
from .expressions import ExpressionsMixin
from .calls import CallsMixin
from .subscripts import SubscriptsMixin
from .statements import StatementsMixin
from .assignments import AssignmentsMixin
from .functions import FunctionsMixin
from .helpers import HelpersMixin


class LLVMIRVisitor(
    ExpressionsMixin,
    CallsMixin,
    SubscriptsMixin,
    StatementsMixin,
    AssignmentsMixin,
    FunctionsMixin,
    HelpersMixin,
    BaseVisitor
):
    """Complete LLVM IR visitor combining all functionality through mixins
    
    This class inherits from multiple mixins to organize the visitor methods:
    - ExpressionsMixin: Expression evaluation (visit_BinOp, visit_Compare, etc.)
    - CallsMixin: Function calls and related operations
    - SubscriptsMixin: Subscript and attribute access
    - StatementsMixin: Control flow statements (if, while, for, etc.)
    - AssignmentsMixin: Variable assignments and initialization
    - FunctionsMixin: Function definitions
    - HelpersMixin: Utility methods and type conversions
    - BaseVisitor: Core initialization and context management
    """
    pass
