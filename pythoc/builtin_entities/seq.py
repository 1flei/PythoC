from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
import ast


class seq(BuiltinFunction):
    """seq(end) or seq(start, end) - Iterator for integer sequences
    
    Uses yield-based implementation for efficient inlining.
    Returns Python range for constant iteration (compile-time unrolling).
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'seq'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle seq(end) or seq(start, end) call
        
        Returns:
        - Python range for constant arguments (enables compile-time unrolling)
        - Calls yield-based counter_yield or seq_yield for runtime iteration
        
        Note: step parameter is not supported yet for performance reasons.
        """
        from .python_type import PythonType
        from ..std.seq import counter, counter_range, counter_range_step

        def all_python(args):
            return all(arg.is_python_value() or arg is None for arg in args)

        if all_python(args):
            vals = [arg.get_python_value() for arg in args]
            py_range = range(*vals)
            return wrap_value(
                py_range,
                kind="python",
                type_hint=PythonType.wrap(py_range, is_constant=True)
            )
        
        if len(args) == 1:
            return counter.handle_call(visitor, func_ref, args, node)
        elif len(args) == 2:
            return counter_range.handle_call(visitor, func_ref, args, node)
        elif len(args) == 3:
            return counter_range_step.handle_call(visitor, func_ref, args, node)
        else:
            logger.error("seq() takes 1 to 3 arguments", node=node, exc_type=ValueError)
