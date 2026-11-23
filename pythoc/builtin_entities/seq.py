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
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle seq(end) or seq(start, end) call
        
        Returns:
        - Python range for constant arguments (enables compile-time unrolling)
        - Calls yield-based counter_yield or seq_yield for runtime iteration
        
        Note: step parameter is not supported yet for performance reasons.
        """
        from .python_type import PythonType
        from ..std.seq import counter, counter_range, counter_range_step

        if not 1 <= len(args) <= 3:
            logger.error('seq() takes 1 to 3 arguments', node=node, exc_type=ValueError)
        if all(arg.is_python_value() for arg in args):
            vals = [arg.get_python_value() for arg in args]
            try:
                py_range = range(*vals)
            except (TypeError, ValueError) as error:
                logger.error(f'seq(): {error}', node=node, exc_type=type(error))
            return wrap_value(
                py_range,
                kind="python",
                type_hint=PythonType.wrap(py_range, is_constant=True)
            )
        
        counter_fn = (counter, counter_range, counter_range_step)[len(args) - 1]
        return counter_fn.handle_call(visitor, func_ref, args, node)
