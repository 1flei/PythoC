from llvmlite import ir
from .base import BuiltinFunction, BuiltinEntity
from ..valueref import wrap_value
from ..logger import logger
import ast


class typeof(BuiltinFunction):
    """typeof(x) - Get the type of a value or return a type
    
    Works in both Python preprocessing and @compile contexts:
    - Python preprocessing: typeof(5) -> pyconst[5], typeof(i32) -> i32
    - In @compile: typeof(x) -> type of x
    
    Returns:
        - Python value -> pyconst[value]
        - PC value -> its type_hint  
        - Type -> the type itself
    
    Examples:
        typeof(5)        # pyconst[5]
        typeof(100)      # pyconst[100]
        typeof(x)        # i32 (if x: i32)
        typeof(i32)      # i32
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'typeof'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    def __new__(cls, value_or_type):
        """Support direct Python-level calls: typeof(5), typeof(i32)"""
        if isinstance(value_or_type, type) and issubclass(value_or_type, BuiltinEntity):
            return value_or_type
        elif isinstance(value_or_type, BuiltinEntity):
            return value_or_type
        else:
            from .python_type import PythonType
            return PythonType.wrap(value_or_type, is_constant=True)
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle typeof(x) call in @compile context
        
        Strategy:
        1. If arg is a type annotation (Name/Attribute/Subscript) -> parse as type
        2. If arg evaluates to ValueRef with Python value -> wrap as pyconst
        3. If arg evaluates to ValueRef with LLVM value -> return its type_hint
        """
        if len(node.args) != 1:
            logger.error(f"typeof() takes exactly 1 argument ({len(node.args)} given)",
                        node=node, exc_type=TypeError)
        
        arg_node = node.args[0]
        
        try:
            pc_type = visitor.type_resolver.parse_annotation(arg_node)
            if pc_type is not None:
                from .python_type import PythonType
                return wrap_value(pc_type, kind="python", type_hint=PythonType.wrap(pc_type))
        except:
            pass
        
        value_ref = visitor.visit_expression(arg_node)
        
        if value_ref.is_python_value():
            from .python_type import PythonType
            py_value = value_ref.value
            pyconst_type = PythonType.wrap(py_value, is_constant=True)
            return wrap_value(pyconst_type, kind="python", type_hint=PythonType.wrap(pyconst_type))
        
        if value_ref.type_hint is not None:
            from .python_type import PythonType
            return wrap_value(value_ref.type_hint, kind="python", type_hint=PythonType.wrap(value_ref.type_hint))
        
        logger.error(f"typeof() cannot determine type of {ast.dump(arg_node)}",
                    node=node, exc_type=TypeError)
