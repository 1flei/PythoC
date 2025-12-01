from llvmlite import ir
from .base import BuiltinFunction
from ..valueref import wrap_value
import ast


class char(BuiltinFunction):
    """char(value) - Convert string or int to i8 character
    
    Converts Python values to i8 type:
    - char("abc") -> i8(ord('a')) - first character of string
    - char("s") -> i8(ord('s')) - single character
    - char("") -> i8(0) - empty string returns null terminator
    - char(48) -> i8(48) - int directly converted to i8
    
    Only accepts int or str Python values.
    Raises TypeError for other types.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'char'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle char(value) call
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs
            node: ast.Call node
            
        Returns:
            ValueRef containing i8 value
        """
        from .python_type import PythonType
        
        if len(args) != 1:
            raise TypeError(f"char() takes exactly 1 argument ({len(args)} given)")
        
        arg = args[0]
        
        if not arg.is_python_value():
            raise TypeError(f"char() only accepts Python int or str values, got {arg.type_hint}")
        
        py_value = arg.get_python_value()
        
        if isinstance(py_value, str):
            if len(py_value) == 0:
                char_value = 0
            else:
                char_value = ord(py_value[0])
        elif isinstance(py_value, int):
            char_value = py_value
        else:
            raise TypeError(f"char() only accepts int or str, got {type(py_value).__name__}")

        python_type = PythonType.wrap(char_value, is_constant=True)
        return wrap_value(char_value, kind="python", type_hint=python_type)
