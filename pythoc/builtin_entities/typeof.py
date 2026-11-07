from .base import BuiltinFunction, BuiltinEntity
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
        """Support direct Python-level calls: typeof(5), typeof(i32)
        
        Python-level semantics:
        - typeof(5)   -> pyconst[5]
        - typeof(T)   -> T        (when T is a PC BuiltinEntity type)
        """
        if isinstance(value_or_type, type) and issubclass(value_or_type, BuiltinEntity):
            # PC type class: typeof(i32) -> i32
            return value_or_type
        elif isinstance(value_or_type, BuiltinEntity):
            # PC type instance: typeof(i32()) -> its type
            return type(value_or_type)
        else:
            # Python value: typeof(5) -> pyconst[5]
            from .python_type import pyconst
            return pyconst[value_or_type]
    
    _borrows_args = True

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Determine the type of an evaluated value without replaying it."""
        from .python_type import PythonType
        from ..valueref import wrap_python_constant

        if len(args) != 1:
            logger.error(f"typeof() takes exactly 1 argument ({len(args)} given)",
                         node=node, exc_type=TypeError)
        value = args[0]
        if value.is_python_value():
            python_value = value.get_python_value()
            if (isinstance(python_value, BuiltinEntity)
                    or isinstance(python_value, type)
                    and (getattr(python_value, 'can_be_type', lambda: False)()
                         or getattr(python_value, '_is_struct', False)
                         or getattr(python_value, '_is_enum', False)
                         or getattr(python_value, '_is_union', False))):
                result = visitor.type_resolver.resolve_type_value(value)
            else:
                result = PythonType.wrap(python_value, is_constant=True)
        else:
            result = value.type_hint
        if result is None:
            logger.error('typeof() cannot determine the argument type',
                         node=node, exc_type=TypeError)
        return wrap_python_constant(result)
