from .base import BuiltinFunction
from ..valueref import wrap_value
import ast


class consume(BuiltinFunction):
    """consume(t: linear) -> void
    
    Consume a linear token, marking it as destroyed.
    The token variable becomes invalid after consumption.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'consume'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle consume(token) call
        
        consume() is a no-op at IR level. The actual consumption happens in visit_Call
        when it calls _transfer_linear_ownership on the argument.
        
        We just validate the argument and return void.
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (ownership already transferred)
            node: ast.Call node
        
        Returns:
            void
        """
        from .types import void
        
        if len(args) != 1:
            raise TypeError("consume() takes exactly 1 argument")
        
        arg_value = args[0]
        if not hasattr(arg_value, 'type_hint') or not arg_value.type_hint:
            raise TypeError(f"consume() argument must have type information (line {node.lineno})")
        
        if hasattr(arg_value.type_hint, 'is_linear') and not arg_value.type_hint.is_linear():
            raise TypeError(f"consume() requires a linear type argument (line {node.lineno})")
        
        return wrap_value(None, kind='python', type_hint=void)
    
    @classmethod
    def _parse_linear_path(cls, node: ast.AST):
        """Parse variable name and index path from AST node
        
        Examples:
            t -> ('t', ())
            t[0] -> ('t', (0,))
            t[1][0] -> ('t', (1, 0))
        
        Returns:
            (var_name, path_tuple)
        """
        path = []
        current = node
        
        while isinstance(current, ast.Subscript):
            if isinstance(current.slice, ast.Constant):
                idx = current.slice.value
                if not isinstance(idx, int):
                    raise TypeError(f"consume() requires integer index, got {type(idx).__name__}")
                path.insert(0, idx)
            elif isinstance(current.slice, ast.Index):
                if isinstance(current.slice.value, ast.Constant):
                    idx = current.slice.value.value
                    if not isinstance(idx, int):
                        raise TypeError(f"consume() requires integer index, got {type(idx).__name__}")
                    path.insert(0, idx)
                else:
                    raise TypeError("consume() requires constant integer index")
            else:
                raise TypeError("consume() requires constant integer index")
            
            current = current.value
        
        if not isinstance(current, ast.Name):
            raise TypeError(f"consume() requires variable name, got {type(current).__name__}")
        
        return current.id, tuple(path)
