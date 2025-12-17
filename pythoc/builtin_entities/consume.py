from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
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
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle consume(token) call
        
        consume() is a no-op at IR level. The actual consumption happens in visit_Call
        when it calls _transfer_linear_ownership on the argument.
        
        We just validate the argument and return void.
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (consume function)
            args: Pre-evaluated arguments (ownership already transferred)
            node: ast.Call node
        
        Returns:
            void
        """
        from .types import void
        
        if len(args) != 1:
            logger.error("consume() takes exactly 1 argument", node=node, exc_type=TypeError)
        
        arg_value = args[0]
        if not hasattr(arg_value, 'type_hint') or not arg_value.type_hint:
            logger.error(f"consume() argument must have type information (line {node.lineno})",
                        node=node, exc_type=TypeError)
        
        if hasattr(arg_value.type_hint, 'is_linear') and not arg_value.type_hint.is_linear():
            logger.error(f"consume() requires a linear type argument (line {node.lineno})",
                        node=node, exc_type=TypeError)
        
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
                    logger.error(f"consume() requires integer index, got {type(idx).__name__}",
                                node=current, exc_type=TypeError)
                path.insert(0, idx)
            elif isinstance(current.slice, ast.Index):
                if isinstance(current.slice.value, ast.Constant):
                    idx = current.slice.value.value
                    if not isinstance(idx, int):
                        logger.error(f"consume() requires integer index, got {type(idx).__name__}",
                                    node=current, exc_type=TypeError)
                    path.insert(0, idx)
                else:
                    logger.error("consume() requires constant integer index",
                                node=current, exc_type=TypeError)
            else:
                logger.error("consume() requires constant integer index",
                            node=current, exc_type=TypeError)
            
            current = current.value
        
        if not isinstance(current, ast.Name):
            logger.error(f"consume() requires variable name, got {type(current).__name__}",
                        node=current, exc_type=TypeError)
        
        return current.id, tuple(path)
