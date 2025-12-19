from llvmlite import ir
import ast
from .base import BuiltinType
from ..valueref import wrap_value, ensure_ir
from ..logger import logger

# Array type (exact copy from entities_impl.py)
class array(BuiltinType):
    """Array type - supports array[T, N] and array[T, N, M, ...] syntax"""
    _is_signed = False
    element_type = None  # Element type
    dimensions = None    # Tuple of dimensions (N,) or (N, M, ...)

    @classmethod
    def is_array(cls) -> bool:
        return True
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes array type for FFI.
        
        Returns ctypes array type (element_ctype * total_size).
        """
        import ctypes
        
        if cls.element_type is None or cls.dimensions is None:
            return ctypes.c_void_p
        
        # Get element ctypes type
        if hasattr(cls.element_type, 'get_ctypes_type'):
            elem_ctype = cls.element_type.get_ctypes_type()
        else:
            return ctypes.c_void_p
        
        if elem_ctype is None:
            return ctypes.c_void_p
        
        # Build nested array type for multi-dimensional arrays
        # array[i32, 2, 3] -> (c_int32 * 3) * 2
        result_type = elem_ctype
        for dim in reversed(cls.dimensions):
            result_type = result_type * dim
        
        return result_type
    
    @classmethod
    def get_name(cls) -> str:
        return 'array'
    
    @classmethod
    def get_type_id(cls) -> str:
        """Generate unique type ID for array types."""
        if cls.element_type and cls.dimensions:
            # Import here to avoid circular dependency
            from ..type_id import get_type_id
            elem_id = get_type_id(cls.element_type)
            dims_str = '_'.join(str(d) for d in cls.dimensions)
            return f'A{elem_id}_{dims_str}'
        return 'Ax'  # unknown array
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM array type
        
        Args:
            module_context: Optional IR module context (passed to element type)
        """
        if cls.element_type is None or cls.dimensions is None:
            logger.error("array type requires element type and dimensions", node=None, exc_type=TypeError)
        
        # Get element LLVM type
        if hasattr(cls.element_type, 'get_llvm_type'):
            elem_llvm = cls.element_type.get_llvm_type(module_context)
        elif isinstance(cls.element_type, ir.Type):
            # ANTI-PATTERN: element_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"array.get_llvm_type: element_type is raw LLVM type {cls.element_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None, exc_type=TypeError)
        else:
            logger.error(f"array.get_llvm_type: unknown element type {cls.element_type}",
                        node=None, exc_type=TypeError)
        
        # Build nested array type for multi-dimensional arrays
        # array[i32, 2, 3] -> [2 x [3 x i32]]
        result_type = elem_llvm
        for dim in reversed(cls.dimensions):
            result_type = ir.ArrayType(result_type, dim)
        
        return result_type
    
    @classmethod
    def _parse_dimensions(cls, dim_nodes, user_globals=None):
        """Parse array dimensions from AST nodes
        
        Args:
            dim_nodes: List of AST nodes representing dimensions
            user_globals: Optional dict to resolve variable names
            
        Returns:
            List of dimension values (integers)
        """
        dimensions = []
        for dim_node in dim_nodes:
            dim_value = None
            if isinstance(dim_node, ast.Constant):
                dim_value = dim_node.value
            elif isinstance(dim_node, ast.Num):  # Python 3.6 compatibility
                dim_value = dim_node.n
            elif isinstance(dim_node, ast.Name) and user_globals is not None:
                # Try to resolve the name from user_globals
                if dim_node.id in user_globals:
                    dim_value = user_globals[dim_node.id]
                    if not isinstance(dim_value, int):
                        logger.error(f"array dimension '{dim_node.id}' must be an integer, got {type(dim_value)}",
                                    node=dim_node, exc_type=TypeError)
                else:
                    logger.error(f"array dimension '{dim_node.id}' not found in scope",
                                node=dim_node, exc_type=TypeError)
            else:
                logger.error(f"array dimensions must be constants or variable names, got {ast.dump(dim_node)}",
                            node=dim_node, exc_type=TypeError)
            
            if dim_value is None:
                logger.error(f"Failed to resolve array dimension: {ast.dump(dim_node)}",
                            node=dim_node, exc_type=TypeError)
            
            dimensions.append(dim_value)
        return dimensions
    
    @classmethod
    def get_size_bytes(cls) -> int:
        """Get size in bytes"""
        if cls.element_type is None or cls.dimensions is None:
            return 0
        
        # Get element size
        if hasattr(cls.element_type, 'get_size_bytes'):
            elem_size = cls.element_type.get_size_bytes()
        else:
            elem_size = 4  # Default
        
        # Calculate total size
        total_elements = 1
        for dim in cls.dimensions:
            total_elements *= dim
        
        return elem_size * total_elements
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # array[T, N]() for zero-initialization
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle array[T, N]() for zero-initialization
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the array type itself)
            args: Pre-evaluated arguments (should be empty)
            node: Original ast.Call node
        """
        if cls.element_type is None or cls.dimensions is None:
            logger.error("array type requires element type and dimensions", node=None, exc_type=TypeError)
        
        if len(args) != 0:
            logger.error(f"array[T, N]() takes no arguments ({len(args)} given)", node=node, exc_type=TypeError)
        # Get LLVM array type
        array_type = cls.get_llvm_type()

        from ..type_converter import TypeConverter
        converter = TypeConverter(visitor)

        zero_array = converter.create_zero_constant(array_type)
        return wrap_value(zero_array, kind="value", type_hint=cls)

    @classmethod
    def get_decay_pointer_type(cls):
        """Get the pointer type that this array should decay to.
        
        Examples:
            array[i32, 10] -> ptr[i32]
            array[array[i32, 5], 3] -> ptr[array[i32, 5]]
        
        Returns:
            ptr[element_type] specialized class
        
        Raises:
            TypeError: If element_type is None
        """
        if cls.element_type is None:
            logger.error("Cannot decay array without element_type", node=None, exc_type=TypeError)
        
        from .types import ptr as ptr_class
        
        # For multi-dimensional arrays, decay to pointer to inner array
        # array[array[i32, 5], 3] -> ptr[array[i32, 5]]
        if cls.dimensions and len(cls.dimensions) > 1:
            # Create inner array type
            # Use tuple unpacking compatible with Python 3.9+
            inner_array = array[(cls.element_type,) + tuple(cls.dimensions[1:])]
            return ptr_class[inner_array]
        
        # For single-dimensional arrays, decay to pointer to element
        # array[i32, 10] -> ptr[i32]
        return ptr_class[cls.element_type]

    @classmethod
    def handle_assign_decay(cls, visitor, value_ref):
        """Handle array decay for assignment without type annotation.
        
        In C, arrays decay to pointers in most expression contexts.
        This method converts an array ValueRef to a pointer ValueRef.
        
        Args:
            visitor: AST visitor instance
            value_ref: ValueRef with array type
            
        Returns:
            ValueRef with decayed pointer type
        """
        decay_ptr_type = cls.get_decay_pointer_type()
        return visitor.type_converter.convert(value_ref, decay_ptr_type)

    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node: ast.Subscript):
        """Handle array value subscript: arr[index] or arr[i, j, k]
        
        Note: Type subscripts (array[T, N]) are now handled by PythonType.handle_subscript
        which extracts items and calls array.handle_type_subscript directly.
        This method only handles value subscripts.
        
        Args:
            visitor: AST visitor instance
            base: Pre-evaluated base object (ValueRef)
            index: Pre-evaluated index (ValueRef)
            node: Original ast.Subscript node
            
        Returns:
            ValueRef with element value
        """
        from llvmlite import ir
        from ..valueref import ensure_ir
        from ..ir_helpers import propagate_qualifiers, strip_qualifiers
        
        # Value subscript: arr[index] or arr[i, j, k]
        # Always decay to ptr for both single dimensional and multi-dimensional indices
        base_type = base.type_hint
        
        # Strip qualifiers to get base array type, then get decay pointer type
        base_array_type = strip_qualifiers(base_type)
        ptr_type = base_array_type.get_decay_pointer_type()
        
        # Propagate qualifiers from array to pointer
        # const[array[i32, 3]] -> const[ptr[i32]]
        ptr_type = propagate_qualifiers(base_type, ptr_type)
        
        ptr_base = visitor.type_converter.convert(base, ptr_type)
        return ptr_type.handle_subscript(visitor, ptr_base, index, node)
    
    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items for array
        
        Args:
            items: Normalized tuple: ((None, element_type), (None, dim1), (None, dim2), ...)
        
        Returns:
            array subclass with element_type and dimensions set
        """
        import builtins
        if not isinstance(items, builtins.tuple):
            items = (items,)
        if len(items) < 2:
            logger.error("array requires at least element type and one dimension", node=None, exc_type=TypeError)
        # First item is element type
        elem_name_opt, element_type = items[0]
        # Remaining items are dimensions
        dimensions = []
        for name_opt, dim in items[1:]:
            if not isinstance(dim, int) or dim <= 0:
                logger.error(f"array dimensions must be positive integers, got {dim}",
                            node=None, exc_type=TypeError)
            dimensions.append(dim)
        elem_name = getattr(element_type, 'get_name', lambda: str(element_type))()
        dims_str = ', '.join(str(d) for d in dimensions)
        return type(
            f'array[{elem_name}, {dims_str}]',
            (array,),
            {
                'element_type': element_type,
                'dimensions': builtins.tuple(dimensions)
            }
        )