from llvmlite import ir
import ast
from .base import BuiltinType
from ..valueref import wrap_value, ensure_ir

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
            raise TypeError("array type requires element type and dimensions")
        
        # Get element LLVM type
        if hasattr(cls.element_type, 'get_llvm_type'):
            elem_llvm = cls.element_type.get_llvm_type(module_context)
        elif isinstance(cls.element_type, ir.Type):
            # ANTI-PATTERN: element_type should be BuiltinEntity, not ir.Type
            raise TypeError(
                f"array.get_llvm_type: element_type is raw LLVM type {cls.element_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead."
            )
        else:
            raise TypeError(f"array.get_llvm_type: unknown element type {cls.element_type}")
        
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
                        raise TypeError(f"array dimension '{dim_node.id}' must be an integer, got {type(dim_value)}")
                else:
                    raise TypeError(f"array dimension '{dim_node.id}' not found in scope")
            else:
                raise TypeError(f"array dimensions must be constants or variable names, got {ast.dump(dim_node)}")
            
            if dim_value is None:
                raise TypeError(f"Failed to resolve array dimension: {ast.dump(dim_node)}")
            
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
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle array[T, N]() for zero-initialization
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (should be empty)
            node: Original ast.Call node
        """
        if cls.element_type is None or cls.dimensions is None:
            raise TypeError("array type requires element type and dimensions")
        
        if len(args) != 0:
            raise TypeError(f"array[T, N]() takes no arguments ({len(args)} given)")
        # Get LLVM array type
        array_type = cls.get_llvm_type()

        from ..type_converter import TypeConverter
        converter = TypeConverter(visitor)

        zero_array = converter.create_zero_constant(array_type)
        return wrap_value(zero_array, kind="value", type_hint=cls)
    
    @classmethod
    def handle_as_type(cls, visitor, node: ast.AST):
        """Handle array type annotation resolution (unified call protocol)
        
        Args:
            visitor: AST visitor instance
            node: AST node
                - ast.Name('array'): returns array base class
                - ast.Subscript('array', (T, N)): returns array[T, N] specialized type
                - ast.Subscript('array', (T, N, M)): returns array[T, N, M] specialized type
        
        Returns:
            array class or array[T, N, ...] specialized type
        """
        # Handle None node (direct class reference)
        if node is None:
            return cls
        
 # array ,
        if isinstance(node, ast.Name):
            return cls
        
 # array[T, N, ...] ,
        if isinstance(node, ast.Subscript):
 # slice
            slice_node = node.slice
            
 # slice_node Tuple: (T, N) (T, N, M, ...)
            if isinstance(slice_node, ast.Tuple):
                elts = slice_node.elts
                if len(elts) < 2:
                    raise TypeError("array requires at least element type and one dimension")
                
 # 
                elem_type_node = elts[0]
                elem_type = visitor.type_resolver.parse_annotation(elem_type_node)
                
                if elem_type is None:
                    raise TypeError(f"Unknown element type in array annotation")
                
                # Remaining elements are dimensions
                # Get user_globals from type_resolver
                user_globals = getattr(visitor.type_resolver, 'user_globals', None)
                dimensions = cls._parse_dimensions(elts[1:], user_globals=user_globals)
                
                # Delegate via normalization -> handle_type_subscript
                raw_items = (elem_type, *dimensions)
                normalized = cls.normalize_subscript_items(raw_items)
                return cls.handle_type_subscript(normalized)
            else:
                raise TypeError("array requires element type and dimensions: array[T, N]")
        
        # Other cases return base class
        return cls

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
            raise TypeError("Cannot decay array without element_type")
        
        from .types import ptr as ptr_class
        
        # For multi-dimensional arrays, decay to pointer to inner array
        # array[array[i32, 5], 3] -> ptr[array[i32, 5]]
        if cls.dimensions and len(cls.dimensions) > 1:
            # Create inner array type
            inner_array = array[cls.element_type, *cls.dimensions[1:]]
            return ptr_class[inner_array]
        
        # For single-dimensional arrays, decay to pointer to element
        # array[i32, 10] -> ptr[i32]
        return ptr_class[cls.element_type]

    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node: ast.Subscript):
        """Handle array subscript operations (unified duck typing protocol)
        
        Supports three modes:
        1. Type subscript (index=None): array[i32, 10] - creates specialized array type
        2. Value subscript (single index): arr[0] - accesses array element
        3. Value subscript (tuple index): matrix[1, 2] - accesses multi-dimensional array element
        
        Args:
            visitor: AST visitor instance
            base: Pre-evaluated base object (ValueRef)
            index: Pre-evaluated index (ValueRef or None for type subscript)
            node: Original ast.Subscript node
            
        Returns:
            For type subscript: specialized array type (ValueRef kind='python')
            For value subscript: ValueRef with element value
        """
        from llvmlite import ir
        from ..valueref import ensure_ir
        from ..ir_helpers import propagate_qualifiers, strip_qualifiers
        
        # Check if this is a type subscript or value subscript
        # Use index=None as the marker (unified protocol!)
        if index is None:
            # Type subscript: array[T, N, M, ...]
            return cls.handle_as_type(visitor, node)
        
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
            raise TypeError("array requires at least element type and one dimension")
        # First item is element type
        elem_name_opt, element_type = items[0]
        # Remaining items are dimensions
        dimensions = []
        for name_opt, dim in items[1:]:
            if not isinstance(dim, int) or dim <= 0:
                raise TypeError(f"array dimensions must be positive integers, got {dim}")
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