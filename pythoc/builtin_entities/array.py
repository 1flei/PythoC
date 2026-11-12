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
    def map_type_components(cls, mapper):
        element = mapper(cls.element_type)
        if element is cls.element_type:
            return cls
        return cls.handle_type_subscript(((None, element),) + tuple(
            (None, dim) for dim in cls.dimensions))

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
        element_type = cls._get_element_type()
        if hasattr(element_type, 'get_ctypes_type'):
            elem_ctype = element_type.get_ctypes_type()
        else:
            return ctypes.c_void_p
        
        if elem_ctype is None:
            return ctypes.c_void_p
        
        # Build nested array type for multi-dimensional arrays
        # array[i32, 2, 3] -> (c_int32 * 3) * 2
        result_type = elem_ctype
        for dim in reversed(cls.dimensions):
            result_type = result_type * dim
        
        from ..target_layout import validate_ctypes_layout
        return validate_ctypes_layout(cls, result_type)
    
    @classmethod
    def get_name(cls) -> str:
        return 'array'
    
    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        from ..type_id import get_type_id
        return get_type_id(cls)

    @classmethod
    def _get_element_type(cls):
        from ..type_resolver import TypeResolver
        element_type = TypeResolver.resolve_type(cls.element_type)
        from ..type_descriptor import contains_linear_storage
        if (cls.dimensions and all(cls.dimensions)
                and contains_linear_storage(element_type)):
            raise TypeError('Arrays of linear values are unsupported; use a struct for ownership tracking')
        return element_type

    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM array type
        
        Args:
            module_context: Optional IR module context (passed to element type)
        """
        if cls.element_type is None or cls.dimensions is None:
            logger.error("array type requires element type and dimensions", node=None, exc_type=TypeError)

        from ..type_graph import prepare_type
        prepare_type(cls)
        
        # Get element LLVM type.  A string element type is an unresolved
        # forward-ref name kept by handle_type_subscript; resolve it here
        # (same lazy resolution as ptr's string pointee), failing loudly if
        # the name was never defined.
        element_type = cls._get_element_type()
        from ..type_resolver import TypeResolver
        TypeResolver.resolve_type(cls.element_type, freeze=True)
        if hasattr(element_type, 'get_llvm_type'):
            elem_llvm = element_type.get_llvm_type(module_context)
        elif isinstance(element_type, ir.Type):
            # ANTI-PATTERN: element_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"array.get_llvm_type: element_type is raw LLVM type {element_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None, exc_type=TypeError)
        else:
            logger.error(f"array.get_llvm_type: unknown element type {element_type}",
                        node=None, exc_type=TypeError)
        
        # Build nested array type for multi-dimensional arrays
        # array[i32, 2, 3] -> [2 x [3 x i32]]
        result_type = elem_llvm
        for dim in reversed(cls.dimensions):
            result_type = ir.ArrayType(result_type, dim)
        
        return result_type
    
    @classmethod
    def get_layout(cls):
        if cls.element_type is None or cls.dimensions is None:
            from ..target_layout import TypeLayout
            return TypeLayout.empty()

        from ..type_graph import prepare_type
        prepare_type(cls)
        from ..target_layout import get_target_layout, get_type_layout
        element_layout = get_type_layout(cls._get_element_type())
        return get_target_layout().array(element_layout, cls.dimensions)

    @classmethod
    def get_size_bytes(cls) -> int:
        return cls.get_layout().allocation_size

    @classmethod
    def get_alignment(cls) -> int:
        return cls.get_layout().alignment
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # array[T, N]() for zero-initialization
    
    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
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
        from ..type_descriptor import can_zero_initialize
        if not can_zero_initialize(cls):
            logger.error('Cannot zero-initialize an array with proof-bearing elements',
                         node=node, exc_type=TypeError)
        # Get LLVM array type
        array_type = cls.get_llvm_type(visitor.module.context)

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
        from ..type_descriptor import array_decay_type
        return array_decay_type(cls)

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
        from ..type_descriptor import array_decay_type
        decay_ptr_type = array_decay_type(value_ref.type_hint)
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
        from ..type_descriptor import array_decay_type
        ptr_type = array_decay_type(base.type_hint)
        ptr_base = visitor.type_converter.convert(base, ptr_type)
        return ptr_type.handle_subscript(visitor, ptr_base, index, node)

    @classmethod
    def handle_add(cls, visitor, left, right, node):
        """Handle array + int: decay to pointer and delegate to ptr.handle_add"""
        from ..type_descriptor import array_decay_type
        ptr_type = array_decay_type(left.type_hint)

        ptr_left = visitor.type_converter.convert(left, ptr_type)
        return ptr_type.handle_add(visitor, ptr_left, right, node)

    @classmethod
    def handle_radd(cls, visitor, left, right, node):
        """Handle int + array: decay to pointer and delegate to ptr.handle_radd"""
        from ..type_descriptor import array_decay_type
        ptr_type = array_decay_type(right.type_hint)

        ptr_right = visitor.type_converter.convert(right, ptr_type)
        return ptr_type.handle_radd(visitor, left, ptr_right, node)

    @classmethod
    def handle_sub(cls, visitor, left, right, node):
        """Handle array - int: decay to pointer and delegate to ptr.handle_sub"""
        from ..type_descriptor import array_decay_type
        ptr_type = array_decay_type(left.type_hint)

        ptr_left = visitor.type_converter.convert(left, ptr_type)
        return ptr_type.handle_sub(visitor, ptr_left, right, node)
    
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
        # First item is element type.  A quoted element type name
        # (``array["T", 8]``) stays a lazy string here and is resolved at
        # IR materialization, mirroring ptr's string pointee; binding via
        # the registry at subscript time would bake in a time-of-check
        # answer from a mutable table.
        elem_name_opt, element_type = items[0]
        from ..type_descriptor import validate_type_argument
        if any(name is not None for name, _ in items):
            raise TypeError('array type arguments cannot have member names')
        validate_type_argument(element_type, 'array element', allow_void=False)
        # Remaining items are dimensions
        dimensions = []
        for name_opt, dim in items[1:]:
            # dim == 0 is the C zero-length array extension: the field
            # contributes its element alignment and no storage. Only a
            # negative dimension is malformed.
            if type(dim) is not int or dim < 0:
                logger.error(f"array dimensions must be non-negative integers, got {dim}",
                            node=None, exc_type=TypeError)
            dimensions.append(dim)
        elem_name = getattr(element_type, 'get_name', lambda: str(element_type))()
        dims_str = ', '.join(str(d) for d in dimensions)
        return type(
            f'array[{elem_name}, {dims_str}]',
            (array,),
            {
                '_pc_specialized': True,
                'element_type': element_type,
                'dimensions': builtins.tuple(dimensions)
            }
        )
