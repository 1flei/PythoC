from llvmlite import ir
from .base import BuiltinFunction, BuiltinEntity, _get_unified_registry
from ..valueref import wrap_value
import ast
import ctypes


class sizeof(BuiltinFunction):
    """sizeof(type) - Get size of a type in bytes"""
    
    @classmethod
    def get_name(cls) -> str:
        return 'sizeof'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call) -> ir.Value:
        """Handle sizeof(type) call
        
        Unified implementation using TypeResolver to parse all types.
        This eliminates ~150 lines of duplicate type parsing logic.
        """
        if len(node.args) != 1:
            raise TypeError(f"sizeof() takes exactly 1 argument ({len(node.args)} given)")
        
        arg = node.args[0]
        
        pc_type = visitor.type_resolver.parse_annotation(arg)
        if pc_type is None:
            raise TypeError(f"sizeof() argument must be a type, got: {ast.dump(arg)}")
        
        size = cls._get_type_size(pc_type, visitor)
        from .python_type import PythonType
        python_type = PythonType.wrap(size, is_constant=True)
        return wrap_value(size, kind="python", type_hint=python_type)
    
    @classmethod
    def _get_type_size(cls, pc_type, visitor) -> int:
        """Get size of a PC type
        
        Args:
            pc_type: BuiltinEntity type (i32, f64, ptr[T], etc.) or Python struct class
            visitor: AST visitor for context
        
        Returns:
            int: Size in bytes
        """
        if hasattr(pc_type, 'get_size_bytes'):
            return pc_type.get_size_bytes()
        
        if isinstance(pc_type, type) and hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            registry = _get_unified_registry()
            struct_name = pc_type.__name__
            if registry.has_struct(struct_name):
                struct_info = registry.get_struct(struct_name)
                return cls._calculate_struct_size(struct_info, registry)
            raise TypeError(f"sizeof(): struct '{struct_name}' not found in registry")
        
        if isinstance(pc_type, ir.Type):
            raise TypeError(
                f"sizeof() received ir.Type ({pc_type}). This is a bug - "
                "use BuiltinEntity (i32, f64, ptr[T], etc.) instead."
            )
        
        raise TypeError(f"sizeof(): unknown or unsupported type {pc_type}")
    
    @classmethod
    def _align_to(cls, size: int, alignment: int) -> int:
        """Align size to the specified alignment boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
    
    @classmethod
    def _get_field_size(cls, field_type, struct_registry) -> int:
        """Get size of a field type"""
        if isinstance(field_type, type) and issubclass(field_type, BuiltinEntity):
            if field_type.can_be_type():
                return field_type.get_size_bytes()
        
        if isinstance(field_type, str):
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(field_type)
            if entity_cls and entity_cls.can_be_type():
                return entity_cls.get_size_bytes()
            if struct_registry.has_struct(field_type):
                struct_info = struct_registry.get_struct(field_type)
                return cls._calculate_struct_size(struct_info, struct_registry)
        
        if hasattr(field_type, '__name__'):
            type_name = field_type.__name__
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(type_name)
            if entity_cls and entity_cls.can_be_type():
                return entity_cls.get_size_bytes()
            if struct_registry.has_struct(type_name):
                struct_info = struct_registry.get_struct(type_name)
                return cls._calculate_struct_size(struct_info, struct_registry)
        
        if hasattr(field_type, 'pointee_type') or (hasattr(field_type, '__name__') and field_type.__name__ == 'ptr'):
            return ctypes.sizeof(ctypes.c_void_p)
        
        raise TypeError("sizeof(): unknown field type for size calculation")
    
    @classmethod
    def _get_field_alignment(cls, field_type, struct_registry) -> int:
        """Get alignment of a field type"""
        if isinstance(field_type, type) and issubclass(field_type, BuiltinEntity):
            if field_type.can_be_type():
                return min(field_type.get_size_bytes(), 8)
        
        if isinstance(field_type, str):
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(field_type)
            if entity_cls and entity_cls.can_be_type():
                return min(entity_cls.get_size_bytes(), 8)
            if struct_registry.has_struct(field_type):
                struct_info = struct_registry.get_struct(field_type)
                return cls._calculate_struct_alignment(struct_info, struct_registry)
        
        if hasattr(field_type, '__name__'):
            type_name = field_type.__name__
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(type_name)
            if entity_cls and entity_cls.can_be_type():
                return min(entity_cls.get_size_bytes(), 8)
            if struct_registry.has_struct(type_name):
                struct_info = struct_registry.get_struct(type_name)
                return cls._calculate_struct_alignment(struct_info, struct_registry)
        
        if hasattr(field_type, 'pointee_type') or (hasattr(field_type, '__name__') and field_type.__name__ == 'ptr'):
            return ctypes.sizeof(ctypes.c_void_p)
        
        raise TypeError("sizeof(): unknown field type for size calculation")
    
    @classmethod
    def _calculate_struct_alignment(cls, struct_info, struct_registry) -> int:
        """Calculate the alignment requirement for a struct"""
        max_alignment = 1
        for field_name, field_type in struct_info.fields:
            field_alignment = cls._get_field_alignment(field_type, struct_registry)
            max_alignment = max(max_alignment, field_alignment)
        return max_alignment
    
    @classmethod
    def _calculate_struct_size(cls, struct_info, struct_registry) -> int:
        """Calculate struct size with proper alignment"""
        current_offset = 0
        max_alignment = 1
        
        for field_name, field_type in struct_info.fields:
            field_size = cls._get_field_size(field_type, struct_registry)
            field_alignment = cls._get_field_alignment(field_type, struct_registry)
            
            max_alignment = max(max_alignment, field_alignment)
            current_offset = cls._align_to(current_offset, field_alignment)
            current_offset += field_size
        
        total_size = cls._align_to(current_offset, max_alignment)
        
        return total_size
