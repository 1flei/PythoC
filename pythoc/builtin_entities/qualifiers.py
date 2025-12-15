from llvmlite import ir
from .base import BuiltinType, BuiltinEntityMeta
from ..logger import logger


class TypeQualifierMeta(BuiltinEntityMeta):
    """Metaclass that forwards unknown attributes to qualified_type"""
    
    # Methods that should NOT be forwarded (qualifier-specific logic)
    _LOCAL_METHODS = frozenset({
        'qualified_type', '_qualifier_flags', 'get_qualifier_name',
        'get_qualifier_flags', 'handle_type_subscript', '__class_getitem__',
        'is_const', 'is_static', 'is_volatile', 'get_name',
        '_normalize_qualifiers', '__init__', '__new__', '__init_subclass__',
        '__mro__', '__bases__', '__dict__', '__module__', '__qualname__',
    })
    
    def __getattribute__(cls, name):
        # For private/dunder methods and local methods, use normal lookup
        if name.startswith('_') or name in TypeQualifierMeta._LOCAL_METHODS:
            return super().__getattribute__(name)
        
        # Check if this attribute is defined in the dynamic subclass's __dict__
        # (not inherited from TypeQualifier base class)
        if name in cls.__dict__:
            return super().__getattribute__(name)
        
        # Try to forward to qualified_type
        try:
            qualified_type = super().__getattribute__('qualified_type')
            if qualified_type is not None and hasattr(qualified_type, name):
                return getattr(qualified_type, name)
        except AttributeError:
            pass
        
        # Fallback to normal attribute lookup (will raise AttributeError if not found)
        return super().__getattribute__(name)


class TypeQualifier(BuiltinType, metaclass=TypeQualifierMeta):
    """Base class for type qualifiers (const, static, volatile)
    
    Automatically forwards protocol methods to qualified_type via metaclass.
    Only qualifier-specific logic (get_qualifier_name, get_qualifier_flags, etc.) 
    needs to be defined here.
    """
    qualified_type = None
    _qualifier_flags = {}  # Override in subclasses: {'const': True, 'static': False, 'volatile': False}
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        """Override in subclasses to return qualifier name"""
        logger.error("get_qualifier_name() must be implemented in subclass",
                    node=None, exc_type=NotImplementedError)
    
    @classmethod
    def get_name(cls) -> str:
        if cls.qualified_type:
            type_name = cls.qualified_type.get_name() if hasattr(cls.qualified_type, 'get_name') else str(cls.qualified_type)
            return f'{cls.get_qualifier_name()}[{type_name}]'
        return cls.get_qualifier_name()
    
    @classmethod
    def get_qualifier_flags(cls):
        """Get all qualifier flags for this type (including nested qualifiers)"""
        flags = cls._qualifier_flags.copy()
        if cls.qualified_type and hasattr(cls.qualified_type, 'get_qualifier_flags'):
            # Merge with nested qualifier flags
            nested_flags = cls.qualified_type.get_qualifier_flags()
            flags.update(nested_flags)
        return flags
    
    @classmethod
    def _normalize_qualifiers(cls, item):
        """Normalize nested qualifiers: const[static[T]] -> const_static[T]
        
        Returns (base_type, qualifier_flags) where qualifier_flags is a dict
        of all qualifiers that should be applied.
        """
        flags = cls._qualifier_flags.copy()
        base_type = item
        
        # Unwrap nested qualifiers and collect flags
        while isinstance(base_type, type) and issubclass(base_type, TypeQualifier):
            # Merge flags: OR operation (any True wins)
            for key in flags:
                if key in base_type._qualifier_flags:
                    flags[key] = flags[key] or base_type._qualifier_flags[key]
            base_type = base_type.qualified_type
        
        return base_type, flags
    
    @classmethod
    def handle_type_subscript(cls, item):
        """Unified type subscript handler for both runtime and compile-time paths
        
        Args:
            item: Normalized tuple from normalize_subscript_items: ((None, type),)
                  or raw type from TypeResolver
        
        Returns:
            Qualifier subclass with qualified_type set
        """
        if item is None:
            logger.error(f"{cls.get_qualifier_name()} requires a type parameter: {cls.get_qualifier_name()}[T]",
                        node=None, exc_type=TypeError)
        
        # Unwrap normalized tuple if needed
        import builtins
        if isinstance(item, builtins.tuple) and len(item) == 1 and isinstance(item[0], builtins.tuple):
            # Normalized format: ((None, type),) -> extract type
            _, actual_type = item[0]
            item = actual_type
        
        # Normalize qualifiers
        base_type, flags = cls._normalize_qualifiers(item)
        
        # Build canonical name: sort qualifiers alphabetically
        qualifier_names = sorted([name for name, enabled in flags.items() if enabled])
        if hasattr(base_type, 'get_name'):
            type_name = base_type.get_name()
        elif hasattr(base_type, '__name__'):
            type_name = base_type.__name__
        else:
            type_name = str(base_type)
        
        # Create class name
        if len(qualifier_names) == 1:
            class_name = f'{qualifier_names[0]}[{type_name}]'
        else:
            # Multiple qualifiers: const_static[T]
            class_name = f'{"_".join(qualifier_names)}[{type_name}]'
        
        # Create methods dict with all is_* methods
        methods = {
            'qualified_type': base_type,
            '_qualifier_flags': flags,
        }
        
        # Add is_* methods based on flags
        if flags.get('const'):
            methods['is_const'] = classmethod(lambda c: True)
        if flags.get('static'):
            methods['is_static'] = classmethod(lambda c: True)
        if flags.get('volatile'):
            methods['is_volatile'] = classmethod(lambda c: True)
        
        return type(
            class_name,
            (cls,),
            methods
        )
    
    def __class_getitem__(cls, item):
        """Runtime path: delegate to BuiltinType normalization and handle_type_subscript"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)


class const(TypeQualifier):
    _qualifier_flags = {'const': True, 'static': False, 'volatile': False}
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'const'
    
    @classmethod
    def is_const(cls) -> bool:
        return True

class static(TypeQualifier):
    _qualifier_flags = {'const': False, 'static': True, 'volatile': False}
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'static'
    
    @classmethod
    def is_static(cls) -> bool:
        return True

class volatile(TypeQualifier):
    _qualifier_flags = {'const': False, 'static': False, 'volatile': True}
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'volatile'
    
    @classmethod
    def is_volatile(cls) -> bool:
        return True