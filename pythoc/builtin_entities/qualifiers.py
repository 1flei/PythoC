from llvmlite import ir
from .base import BuiltinType, BuiltinEntityMeta
from ..logger import logger


class TypeQualifierMeta(BuiltinEntityMeta):
    """Metaclass that forwards unknown attributes to qualified_type"""
    
    # Methods that should NOT be forwarded (qualifier-specific logic)
    _LOCAL_METHODS = frozenset({
        'qualified_type', '_qualifier_flags', 'get_qualifier_name',
        'get_qualifier_flags', 'handle_type_subscript', '__class_getitem__',
        'is_const', 'is_static', 'is_thread_local', 'is_volatile', 'get_name',
        '_normalize_qualifiers', '__init__', '__new__', '__init_subclass__',
        '__mro__', '__bases__', '__dict__', '__module__', '__qualname__',
        'get_llvm_type', 'get_size_bytes', 'get_type_id',
    })
    
    def __getattribute__(cls, name):
        # For private/dunder methods and local methods, use normal lookup
        if name.startswith('_') or name in TypeQualifierMeta._LOCAL_METHODS:
            return super().__getattribute__(name)
        
        # Check if this attribute is defined in the dynamic subclass's __dict__
        # (not inherited from TypeQualifier base class)
        if name in cls.__dict__:
            return super().__getattribute__(name)
        
        # Try to forward to qualified_type.  A string inner type (quoted
        # forward reference like static["Box"]) resolves through the
        # forward-ref registry before the attribute is forwarded, mirroring
        # the lazy resolution in _resolve_qualified_type.
        try:
            qualified_type = super().__getattribute__('qualified_type')
            if isinstance(qualified_type, str):
                from ..forward_ref import get_defined_type
                resolved = get_defined_type(qualified_type)
                if resolved is not None:
                    qualified_type = resolved
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
    def _resolve_qualified_type(cls):
        """Resolve ``qualified_type``, lazily dereferencing forward refs.

        A quoted inner annotation (``static["Box"]``) may resolve to a plain
        string when the type name is not yet bound at class-decoration time.
        Like ``ptr`` (which keeps the string in ``pointee_type`` and resolves
        it inside ``get_llvm_type``), the qualifier resolves the string
        through the forward-ref registry at use time.
        """
        base_type = cls.qualified_type
        if isinstance(base_type, str):
            from ..forward_ref import get_defined_type
            resolved = get_defined_type(base_type)
            if resolved is None:
                logger.error(
                    f"{cls.get_qualifier_name()}: unresolved forward "
                    f"reference '{base_type}'",
                    node=None, exc_type=NameError,
                )
            base_type = resolved
        return base_type

    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get the LLVM type of the qualified (inner) type.

        Defined locally so the metaclass does not forward it to
        ``qualified_type``: a string inner type (unresolved forward ref)
        would make the forwarding return ``None`` and crash IR materialization
        with llvmlite's ``assert isinstance(typ, types.Type)``.  An
        unspecialized qualifier (no inner type) falls back to the base
        implementation, as the forwarding did before.
        """
        base_type = cls._resolve_qualified_type()
        if base_type is None:
            return super().get_llvm_type(module_context)
        return base_type.get_llvm_type(module_context)

    @classmethod
    def get_size_bytes(cls):
        """Size of the qualified (inner) type; resolved locally for the
        same reason as get_llvm_type."""
        base_type = cls._resolve_qualified_type()
        if base_type is None:
            return super().get_size_bytes()
        return base_type.get_size_bytes()

    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        base_type = cls._resolve_qualified_type()
        if base_type is None:
            return super().get_type_id(_visited)
        from ..type_id import get_type_id
        return get_type_id(base_type, _visited)
    
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
            '_pc_specialized': True,
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
    _qualifier_flags = {
        'const': True,
        'static': False,
        'thread_local': False,
        'volatile': False,
    }
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'const'
    
    @classmethod
    def is_const(cls) -> bool:
        return True

class static(TypeQualifier):
    _qualifier_flags = {
        'const': False,
        'static': True,
        'thread_local': False,
        'volatile': False,
    }
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'static'
    
    @classmethod
    def is_static(cls) -> bool:
        return True


class thread_local(TypeQualifier):
    _qualifier_flags = {
        'const': False,
        'static': False,
        'thread_local': True,
        'volatile': False,
    }

    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'thread_local'

    @classmethod
    def is_thread_local(cls) -> bool:
        return True


class volatile(TypeQualifier):
    _qualifier_flags = {
        'const': False,
        'static': False,
        'thread_local': False,
        'volatile': True,
    }
    
    @classmethod
    def get_qualifier_name(cls) -> str:
        return 'volatile'
    
    @classmethod
    def is_volatile(cls) -> bool:
        return True