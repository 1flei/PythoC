"""IR generation helpers for type qualifiers (const, volatile)"""
from llvmlite import ir
from typing import Any


def strip_qualifiers(pc_type: Any) -> Any:
    """Strip all qualifiers from a type, returning the base type
    
    Example:
        const[volatile[i32]] -> i32
        static[const[array[i32, 5]]] -> array[i32, 5]
    """
    if pc_type is None:
        return None
    
    # Keep stripping qualifiers until we reach the base type
    while hasattr(pc_type, 'qualified_type') and pc_type.qualified_type is not None:
        pc_type = pc_type.qualified_type
    
    return pc_type


def propagate_qualifiers(from_type: Any, to_type: Any) -> Any:
    """Propagate qualifiers from one type to another
    
    When accessing a member of a const/volatile object, the member should
    also be const/volatile.
    
    Args:
        from_type: Source type (may have qualifiers)
        to_type: Target type (base type without qualifiers)
    
    Returns:
        Target type with qualifiers from source type applied
    
    Example:
        propagate_qualifiers(const[array[i32, 5]], i32) -> const[i32]
        propagate_qualifiers(volatile[ptr[i32]], i32) -> volatile[i32]
        propagate_qualifiers(const[volatile[struct]], field_type) -> const[volatile[field_type]]
    """
    if from_type is None or to_type is None:
        return to_type
    
    # Import here to avoid circular dependency
    from .builtin_entities.qualifiers import const, volatile, static
    
    # Collect qualifiers from source type (in order from outer to inner)
    qualifiers = []
    current = from_type
    
    while True:
        if isinstance(current, type):
            # Check if this is a TypeQualifier by checking for qualified_type
            if hasattr(current, 'qualified_type') and hasattr(current, '_qualifier_flags'):
                # This is a qualifier type - check which qualifier it is
                flags = current._qualifier_flags
                if flags.get('const'):
                    qualifiers.append(const)
                if flags.get('volatile'):
                    qualifiers.append(volatile)
                # Note: static is not propagated to members
                # static applies to storage duration, not to the object itself
                
                # Move to inner type
                if current.qualified_type is not None:
                    current = current.qualified_type
                    continue
        break
    
    # Apply qualifiers to target type (in reverse order to maintain nesting)
    result = to_type
    for qualifier in reversed(qualifiers):
        result = qualifier[result]
    
    return result


def is_qualifier_type(pc_type: Any, qualifier_name: str) -> bool:
    """Check if a PC type has a specific qualifier"""
    if pc_type is None:
        return False
    
    # Check if type has get_qualifier_flags method
    if hasattr(pc_type, 'get_qualifier_flags'):
        flags = pc_type.get_qualifier_flags()
        return flags.get(qualifier_name, False)
    
    # Check for is_* methods as fallback
    check_method = f'is_{qualifier_name}'
    if hasattr(pc_type, check_method):
        method = getattr(pc_type, check_method)
        if callable(method):
            try:
                return method()
            except:
                pass
    
    return False


def is_const(pc_type: Any) -> bool:
    """Check if type is const qualified"""
    return is_qualifier_type(pc_type, 'const')


def is_volatile(pc_type: Any) -> bool:
    """Check if type is volatile qualified"""
    return is_qualifier_type(pc_type, 'volatile')


def is_static(pc_type: Any) -> bool:
    """Check if type is static qualified"""
    return is_qualifier_type(pc_type, 'static')


def make_load_volatile(load_inst: ir.LoadInstr) -> ir.LoadInstr:
    """Make a load instruction volatile by monkey-patching its descr method
    
    LLVM volatile loads prevent optimization that would eliminate or reorder
    memory accesses, which is essential for memory-mapped I/O and multi-threading.
    """
    def volatile_descr(buf):
        """Custom descr that adds volatile keyword
        
        Format for opaque pointers (LLVM 15+):
            %result = load volatile type, ptr %ptr, align N
        Format for typed pointers (LLVM <15):
            %result = load volatile type, type* %ptr, align N
        """
        [val] = load_inst.operands
        align = ' align %d' % load_inst.align if load_inst.align is not None else ''
        
        # Check if using opaque pointers (pointer type has no pointee attribute or is_opaque)
        if hasattr(val.type, 'is_opaque') and val.type.is_opaque:
            # Opaque pointer format: load volatile type, ptr %ptr
            buf.append("load volatile {0}, {1} {2}{3}{4}\n".format(
                load_inst.type,
                val.type,
                val.get_reference(),
                align,
                load_inst._stringify_metadata(leading_comma=True)
            ))
        else:
            # Typed pointer format: load volatile type, type* %ptr
            buf.append("load volatile {0}, {1} {2}{3}{4}\n".format(
                load_inst.type,
                val.type,
                val.get_reference(),
                align,
                load_inst._stringify_metadata(leading_comma=True)
            ))
    
    load_inst.descr = volatile_descr
    load_inst._is_volatile = True
    return load_inst


def make_store_volatile(store_inst: ir.StoreInstr) -> ir.StoreInstr:
    """Make a store instruction volatile by monkey-patching its descr method
    
    LLVM volatile stores prevent optimization that would eliminate or reorder
    memory accesses, which is essential for memory-mapped I/O and multi-threading.
    """
    def volatile_descr(buf):
        """Custom descr that adds volatile keyword
        
        Format for opaque pointers (LLVM 15+):
            store volatile type %value, ptr %ptr, align N
        Format for typed pointers (LLVM <15):
            store volatile type %value, type* %ptr, align N
        """
        [val, ptr] = store_inst.operands
        align = ' align %d' % store_inst.align if store_inst.align is not None else ''
        
        # Check if using opaque pointers
        if hasattr(ptr.type, 'is_opaque') and ptr.type.is_opaque:
            # Opaque pointer format: store volatile type %value, ptr %ptr
            buf.append("store volatile {0} {1}, {2} {3}{4}{5}\n".format(
                val.type,
                val.get_reference(),
                ptr.type,
                ptr.get_reference(),
                align,
                store_inst._stringify_metadata(leading_comma=True)
            ))
        else:
            # Typed pointer format: store volatile type %value, type* %ptr
            buf.append("store volatile {0} {1}, {2} {3}{4}{5}\n".format(
                val.type,
                val.get_reference(),
                ptr.type,
                ptr.get_reference(),
                align,
                store_inst._stringify_metadata(leading_comma=True)
            ))
    
    store_inst.descr = volatile_descr
    store_inst._is_volatile = True
    return store_inst


def safe_load(builder: ir.IRBuilder, ptr: ir.Value, pc_type: Any = None, 
              name: str = '', align: Any = None) -> ir.LoadInstr:
    """Load with automatic volatile handling based on type qualifiers
    
    Args:
        builder: LLVM IR builder
        ptr: Pointer to load from
        pc_type: PC type hint (will check for volatile qualifier)
        name: Optional name for the loaded value
        align: Optional alignment
    
    Returns:
        Load instruction (volatile if type is volatile-qualified)
    """
    load_inst = builder.load(ptr, name=name, align=align)
    
    # Check if type is volatile
    if is_volatile(pc_type):
        make_load_volatile(load_inst)
    
    return load_inst


def safe_store(builder: ir.IRBuilder, value: ir.Value, ptr: ir.Value, 
               pc_type: Any = None, align: Any = None) -> ir.StoreInstr:
    """Store with automatic volatile handling and const checking
    
    Args:
        builder: LLVM IR builder
        value: Value to store
        ptr: Pointer to store to
        pc_type: PC type hint (will check for const/volatile qualifiers)
        align: Optional alignment
    
    Returns:
        Store instruction (volatile if type is volatile-qualified)
    
    Raises:
        TypeError: If attempting to modify const-qualified type
    """
    # Check if trying to modify const
    if is_const(pc_type):
        type_name = pc_type.get_name() if hasattr(pc_type, 'get_name') else str(pc_type)
        raise TypeError(f"Cannot modify const-qualified type: {type_name}")
    
    store_inst = builder.store(value, ptr, align=align)
    
    # Check if type is volatile
    if is_volatile(pc_type):
        make_store_volatile(store_inst)
    
    return store_inst
