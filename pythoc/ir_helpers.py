"""IR generation helpers for type qualifiers (const, volatile)"""
from llvmlite import ir
from typing import Any, Optional
from .logger import logger


def alloca_in_entry(builder, llvm_type, name=""):
    """Allocate a fixed compiler scratch slot once per function invocation."""
    llvm_builder = builder if isinstance(builder, ir.IRBuilder) else builder.ir_builder
    with llvm_builder.goto_entry_block():
        return llvm_builder.alloca(llvm_type, name=name)


def strip_qualifiers(pc_type: Any) -> Any:
    """Strip qualifier wrappers (const, volatile, static) from a type.

    Delegates to the canonical implementation in type_converter which
    correctly preserves refined types.

    Example:
        const[volatile[i32]] -> i32
        static[const[array[i32, 5]]] -> array[i32, 5]
        refined[i32, pred] -> refined[i32, pred]  (NOT stripped)
    """
    from .type_converter import strip_qualifiers as _canonical_strip
    return _canonical_strip(pc_type)


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
    
    from .builtin_entities.qualifiers import const, volatile
    from .type_descriptor import _flags
    flags = _flags(from_type)
    from .type_converter import get_base_type, strip_qualifiers
    source = strip_qualifiers(from_type)
    base = get_base_type(source)
    if (getattr(source, '_is_refined', False)
            and (getattr(base, '_field_types', None) is not None
                 or getattr(base, 'dimensions', None) is not None)):
        # Mutating a subobject cannot establish the predicate on its owner.
        # A copy with its proof erased can still be modified and re-proved.
        flags = dict(flags, const=True)
    result = const[to_type] if flags.get('const') else to_type
    return volatile[result] if flags.get('volatile') else result


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


def is_thread_local(pc_type: Any) -> bool:
    """Check if type is thread_local qualified"""
    return is_qualifier_type(pc_type, 'thread_local')


def is_link_time_constant(value: Any) -> bool:
    """Check if an IR value is usable inside a constant (static) initializer.

    ir.Constant covers folded constants. ir.GlobalValue subclasses
    (ir.Function, ir.GlobalVariable) are link-time constants and valid
    elements of constant aggregate initializers, even though llvmlite does
    not model them as ir.Constant.
    """
    return isinstance(value, (ir.Constant, ir.GlobalValue))


# Linkage kinds supported by @compile(linkage=...). They map to the C
# toolchain concepts of a plain external definition (default), a static
# function (internal), and inline/ODR definitions (weak_odr / linkonce_odr).
FUNC_LINKAGE_KINDS = frozenset({'external', 'internal', 'weak_odr', 'linkonce_odr'})


def validate_func_linkage(linkage: Optional[str],
                          func_name: Optional[str] = None) -> Optional[str]:
    """Validate a user-requested function linkage kind.

    Returns the normalized linkage: None for the default external linkage,
    so not specifying linkage keeps the emitted IR byte-identical to before.
    Unsupported values raise a loud error.
    """
    if linkage is None:
        return None
    if linkage not in FUNC_LINKAGE_KINDS:
        logger.error(
            f"Unsupported linkage '{linkage}'"
            + (f" for function '{func_name}'" if func_name else "")
            + f"; expected one of {sorted(FUNC_LINKAGE_KINDS)}",
            node=None, exc_type=TypeError)
    return None if linkage == 'external' else linkage


def make_memory_volatile(instruction):
    """Preserve the complete LLVM memory descriptor while marking volatility."""
    if not isinstance(instruction, ir.Instruction) or getattr(instruction, '_is_volatile', False):
        return instruction
    marker = instruction.opname + ' '
    original = instruction.descr

    def volatile_descr(buffer):
        pieces = []
        original(pieces)
        buffer.append(''.join(pieces).replace(marker, marker + 'volatile ', 1))

    instruction.descr = volatile_descr
    instruction._is_volatile = True
    return instruction


def make_load_volatile(load_inst):
    return make_memory_volatile(load_inst)


def make_store_volatile(store_inst):
    return make_memory_volatile(store_inst)


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
    if is_volatile(pc_type):
        volatile_load = getattr(builder, 'load_volatile', None)
        if volatile_load is not None:
            return volatile_load(ptr, name=name, align=align)
    load_inst = builder.load(ptr, name=name, align=align)
    
    # Check if type is volatile
    if is_volatile(pc_type):
        make_load_volatile(load_inst)
    
    return load_inst


def check_const_store(pc_type, node=None):
    """Apply the common const-mutation check before emitting a store."""
    from .type_descriptor import contains_const_storage
    if contains_const_storage(pc_type):
        type_name = pc_type.get_name() if hasattr(pc_type, 'get_name') else str(pc_type)
        logger.error(f"Cannot reassign to const variable of type {type_name}", node=node)


def safe_store(builder: ir.IRBuilder, value: ir.Value, ptr: ir.Value,
               pc_type: Any = None, align: Any = None, node: Any = None) -> ir.StoreInstr:
    """Store with automatic volatile handling and const checking
    
    Args:
        builder: LLVM IR builder
        value: Value to store
        ptr: Pointer to store to
        pc_type: PC type hint (will check for const/volatile qualifiers)
        align: Optional alignment
        node: AST node for error reporting
    
    Returns:
        Store instruction (volatile if type is volatile-qualified)
    
    Raises:
        RuntimeError: If attempting to modify const-qualified type
    """
    # Check if trying to modify const
    check_const_store(pc_type, node)
    
    if is_volatile(pc_type):
        volatile_store = getattr(builder, 'store_volatile', None)
        if volatile_store is not None:
            return volatile_store(value, ptr, align=align)
    store_inst = builder.store(value, ptr, align=align)
    
    # Check if type is volatile
    if is_volatile(pc_type):
        make_store_volatile(store_inst)
    
    return store_inst
