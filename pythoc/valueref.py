from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING, Union, Any as AnyType
from llvmlite import ir
import ast
from .logger import logger
from .type_check import is_struct_type, is_enum_type

if TYPE_CHECKING:
    from typing import Any


def extract_constant_index(index_value: 'ValueRef', context: str = "subscript") -> int:
    """Extract constant integer index from a ValueRef for subscript operations.
    
    This function handles compile-time constant index extraction for operations like:
    - struct[i], union[i], enum[i] (requires compile-time constant)
    - varargs[i] (requires compile-time constant)
    
    Args:
        index_value: ValueRef containing the index value
        context: Description of the subscript context for error messages
    
    Returns:
        int: Python integer constant value
    
    Raises:
        TypeError: If index is not a constant (runtime value) or not an integer
    
    Usage:
        index_value = visitor.visit_expression(node.slice)
        idx = extract_constant_index(index_value, "struct subscript")
        field = struct._field_types[idx]
    """
    if index_value.is_python_value():
        # PythonType - extract Python object
        py_obj = index_value.type_hint.get_python_object()
        if not isinstance(py_obj, int):
            raise TypeError(f"{context} index must be an integer, got {type(py_obj).__name__}")
        return py_obj
    
    # LLVM value - must be ir.Constant
    index_ir = index_value.value
    if not isinstance(index_ir, ir.Constant):
        raise TypeError(f"{context} index must be a constant, got runtime value")
    
    # Extract constant value
    const_val = index_ir.constant
    if not isinstance(const_val, int):
        raise TypeError(f"{context} index must be an integer constant, got {type(const_val).__name__}")
    
    return const_val

@dataclass
class ValueRef:
    """Wrapper around LLVM IR value or Python value with type metadata.
    
    Enhanced to support both PC types (LLVM IR) and Python types (Python objects).
    Supports unified lvalue/rvalue by optionally carrying address alongside value.
    
    Attributes:
        kind: 'address' | 'pointer' | 'value' | 'python'
              - 'address': lvalue (can be assigned to, e.g., variable, field)
              - 'pointer': pointer value (can be dereferenced)
              - 'value': regular LLVM value (cannot be assigned to)
              - 'python': Python runtime value (not compiled)
        value: Union[ir.Value, Any] - LLVM IR value for PC types, Python object for Python types
        type_hint: Language-level type information (PC type or PythonType)
                   This is the single source of truth for type information.
                   LLVM types are derived from this via get_llvm_type().
        address: Optional address (pointer) for lvalue operations
                 When present, enables use in both lvalue and rvalue contexts
        source_node: original AST node for debugging (optional)
        var_name: Optional variable name for linear tracking (lvalue only)
        linear_path: Optional index path for linear token tracking (lvalue only)
    """
    kind: str
    value: Union[ir.Value, AnyType]  # Changed from ir_value to value
    type_hint: AnyType  # PC type or PythonType (BuiltinEntity subclass)
    address: Optional[ir.Value] = None  # Address for lvalue operations
    source_node: Optional[ast.AST] = None  # Source AST node
    var_name: Optional[str] = None  # Variable name for linear tracking
    linear_path: Optional[tuple] = None  # Index path for linear tracking
    vref_id: Optional[str] = None  # Unique ID for ValueRef tracking (replaces temp_linear_id)
    
    def __post_init__(self):
        """Validate ValueRef after initialization."""
        # There must be a type hint
        if self.type_hint is None:
            raise ValueError("type_hint must be provided")
        # CRITICAL: Prevent double-wrapping ValueRef
        if isinstance(self.value, ValueRef):
            raise TypeError(
                f"BUG: ValueRef.value is another ValueRef (double-wrapping detected). "
                f"inner.kind={self.value.kind}, inner.type_hint={self.value.type_hint}, "
                f"outer.kind={self.kind}, outer.type_hint={self.type_hint}"
            )
    
    # Backward compatibility property
    @property
    def ir_value(self) -> ir.Value:
        """Backward-compatible accessor for LLVM IR value.
        
        Raises TypeError if this is a Python value.
        """
        if self.is_python_value():
            raise TypeError(
                f"Cannot get ir_value from Python type {self.type_hint.get_name() if hasattr(self.type_hint, 'get_name') else self.type_hint}"
            )
        return self.value
    
    def is_python_value(self) -> bool:
        """Check if this ValueRef holds a Python value."""
        # Check kind first - if explicitly marked as python, trust it
        if self.kind == "python":
            return True
        # Otherwise check type_hint
        if self.type_hint and hasattr(self.type_hint, 'is_python_type'):
            return self.type_hint.is_python_type()
        return False
    
    def is_llvm_value(self) -> bool:
        """Check if this ValueRef holds an LLVM IR value."""
        return not self.is_python_value()
    
    def get_python_value(self) -> AnyType:
        """Get the Python value.
        
        Raises TypeError if this is an LLVM value.
        """
        if not self.is_python_value():
            raise TypeError(
                f"Cannot get Python value from LLVM type {self.type_hint.get_name() if hasattr(self.type_hint, 'get_name') else self.type_hint}"
            )
        return self.value

    @property
    def type(self) -> ir.Type:
        """Backward-compatible type accessor.
        
        Returns LLVM type for PC types, raises error for Python types.
        """
        if self.is_python_value():
            raise TypeError(
                f"Python type {self.type_hint.get_name() if hasattr(self.type_hint, 'get_name') else self.type_hint} has no LLVM type"
            )
        return self.value.type

    def unwrap(self) -> ir.Value:
        """Get underlying LLVM IR value.
        
        Raises TypeError if this is a Python value.
        """
        if self.is_python_value():
            raise TypeError(
                f"Cannot unwrap Python value to LLVM IR"
            )
        return self.value

    def is_pointer(self) -> bool:
        """Check if this is a pointer-typed value.
        Prefer BuiltinType semantics when available, fallback to LLVM type.
        """
        if self.is_python_value():
            return False
        if self.type_hint is not None:
            # Explicit pointer flag on BuiltinType
            if hasattr(self.type_hint, '_is_pointer') and getattr(self.type_hint, '_is_pointer'):
                return True
        return isinstance(self.value.type, ir.PointerType)

    def pointee(self) -> Optional[ir.Type]:
        """Get pointee type if this is a pointer."""
        if self.is_python_value():
            return None
        return self.value.type.pointee if isinstance(self.value.type, ir.PointerType) else None
    
    def get_llvm_type(self) -> ir.Type:
        """Get LLVM type for this value.
        
        Prioritizes actual IR value type over type_hint to ensure consistency.
        This avoids mismatches between IdentifiedStructType and LiteralStructType.
        Raises TypeError for Python types.
        """
        if self.is_python_value():
            raise TypeError(
                f"Python type {self.type_hint.get_name() if hasattr(self.type_hint, 'get_name') else self.type_hint} has no LLVM type"
            )
        # Prioritize actual IR value type for consistency (only for IR values)
        if isinstance(self.value, ir.Value):
            return self.value.type
        raise TypeError("Cannot determine LLVM type for value")

    # Common IR operations, always return ValueRef
    def load(self, builder: ir.IRBuilder, name: Optional[str] = None) -> "ValueRef":
        """Load value from pointer, preserving type information.
        
        Raises TypeError for Python values.
        """
        if self.is_python_value():
            raise TypeError("Cannot load from Python value")
        loaded = builder.load(self.unwrap(), name=name) if self.is_pointer() else self.unwrap()
        # Determine new type hint: if we loaded from a pointer, use its pointee BuiltinEntity when available
        new_type_hint = self.type_hint
        if self.is_pointer():
            pointee_hint = getattr(self.type_hint, 'pointee_type', None)
            if pointee_hint is not None:
                new_type_hint = pointee_hint
        return ValueRef(
            kind="value",
            value=loaded,
            type_hint=new_type_hint,
            source_node=self.source_node
        )

    def gep(self, builder: ir.IRBuilder, indices: Sequence[ir.Value], 
            result_type_hint: Optional['Any'] = None) -> "ValueRef":
        """Get element pointer, optionally updating type hint.
        
        Raises TypeError for Python values.
        """
        if self.is_python_value():
            raise TypeError("Cannot perform GEP on Python value")
        gep_val = builder.gep(self.unwrap(), indices, inbounds=True)
        return ValueRef(
            kind="value",
            value=gep_val,
            type_hint=result_type_hint or self.type_hint,
            source_node=self.source_node
        )

    def bitcast(self, builder: ir.IRBuilder, target_type: ir.Type,
                target_type_hint: Optional['Any'] = None) -> "ValueRef":
        """Bitcast to target type, optionally updating type hint.
        
        Raises TypeError for Python values.
        """
        if self.is_python_value():
            raise TypeError("Cannot bitcast Python value")
        casted = builder.bitcast(self.unwrap(), target_type)
        kind = self.kind if isinstance(target_type, ir.PointerType) else "value"
        return ValueRef(
            kind=kind, 
            value=casted,
            type_hint=target_type_hint or self.type_hint,
            source_node=self.source_node
        )
    
    def as_lvalue(self) -> "ValueRef":
        """Convert to lvalue representation (kind='address').
        
        Extracts the address from this ValueRef for use in assignment contexts.
        Special handling for pyconst_field kind.
        
        Returns:
            ValueRef with kind='address' or 'pyconst_field' and value set to the address
            
        Raises:
            ValueError: If this ValueRef doesn't have an address
            TypeError: If this is a Python value
        """
        # Special case: pyconst_field is already an lvalue (with special semantics)
        if self.kind == 'pyconst_field':
            return self
        
        if self.is_python_value():
            raise TypeError("Cannot convert Python value to lvalue")
        if self.address is not None:
            # Has explicit address field - use it
            return ValueRef(
                kind='address',
                value=self.address,
                type_hint=self.type_hint,
                address=None,
                source_node=self.source_node,
                var_name=self.var_name,
                linear_path=self.linear_path
            )
        elif self.kind == 'address':
            # Already an lvalue - return as-is
            return self
        else:
            raise ValueError(f"Cannot convert ValueRef with kind='{self.kind}' and no address to lvalue")
    
    def __repr__(self) -> str:
        """String representation with type."""
        type_str = ""
        if self.type_hint:
            type_name = self.type_hint.get_name() if hasattr(self.type_hint, 'get_name') else str(self.type_hint)
            type_str = f", type={type_name}"
        
        value_str = ""
        if self.is_python_value():
            value_str = f", py_value={repr(self.value)}"
        
        addr_str = ", +addr" if self.address is not None else ""
        return f"ValueRef(kind={self.kind}{type_str}{value_str}{addr_str})"

    def is_struct_value(self):
        return is_struct_type(self.type_hint)

    def get_pc_type(self):
        from .builtin_entities.python_type import pyconst
        # Consider different kind
        if self.kind == 'python':
            return pyconst[self.value]
        return self.type_hint

    def get_ir_value(self) -> ir.Value:
        if self.is_python_value():
            # Return empty struct {} for pyconst (zero-sized type)
            return ir.Constant(ir.LiteralStructType([]), [])
        return self.value

    # def __getattribute__(self, name):
    #     # List of fields to track
    #     # if name in ('kind', 'value', 'type_hint', 'address', 'source_node', 
    #     #             'var_name', 'linear_path', 'vref_id', 'ir_value'):
    #     if name in ('kind', ):
    #         import traceback
    #         import sys
    #         # Print stack trace showing where the access happened
    #         stack = traceback.extract_stack()[:-1]  # Exclude this frame
    #         caller = stack[-1]
    #         if not caller.filename.endswith('valueref.py'):
    #             print(f"[VALUEREF ACCESS] .{name} at {caller.filename}:{caller.lineno} in {caller.name}", 
    #                 file=sys.stderr)
    #             pass
    #     return object.__getattribute__(self, name)



def ensure_ir(value: Union[ir.Value, ValueRef], visitor=None) -> ir.Value:
    """Return underlying ir.Value for either ValueRef or ir.Value.
    
    This function is backward compatible and works with both old and new ValueRef.
    Automatically promotes Python values to LLVM constants if possible.
    Special handling for pyconst_field kind.
    
    Args:
        value: ValueRef or ir.Value to extract IR from
        visitor: Optional visitor for context (needed for string constants)
    """
    if isinstance(value, ValueRef):
        return value.get_ir_value()
    if isinstance(value, ir.Value):
        return value
    raise TypeError(f"Cannot get IR value from {type(value)}")


def get_type(value: Union[ir.Value, ValueRef]):
    """Get LLVM type for either ValueRef or ir.Value.
    
    For ValueRef, derives from type_hint if available, otherwise from value.type.
    Raises TypeError for Python types.
    """
    if isinstance(value, ValueRef):
        return value.get_llvm_type()
    return value.type


def get_type_hint(value: ValueRef) -> Optional['Any']:
    """Get type hint from ValueRef if available."""
    if isinstance(value, ValueRef):
        return value.type_hint
    return None


# Backward compatibility alias
def get_pc_type(value: ValueRef) -> Optional['Any']:
    """Get type hint from ValueRef (backward compatibility)."""
    return get_type_hint(value)


def _infer_kind(value: Union[ir.Value, AnyType], 
                type_hint: Optional['Any'],
                address: Optional[ir.Value]) -> str:
    """Infer the kind of ValueRef based on value and type_hint.
    
    Inference rules:
    1. If address is provided -> 'address' (lvalue with address)
    2. If type_hint is a Python type -> 'python'
    3. If value is an ir.Value with pointer type -> 'pointer' or 'address'
    4. Otherwise -> 'value'
    
    Args:
        value: The value to wrap
        type_hint: Optional type hint
        address: Optional address for lvalue
    
    Returns:
        Inferred kind string
    """
    # Rule 1: If address is provided, it's an lvalue
    if address is not None:
        return 'address'
    
    # Rule 2: Check if type_hint indicates a Python type
    if type_hint is not None:
        if hasattr(type_hint, 'is_python_type') and type_hint.is_python_type():
            return 'python'
    
    # Rule 3: Check if value is an LLVM pointer type
    if isinstance(value, ir.Value):
        if isinstance(value.type, ir.PointerType):
            # Pointer value - could be 'pointer' or 'address'
            # Default to 'pointer' (rvalue pointer)
            return 'pointer'
        else:
            # Non-pointer LLVM value
            return 'value'
    
    # Rule 4: Non-LLVM value (Python object)
    # If we reach here without a Python type_hint, it's likely a Python value
    return 'python'


def wrap_value(value: Union[ir.Value, AnyType],
               kind: Optional[str] = None, 
               type_hint: Optional['Any'] = None,
               address: Optional[ir.Value] = None,  # Address for lvalue operations
               source_node: Optional[ast.AST] = None,
               var_name: Optional[str] = None,  # Variable name for linear tracking
               linear_path: Optional[tuple] = None,  # Linear path for tracking
               vref_id: Optional[str] = None,
               node: Optional[ast.AST] = None) -> ValueRef:  # Unique ID for ValueRef tracking
    """Wrap a value (LLVM IR or Python) in ValueRef with type information.
    
    Args:
        value: LLVM IR value for PC types, or Python object for Python types
        kind: Optional value kind ('address', 'value', 'python', 'pointer').
              If None, will be inferred from value and type_hint.
        type_hint: Optional language-level type information (BuiltinType or PythonType class)
        address: Optional address (pointer) for lvalue operations
        source_node: Optional source AST node for debugging
        var_name: Optional variable name for linear token tracking
        linear_path: Optional index path tuple for linear token tracking
        vref_id: Optional unique ID for ValueRef tracking
        node: Optional AST node for error reporting
    
    Returns:
        ValueRef with type information
    """
    # CRITICAL: Prevent double-wrapping ValueRef
    if isinstance(value, ValueRef):
        raise TypeError(
            f"BUG: Attempting to wrap a ValueRef in another ValueRef. "
            f"value.kind={value.kind}, value.type_hint={value.type_hint}, "
            f"requested kind={kind}, requested type_hint={type_hint}"
        )

    # Infer kind if not provided
    if kind is None:
        kind = _infer_kind(value, type_hint, address)

    # Valid kinds: address, value, python, pointer, pyconst_field, varargs
    valid_kinds = {'address', 'value', 'python', 'pointer', 'pyconst_field', 'varargs'}
    if kind not in valid_kinds:
        raise ValueError(f"Invalid kind: {kind}, must be one of {valid_kinds}")
    
    # Python types can use 'python', 'pyconst_field', or 'address' (for zero-sized alloca)
    # The 'address' kind is allowed because pyconst has zero-sized LLVM type {}
    if hasattr(type_hint, 'is_python_type') and type_hint.is_python_type():
        if kind not in ('python', 'pyconst_field', 'address'):
            logger.error(f"Python type requires kind='python', 'pyconst_field', or 'address', got '{kind}', type={type_hint}", node)
    
    # address kind requires address field (except for special cases)
    if kind == "address" and address is None:
        raise ValueError("Address kind requires address")

    if kind == "value" and address is not None:
        kind = "address"
        raise ValueError("Value kind cannot have address")
    
    return ValueRef(
        kind=kind,
        value=value,
        type_hint=type_hint,
        address=address,
        source_node=source_node,
        var_name=var_name,
        linear_path=linear_path,
        vref_id=vref_id
    )
