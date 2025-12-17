"""
Python Type Wrapper and pyconst Implementation

This module provides:
1. PythonType - wrapper for external Python objects (legacy)
2. pyconst[value] - zero-sized compile-time constant type

Key concept: pyconst[value] encodes constant values in the type system,
enabling zero-overhead conditional struct layouts.
"""

from .base import BuiltinEntity
from typing import Any, Optional
from ..valueref import wrap_value
from ..logger import logger
import ast


class _PythonTypeBase(BuiltinEntity):
    """Base class for PythonType - not registered as builtin entity.
    
    The underscore prefix prevents automatic registration.
    """
    pass


class PythonType(_PythonTypeBase):
    """Wrapper for compile-time constant values (implements pyconst semantics).
    
    PythonType now implements pyconst[value] semantics:
    - Zero-sized type (no LLVM representation)
    - Value encoded in type identity
    - Type checking: pyconst[5] != pyconst[3]
    - Field access returns constant (no load)
    - Field assignment type-checks value match
    
    Legacy usage: Wraps external Python objects for compile-time evaluation.
    New usage: Represents pyconst[value] types.
    
    Note: PythonType instances are created via PythonType.wrap() or pyconst[value],
    not registered as builtin entities since each instance wraps a different value.
    """
    
    def __init__(self, python_obj: Any, is_constant: bool = False):
        """Initialize a PythonType instance wrapping a Python object.
        
        Args:
            python_obj: The Python object to wrap
            is_constant: Whether this is a compile-time constant
        """
        self._python_object = python_obj
        self._python_type = type(python_obj)
        
        # Determine if this can be a compile-time constant
        # Include lists, tuples, dicts for compile-time subscripting
        # Include callable objects (functions) for compile-time evaluation
        self._is_constant = is_constant or isinstance(python_obj, (int, float, str, bool, type(None), list, tuple, dict)) or callable(python_obj)
        self._constant_value = python_obj if self._is_constant else None
    
    @classmethod
    def wrap(cls, python_obj: Any, is_constant: bool = False):
        """Create a PythonType wrapper for a Python object.
        
        Args:
            python_obj: The Python object to wrap
            is_constant: Whether this is a compile-time constant
            
        Returns:
            A new PythonType instance wrapping the object
        """
        return cls(python_obj, is_constant)
    
    @classmethod
    def get_name(cls) -> str:
        """Return the entity name."""
        return "pyconst"
    
    def get_instance_name(self) -> str:
        """Return the instance-specific name for this pyconst[value]."""
        if self._is_constant:
            return f"pyconst[{repr(self._constant_value)}]"
        if self._python_type:
            return f"python_{self._python_type.__name__}"
        return "python_object"
    
    def can_be_type(self) -> bool:
        """pyconst types can be used as type hints."""
        return True
    
    def get_size_bytes(self) -> int:
        """pyconst is zero-sized."""
        if self._is_constant:
            return 0
        return 0  # Non-constant Python types also have no LLVM representation
    
    def get_type_id(self) -> str:
        """Get type ID for pyconst[value]."""
        if self._is_constant:
            # Use value repr for type ID
            value_repr = repr(self._constant_value)
            # Sanitize for valid identifier
            value_repr = value_repr.replace("'", "").replace('"', "").replace(" ", "_")
            return f"pyconst_{value_repr}"
        return "python_object"
    
    def can_be_called(self) -> bool:
        """Python objects might be callable."""
        if self._python_object is not None:
            return callable(self._python_object)
        return False
    
    def is_python_type(self) -> bool:
        """Identify this as a Python type."""
        return True
    
    def is_constant(self) -> bool:
        """Check if this is a compile-time constant."""
        return self._is_constant
    
    def get_python_object(self) -> Any:
        """Get the wrapped Python object."""
        return self._python_object
    
    def get_constant_value(self) -> Any:
        """Get the constant value if this is a constant."""
        if not self._is_constant:
            logger.error(f"{self.get_instance_name()} is not a constant", node=None, exc_type=ValueError)
        return self._constant_value
    
    def get_llvm_type(self, module_context=None):
        """pyconst types are zero-sized, represented as empty struct {}.
        
        Returns an empty struct type {} for compatibility with struct layout.
        This allows struct[pyconst[1], pyconst[2]] to become {{}, {}}.
        """
        from llvmlite import ir
        # Return empty struct {} for zero-sized representation
        return ir.LiteralStructType([])
    
    def handle_field_access(self, visitor, base, field_index, field_name, node):
        """Handle field access for pyconst fields in structs.
        
        Returns a special ValueRef that can be used both as a value and as an lvalue.
        
        Args:
            visitor: AST visitor
            base: Base struct ValueRef (ignored for pyconst)
            field_index: Index of this field in struct (unused)
            field_name: Name of the field (for error messages)
            node: AST node (for error reporting)
        
        Returns:
            ValueRef with constant value and special metadata for assignment
        """
        if not self._is_constant:
            logger.error(f"Cannot access non-constant Python field '{field_name}'", node=node, exc_type=TypeError)
        
        # Create a ValueRef that acts as both constant value and assignable lvalue
        # Strategy: Return as Python value but with special attributes
        from ..valueref import wrap_value
        from llvmlite import ir
        
        # Create null pointer as dummy address
        null_ptr = ir.Constant(ir.IntType(8).as_pointer(), None)
        
        # Return ValueRef with:
        # - kind="pyconst_field" for lvalue protocol
        # - But also with the actual constant value for expression use
        result = wrap_value(
            self._constant_value,  # The actual constant value
            kind="pyconst_field",
            type_hint=self,
            address=null_ptr  # Dummy address for lvalue protocol
        )
        # Store metadata for type checking during assignment
        result._pyconst_value = self._constant_value
        result._pyconst_type = self
        return result
    
    def handle_field_assignment(self, visitor, base, field_index, field_name, value_ref, node):
        """Handle field assignment for pyconst fields in structs.
        
        Type checking rule: Assignment must match exact value
        - x.field = 5 where field: pyconst[5] -> OK
        - x.field = 3 where field: pyconst[5] -> ERROR
        
        Args:
            visitor: AST visitor
            base: Base struct ValueRef
            field_index: Index of this field in struct
            field_name: Name of the field
            value_ref: Value being assigned
            node: AST node
        
        Returns:
            None (assignment is a no-op for zero-sized fields)
        """
        if not self._is_constant:
            logger.error(f"Cannot assign to non-constant Python field '{field_name}'", node=node, exc_type=TypeError)
        
        # Extract the value being assigned
        if value_ref.is_python_value():
            assigned_value = value_ref.value
        elif hasattr(value_ref.value, 'constant'):
            # LLVM constant
            assigned_value = value_ref.value.constant
        else:
            logger.error(
                f"Cannot assign runtime value to pyconst field '{field_name}'. "
                f"pyconst fields require compile-time constant values.",
                node=node, exc_type=TypeError
            )
        
        # Type check: value must match exactly
        if assigned_value != self._constant_value:
            logger.error(
                f"Type mismatch: cannot assign {repr(assigned_value)} to field '{field_name}' "
                f"of type pyconst[{repr(self._constant_value)}]. "
                f"pyconst fields require exact value match.",
                node=node, exc_type=TypeError
            )
        
        # Assignment is valid but is a no-op (zero-sized field)
    
    def handle_call(self, visitor, func_ref, args, node: ast.Call):
        """Handle calling a Python object.
        
        Strategy:
        1. If the Python object has a 'handle_call' attribute (function): 
           call it with (visitor, func_ref, args, node) - allows custom IR generation
        2. Otherwise, if constant and callable: evaluate at compile time using constant args
        3. Otherwise: raise error with helpful message
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable
            args: Pre-evaluated arguments (list of ValueRef)
            node: ast.Call node
        """
        if self._is_constant and callable(self._python_object):
            # Check if the callable has a custom handle_call method
            if hasattr(self._python_object, 'handle_call') and callable(self._python_object.handle_call):
                # Call the custom handler with visitor, func_ref, args, and node
                # This allows Python functions to generate IR directly
                return self._python_object.handle_call(visitor, func_ref, args, node)
            
            # Default behavior: compile-time evaluation
            return self._eval_call(visitor, node, self._python_object)
        
        logger.error(
            f"Cannot call Python object '{self._python_type.__name__}' in compiled code.\n"
            f"Hint: Only compile-time callable constants with constant arguments are supported.",
            node=node, exc_type=NotImplementedError
        )
    
    def handle_attribute(self, visitor, base, attr_name, node: ast.Attribute):
        """Handle attribute access on a Python object.
        
        Returns the attribute as a new PythonType wrapper.
        If the attribute is callable, it can be called later via handle_call.
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            attr_name: Attribute name (str)
            node: ast.Attribute node
        """
        # Check if the Python object has its own handle_attribute method
        if hasattr(self._python_object, 'handle_attribute') and callable(self._python_object.handle_attribute):
            # Delegate to the object's handle_attribute
            return self._python_object.handle_attribute(visitor, base, attr_name, node)
        
        if not self._is_constant:
            logger.error(
                f"Cannot access attribute '{attr_name}' on non-constant Python object '{self._python_type.__name__}'",
                node=node, exc_type=NotImplementedError
            )
        
        try:
            attr_value = getattr(self._python_object, attr_name)
        except AttributeError:
            logger.error(
                f"Python object '{self._python_type.__name__}' has no attribute '{attr_name}'",
                node=node, exc_type=AttributeError
            )
        
        # Wrap the attribute as a new PythonType
        from ..valueref import wrap_value
        attr_type = PythonType.wrap(attr_value, is_constant=True)
        
        # Store reference to the base object for method calls
        # This allows us to bind 'self' when the method is called
        if callable(attr_value) and hasattr(attr_value, '__self__'):
            # It's a bound method, keep it as-is
            pass
        
        return wrap_value(attr_value, kind="python", type_hint=attr_type)
    
    def handle_method_call(self, visitor, node: ast.Call, method_name: str):
        """Handle calling a method on a Python object at compile time.
        
        DEPRECATED: Use handle_attribute + handle_call instead.
        This is kept for backward compatibility.
        """
        if not self._is_constant:
            logger.error(
                f"Cannot call method '{method_name}' on non-constant Python object '{self._python_type.__name__}'",
                node=node, exc_type=NotImplementedError
            )
        base_obj = self._python_object
        try:
            target = getattr(base_obj, method_name)
        except AttributeError:
            logger.error(f"Python object '{self._python_type.__name__}' has no attribute '{method_name}'",
                        node=node, exc_type=AttributeError)
        if not callable(target):
            logger.error(f"Attribute '{method_name}' of '{self._python_type.__name__}' is not callable",
                        node=node, exc_type=TypeError)
        return self._eval_call(visitor, node, target)
    
    def _eval_call(self, visitor, node: ast.Call, target_callable):
        """Evaluate a Python callable with arguments (constant or runtime)."""
        from llvmlite import ir
        
        # Evaluate and extract arguments
        args_py = []
        for arg_node in node.args:
            # First, check if arg_node is a simple constant in AST
            if isinstance(arg_node, ast.Constant):
                # Use the Python constant value directly
                args_py.append(arg_node.value)
                continue
            
            # Otherwise, evaluate the expression
            valref = visitor.visit_expression(arg_node)
            
            # Extract appropriate value based on ValueRef kind
            if hasattr(valref, 'is_python_value') and valref.is_python_value():
                # Python value: use .value directly (the Python object)
                args_py.append(valref.value)
            else:
                # LLVM value: check if it's a constant we can extract
                if isinstance(valref.value, ir.Constant):
                    # Extract constant value if possible
                    if isinstance(valref.value.type, ir.IntType):
                        args_py.append(valref.value.constant)
                    elif isinstance(valref.value.type, (ir.FloatType, ir.DoubleType)):
                        args_py.append(valref.value.constant)
                    else:
                        # Cannot extract, pass ValueRef
                        args_py.append(valref)
                else:
                    # Runtime LLVM value - pass the ValueRef
                    args_py.append(valref)
        
        try:
            result = target_callable(*args_py)
        except Exception as e:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Python call failed: {e}\nOriginal traceback:\n{tb_str}", node=node, exc_type=ValueError)
        
        # Wrap result back
        return self._wrap_constant_result(visitor, result)
    
    def handle_subscript(self, visitor, base, index, node: ast.Subscript):
        """Handle subscripting a Python object (always type subscript for PC types).
        
        Design principle:
        - If _python_object is a PC type class (ptr, struct, etc.): type subscript
        - If _python_object is a Python sequence (list, dict): compile-time indexing
        
        For type subscripts:
        - ptr[i32]: index is pyconst[i32]
        - struct[x: i32]: index is refined[struct[...], "slice"]
        - struct[x: i32, y: f64]: index is refined[struct[...], "tuple"] of slices
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            index: Pre-evaluated index (ValueRef) - always provided now
            node: ast.Subscript node
        """
        from ..valueref import wrap_value
        
        # Check if the Python object is a PC type class with handle_type_subscript
        # This handles PC type classes (ptr, struct, array, func, etc.)
        if isinstance(self._python_object, type) and hasattr(self._python_object, 'handle_type_subscript'):
            # Extract value from index ValueRef and pass to normalize_subscript_items
            # normalize_subscript_items handles all formats: slice, tuple, refined[..., "slice/tuple"]
            if index.is_python_value():
                items = index.value
            elif index.type_hint is not None and hasattr(index.type_hint, '_field_types'):
                # Non-python value with struct type_hint (from visit_Tuple without all-python special case)
                # Extract items from the struct's _field_types
                items = self._extract_struct_type_items(index.type_hint)
            else:
                logger.error(f"Type subscript index must be a python value, got {index}", node=node, exc_type=TypeError)
            
            normalized = self._python_object.normalize_subscript_items(items)
            result_type = self._python_object.handle_type_subscript(normalized)
            return wrap_value(result_type, kind="python", type_hint=PythonType.wrap(result_type))

        # Python sequence subscript (list, dict, etc.)
        # Extract the index value
        if index.is_python_value():
            index_val = index.value
        else:
            # Try to extract constant from AST node
            import ast
            if isinstance(node.slice, ast.Constant):
                index_val = node.slice.value
            else:
                logger.error(f"List subscript requires constant index at compile time, got {type(node.slice)}",
                            node=node, exc_type=TypeError)
        
        result = self._python_object[index_val]
        return self._wrap_constant_result(visitor, result)
    
    def _extract_struct_type_items(self, struct_type):
        """Extract items from a struct type (from visit_Tuple without all-python special case).
        
        This handles the case where visit_Tuple creates a runtime struct instead of
        a refined[struct[...], "tuple"]. The struct's _field_types contain pyconst
        or refined types that we need to extract.
        
        Args:
            struct_type: A struct type with _field_types attribute
        
        Returns:
            Tuple of items suitable for normalize_subscript_items
        """
        from .refined import RefinedType
        
        field_types = getattr(struct_type, '_field_types', [])
        if not field_types:
            logger.error(f"Struct type has no field types: {struct_type}", node=None, exc_type=TypeError)
        
        items = []
        for field_type in field_types:
            # Extract actual value from pyconst wrapper
            actual_value = field_type
            if hasattr(field_type, '_python_object'):
                actual_value = field_type._python_object
            elif hasattr(field_type, 'get_python_object'):
                actual_value = field_type.get_python_object()
            
            # Check if it's a refined type with "slice" tag
            if isinstance(actual_value, type) and issubclass(actual_value, RefinedType):
                tags = getattr(actual_value, '_tags', [])
                if "slice" in tags:
                    # Use base class method to extract slice
                    from .base import BuiltinType
                    items.append(BuiltinType._extract_slice_from_refined(actual_value))
                    continue
            
            items.append(actual_value)
        
        return tuple(items)
    
    def _wrap_constant_result(self, visitor, result):
        """Wrap a compile-time evaluation result.
        
        Keep Python values as PythonType until conversion is needed.
        If result is a ValueRef, return it directly.
        """
        from ..valueref import wrap_value, ValueRef
        
        # If result is already a ValueRef, return it directly
        if isinstance(result, ValueRef):
            return result
        
        # Keep all Python values as PythonType (lazy conversion)
        # Conversion to LLVM will happen when needed (assignment, return, etc.)
        python_type = PythonType.wrap(result, is_constant=True)
        return wrap_value(result, kind="python", type_hint=python_type)
    
    def promote_to_pc_type(self, target_pc_type):
        """Promote this Python value to a specific PC type.
        
        Args:
            target_pc_type: Target PC type (i32, f64, etc.)
        
        Returns:
            ValueRef with LLVM IR value and target PC type hint
        """
        if not self._is_constant:
            logger.error(f"Cannot promote non-constant Python value to PC type", node=None, exc_type=TypeError)
        
        from llvmlite import ir
        from ..valueref import wrap_value
        
        python_val = self._python_object
        
        # Get target LLVM type
        target_llvm_type = target_pc_type.get_llvm_type()
        
        # Convert based on target type
        if isinstance(target_llvm_type, ir.IntType):
            if not isinstance(python_val, (bool, int)):
                logger.error(f"Cannot convert {type(python_val).__name__} to integer type",
                            node=None, exc_type=TypeError)
            ir_val = ir.Constant(target_llvm_type, int(python_val))
            return wrap_value(ir_val, kind="value", type_hint=target_pc_type)
        elif isinstance(target_llvm_type, (ir.FloatType, ir.DoubleType)):
            if not isinstance(python_val, (int, float)):
                logger.error(f"Cannot convert {type(python_val).__name__} to float type",
                            node=None, exc_type=TypeError)
            ir_val = ir.Constant(target_llvm_type, float(python_val))
            return wrap_value(ir_val, kind="value", type_hint=target_pc_type)
        else:
            logger.error(f"Cannot promote Python value to PC type {target_pc_type}",
                        node=None, exc_type=TypeError)
    
    def promote_to_default_pc_type(self):
        """Promote this Python value to default PC type based on Python type.
        
        Default promotions:
        - bool -> i32 (C default)
        - int -> i32 (if fits) or i64
        - float -> f64
        
        Returns:
            ValueRef with LLVM IR value and default PC type hint
        """
        if not self._is_constant:
            logger.error(f"Cannot promote non-constant Python value to PC type", node=None, exc_type=TypeError)
        
        from llvmlite import ir
        from ..valueref import wrap_value
        
        python_val = self._python_object
        
        if isinstance(python_val, bool):
            from .types import i32
            ir_val = ir.Constant(ir.IntType(32), int(python_val))
            return wrap_value(ir_val, kind="value", type_hint=i32)
        elif isinstance(python_val, int):
            from .types import i32, i64
            if -2**31 <= python_val < 2**31:
                ir_val = ir.Constant(ir.IntType(32), python_val)
                return wrap_value(ir_val, kind="value", type_hint=i32)
            else:
                ir_val = ir.Constant(ir.IntType(64), python_val)
                return wrap_value(ir_val, kind="value", type_hint=i64)
        elif isinstance(python_val, float):
            from .types import f64
            ir_val = ir.Constant(ir.DoubleType(), python_val)
            return wrap_value(ir_val, kind="value", type_hint=f64)
        else:
            logger.error(f"Cannot promote Python type {type(python_val).__name__} to PC type",
                        node=None, exc_type=TypeError)


def is_python_type(type_hint) -> bool:
    """Check if a type hint is a PythonType."""
    if type_hint is None:
        return False
    return isinstance(type_hint, PythonType)


class pyconst(BuiltinEntity):
    """Zero-sized type encoding compile-time constant values.
    
    Usage:
        pyconst[5]       # Type representing constant 5
        pyconst[100]     # Type representing constant 100
        
    Properties:
        - Zero-sized: sizeof(pyconst[value]) == 0
        - Type checking: pyconst[5] != pyconst[3]
        - In structs: zero-sized fields
        - Field access: always returns constant (no load)
        - Field assignment: type-checks value match
    
    Implementation:
        pyconst[value] is syntactic sugar for PythonType.wrap(value, is_constant=True)
    """
    
    @classmethod
    def get_name(cls) -> str:
        return "pyconst"
    
    @classmethod
    def can_be_type(cls) -> bool:
        return True
    
    @classmethod
    def __class_getitem__(cls, value):
        """Support pyconst[value] subscript syntax.
        
        Args:
            value: Compile-time constant value
        
        Returns:
            PythonType instance representing pyconst[value]
        """
        return PythonType.wrap(value, is_constant=True)

