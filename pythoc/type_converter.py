"""
Centralized type conversion service for LLVM IR types.

This module provides a single place for value conversions and small utilities
needed during IR generation. ALL EXTERNAL CALLERS MUST PASS PC TYPES (BuiltinEntity
classes). Raw LLVM types are used only internally for IR instruction selection.
Reverse inference from LLVM to PC is intentionally not supported.
"""

from typing import Optional, Union, Any, Tuple, cast
from llvmlite import ir

from .valueref import ValueRef, ensure_ir, wrap_python_constant, wrap_value, get_type, get_type_hint
from .type_check import is_struct_type, is_enum_type
from .logger import logger
from .builtin_entities.base import BuiltinType
from .schema_protocol import get_schema_field_types, is_schema_type
import ast


def raise_if_compile_level_callable(python_val):
    """Uniform boundary error for compile-level callables.

    Compile-level callables (nested-def closures, lambdas, yield-function
    placeholders, @inline wrappers) are compile-time-only values: calling
    one inline-expands it. Using one as a *runtime* value (storing into a
    PC-typed variable, converting to func[...], returning from a compiled
    function, arithmetic, ...) is always an error; raise one uniform
    message for it here. No-op for anything else.
    """
    if not getattr(python_val, '_pc_compile_level_callable', False):
        return
    name = (getattr(python_val, 'name', None)
            or getattr(python_val, '__name__', None)
            or type(python_val).__name__)
    raise TypeError(
        f"Compile-level callable '{name}' cannot be used as a runtime value; "
        f"it can only be called (which inline-expands it at compile time). "
        f"Convert it explicitly with instantiate(...) where supported"
    )


def strip_qualifiers(pc_type):
    """Strip type qualifiers (const, volatile, static) but NOT refined types
    
    Returns the underlying type by stripping qualifiers ONLY.
    Refined types are NOT stripped because they have semantic meaning
    (runtime constraints) and are not mere compile-time qualifiers.
    
    Example:
        const[i32] -> i32
        volatile[ptr[i32]] -> ptr[i32]
        static[const[i32]] -> i32 (strips nested qualifiers)
        refined[is_positive] -> refined[is_positive] (NOT stripped!)
    """
    if pc_type is None:
        return None
    
    # Refined types are NOT qualifiers - they carry semantic constraints
    # Do NOT strip them!
    if hasattr(pc_type, '_is_refined') and pc_type._is_refined:
        return pc_type
    
    # Check if this is a qualifier type
    if hasattr(pc_type, 'qualified_type') and pc_type.qualified_type is not None:
        # Recursively strip nested qualifiers
        return strip_qualifiers(pc_type.qualified_type)
    
    return pc_type


def get_base_type(pc_type):
    """Get the base type, stripping both qualifiers AND refined types

    This is used for operations that need to work with the underlying
    value type (e.g., arithmetic operations, type category checks).

    Example:
        const[i32] -> i32
        refined[is_positive] -> i32
        const[refined[is_positive]] -> i32
    """
    if pc_type is None:
        return None

    # First strip qualifiers
    pc_type = strip_qualifiers(pc_type)

    # Then strip refined types to get to the base type
    if hasattr(pc_type, '_is_refined') and pc_type._is_refined:
        if hasattr(pc_type, '_is_single_param') and pc_type._is_single_param:
            # Single-parameter refined type: get underlying type
            if pc_type._param_types and len(pc_type._param_types) > 0:
                underlying_type = pc_type._param_types[0]
                # Recursively process in case underlying type is also refined
                return get_base_type(underlying_type)
        # Multi-parameter refined type: keep as is (it's a struct)
        return pc_type

    return pc_type


def forget_refinement(pc_type):
    """Strip refinement from a type for operation results

    This is the "forget" operation for refinement types at operation boundaries.
    When a value participates in arithmetic/bitwise operations, the result
    should NOT retain the refinement because predicates are not closed
    under most operations.

    Example:
        refined[i32, is_positive] -> i32
        refined[i32, is_positive, "checked"] -> i32
        const[refined[i32, is_positive]] -> const[i32]
        i32 -> i32 (no change for non-refined)

    This differs from get_base_type() in that:
    - get_base_type() also strips qualifiers
    - forget_refinement() preserves qualifiers, only strips refinement

    Use cases:
    - Arithmetic operations: x + y where x is refined
    - Bitwise operations: x | y where x is refined
    - Unary operations: -x where x is refined
    """
    if pc_type is None:
        return None

    # Handle refined types directly
    if hasattr(pc_type, '_is_refined') and pc_type._is_refined:
        # For single-param refined types, return the base type
        if hasattr(pc_type, '_base_type') and pc_type._base_type is not None:
            return pc_type._base_type
        # For multi-param refined (struct), return the struct type
        if hasattr(pc_type, '_struct_type') and pc_type._struct_type is not None:
            return pc_type._struct_type
        # Fallback: try param_types
        if hasattr(pc_type, '_param_types') and pc_type._param_types:
            if len(pc_type._param_types) == 1:
                return pc_type._param_types[0]
        # If nothing else works, return as-is
        return pc_type

    # Handle qualified types that wrap refined types (e.g., const[refined[i32, pred]])
    if hasattr(pc_type, 'qualified_type') and pc_type.qualified_type is not None:
        inner = forget_refinement(pc_type.qualified_type)
        if inner != pc_type.qualified_type:
            # Reconstruct qualifier wrapper around the stripped inner type
            from .ir_helpers import propagate_qualifiers
            return propagate_qualifiers(pc_type, inner)

    return pc_type


class TypeConverter:
    """
    Centralized type conversion service for LLVM IR types.

    - Integer to integer (sext/zext/trunc)
    - Integer to float (sitofp/uitofp)
    - Float to integer (fptosi/fptoui)
    - Float to float (fpext/fptrunc)
    - Pointer to pointer (bitcast)
    - Pointer <-> integer (ptrtoint/inttoptr)
    - Boolean conversions
    - Zero constant creation
    """

    def __init__(self, visitor):
        self._visitor = visitor
        self._conversion_registry = self._build_conversion_registry()

    @property
    def builder(self):
        return self._visitor.builder

    def convert(
        self,
        value: Union[ir.Value, ValueRef],
        target_type: type,
        node: Optional[ast.AST] =None
    ) -> ValueRef:
        """
        Convert value to target PC type using appropriate LLVM instruction.

        Args:
            value: Source value (ValueRef or ir.Value)
            target_type: Target PC type class (must have get_llvm_type method)
        
        Returns:
            ValueRef with correct type_hint
        """
        if isinstance(value, ir.Value):
            raise ValueError(f"Cannot convert raw LLVM value {value}")
        if value.has_place() or value.has_storage():
            from .valueref import read_rvalue
            value = read_rvalue(value, self.builder)
        if value.is_python_value():
            raise_if_compile_level_callable(value.get_python_value())
        
        # A qualifier can wrap a refinement. Decide the semantic relation
        # before the LLVM fast path or storage qualifier erasure.
        from .type_descriptor import can_convert
        target_is_refined = getattr(strip_qualifiers(target_type), '_is_refined', False)
        if target_is_refined and not can_convert(
                value.type_hint, target_type, mode='explicit', value=value):
            target_name = target_type.get_name() if hasattr(target_type, 'get_name') else str(target_type)
            source_name = value.type_hint.get_name() if value.type_hint and hasattr(value.type_hint, 'get_name') else str(value.type_hint)
            logger.error(f"Cannot directly convert from {source_name} to refined type {target_name}. "
                f"Refined types must be constructed using assume() or refine().",
                node, exc_type=TypeError)
        if not can_convert(value.type_hint, target_type, mode='explicit', value=value):
            from .type_descriptor import conversion_failure
            detail = conversion_failure(value.type_hint, target_type, value)
            if detail is not None:
                raise TypeError(detail)
            source_name = (value.type_hint.get_name()
                           if value.type_hint is not None and hasattr(value.type_hint, 'get_name')
                           else str(value.type_hint))
            target_name = (target_type.get_name() if hasattr(target_type, 'get_name')
                           else str(target_type))
            logger.error(f"Cannot convert {source_name} to {target_name}",
                         node, exc_type=TypeError)
        
        # Strip qualifiers for comparison and conversion
        stripped_target = strip_qualifiers(target_type)
        stripped_source = strip_qualifiers(value.type_hint) if value.type_hint else None

        # C-style discard cast: converting any already-evaluated value to void
        # is a no-op; the value's side effects happened before the conversion.
        from .builtin_entities.types import void as _void_type
        if stripped_target is _void_type:
            return wrap_value(None, kind="python", type_hint=_void_type)

        from .type_descriptor import same_type
        if stripped_source is not None and same_type(stripped_source, stripped_target):
            # Equal PythoC types can still carry different LLVM pointer
            # representations while an identified record is being completed.
            # Skip lowering only when the actual IR type already matches.
            if value.is_python_value():
                return value.clone(type_hint=target_type)
            base = get_base_type(target_type)
            module = getattr(self._visitor, 'module', None)
            module_context = getattr(module, 'context', None)
            target_ir_type = base.get_llvm_type(module_context)
            source_ir_type = ensure_ir(value).type
            if (source_ir_type == target_ir_type
                    or (value.has_place()
                        and isinstance(source_ir_type, ir.PointerType)
                        and source_ir_type.pointee == target_ir_type)):
                return value.clone(type_hint=target_type)
        
        # Step 0: Handle PythonType (pyconst) target - type checking only, no IR conversion
        from .builtin_entities.python_type import PythonType
        if isinstance(stripped_target, PythonType):
            # Target is pyconst[value] - this is a zero-sized type
            # Only type checking is needed, no actual conversion
            if value.is_python_value():
                assigned_value = value.get_python_value()
            elif hasattr(value.value, 'constant'):
                # LLVM constant - extract value
                assigned_value = value.value.constant
            else:
                raise TypeError(
                    f"Cannot assign runtime value to pyconst field. "
                    f"pyconst fields require compile-time constant values."
                )
            
            # Type check: value must match exactly
            expected_value = stripped_target.get_constant_value()
            if assigned_value != expected_value:
                raise TypeError(
                    f"Type mismatch: cannot assign {repr(assigned_value)} to "
                    f"{stripped_target.get_instance_name()}. "
                    f"Expected value: {repr(expected_value)}"
                )
            
            # Return a pyconst ValueRef (no actual IR value needed)
            return wrap_value(
                assigned_value,
                kind="python",
                type_hint=target_type
            )
        
        # Step 1: Handle @compile / @extern wrapper -> func pointer
        # When source is a Python value that is a callable wrapper, lower it
        # to a func pointer.  This path handles wrappers used as *values*
        # (e.g. `return add`, `op = add`, function pointer arguments).
        # The *call* path goes through handle_call -> callable_lowering directly.
        if isinstance(value, ValueRef) and value.is_python_value():
            python_val = value.get_python_value()
            from .builtin_entities.func import func as func_type_cls
            target_is_func = (
                isinstance(stripped_target, type)
                and issubclass(stripped_target, func_type_cls)
            )
            if hasattr(python_val, '_is_compiled') and python_val._is_compiled:
                if getattr(python_val, '_is_parametric', False):
                    logger.error(
                        "A parametric function cannot be used as a first-class value "
                        "without supplying its compile-time arguments first.",
                        node=node, exc_type=TypeError,
                    )
                from .callable_lowering import lower_compile_wrapper
                caller_group_key = getattr(self._visitor, 'current_group_key', None)
                lowered = lower_compile_wrapper(
                    python_val, self._visitor.module, caller_group_key,
                    node=node,
                )
                # If the requested target is not a function type (e.g. ptr[void]),
                # continue converting from the lowered function pointer.
                if not target_is_func:
                    return self.convert(lowered, target_type, node)
                _module_ctx = getattr(getattr(self._visitor, 'module', None), 'context', None)
                if not self._func_types_compatible(lowered.type_hint, stripped_target, _module_ctx):
                    func_name = getattr(python_val, '__name__', repr(python_val))
                    logger.error(
                        f"Function '{func_name}' has signature {lowered.type_hint.get_name()}, "
                        f"but is being used as {stripped_target.get_name()}",
                        node=node, exc_type=TypeError,
                    )
                return lowered.clone(type_hint=target_type)
            if getattr(python_val, '_is_extern', False):
                from .callable_lowering import lower_extern_wrapper
                caller_group_key = getattr(self._visitor, 'current_group_key', None)
                lowered = lower_extern_wrapper(
                    python_val, self._visitor.module, caller_group_key,
                    node=node,
                )
                if not target_is_func:
                    return self.convert(lowered, target_type, node)
                _module_ctx = getattr(getattr(self._visitor, 'module', None), 'context', None)
                if not self._func_types_compatible(lowered.type_hint, stripped_target, _module_ctx):
                    func_name = getattr(python_val, 'func_name', repr(python_val))
                    logger.error(
                        f"Extern function '{func_name}' has signature {lowered.type_hint.get_name()}, "
                        f"but is being used as {stripped_target.get_name()}",
                        node=node, exc_type=TypeError,
                    )
                return lowered.clone(type_hint=target_type)
            # Otherwise, auto-promote Python values to PC values
            value = self._promote_python_to_pc(python_val, stripped_target)
            return value
        
        # For actual conversion operations, use base types (strip refined types too)
        base_target = get_base_type(target_type)
        base_source = get_base_type(value.type_hint) if value.type_hint else None

        # Step 2: (enum->int restriction moved to ImplicitCoercer)

        # Step 3: Handle struct to enum conversion
        # struct[pyconst[Tag], payload] -> EnumType
        if is_struct_type(base_source) and is_enum_type(base_target):
            return self._convert_struct_to_enum(value, base_source, base_target, target_type)

        # Step 4: Handle struct to struct conversion (field-by-field)
        if is_struct_type(base_source) and is_struct_type(base_target):
            if not can_convert(value.type_hint, target_type, mode='explicit', value=value):
                raise TypeError(
                    f'Cannot convert {base_source.get_name()} to '
                    f'{base_target.get_name()}')
            return self._convert_struct_to_struct(value, base_source, base_target, target_type)

        # Validate target is a PC type (use base version)
        if not isinstance(base_target, type) or not hasattr(base_target, 'get_llvm_type'):
            raise TypeError(f"Target type {target_type} is not a valid PC builtin type with get_llvm_type()")

        # Extract LLVM type from pythoc type (use base version)
        # All PC types now accept module_context parameter uniformly
        module = getattr(self._visitor, 'module', None)
        module_context = getattr(module, 'context', None)
        target_llvm_type = base_target.get_llvm_type(module_context)
        
        # Get source LLVM type
        source_ir = ensure_ir(value)
        source_type = source_ir.type

        # Fast path: same IR type
        if source_type == target_llvm_type:
            if value.type_hint is not None and not can_convert(
                    value.type_hint, target_type, mode='explicit', value=value):
                raise TypeError(
                    f'Cannot convert {value.type_hint.get_name()} to '
                    f'{target_type.get_name()}')
            return wrap_value(source_ir, kind="value", type_hint=target_type)

        # Infer signedness from pythoc types (use base versions)
        source_is_unsigned = False
        target_is_unsigned = False
        source_pc_type = base_source
        if hasattr(source_pc_type, 'is_signed'):
            source_is_unsigned = not source_pc_type.is_signed()
        if hasattr(base_target, 'is_signed'):
            target_is_unsigned = not base_target.is_signed()

        # Dispatch via registry (uses LLVM types for IR instruction selection)
        conversion_key = (type(source_type), type(target_llvm_type))
        converter_func = self._conversion_registry.get(conversion_key)
        if converter_func is None:
            value_hint = get_type_hint(value) if hasattr(value, '__class__') else None
            logger.error(
                f"No conversion from {source_type} to {target_llvm_type}"
                f"  value type: {type(value)}, value hint: {value_hint}"
                f"  target_type: {target_type}", node)

        return converter_func(
            value,
            source_type,
            target_llvm_type,
            source_is_unsigned,
            target_is_unsigned,
            target_type,
        )

    def _promote_python_to_pc(self, python_val, target_type) -> ValueRef:
        """Promote Python primitive or literal carrier value to a PC-typed ValueRef."""
        from .builtin_entities.python_type import PythonType
        from .literal_protocol import get_sequence_elements, is_mapping_carrier, is_sequence_carrier

        # @compile / @extern function wrappers used as rvalues (e.g. function
        # pointer comparisons) lower to func pointers via the general convert
        # path rather than primitive promotion.
        if hasattr(python_val, '_is_compiled') or getattr(python_val, '_is_extern', False):
            wrapped = wrap_value(python_val, kind="python", type_hint=PythonType.wrap(python_val))
            return self.convert(wrapped, target_type)

        if is_sequence_carrier(python_val):
            sequence_items = list(get_sequence_elements(python_val))

            if hasattr(target_type, '_is_enum') and target_type._is_enum:
                target_type_ref = wrap_python_constant(target_type)
                arg_refs = []
                for item in sequence_items:
                    if isinstance(item, ValueRef):
                        arg_refs.append(item)
                    else:
                        item_py_type = PythonType.wrap(item, is_constant=True)
                        arg_refs.append(wrap_value(item, kind="python", type_hint=item_py_type))
                if len(arg_refs) not in (1, 2):
                    raise TypeError(f"Enum initialization requires 1 or 2 elements, got {len(arg_refs)}")
                return self._visitor.value_dispatcher.handle_type_call(target_type_ref, arg_refs, None)

            if hasattr(target_type, "is_array") and target_type.is_array():
                list_carrier = self._sequence_literal_to_pc_list(python_val)
                return self._convert_pc_list_to_array(list_carrier, target_type)

            if is_schema_type(target_type):
                return self._convert_tuple_to_struct(sequence_items, target_type)

            raise TypeError(f"Cannot promote sequence literal {python_val} to PC type {target_type}")

        if is_mapping_carrier(python_val):
            if is_schema_type(target_type):
                return self._convert_dict_to_struct(python_val, target_type)
            raise TypeError(f"Cannot promote mapping literal to PC type {target_type}")

        if not isinstance(python_val, (int, float, bool, str, type(None))):
            raise_if_compile_level_callable(python_val)
            raise TypeError(
                f"Cannot promote Python value of type {type(python_val).__name__} to PC type. "
                f"Only primitive types and literal carriers are supported. Got: {repr(python_val)}"
            )

        if hasattr(target_type, '_is_enum') and target_type._is_enum and isinstance(python_val, int):
            target_type_ref = wrap_python_constant(target_type)
            tag_py_type = PythonType.wrap(python_val, is_constant=True)
            tag_ref = wrap_value(python_val, kind="python", type_hint=tag_py_type)
            return self._visitor.value_dispatcher.handle_type_call(target_type_ref, [tag_ref], None)
        
        if isinstance(python_val, str):
            # If the target is a byte-sized integer array, treat the string as an
            # array initializer (C semantics: copy characters, append implicit
            # null terminator if there is room, zero-fill the rest).
            if self._is_byte_array_type(target_type):
                return self._convert_string_to_array_value(python_val, target_type)

            # Otherwise create global string constant as ptr[i8]
            from .builtin_entities import ptr, i8
            str_const = self._visitor._create_string_constant(python_val)
            string_pointer = wrap_value(str_const, kind="value", type_hint=ptr[i8])
            from .type_descriptor import same_type
            if same_type(ptr[i8], target_type):
                return string_pointer
            return self.convert(string_pointer, target_type)

        # Get module_context for types that require it (e.g., structs with forward refs)
        module = getattr(self._visitor, 'module', None)
        module_context = getattr(module, 'context', None) if module else None
        llvm_type = target_type.get_llvm_type(module_context)

        if isinstance(python_val, bool):
            python_val = int(python_val)
        if (getattr(target_type, '_is_bool', False)
                and isinstance(python_val, (int, float))):
            python_val = int(python_val != 0)

        if isinstance(llvm_type, (ir.ArrayType, ir.BaseStructType)):
            if not isinstance(python_val, (int, float)) or python_val != 0:
                raise TypeError(
                    "Aggregate initialization requires zero or an explicit "
                    "sequence of field/element values"
                )
            return wrap_value(
                self.create_zero_constant(llvm_type),
                kind="value", type_hint=target_type,
            )
        
        # Handle pointer types specially - convert integer to pointer via inttoptr.
        # Python integer constants are compile-time values, so the resulting
        # pointer is also a constant expression (matching C's address constants).
        if hasattr(target_type, 'is_pointer') and target_type.is_pointer():
            if isinstance(python_val, int):
                from .builtin_entities import i64
                int_val = ir.Constant(i64.get_llvm_type(), python_val)
                ptr_val = ir.Constant.inttoptr(int_val, llvm_type)
                return wrap_value(ptr_val, kind="value", type_hint=target_type)
            else:
                raise TypeError(f"Cannot promote Python {type(python_val).__name__} to pointer type")

        # Handle float -> integer conversion (e.g. pyconst float used in i32 context).
        # Truncate toward zero to match C semantics.
        if isinstance(python_val, float) and hasattr(target_type, '_is_integer') and target_type._is_integer:
            python_val = int(python_val)

        ir_val = ir.Constant(llvm_type, python_val)
        return wrap_value(ir_val, kind="value", type_hint=target_type)

    @staticmethod
    def _func_types_compatible(source_func_type, target_func_type, module_context=None) -> bool:
        """Compare callable signatures in the PythoC type layer."""
        from .type_descriptor import can_convert
        return can_convert(source_func_type, target_func_type, mode='implicit')

    @staticmethod
    def _is_byte_array_type(pc_type) -> bool:
        """Return True if pc_type is an array whose element type is one byte."""
        base_type = strip_qualifiers(pc_type)
        if not (hasattr(base_type, 'is_array') and base_type.is_array()):
            return False
        elem_type = getattr(base_type, 'element_type', None)
        if elem_type is None:
            return False
        if not (hasattr(elem_type, '_is_integer') and elem_type._is_integer):
            return False
        if getattr(elem_type, '_is_bool', False):
            return False
        return getattr(elem_type, '_size_bytes', None) == 1

    def _convert_string_to_array_value(self, string_val: str, target_array_type) -> ValueRef:
        """Convert a Python string literal into a fixed-size byte array value.

        Follows C string literal initialization rules for char arrays:
        - If the string fits with room for a null terminator, copy the bytes,
          append '\0', and zero-fill any remaining slots.
        - If the string exactly fills the array, copy the bytes as-is.
        - If the string is too long, raise a type error.

        The result is an aggregate LLVM value (not a pointer) so it can be used
        both as a direct array initializer and as a nested aggregate element
        (e.g. a struct field of type array[i8, N]).
        """
        from .literal_protocol import materialize_sequence_value
        from .builtin_entities.pc_list import pc_list
        from .valueref import wrap_python_constant, wrap_value

        if not self._is_byte_array_type(target_array_type):
            raise TypeError(
                f"Cannot convert string literal to non-byte-array type {target_array_type}"
            )

        bytes_data = string_val.encode('utf-8')
        dims = getattr(target_array_type, 'dimensions', None) or ()
        total_size = 1
        for dim in dims:
            total_size *= dim

        if len(bytes_data) < total_size:
            byte_values = list(bytes_data) + [0]  # implicit null terminator
        elif len(bytes_data) == total_size:
            byte_values = list(bytes_data)
        else:
            raise TypeError(
                f"String literal is too long for {target_array_type}: "
                f"{len(bytes_data)} bytes given, {total_size} bytes available"
            )

        # Zero-fill any remaining slots.
        while len(byte_values) < total_size:
            byte_values.append(0)

        elements = [wrap_python_constant(b) for b in byte_values]
        pc_list_carrier = pc_list.from_elements(elements)
        array_value = materialize_sequence_value(
            self._visitor, pc_list_carrier, target_array_type)
        return wrap_value(
            array_value,
            kind="value",
            type_hint=target_array_type,
        )

    @staticmethod
    def infer_default_pc_type_from_python(python_val):
        """Infer default PC type from Python value
        
        Args:
            python_val: Python value (int, float, bool, str)
            
        Returns:
            PC type class (i64, f64, bool_type, ptr[i8])
            
        Raises:
            TypeError: If Python type cannot be promoted to PC type
        """
        from .builtin_entities import i64, f64, ptr, i8, bool as bool_type
        
        if isinstance(python_val, bool):
            return bool_type
        elif isinstance(python_val, int):
            return i64
        elif isinstance(python_val, float):
            return f64
        elif isinstance(python_val, str):
            return ptr[i8]
        else:
            raise_if_compile_level_callable(python_val)
            raise TypeError(f"Cannot infer PC type from Python type {type(python_val).__name__}")
    
    def promote_to_pc_default(self, python_val) -> ValueRef:
        """Promote Python value to default PC type
        
        Uses infer_default_pc_type_from_python to determine target type,
        then creates appropriate LLVM constant.
        """
        pc_type = self.infer_default_pc_type_from_python(python_val)
        
        if isinstance(python_val, bool):
            ir_val = ir.Constant(ir.IntType(1), int(python_val))
            return wrap_value(ir_val, kind="value", type_hint=pc_type)
        elif isinstance(python_val, int):
            ir_val = ir.Constant(ir.IntType(64), python_val)
            return wrap_value(ir_val, kind="value", type_hint=pc_type)
        elif isinstance(python_val, float):
            ir_val = ir.Constant(ir.DoubleType(), python_val)
            return wrap_value(ir_val, kind="value", type_hint=pc_type)
        elif isinstance(python_val, str):
            str_const = self._visitor._create_string_constant(python_val)
            return wrap_value(str_const, kind="value", type_hint=pc_type)
        else:
            raise TypeError(f"Cannot promote Python type {type(python_val).__name__} to PC type")

    def _materialize_store_operand(
        self,
        value: ValueRef,
        target_pc_type: type[BuiltinType],
    ) -> ir.Value:
        """Return an IR operand suitable for storing into target_pc_type.

        Some runtime pcvalues, notably arrays, are represented by a pointer to
        storage even when the semantic target is the aggregate value itself.
        In that case we need to load once before the store.
        """
        value_ir = ensure_ir(value)
        if value.is_python_value():
            return value_ir

        target_base_type = cast(type[BuiltinType], strip_qualifiers(target_pc_type))
        if not target_base_type.is_array():
            return value_ir

        target_llvm_type = target_base_type.get_llvm_type(self._visitor.module.context)
        if value.pointee() == target_llvm_type:
            return self.builder.load(value_ir)
        return value_ir

    def _sequence_literal_to_pc_list(self, sequence_value):
        """Convert a generic sequence literal carrier to pc_list, preserving ValueRefs."""
        from .builtin_entities.pc_list import pc_list
        from .builtin_entities.python_type import PythonType
        from .literal_protocol import get_sequence_elements, is_pc_list_type, is_sequence_carrier

        if is_pc_list_type(sequence_value):
            return sequence_value

        if not is_sequence_carrier(sequence_value):
            raise TypeError(f"Expected sequence literal carrier, got {type(sequence_value).__name__}")

        elements = []
        for item in get_sequence_elements(sequence_value):
            if isinstance(item, ValueRef):
                elements.append(item)
                continue
            if is_sequence_carrier(item):
                nested_carrier = self._sequence_literal_to_pc_list(item)
                nested_type = PythonType.wrap(nested_carrier, is_constant=True)
                elements.append(wrap_value(nested_carrier, kind="python", type_hint=nested_type))
                continue
            elements.append(wrap_python_constant(item))

        return pc_list.from_elements(elements)

    def _convert_list_to_array(self, python_list, target_array_type):
        """Convert a Python list/tuple literal to an array value (pointer)."""
        return self._convert_pc_list_to_array(
            self._sequence_literal_to_pc_list(python_list), target_array_type)

    def _convert_pc_list_to_array(self, pc_list_type, target_array_type):
        """Materialize a sequence-literal carrier into an array value (pointer).

        Arrays are lvalues here, so the assembled aggregate (an ir.Constant when
        every element folds, an insertvalue chain otherwise) is stored once into
        an entry-block alloca and returned as a pointer. Both the constant and
        runtime cases go through the same single-pass materializer.
        """
        from .literal_protocol import materialize_sequence_value

        array_value = materialize_sequence_value(
            self._visitor, pc_list_type, target_array_type)
        array_llvm_type = target_array_type.get_llvm_type(self._visitor.module.context)
        tmp_alloca = self._visitor._create_alloca_in_entry(array_llvm_type, "array_literal")
        self.builder.store(array_value, tmp_alloca)
        return wrap_value(tmp_alloca, kind="value", type_hint=target_array_type)

    def try_const_aggregate(self, value_ref, target_pc_type):
        """Fold a braced aggregate initializer to an ir.Constant, or return None.

        Thin entry over the shared carrier lowering: a static/global aggregate
        initialized by a sequence literal whose elements are all compile-time
        constants becomes a constant aggregate seed instead of runtime element
        stores. Non-constant elements (or unsupported shapes) return None so the
        caller falls back to the normal conversion path.
        """
        from .literal_protocol import (
            is_sequence_carrier, lower_sequence_to_constant,
        )

        if not (isinstance(value_ref, ValueRef) and value_ref.is_python_value()):
            return None
        carrier = value_ref.get_python_value()
        if not is_sequence_carrier(carrier):
            return None
        from .type_descriptor import can_convert
        resolver = getattr(getattr(self._visitor, 'implicit_coercer', None),
                           '_resolve_lazy_type_name', None)
        if not can_convert(value_ref.type_hint, target_pc_type,
                           mode='implicit', value=value_ref, context=resolver):
            raise TypeError(
                f'Cannot implicitly convert static aggregate initializer '
                f'to {target_pc_type.get_name()}')
        return lower_sequence_to_constant(self._visitor, carrier, target_pc_type)

    def _convert_tuple_to_struct(self, python_tuple, target_struct_type):
        """Convert a tuple/list literal to a struct value.

        Goes through the same single-pass materializer as arrays: the result is
        an ir.Constant when every field folds to a constant, or an insertvalue
        chain otherwise (handling nested struct/array fields and zero-fill).
        """
        from .literal_protocol import materialize_sequence_value

        field_types = get_schema_field_types(target_struct_type)
        if field_types is None:
            raise TypeError(f"Cannot get field types from {target_struct_type}")
        if len(python_tuple) > len(field_types):
            raise TypeError(
                f"Tuple length {len(python_tuple)} exceeds struct field count {len(field_types)}"
            )

        struct_value = materialize_sequence_value(
            self._visitor, python_tuple, target_struct_type)
        return wrap_value(struct_value, kind="value", type_hint=target_struct_type)

    def _convert_dict_to_struct(self, mapping_val, target_struct_type):
        """Convert a mapping carrier (pc_dict) to a struct value.

        Only accepts keys that are a subset of the struct's field names.
        Missing fields cause an error. Extra keys cause an error.

        This is the symmetric counterpart of ``_convert_tuple_to_struct``
        for ``**kwargs: T`` where ``T`` is a struct type.
        """
        from .literal_protocol import get_mapping_entries
        from .schema_protocol import get_schema_field_names, get_schema_field_types

        field_names = get_schema_field_names(target_struct_type)
        field_types = get_schema_field_types(target_struct_type)

        if field_types is None:
            raise TypeError(f"Cannot get field types from {target_struct_type}")
        if any(n is None for n in field_names):
            raise TypeError(
                f"pc_dict -> struct requires all fields to be named, "
                f"but {target_struct_type} has field_names={field_names}"
            )

        entries = get_mapping_entries(mapping_val)

        # Build key -> value map, extracting string keys from ValueRef wrappers
        key_value_map = {}
        for key_ref, val_ref in entries:
            if isinstance(key_ref, ValueRef) and key_ref.is_python_value():
                key_str = key_ref.get_python_value()
            elif isinstance(key_ref, str):
                key_str = key_ref
            else:
                raise TypeError(
                    f"pc_dict key must be a string, got {type(key_ref)}"
                )
            if key_str in key_value_map:
                raise TypeError(f"duplicate key in pc_dict: '{key_str}'")
            key_value_map[key_str] = val_ref

        # Validate: all dict keys must be valid field names
        valid_names = set(field_names)
        for key_str in key_value_map:
            if key_str not in valid_names:
                logger.error(
                    f"pc_dict key '{key_str}' is not a field of "
                    f"{target_struct_type} (valid: {', '.join(field_names)})",
                    node=None,
                    exc_type=TypeError,
                )

        # Validate: all struct fields must be present in the dict
        missing = [n for n in field_names if n not in key_value_map]
        if missing:
            logger.error(
                f"pc_dict is missing fields for {target_struct_type}: "
                f"{', '.join(missing)}",
                node=None,
                exc_type=TypeError,
            )

        # Build field values in struct field order
        field_values = []
        for fname, ftype in zip(field_names, field_types):
            elem = key_value_map[fname]
            if isinstance(elem, ValueRef):
                field_val = self.convert(elem, ftype)
                field_values.append(ensure_ir(field_val))
            else:
                from .builtin_entities import PythonType
                py_valueref = PythonType.wrap(elem, is_constant=True)
                py_valueref = wrap_value(elem, kind="python", type_hint=py_valueref)
                field_val = self.convert(py_valueref, ftype)
                field_values.append(ensure_ir(field_val))

        from .literal_protocol import assemble_aggregate_value
        struct_value = assemble_aggregate_value(
            self._visitor, target_struct_type, field_values)
        return wrap_value(struct_value, kind="value", type_hint=target_struct_type)

    def _convert_struct_to_struct(self, value, source_struct_type, target_struct_type, original_target_type):
        """Convert struct to struct by field-by-field conversion.
        
        This enables implicit conversion like:
            struct[pyconst[42], pyconst[3.14], pyconst[1]] -> struct[i32, f64, i32]
        
        Each field is converted individually using the standard convert() method.
        
        Args:
            value: Source ValueRef with struct type
            source_struct_type: Source struct type (base type, qualifiers stripped)
            target_struct_type: Target struct type (base type, qualifiers stripped)
            original_target_type: Original target type (may have qualifiers)
        
        Returns:
            ValueRef with converted struct value
        """
        source_fields = get_schema_field_types(source_struct_type)
        target_fields = get_schema_field_types(target_struct_type)
        
        # Check field count matches
        if len(source_fields) != len(target_fields):
            raise TypeError(
                f"Cannot convert struct with {len(source_fields)} fields to struct with {len(target_fields)} fields"
            )
        
        # Get all source field values using get_all_fields
        source_field_values = source_struct_type.get_all_fields(self._visitor, value, None)
        
        # Convert each field
        converted_fields = []
        for i, (src_field_vref, target_field_type) in enumerate(zip(source_field_values, target_fields)):
            # Convert field to target type
            converted_field = self.convert(src_field_vref, target_field_type)
            converted_fields.append(ensure_ir(converted_field))
        
        # Build target struct value
        from .literal_protocol import assemble_aggregate_value
        struct_value = assemble_aggregate_value(
            self._visitor, target_struct_type, converted_fields)
        
        return wrap_value(struct_value, kind="value", type_hint=original_target_type)

    def _convert_struct_to_enum(self, value, source_struct_type, target_enum_type, original_target_type):
        """Convert struct to enum when first field is pyconst tag.
        
        This enables implicit conversion like:
            struct[pyconst[Result.Ok], pyconst[42]] -> Result
        
        The first field must be a pyconst containing the enum tag value.
        The payload type is inferred from the tag's corresponding variant.
        
        Args:
            value: Source ValueRef with struct type
            source_struct_type: Source struct type (base type, qualifiers stripped)
            target_enum_type: Target enum type
            original_target_type: Original target type (may have qualifiers)
        
        Returns:
            ValueRef with enum value
        """
        from .builtin_entities.python_type import PythonType
        
        source_fields = get_schema_field_types(source_struct_type)
        
        # Need at least 1 field (tag), optionally 2 (tag + payload)
        if len(source_fields) < 1 or len(source_fields) > 2:
            raise TypeError(
                f"Cannot convert struct with {len(source_fields)} fields to enum. "
                f"Expected struct[pyconst[tag]] or struct[pyconst[tag], payload]"
            )
        
        # First field must be pyconst (compile-time constant tag)
        tag_field_type = source_fields[0]
        if not isinstance(tag_field_type, PythonType) or not tag_field_type.is_constant():
            raise TypeError(
                f"Cannot convert struct to enum: first field must be pyconst[tag], "
                f"got {tag_field_type}"
            )
        
        # Extract tag value from pyconst
        tag_value = tag_field_type.get_constant_value()
        
        # Find which variant this tag corresponds to
        variant_idx = None
        variant_payload_type = None
        variant_names = target_enum_type._variant_names or []
        variant_types = target_enum_type._variant_types or []
        tag_values = target_enum_type._tag_values
        
        if isinstance(tag_values, dict):
            # @enum decorator format: Dict[str, int]
            for idx, (var_name, var_type) in enumerate(zip(variant_names, variant_types)):
                if var_name in tag_values and tag_values[var_name] == tag_value:
                    variant_idx = idx
                    variant_payload_type = var_type
                    break
        elif tag_values:
            # enum[...] subscript format: may be list of tags
            for idx, (var_name, var_type) in enumerate(zip(variant_names, variant_types)):
                if hasattr(target_enum_type, var_name) and getattr(target_enum_type, var_name) == tag_value:
                    variant_idx = idx
                    variant_payload_type = var_type
                    break
        
        if variant_idx is None:
            raise TypeError(
                f"Tag value {tag_value} does not match any variant of enum {target_enum_type.get_name()}"
            )
        
        # Get source field values
        source_field_values = source_struct_type.get_all_fields(self._visitor, value, None)
        tag_vref = source_field_values[0]
        
        # Build enum using handle_call
        from .builtin_entities.types import void
        
        if len(source_fields) == 1:
            # No payload - just tag
            from .type_descriptor import same_type
            if variant_payload_type is not None and not same_type(variant_payload_type, void):
                raise TypeError(
                    f"Variant {variant_names[variant_idx]} requires payload of type {variant_payload_type}, "
                    f"but struct has no payload field"
                )
            target_enum_ref = wrap_python_constant(target_enum_type)
            return self._visitor.value_dispatcher.handle_type_call(
                target_enum_ref,
                [tag_vref],
                None,
            )
        else:
            # Has payload
            payload_vref = source_field_values[1]
            
            # Check if payload is void variant
            if variant_payload_type is None or same_type(variant_payload_type, void):
                raise TypeError(
                    f"Variant {variant_names[variant_idx]} has no payload, "
                    f"but struct has payload field"
                )
            
            # Convert payload to variant's payload type if needed
            payload_vref = self.convert(payload_vref, variant_payload_type)
            
            target_enum_ref = wrap_python_constant(target_enum_type)
            return self._visitor.value_dispatcher.handle_type_call(
                target_enum_ref,
                [tag_vref, payload_vref],
                None,
            )

    def _build_conversion_registry(self):
        """Build dispatch table for conversions using LLVM types."""
        # llvmlite pointer type class may be private; derive dynamically
        ptr_type_class = type(ir.PointerType(ir.IntType(8)))
        
        # Get BFloatType and FP128Type if available (from monkey patch)
        bfloat_type_class = getattr(ir, 'BFloatType', None)
        fp128_type_class = getattr(ir, 'FP128Type', None)
        
        # Build registry with all float types
        float_types = [ir.HalfType, ir.FloatType, ir.DoubleType]
        if bfloat_type_class:
            float_types.append(bfloat_type_class)
        if fp128_type_class:
            float_types.append(fp128_type_class)
        
        registry = {
            # Integer to integer
            (ir.IntType, ir.IntType): self._convert_int_to_int,
            # Array value -> pointer (decay)
            (ir.ArrayType, ptr_type_class): self._convert_array_value_to_ptr,
            # Pointer conversions
            (ptr_type_class, ptr_type_class): self._convert_ptr_to_ptr,
            (ptr_type_class, ir.IntType): self._convert_ptr_to_int,
            (ir.IntType, ptr_type_class): self._convert_int_to_ptr,
        }
        
        # Add integer <-> float conversions for all float types
        for float_type in float_types:
            registry[(ir.IntType, float_type)] = self._convert_int_to_float
            registry[(float_type, ir.IntType)] = self._convert_float_to_int
        
        # Add float <-> float conversions for all combinations
        for src_float_type in float_types:
            for dst_float_type in float_types:
                registry[(src_float_type, dst_float_type)] = self._convert_float_to_float
        
        return registry

    def _convert_int_to_int(
        self,
        value,
        source_type: ir.IntType,
        target_type: ir.IntType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)
        target_is_bool = getattr(get_base_type(type_hint), '_is_bool', False)

        # Constant-fold integer conversions so compile-time values like offsetof()
        # remain constants when assigned to differently-sized fields.
        # We compute the resulting integer value directly instead of emitting a
        # cast constexpr (sext/zext/trunc), because newer LLVM versions reject
        # those constexprs as instruction operands.
        if isinstance(value_ir, ir.Constant) and value_ir.constant is not None and value_ir.constant is not ir.Undefined:
            from .type_descriptor import integer_value
            v = int(integer_value(value.type_hint, value_ir.constant))
            if target_is_bool:
                v = int(v != 0)
            elif source_type.width >= target_type.width or target_is_unsigned:
                v &= (1 << target_type.width) - 1
            result = ir.Constant(target_type, v)
            return wrap_value(result, kind="value", type_hint=type_hint)

        if source_type.width < target_type.width:
            result = (
                self.builder.zext(value_ir, target_type)
                if source_is_unsigned
                else self.builder.sext(value_ir, target_type)
            )
        elif source_type.width > target_type.width:
            # Special case: converting to bool should use comparison, not truncation
            # i1/u1 remain integers even though they share bool's LLVM type.
            if target_is_bool:
                # Convert to bool: compare != 0
                result = self.builder.icmp_signed('!=', value_ir, ir.Constant(source_type, 0))
            else:
                result = self.builder.trunc(value_ir, target_type)
        else:
            result = value_ir
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_int_to_float(
        self,
        value,
        source_type: ir.IntType,
        target_type,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)

        # Constant-fold integer -> float conversion by computing the float value
        # directly. Avoids emitting a sitofp/uitofp constexpr, which LLVM 20+
        # rejects as an instruction operand.
        if isinstance(value_ir, ir.Constant) and isinstance(value_ir.constant, int) and value_ir.constant is not ir.Undefined:
            from .type_descriptor import integer_value
            v = integer_value(value.type_hint, value_ir.constant)
            result = ir.Constant(target_type, float(v))
            return wrap_value(result, kind="value", type_hint=type_hint)

        result = (
            self.builder.uitofp(value_ir, target_type)
            if source_is_unsigned
            else self.builder.sitofp(value_ir, target_type)
        )
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_float_to_int(
        self,
        value,
        source_type,
        target_type: ir.IntType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)
        if getattr(get_base_type(type_hint), '_is_bool', False):
            if isinstance(value_ir, ir.Constant):
                result = ir.Constant(target_type, int(value_ir.constant != 0))
            else:
                result = self.builder.fcmp_unordered(
                    '!=', value_ir, ir.Constant(source_type, 0.0))
            return wrap_value(result, kind="value", type_hint=type_hint)


        # Constant-fold float -> integer conversion by computing the integer value
        # directly. Avoids emitting a fptosi/fptoui constexpr, which LLVM 20+
        # rejects as an instruction operand.
        if isinstance(value_ir, ir.Constant) and isinstance(value_ir.constant, float):
            v = value_ir.constant
            # C-style truncation toward zero
            int_val = int(v)
            if target_is_unsigned:
                tw = target_type.width
                int_val = int_val & ((1 << tw) - 1)
            result = ir.Constant(target_type, int_val)
            return wrap_value(result, kind="value", type_hint=type_hint)

        result = (
            self.builder.fptoui(value_ir, target_type)
            if target_is_unsigned
            else self.builder.fptosi(value_ir, target_type)
        )
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_float_to_float(
        self,
        value,
        source_type,
        target_type,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)

        # Determine source and target bit widths
        # Map types to bit widths: half/bf16=16, float=32, double=64, fp128=128
        def get_float_bits(float_type):
            type_class = type(float_type)
            if type_class == ir.HalfType or type_class.__name__ == 'BFloatType':
                return 16
            elif type_class == ir.FloatType:
                return 32
            elif type_class == ir.DoubleType:
                return 64
            elif type_class.__name__ == 'FP128Type':
                return 128
            else:
                # Fallback: unknown float type
                raise TypeError(f"Unknown float type: {type_class}")

        source_bits = get_float_bits(source_type)
        target_bits = get_float_bits(target_type)

        # Constant-fold float conversions by computing the target float value
        # directly. Avoids emitting fpext/fptrunc constexprs, which LLVM 20+
        # rejects as instruction operands. Only fold 32/64-bit IEEE floats.
        if (
            isinstance(value_ir, ir.Constant)
            and isinstance(value_ir.constant, float)
            and source_bits in (32, 64)
            and target_bits in (32, 64)
        ):
            v = value_ir.constant
            if source_bits < target_bits:
                # float -> double: Python float already has double precision
                result = ir.Constant(target_type, v)
            elif source_bits > target_bits:
                # double -> float: round to single precision
                import struct
                v32 = struct.unpack('f', struct.pack('f', v))[0]
                result = ir.Constant(target_type, v32)
            else:
                result = ir.Constant(target_type, v)
            return wrap_value(result, kind="value", type_hint=type_hint)

        if source_bits < target_bits:
            # Extend to larger type
            result = self.builder.fpext(value_ir, target_type)
        elif source_bits > target_bits:
            # Truncate to smaller type
            result = self.builder.fptrunc(value_ir, target_type)
        else:
            # Same bit width, no conversion needed
            result = value_ir
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_array_value_to_ptr(
        self,
        value,
        source_type: ir.ArrayType,
        target_type: ir.PointerType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        # Case 1: Already pointer to array, decay with GEP [0, 0]
        val_ir = ensure_ir(value)
        if isinstance(val_ir.type, ir.PointerType) and isinstance(val_ir.type.pointee, ir.ArrayType):
            zero = ir.Constant(ir.IntType(32), 0)
            elem_ptr = self.builder.gep(val_ir, [zero, zero], inbounds=True)
            if elem_ptr.type != target_type:
                elem_ptr = self.builder.bitcast(elem_ptr, target_type)
            return wrap_value(elem_ptr, kind="value", type_hint=type_hint)

        # Fallback: use address field if available
        if isinstance(value, ValueRef) and value.has_place():
            addr_ir = ensure_ir(value.require_place())
            if isinstance(addr_ir.type, ir.PointerType) and isinstance(addr_ir.type.pointee, ir.ArrayType):
                zero = ir.Constant(ir.IntType(32), 0)
                elem_ptr = self.builder.gep(addr_ir, [zero, zero], inbounds=True)
                if elem_ptr.type != target_type:
                    elem_ptr = self.builder.bitcast(elem_ptr, target_type)
                return wrap_value(elem_ptr, kind="value", type_hint=type_hint)

        # This should not happen in correct C semantics
        raise TypeError(
            f"Array value conversion failed: expected pointer-to-array, got {val_ir.type}. "
            f"Arrays should not exist as values in C semantics."
        )

    def _convert_ptr_to_ptr(
        self,
        value,
        source_type: ir.PointerType,
        target_type: ir.PointerType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)
        # Array decay: [N x T]* -> T* (and then to the target pointer type)
        if isinstance(source_type.pointee, ir.ArrayType) and (
            isinstance(target_type.pointee, ir.Type) and not isinstance(target_type.pointee, ir.ArrayType)
        ):
            zero = ir.Constant(ir.IntType(32), 0)
            elem_ptr = self.builder.gep(value_ir, [zero, zero], inbounds=True)
            if elem_ptr.type != target_type:
                elem_ptr = self.builder.bitcast(elem_ptr, target_type)
            return wrap_value(elem_ptr, kind="value", type_hint=type_hint)
        # Null constants: keep null
        if isinstance(value_ir, ir.Constant) and value_ir.constant is None:
            result = ir.Constant(target_type, None)
        else:
            result = self.builder.bitcast(value_ir, target_type)
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_ptr_to_int(
        self,
        value,
        source_type: ir.PointerType,
        target_type: ir.IntType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)
        if getattr(get_base_type(type_hint), '_is_bool', False):
            if isinstance(value_ir, ir.Constant) and value_ir.constant is None:
                result = ir.Constant(target_type, 0)
            else:
                result = self.builder.icmp_unsigned(
                    '!=', value_ir, ir.Constant(source_type, None))
            return wrap_value(result, kind="value", type_hint=type_hint)

        if isinstance(value_ir, ir.Constant) and value_ir.constant is None:
            result = ir.Constant(target_type, 0)
        else:
            result = self.builder.ptrtoint(value_ir, target_type)
        return wrap_value(result, kind="value", type_hint=type_hint)

    def _convert_int_to_ptr(
        self,
        value,
        source_type: ir.IntType,
        target_type: ir.PointerType,
        source_is_unsigned: bool,
        target_is_unsigned: bool,
        type_hint,
    ) -> ValueRef:
        value_ir = ensure_ir(value)
        if isinstance(value_ir, ir.Constant):
            # Constant integer -> pointer is a constant expression, usable in
            # static/global initializers (C address constant semantics).
            result = ir.Constant.inttoptr(value_ir, target_type)
        else:
            result = self.builder.inttoptr(value_ir, target_type)
        return wrap_value(result, kind="value", type_hint=type_hint)

    def to_boolean(self, value: Union[ir.Value, ValueRef]) -> ir.Value:
        """Convert any value to boolean (i1) via the unified value dispatcher."""
        return self._visitor.value_dispatcher.to_boolean(value)

    def create_zero_constant(self, llvm_type: ir.Type) -> ir.Constant:
        """Create a zero constant for the given LLVM type."""
        if isinstance(llvm_type, ir.IntType):
            return ir.Constant(llvm_type, 0)
        elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
            return ir.Constant(llvm_type, 0.0)
        elif isinstance(llvm_type, ir.PointerType):
            return ir.Constant(llvm_type, None)
        elif isinstance(llvm_type, ir.ArrayType):
            return ir.Constant(llvm_type, None)
        elif isinstance(llvm_type, ir.BaseStructType):
            return ir.Constant(llvm_type, None)
        elif isinstance(llvm_type, ir.VoidType):
            raise TypeError("Cannot create zero constant for void type")
        else:
            raise TypeError(f"Cannot create zero constant for {llvm_type}")

    @staticmethod
    def is_llvm_integer_type(llvm_type: ir.Type, width: Optional[int] = None) -> bool:
        if not isinstance(llvm_type, ir.IntType):
            return False
        return width is None or llvm_type.width == width

    @staticmethod
    def is_llvm_float_type(llvm_type: ir.Type) -> bool:
        return isinstance(llvm_type, (ir.FloatType, ir.DoubleType))

    @staticmethod
    def is_llvm_pointer_type(llvm_type: ir.Type) -> bool:
        return isinstance(llvm_type, ir.PointerType)

    @staticmethod
    def create_int_constant(llvm_type: ir.Type, value: int) -> ir.Constant:
        if not isinstance(llvm_type, ir.IntType):
            raise TypeError(f"Expected integer LLVM type, got {llvm_type}")
        return ir.Constant(llvm_type, value)

    def create_bool_constant(self, value: bool) -> ir.Constant:
        from .builtin_entities import bool as bool_type

        bool_llvm_type = bool_type.get_llvm_type(self._visitor.module.context)
        return self.create_int_constant(bool_llvm_type, int(bool(value)))

    def promote_to_float(self, value: Union[ir.Value, ValueRef], target_pc_type) -> ValueRef:
        """Promote integer value to float PC type (f32/f64)."""
        if not isinstance(value, ValueRef) or value.type_hint is None:
            raise TypeError("promote_to_float requires ValueRef with type_hint")
        
        # Strip qualifiers from target type
        target_pc_type = strip_qualifiers(target_pc_type)
        
        if not (hasattr(target_pc_type, '_is_float') and target_pc_type._is_float):
            raise TypeError(f"promote_to_float target must be float type (f32/f64), got {target_pc_type}")
        
        # Handle Python values first
        if isinstance(value, ValueRef) and value.is_python_value():
            python_val = value.get_python_value()
            value = self._promote_python_to_pc(python_val, target_pc_type)
        
        # Get base type for source (stripping both qualifiers and refined types)
        source_pc_type = get_base_type(value.type_hint)
        if hasattr(source_pc_type, '_is_float') and source_pc_type._is_float:
            if source_pc_type == target_pc_type:
                return value
            return self.convert(value, target_pc_type)
        if hasattr(source_pc_type, '_is_integer') and source_pc_type._is_integer:
            return self.convert(value, target_pc_type)
        raise TypeError(f"Cannot promote {source_pc_type} to float")

    def unify_integer_types(self, left: Union[ir.Value, ValueRef], right: Union[ir.Value, ValueRef]) -> Tuple[ValueRef, ValueRef]:
        """Unify two integer operands to the same width using PC type hints."""
        from .type_descriptor import common_numeric_type
        target = common_numeric_type(left.type_hint, right.type_hint)
        if target is None or getattr(target, '_is_float', False):
            return left, right
        coercer = ImplicitCoercer(self)
        return coercer.coerce(left, target), coercer.coerce(right, target)

    @staticmethod
    def infer_unified_pc_type(left_pc_type, right_pc_type):
        """Project the common value relation without consulting LLVM types."""
        from .type_descriptor import common_value_type
        return common_value_type(left_pc_type, right_pc_type)

    def integer_operand(self, value, node=None):
        """Validate an index or offset and materialize it at pointer precision."""
        from .builtin_entities.types import i64
        from .type_descriptor import is_integer_operand
        if not is_integer_operand(value.type_hint, value=value):
            logger.error('Pointer offset or array index must have an integer type',
                         node=node, exc_type=TypeError)
        return ImplicitCoercer(self).coerce(value, i64, node)

    def unify_shift_types(self, left, right):
        """Shifts preserve the left operand's PythoC width and signedness."""
        for value in (left, right):
            if (value.is_python_value()
                    and type(value.get_python_value()) not in (int, bool)):
                raise TypeError('Shift operands must have integer types')
        target = get_base_type(right.type_hint if left.is_python_value()
                               else left.type_hint)
        right_type = get_base_type(right.type_hint)

        def integer(item):
            return (getattr(item, '_is_integer', False)
                    or getattr(item, '_is_bool', False))

        if not integer(target) or (not right.is_python_value()
                                   and not integer(right_type)):
            raise TypeError('Shift operands must have integer types')
        coercer = ImplicitCoercer(self)
        return coercer.coerce(left, target), coercer.coerce(right, target)

    def unify_binop_types(self, left: Union[ir.Value, ValueRef], right: Union[ir.Value, ValueRef]) -> Tuple[ValueRef, ValueRef, bool]:
        """Unify operands for binary operations via PC types."""
        if left.is_python_value() and right.is_python_value():
            # Both are Python values - return as-is, let caller handle Python-level operations
            return left, right, False
        if not right.is_python_value() and left.is_python_value():
            left = self._promote_python_to_pc(left.get_python_value(), get_base_type(right.type_hint))
        if not left.is_python_value() and right.is_python_value():
            right = self._promote_python_to_pc(right.get_python_value(), get_base_type(left.type_hint))
        if not isinstance(left, ValueRef) or left.type_hint is None:
            raise TypeError("unify_binop_types requires ValueRef with type_hint")
        if not isinstance(right, ValueRef) or right.type_hint is None:
            raise TypeError("unify_binop_types requires ValueRef with type_hint")
        from .type_descriptor import common_numeric_type
        target = common_numeric_type(left.type_hint, right.type_hint)
        if target is None:
            return left, right, False
        coercer = ImplicitCoercer(self)
        return (coercer.coerce(left, target), coercer.coerce(right, target),
                bool(getattr(target, '_is_float', False)))


class ImplicitCoercer:
    """Single entry point for all *implicit* conversions.

    Implicit conversions happen at:
    - assignment (x: T = value)
    - function argument binding
    - return statements

    Explicit casts (ptr[T](x), i32(x)) bypass this and call TypeConverter.convert() directly.

    Policy:
    - Same type (after strip_qualifiers) -> delegate to convert()
    - ptr -> ptr: only if null constant, either side is void ptr, or same pointee
    - int -> ptr: rejected (must use explicit cast)
    - enum -> int: rejected (must use tag extraction)
    - All other cases: delegate to convert() (int->int, float, etc.)
    """

    def __init__(self, type_converter: TypeConverter):
        self._tc = type_converter

    def _resolve_lazy_type_name(self, name):
        """Resolve a quoted type name against the compiling module's visible
        namespace (plain ``A = SomeType`` aliases), then the session
        forward-ref registry.  Returns None when unresolvable."""
        from .builtin_entities.base import lookup_ctx_type_name
        visitor = getattr(self._tc, '_visitor', None)
        resolved = lookup_ctx_type_name(getattr(visitor, 'ctx', None), name)
        if resolved is not None:
            return resolved
        from .forward_ref import get_defined_type
        return get_defined_type(name)

    def coerce(self, value: ValueRef, target_type, node=None) -> ValueRef:
        """Implicit conversion: checks policy, then delegates to TypeConverter.convert()."""
        source_type = value.type_hint
        from .type_descriptor import can_convert
        if value.is_python_value():
            raise_if_compile_level_callable(value.get_python_value())

        def check_relation():
            if not can_convert(source_type, target_type, mode='implicit',
                               value=value, context=self._resolve_lazy_type_name):
                from .type_descriptor import conversion_failure
                detail = conversion_failure(source_type, target_type, value)
                if detail is not None:
                    raise TypeError(detail)
                source_name = (source_type.get_name() if hasattr(source_type, 'get_name')
                               else str(source_type))
                target_name = (target_type.get_name() if hasattr(target_type, 'get_name')
                               else str(target_type))
                logger.error(f"Cannot implicitly convert '{source_name}' to '{target_name}'",
                             node, exc_type=TypeError)

        if getattr(strip_qualifiers(target_type), '_is_refined', False) and not can_convert(
                source_type, target_type, mode='implicit',
                context=self._resolve_lazy_type_name):
            logger.error(
                f"Cannot implicitly convert to refined type {target_type.get_name()}; "
                "use assume() or refine().", node, exc_type=TypeError)

        # Early exit: same type after stripping qualifiers
        stripped_source = strip_qualifiers(source_type) if source_type else None
        stripped_target = strip_qualifiers(target_type)
        if stripped_source is not None and stripped_source == stripped_target:
            check_relation()
            return self._tc.convert(value, target_type, node)

        # Early exit: same type_id (handles SpecializedPtr identity issues)
        if stripped_source is not None and stripped_source is not stripped_target:
            from .type_descriptor import same_type
            if same_type(stripped_source, stripped_target,
                         resolver=self._resolve_lazy_type_name):
                check_relation()
                return self._tc.convert(value, target_type, node)

        base_source = get_base_type(source_type) if source_type else None
        base_target = get_base_type(target_type)

        if (is_struct_type(base_source) and is_struct_type(base_target)
                and not can_convert(source_type, target_type, mode='implicit',
                                    context=self._resolve_lazy_type_name)):
            logger.error(
                f"Cannot implicitly convert '{base_source.get_name()}' to "
                f"'{base_target.get_name()}': incompatible fields.",
                node, exc_type=TypeError)

        # --- Pointer policy ---
        source_is_ptr = base_source is not None and hasattr(base_source, '_is_pointer') and base_source._is_pointer
        target_is_ptr = base_target is not None and hasattr(base_target, '_is_pointer') and base_target._is_pointer

        # Reject implicit Python int -> pointer (prefer nullptr; allow explicit cast only).
        if target_is_ptr and value.is_python_value():
            py_val = value.get_python_value()
            if isinstance(py_val, int) and not isinstance(py_val, bool):
                target_name = base_target.get_name() if base_target and hasattr(base_target, 'get_name') else str(base_target)
                logger.error(
                    f"Cannot implicitly convert Python int constant to pointer type '{target_name}'. "
                    f"Use nullptr or an explicit cast: {target_name}(value)",
                    node, exc_type=TypeError)

        if source_is_ptr and target_is_ptr:
            # Null pointer constant -> any pointer: allowed
            if self.is_null_pointer_constant(value):
                check_relation()
                return self._tc.convert(value, target_type, node)
            # The PythoC relation checks void pointers, function signatures,
            # pointee qualifiers and aliases in one place.
            if self.are_compatible_pointers(
                    base_source, base_target,
                    resolve_name=self._resolve_lazy_type_name):
                check_relation()
                return self._tc.convert(value, target_type, node)
            # Incompatible pointers
            source_name = base_source.get_name() if hasattr(base_source, 'get_name') else str(base_source)
            target_name = base_target.get_name() if hasattr(base_target, 'get_name') else str(base_target)
            logger.error(
                f"Cannot implicitly convert '{source_name}' to '{target_name}'. "
                f"Use explicit cast: {target_name}(value)",
                node, exc_type=TypeError)

        # int -> ptr: rejected
        if (base_source is not None and hasattr(base_source, '_is_integer') and base_source._is_integer
                and target_is_ptr):
            source_name = base_source.get_name() if hasattr(base_source, 'get_name') else str(base_source)
            target_name = base_target.get_name() if hasattr(base_target, 'get_name') else str(base_target)
            logger.error(
                f"Cannot implicitly convert integer '{source_name}' to pointer type '{target_name}'. "
                f"Use nullptr or an explicit cast: {target_name}(value)",
                node, exc_type=TypeError)

        # enum -> int: rejected (moved from TypeConverter.convert)
        if is_enum_type(base_source) and base_target is not None and hasattr(base_target, 'is_signed'):
            source_name = base_source.get_name() if hasattr(base_source, 'get_name') else str(base_source)
            target_name = base_target.get_name() if hasattr(base_target, 'get_name') else str(base_target)
            logger.error(
                f"Cannot implicitly convert enum '{source_name}' to integer type '{target_name}'. "
                f"Use explicit tag extraction: value[0]",
                node, exc_type=TypeError)

        # All other cases: delegate to convert()
        check_relation()
        return self._tc.convert(value, target_type, node)

    # --- Policy predicates ---

    @staticmethod
    def is_void_pointer(pc_type) -> bool:
        """True if pc_type is a void pointer.

        Void pointer canonicalization:

        - User-facing: ptr[void]
        - Implementation detail: ptr[void] is currently canonicalized to an unspecialized ptr
          (pointee_type is None). Both lower to i8*.
        """
        base = get_base_type(pc_type)
        if base is None:
            return False
        from .builtin_entities.types import ptr
        if not (isinstance(base, type) and issubclass(base, ptr)):
            return False
        pointee = getattr(base, 'pointee_type', ...)
        if pointee is None:
            return True
        from .builtin_entities.types import void
        return pointee is void

    @staticmethod
    def is_null_pointer_constant(value: ValueRef) -> bool:
        """True if value is a null pointer constant (ir.Constant with None)."""
        from llvmlite import ir as llvm_ir
        if isinstance(value, ValueRef):
            ir_val = value.ir_value
            if isinstance(ir_val, llvm_ir.Constant) and isinstance(ir_val.type, llvm_ir.PointerType):
                return ir_val.constant is None
        return False

    @staticmethod
    def are_compatible_pointers(source_type, target_type, resolve_name=None) -> bool:
        """Directional PythoC pointer compatibility, including pointee qualifiers."""
        from .type_descriptor import can_convert
        return can_convert(source_type, target_type, mode='implicit',
                           context=resolve_name)
