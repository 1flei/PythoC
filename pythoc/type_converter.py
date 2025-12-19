"""
Centralized type conversion service for LLVM IR types.

This module provides a single place for value conversions and small utilities
needed during IR generation. ALL EXTERNAL CALLERS MUST PASS PC TYPES (BuiltinEntity
classes). Raw LLVM types are used only internally for IR instruction selection.
Reverse inference from LLVM to PC is intentionally not supported.
"""

from typing import Optional, Union, Any, Tuple
from llvmlite import ir

from .valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from .type_check import is_struct_type, is_enum_type
from .logger import logger
import ast


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
        
        # Check if target is a refined type but source is not
        # This check MUST happen before stripping qualifiers, because refined types
        # can only be constructed via assume() or refine(), not by direct conversion
        target_is_refined = hasattr(target_type, '_is_refined') and target_type._is_refined
        source_is_refined = value.type_hint and hasattr(value.type_hint, '_is_refined') and value.type_hint._is_refined
        
        if target_is_refined and not source_is_refined:
            # Direct conversion from base type to refined type is not allowed
            target_name = target_type.get_name() if hasattr(target_type, 'get_name') else str(target_type)
            source_name = value.type_hint.get_name() if value.type_hint and hasattr(value.type_hint, 'get_name') else str(value.type_hint)
            logger.error(f"Cannot directly convert from {source_name} to refined type {target_name}. "
                f"Refined types must be constructed using assume() or refine().", node)
            # raise TypeError(
            #     f"Cannot directly convert from {source_name} to refined type {target_name}. "
            #     f"Refined types must be constructed using assume() or refine()."
            # )
        
        # Strip qualifiers for comparison and conversion
        stripped_target = strip_qualifiers(target_type)
        stripped_source = strip_qualifiers(value.type_hint) if value.type_hint else None
        
        if stripped_source == stripped_target:
            # Types match after stripping qualifiers, just update type_hint
            return wrap_value(
                value.ir_value,
                kind=value.kind,
                type_hint=target_type,  # Keep original qualifiers
                address=value.address,
                var_name=value.var_name,
                linear_path=value.linear_path
            )
        
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
        
        # Step 1: Handle @compile wrapper -> func conversion
        # When source is a Python value that is a @compile wrapper, convert to func pointer
        if isinstance(value, ValueRef) and value.is_python_value():
            python_val = value.get_python_value()
            if hasattr(python_val, '_is_compiled') and python_val._is_compiled:
                # This is a @compile wrapper - convert to func pointer
                return self._convert_compile_wrapper_to_func(python_val, target_type)
            # Otherwise, auto-promote Python values to PC values
            value = self._promote_python_to_pc(python_val, stripped_target)
            return value
        
        # For actual conversion operations, use base types (strip refined types too)
        base_target = get_base_type(target_type)
        base_source = get_base_type(value.type_hint) if value.type_hint else None
        
        # Step 2: Handle enum to integer conversion (extract tag field)
        if is_enum_type(base_source) and hasattr(base_target, 'is_signed'):
            # Converting enum to integer - extract the tag field
            # Enum is struct { tag, payload }, extract field 0 (tag)
            enum_val = ensure_ir(value)
            tag_val = self.builder.extract_value(enum_val, 0, name="enum_tag")
            tag_ref = wrap_value(tag_val, kind="value", type_hint=base_source._tag_type)
            # Convert tag to target integer type
            return self.convert(tag_ref, target_type)

        # Step 3: Handle struct to enum conversion
        # struct[pyconst[Tag], payload] -> EnumType
        if is_struct_type(base_source) and is_enum_type(base_target):
            return self._convert_struct_to_enum(value, base_source, base_target, target_type)

        # Step 4: Handle struct to struct conversion (field-by-field)
        if is_struct_type(base_source) and is_struct_type(base_target):
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
        """Promote Python primitive value to a PC-typed ValueRef.
        
        Args:
            python_val: Python value to promote
            target_type: Optional target PC type (for context-aware promotion)
        """
        # Handle refined tuple types - extract the actual tuple values
        # This happens when visit_Tuple returns refined[struct[...], "tuple"]
        from .builtin_entities.refined import RefinedType
        if isinstance(python_val, type) and issubclass(python_val, RefinedType):
            tags = getattr(python_val, '_tags', [])
            if "tuple" in tags:
                # Extract tuple values from refined type
                base_struct = python_val._base_type
                if hasattr(base_struct, '_field_types'):
                    tuple_values = []
                    for field_type in base_struct._field_types:
                        if hasattr(field_type, '_python_object'):
                            tuple_values.append(field_type._python_object)
                        elif hasattr(field_type, 'get_python_object'):
                            tuple_values.append(field_type.get_python_object())
                        else:
                            tuple_values.append(field_type)
                    python_val = tuple(tuple_values)
        
        # Validate python_val is a supported primitive type
        # Allow: int, float, bool, str, None, list, tuple (for array/struct initialization)
        # Reject: type objects, classes, functions, modules, etc.
        if not isinstance(python_val, (int, float, bool, str, type(None), list, tuple)):
            raise TypeError(
                f"Cannot promote Python value of type {type(python_val).__name__} to PC type. "
                f"Only primitive types (int, float, bool, str, None) and collections (list, tuple) are supported. "
                f"Got: {repr(python_val)}"
            )
        
        # Handle enum initialization from tuple or int
        if hasattr(target_type, '_is_enum') and target_type._is_enum:
            if isinstance(python_val, (tuple, list)):
                # Tuple initialization: (tag, payload) or (tag,)
                from .builtin_entities.python_type import PythonType
                
                if len(python_val) == 1:
                    # Single element: (tag,) for void variants
                    tag_py_type = PythonType.wrap(python_val[0], is_constant=True)
                    tag_ref = wrap_value(python_val[0], kind="python", type_hint=tag_py_type)
                    return target_type.handle_call(self._visitor, target_type, [tag_ref], None)
                elif len(python_val) == 2:
                    # Two elements: (tag, payload)
                    tag_py_type = PythonType.wrap(python_val[0], is_constant=True)
                    tag_ref = wrap_value(python_val[0], kind="python", type_hint=tag_py_type)
                    payload_py_type = PythonType.wrap(python_val[1], is_constant=True)
                    payload_ref = wrap_value(python_val[1], kind="python", type_hint=payload_py_type)
                    return target_type.handle_call(self._visitor, target_type, [tag_ref, payload_ref], None)
                else:
                    raise TypeError(f"Enum initialization requires 1 or 2 elements, got {len(python_val)}")
            elif isinstance(python_val, int):
                # Direct tag initialization for void variants
                from .builtin_entities.python_type import PythonType
                tag_py_type = PythonType.wrap(python_val, is_constant=True)
                tag_ref = wrap_value(python_val, kind="python", type_hint=tag_py_type)
                return target_type.handle_call(self._visitor, target_type, [tag_ref], None)
        
        # Handle list/tuple to array conversion
        if isinstance(python_val, list):
            # Check if target_type is array
            if hasattr(target_type, "is_array") and target_type.is_array():
                return self._convert_list_to_array(python_val, target_type)
            # Otherwise, cannot promote (no target type info)
            raise TypeError(f"Cannot promote Python {python_val} to PC type {target_type}")
        
        # Handle tuple to struct conversion
        if isinstance(python_val, tuple):
            # Check if target_type is struct
            if hasattr(target_type, '_field_types'):
                return self._convert_tuple_to_struct(python_val, target_type)
            raise TypeError(f"Cannot promote Python tuple to PC type {target_type}")
        
        if isinstance(python_val, str):
            # Create global string constant
            from .builtin_entities import ptr, i8
            str_const = self._visitor._create_string_constant(python_val)
            return wrap_value(str_const, kind="value", type_hint=ptr[i8])

        # Get module_context for types that require it (e.g., structs with forward refs)
        module = getattr(self._visitor, 'module', None)
        module_context = getattr(module, 'context', None) if module else None
        llvm_type = target_type.get_llvm_type(module_context)
        
        # Handle pointer types specially - convert integer to pointer via inttoptr
        if hasattr(target_type, 'is_pointer') and target_type.is_pointer():
            if isinstance(python_val, int):
                # Create integer constant first, then convert to pointer
                from .builtin_entities import i64
                int_val = ir.Constant(i64.get_llvm_type(), python_val)
                ptr_val = self._visitor.builder.inttoptr(int_val, llvm_type)
                return wrap_value(ptr_val, kind="value", type_hint=target_type)
            else:
                raise TypeError(f"Cannot promote Python {type(python_val).__name__} to pointer type")
        
        ir_val = ir.Constant(llvm_type, python_val)
        return wrap_value(ir_val, kind="value", type_hint=target_type)
        
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
            raise TypeError(f"Cannot infer PC type from Python type {type(python_val).__name__}")
    
    def _convert_compile_wrapper_to_func(self, wrapper, target_type) -> ValueRef:
        """Convert @compile wrapper to func pointer type.
        
        This handles the conversion of a @compile decorated function to a function
        pointer. The wrapper contains all the metadata needed to declare/get the
        IR function and build the appropriate func type.
        
        Args:
            wrapper: The @compile wrapper function object
            target_type: Target func type (may be None for auto-inference)
        
        Returns:
            ValueRef with func pointer type_hint
        """
        from .registry import get_unified_registry
        from .effect import get_current_compilation_context
        from .builtin_entities import func as func_type_cls
        from llvmlite import ir
        
        registry = get_unified_registry()
        func_name = wrapper._original_name
        
        # Check if this wrapper has a specific mangled name
        lookup_mangled = getattr(wrapper, '_mangled_name', None)
        if lookup_mangled:
            func_info = registry.get_function_info_by_mangled(lookup_mangled)
            if not func_info:
                raise NameError(f"Function '{func_name}' with mangled '{lookup_mangled}' not found in registry")
            actual_func_name = func_info.mangled_name
        else:
            func_info = registry.get_function_info(func_name)
            if not func_info:
                raise NameError(f"Function '{func_name}' not found in registry")
            actual_func_name = func_info.mangled_name if func_info.mangled_name else func_name
        
        # Handle transitive effect propagation
        compilation_ctx = get_current_compilation_context()
        
        if compilation_ctx and not lookup_mangled:
            ctx_suffix = compilation_ctx.get('suffix')
            ctx_effects = compilation_ctx.get('effect_overrides', {})
            ctx_group_key = compilation_ctx.get('group_key')
            
            if ctx_suffix and ctx_effects:
                func_effect_deps = func_info.effect_dependencies if func_info else set()
                overridden_effects = set(ctx_effects.keys())
                
                if func_effect_deps & overridden_effects:
                    suffix_mangled_name = f"{func_name}_{ctx_suffix}"
                    suffix_info = registry.get_function_info_by_mangled(suffix_mangled_name)
                    
                    if not suffix_info:
                        logger.debug(f"Generating transitive suffix version: {suffix_mangled_name}")
                        original_wrapper = func_info.wrapper if func_info else None
                        
                        if original_wrapper and hasattr(original_wrapper, '__wrapped__'):
                            from .effect import restore_effect_context
                            from .decorators.compile import compile as compile_decorator
                            with restore_effect_context(ctx_effects):
                                compile_decorator(
                                    original_wrapper.__wrapped__,
                                    suffix=ctx_suffix,
                                    _effect_group_key=ctx_group_key
                                )
                            suffix_info = registry.get_function_info_by_mangled(suffix_mangled_name)
                    
                    if suffix_info:
                        func_info = suffix_info
                        actual_func_name = suffix_mangled_name
                        logger.debug(f"Using transitive suffix version: {actual_func_name}")
        
        # Get or declare the function in the module
        module = self._visitor.module
        module_context = module.context
        
        try:
            ir_func = module.get_global(actual_func_name)
        except KeyError:
            # Declare the function with proper ABI handling via builder
            param_llvm_types = []
            for param_name in func_info.param_names:
                param_type = func_info.param_type_hints.get(param_name)
                if param_type and hasattr(param_type, 'get_llvm_type'):
                    param_llvm_types.append(param_type.get_llvm_type(module_context))
                else:
                    param_llvm_types.append(ir.IntType(32))
            
            if func_info.return_type_hint and hasattr(func_info.return_type_hint, 'get_llvm_type'):
                return_llvm_type = func_info.return_type_hint.get_llvm_type(module_context)
            else:
                return_llvm_type = ir.VoidType()
            
            # Use LLVMBuilder to declare function with C ABI
            from .builder import LLVMBuilder
            temp_builder = LLVMBuilder()
            func_wrapper = temp_builder.declare_function(
                module, actual_func_name,
                param_llvm_types, return_llvm_type
            )
            ir_func = func_wrapper.ir_function
        
        # Build func type hint
        param_types = [func_info.param_type_hints[p] for p in func_info.param_names]
        if param_types:
            func_type_hint = func_type_cls[param_types, func_info.return_type_hint]
        else:
            func_type_hint = func_type_cls[[], func_info.return_type_hint]
        
        return wrap_value(ir_func, kind='pointer', type_hint=func_type_hint)

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

    def _convert_list_to_array(self, python_list, target_array_type):
        """Convert Python list/tuple to array constant.
        
        Args:
            python_list: Python list or tuple
            target_array_type: Target array type (array[T, N])
        
        Returns:
            ValueRef with pointer to array
        """        
        array_llvm_type = target_array_type.get_llvm_type(self._visitor.module.context)
        
        # Build the array constant recursively
        array_const = self._build_array_constant_recursive(python_list, target_array_type)
        
        # Arrays cannot exist as values in C semantics - materialize to memory immediately
        tmp_alloca = self._visitor._create_alloca_in_entry(array_llvm_type, "array_literal")
        self.builder.store(array_const, tmp_alloca)
        
        # Return as pointer to array
        return wrap_value(tmp_alloca, kind="value", type_hint=target_array_type)
    
    def _build_array_constant_recursive(self, python_list, target_array_type):
        """Build array constant recursively (returns IR constant, not ValueRef)"""
        pc_elem_type = target_array_type.element_type
        dimensions = target_array_type.dimensions
        if not isinstance(dimensions, (list, tuple)):
            dimensions = [dimensions]
        
        # Get LLVM types
        elem_llvm_type = pc_elem_type.get_llvm_type(self._visitor.module.context)
        array_llvm_type = target_array_type.get_llvm_type(self._visitor.module.context)
        
        # Handle 1D array
        if len(dimensions) == 1:
            size = dimensions[0]
            
            # Convert each element
            elem_constants = []
            for i, py_elem in enumerate(python_list):
                if i >= size:
                    break
                
                # Wrap Python value as PythonType ValueRef
                from .builtin_entities.python_type import PythonType
                py_valueref = PythonType.wrap(py_elem, is_constant=True)
                py_valueref = wrap_value(py_elem, kind="python", type_hint=py_valueref)
                
                # Convert to element type
                elem_val = self.convert(py_valueref, pc_elem_type)
                elem_constants.append(ensure_ir(elem_val))
            
            # Zero-fill remaining elements if list is shorter than array
            if len(elem_constants) < size:
                zero_val = self.create_zero_constant(elem_llvm_type)
                elem_constants.extend([zero_val] * (size - len(elem_constants)))
            
            # Create and return array constant (IR value, not ValueRef)
            return ir.Constant(array_llvm_type, elem_constants)
        
        # Handle multi-dimensional array (recursive)
        else:
            # For multi-dim array like array[array[T, M], N]
            # python_list should be a nested list [[...], [...], ...]
            outer_size = dimensions[0]
            
            # Build inner array type
            inner_dims = dimensions[1:]
            from .builtin_entities import array
            
            # Create inner array type: array[i32, 3, 4] for original array[i32, 2, 3, 4]
            # Use tuple unpacking compatible with Python 3.9+
            inner_array_type = array[(pc_elem_type,) + tuple(inner_dims)]
            
            # Convert each sub-list
            elem_constants = []
            for i, py_sublist in enumerate(python_list):
                if i >= outer_size:
                    break
                
                if not isinstance(py_sublist, (list, tuple)):
                    raise TypeError(f"Expected nested list for multi-dimensional array, got {type(py_sublist).__name__}")
                
                # Recursively build inner array constant (returns IR constant)
                inner_const = self._build_array_constant_recursive(py_sublist, inner_array_type)
                elem_constants.append(inner_const)
            
            # Zero-fill remaining elements
            if len(elem_constants) < outer_size:
                inner_llvm_type = inner_array_type.get_llvm_type(self._visitor.module.context)
                zero_val = self.create_zero_constant(inner_llvm_type)
                elem_constants.extend([zero_val] * (outer_size - len(elem_constants)))
            
            # Create and return outer array constant (IR value, not ValueRef)
            return ir.Constant(array_llvm_type, elem_constants)

    def _convert_tuple_to_struct(self, python_tuple, target_struct_type):
        """Convert Python tuple to struct value.
        
        Handles various element types:
        - Python primitives (int, float, bool, str)
        - Python tuples/lists (nested structures)
        - ValueRef (already-evaluated expressions like union() calls)
        
        Args:
            python_tuple: Python tuple or list (may contain ValueRef elements)
            target_struct_type: Target struct type (BuiltinEntity subclass)
        
        Returns:
            ValueRef with struct value
        """
        # Get field types from struct
        if hasattr(target_struct_type, '_field_types') and target_struct_type._field_types:
            field_types = target_struct_type._field_types
        else:
            raise TypeError(f"Cannot get field types from {target_struct_type}")
        
        # Check length
        if len(python_tuple) != len(field_types):
            raise TypeError(
                f"Tuple length {len(python_tuple)} does not match struct field count {len(field_types)}"
            )
        
        # Convert each element to corresponding field type
        field_values = []
        for i, (elem, field_type) in enumerate(zip(python_tuple, field_types)):
            # Case 1: elem is already a ValueRef (e.g., from union() call or other expressions)
            if isinstance(elem, ValueRef):
                # Convert to field type if needed
                if elem.type_hint != field_type:
                    field_val = self.convert(elem, field_type)
                else:
                    field_val = elem
                field_values.append(ensure_ir(field_val))
            
            # Case 2: elem is a Python tuple/list - recursively convert
            elif isinstance(elem, (tuple, list)):
                # Check if field_type is struct/array/union
                if hasattr(field_type, '_field_types'):
                    # Nested struct
                    field_val = self._convert_tuple_to_struct(elem, field_type)
                    field_values.append(ensure_ir(field_val))
                elif hasattr(field_type, 'is_array') and field_type.is_array():
                    # Nested array
                    field_val = self._convert_list_to_array(elem, field_type)
                    field_values.append(ensure_ir(field_val))
                elif hasattr(field_type, 'get_name') and field_type.get_name() == 'union':
                    # Union field - create undefined union (Python tuples cannot directly initialize unions)
                    # User should use union[...]() constructor explicitly
                    raise TypeError(
                        f"Cannot initialize union field from tuple. "
                        f"Use explicit union constructor: union[...]() "
                    )
                else:
                    raise TypeError(f"Cannot convert {type(elem)} to field type {field_type}")
            
            # Case 3: elem is a Python primitive - wrap and convert
            else:
                from .builtin_entities import PythonType
                py_valueref = PythonType.wrap(elem, is_constant=True)
                py_valueref = wrap_value(elem, kind="python", type_hint=py_valueref)
                
                # Convert to field type
                field_val = self.convert(py_valueref, field_type)
                field_values.append(ensure_ir(field_val))
        
        # Create struct constant
        struct_llvm_type = target_struct_type.get_llvm_type(self._visitor.module.context)

        # Build struct value
        struct_value = ir.Constant(struct_llvm_type, ir.Undefined)
        for i, field_val in enumerate(field_values):
            struct_value = self._visitor.builder.insert_value(struct_value, field_val, i)
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
        # Get field types
        source_fields = source_struct_type._field_types if source_struct_type._field_types else []
        target_fields = target_struct_type._field_types if target_struct_type._field_types else []
        
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
        target_llvm_type = target_struct_type.get_llvm_type(self._visitor.module.context)
        struct_value = ir.Constant(target_llvm_type, ir.Undefined)
        for i, field_val in enumerate(converted_fields):
            struct_value = self._visitor.builder.insert_value(struct_value, field_val, i)
        
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
        
        # Get source struct fields
        source_fields = source_struct_type._field_types if source_struct_type._field_types else []
        
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
            if variant_payload_type is not None and variant_payload_type != void:
                raise TypeError(
                    f"Variant {variant_names[variant_idx]} requires payload of type {variant_payload_type}, "
                    f"but struct has no payload field"
                )
            return target_enum_type.handle_call(self._visitor, target_enum_type, [tag_vref], None)
        else:
            # Has payload
            payload_vref = source_field_values[1]
            
            # Check if payload is void variant
            if variant_payload_type is None or variant_payload_type == void:
                raise TypeError(
                    f"Variant {variant_names[variant_idx]} has no payload, "
                    f"but struct has payload field"
                )
            
            # Convert payload to variant's payload type if needed
            payload_type = payload_vref.type_hint
            if payload_type != variant_payload_type:
                payload_vref = self.convert(payload_vref, variant_payload_type)
            
            return target_enum_type.handle_call(self._visitor, target_enum_type, [tag_vref, payload_vref], None)

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
        if source_type.width < target_type.width:
            result = (
                self.builder.zext(value_ir, target_type)
                if source_is_unsigned
                else self.builder.sext(value_ir, target_type)
            )
        elif source_type.width > target_type.width:
            # Special case: converting to bool (i1) should use comparison, not truncation
            # bool(42) should be True, not False (42's lowest bit is 0)
            if target_type.width == 1:
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
            return wrap_value(elem_ptr, kind="value", type_hint=type_hint)
        
        # Fallback: use address field if available
        if isinstance(value, ValueRef) and value.address is not None:
            addr_ir = ensure_ir(value.address)
            if isinstance(addr_ir.type, ir.PointerType) and isinstance(addr_ir.type.pointee, ir.ArrayType):
                zero = ir.Constant(ir.IntType(32), 0)
                elem_ptr = self.builder.gep(addr_ir, [zero, zero], inbounds=True)
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
        # Array decay: [N x T]* -> T*
        if isinstance(source_type.pointee, ir.ArrayType) and (
            isinstance(target_type.pointee, ir.Type) and not isinstance(target_type.pointee, ir.ArrayType)
        ):
            zero = ir.Constant(ir.IntType(32), 0)
            elem_ptr = self.builder.gep(value_ir, [zero, zero], inbounds=True)
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
        if isinstance(value_ir, ir.Constant) and value_ir.constant == 0:
            result = ir.Constant(target_type, None)
        else:
            result = self.builder.inttoptr(value_ir, target_type)
        return wrap_value(result, kind="value", type_hint=type_hint)

    def to_boolean(self, value: Union[ir.Value, ValueRef]) -> ir.Value:
        """Convert any value to boolean (i1)."""
        pc_type = None
        if isinstance(value, ValueRef) and value.type_hint is not None:
            pc_type = value.type_hint
        llvm_type = get_type(value)
        value_ir = ensure_ir(value)
        # Prefer PC type dispatch
        if pc_type is not None:
            if hasattr(pc_type, '_is_integer') and pc_type._is_integer:
                if llvm_type.width == 1:
                    return value_ir
                return self.builder.icmp_signed('!=', value_ir, ir.Constant(llvm_type, 0))
            elif hasattr(pc_type, '_is_float') and pc_type._is_float:
                return self.builder.fcmp_ordered('!=', value_ir, ir.Constant(llvm_type, 0.0))
            elif hasattr(pc_type, 'pointee_type'):
                null_ptr = ir.Constant(llvm_type, None)
                return self.builder.icmp_signed('!=', value_ir, null_ptr)
            else:
                raise TypeError(f"Cannot convert pc_type {pc_type} to boolean")
        raise TypeError(f"Cannot get pc_type from value")

    def create_zero_constant(self, llvm_type: ir.Type) -> ir.Constant:
        """Create a zero constant for the given LLVM type."""
        if isinstance(llvm_type, ir.IntType):
            return ir.Constant(llvm_type, 0)
        elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
            return ir.Constant(llvm_type, 0.0)
        elif isinstance(llvm_type, ir.PointerType):
            return ir.Constant(llvm_type, None)
        elif isinstance(llvm_type, ir.ArrayType):
            elem_zero = self.create_zero_constant(llvm_type.element)
            return ir.Constant(llvm_type, [elem_zero] * llvm_type.count)
        elif isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            field_zeros = [self.create_zero_constant(ft) for ft in llvm_type.elements]
            return ir.Constant(llvm_type, field_zeros)
        elif isinstance(llvm_type, ir.VoidType):
            raise TypeError("Cannot create zero constant for void type")
        else:
            raise TypeError(f"Cannot create zero constant for {llvm_type}")

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
        left_type = get_type(left)
        right_type = get_type(right)
        # If not both integers, return as-is (must be ValueRef)
        if not isinstance(left_type, ir.IntType) or not isinstance(right_type, ir.IntType):
            if not isinstance(left, ValueRef) or not isinstance(right, ValueRef):
                raise TypeError("unify_integer_types requires ValueRef inputs with type_hint")
            return left, right
        # Ensure we have type hints
        if not isinstance(left, ValueRef) or left.type_hint is None:
            raise TypeError("unify_integer_types requires ValueRef with type_hint")
        if not isinstance(right, ValueRef) or right.type_hint is None:
            raise TypeError("unify_integer_types requires ValueRef with type_hint")
        # Same width
        if left_type.width == right_type.width:
            return left, right
        # Promote narrower to wider using the wider one's PC type
        if left_type.width < right_type.width:
            left_promoted = self.convert(left, right.type_hint)
            return left_promoted, right
        else:
            right_promoted = self.convert(right, left.type_hint)
            return left, right_promoted

    def unify_binop_types(self, left: Union[ir.Value, ValueRef], right: Union[ir.Value, ValueRef]) -> Tuple[ValueRef, ValueRef, bool]:
        """Unify operands for binary operations via PC types."""
        if left.is_python_value() and right.is_python_value():
            # Both are Python values - return as-is, let caller handle Python-level operations
            return left, right, False
        if not right.is_python_value() and left.is_python_value():
            left = self._promote_python_to_pc(left.get_python_value(), right.type_hint)
        if not left.is_python_value() and right.is_python_value():
            right = self._promote_python_to_pc(right.get_python_value(), left.type_hint)
        if not isinstance(left, ValueRef) or left.type_hint is None:
            raise TypeError("unify_binop_types requires ValueRef with type_hint")
        if not isinstance(right, ValueRef) or right.type_hint is None:
            raise TypeError("unify_binop_types requires ValueRef with type_hint")
        left_pc_type = left.type_hint
        right_pc_type = right.type_hint
        left_is_float = hasattr(left_pc_type, '_is_float') and left_pc_type._is_float
        right_is_float = hasattr(right_pc_type, '_is_float') and right_pc_type._is_float
        if left_is_float or right_is_float:
            from .builtin_entities import f64
            if left_is_float and right_is_float:
                target_pc_type = f64 if (left_pc_type == f64 or right_pc_type == f64) else left_pc_type
            elif left_is_float:
                target_pc_type = left_pc_type
            else:
                target_pc_type = right_pc_type
            if not left_is_float:
                left = self.promote_to_float(left, target_pc_type)
            elif left_pc_type != target_pc_type:
                left = self.convert(left, target_pc_type)
            if not right_is_float:
                right = self.promote_to_float(right, target_pc_type)
            elif right_pc_type != target_pc_type:
                right = self.convert(right, target_pc_type)
            return left, right, True
        left, right = self.unify_integer_types(left, right)
        return left, right, False
