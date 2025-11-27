"""
Assignments mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..logger import logger
from ..ir_helpers import safe_store, safe_load, is_const, is_volatile
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access


def _get_struct_field_types(pc_type):
    """Helper to get field types from struct type
    
    Args:
        pc_type: PC struct type (StructType)
        
    Returns:
        List of field types, or None if not available
    """
    if not pc_type:
        return None
    
    # All structs now use unified system (_field_types)
    if hasattr(pc_type, '_field_types'):
        return pc_type._field_types
    
    return None


class AssignmentsMixin:
    """Mixin containing assignments-related visitor methods"""
    
    def visit_lvalue(self, node: ast.AST) -> ValueRef:
        """Compute lvalue for assignment target, returns ValueRef with kind='address'
        
        Unified implementation: delegates to visit_expression which returns ValueRef
        with both loaded value and address, then extracts the address for lvalue context.
        """
        if isinstance(node, ast.Tuple):
            raise ValueError("Tuple unpacking should be handled by caller")
        
        result = self.visit_expression(node)
        return result.as_lvalue()

    def visit_lvalue_or_define(self, node: ast.AST, value_ref: ValueRef, pc_type=None, source="inference") -> ValueRef:
        """Visit lvalue or define new variable if it doesn't exist
        
        Args:
            node: AST node (usually ast.Name)
            value_ref: ValueRef to infer type from (optional)
            pc_type: Explicit PC type (optional, overrides inference)
            source: Source for variable declaration
            
        Returns:
            ValueRef with kind='address' (lvalue), or None if Python constant was created
        """
        if not isinstance(node, ast.Name):
            # For complex expressions, just return lvalue
            return self.visit_lvalue(node)
        
        var_info = self.lookup_variable(node.id)
        if var_info:
            # Variable exists, return lvalue
            return self.visit_lvalue(node)
        else:
            # Variable doesn't exist, create it
            # Special case: Python values without explicit pc_type should remain as Python constants
            if pc_type is None and value_ref is not None and value_ref.is_python_value():
                # Python constant (e.g., x = 1, MyType = i32)
                # Store as Python constant without allocating LLVM storage
                self.declare_variable(
                    name=node.id,
                    type_hint=value_ref.type_hint,
                    alloca=None,  # No LLVM storage needed
                    value_ref=value_ref,  # Store ValueRef directly
                    source="python_constant",
                    line_number=getattr(node, 'lineno', 0)
                )
                return None  # No lvalue for Python constants
            
            # PC value or explicit pc_type: create alloca
            if pc_type is None and value_ref is not None:
                pc_type = self.infer_pc_type_from_value(value_ref)
            
            if pc_type is None:
                raise TypeError(f"Cannot determine type for new variable '{node.id}'")
            
            # Create alloca and declare variable
            llvm_type = pc_type.get_llvm_type(self.module.context)
            alloca = self._create_alloca_in_entry(llvm_type, f"{node.id}_addr")
            
            self.declare_variable(
                name=node.id,
                type_hint=pc_type,
                alloca=alloca,
                source=source,
                line_number=getattr(node, 'lineno', 0)
            )
            
            # Return lvalue for the new variable
            from ..valueref import ValueRef
            return ValueRef(
                kind='address',
                value=alloca,
                type_hint=pc_type,
                address=alloca
            )
    
    def _store_to_lvalue(self, lvalue: ValueRef, rvalue: ValueRef):
        """Store value to lvalue with type conversion and qualifier checks
        
        Special handling for pyconst fields.
        """
        # Special case: pyconst field assignment
        if lvalue.kind == "pyconst_field":
            # Type-check the value
            if not hasattr(lvalue, '_pyconst_value') or not hasattr(lvalue, '_pyconst_type'):
                raise TypeError("Invalid pyconst field lvalue")
            
            expected_value = lvalue._pyconst_value
            pyconst_type = lvalue._pyconst_type
            
            # Extract assigned value
            if rvalue.is_python_value():
                assigned_value = rvalue.value
            elif hasattr(rvalue.value, 'constant'):
                # LLVM constant
                assigned_value = rvalue.value.constant
            else:
                raise TypeError(
                    f"Cannot assign runtime value to pyconst field. "
                    f"pyconst fields require compile-time constant values."
                )
            
            # Type check: value must match exactly
            if assigned_value != expected_value:
                raise TypeError(
                    f"Type mismatch: cannot assign {repr(assigned_value)} to pyconst field "
                    f"of type {pyconst_type.get_instance_name()}. "
                    f"Expected value: {repr(expected_value)}"
                )
            
            # Assignment is valid but is a no-op (zero-sized field)
            return
        
        assert lvalue.kind == 'address', f"Expected lvalue with kind='address', got kind='{lvalue.kind}'"
        
        target_pc_type = lvalue.type_hint
        target_llvm_type = lvalue.ir_value.type.pointee
        
        # Convert value to target type (type_converter will handle Python value promotion)
        source_llvm_type = get_type(rvalue) if not rvalue.is_python_value() else None
        
        if source_llvm_type is None or source_llvm_type != target_llvm_type:
            # Convert using PC type directly (this includes Python value promotion)
            rvalue = self.type_converter.convert(rvalue, target_pc_type)
        
        # Special handling for arrays: if rvalue is already a pointer to array,
        # we need to copy the array contents (load + store), not store the pointer
        rvalue_ir = ensure_ir(rvalue)
        if isinstance(rvalue_ir.type, ir.PointerType) and isinstance(rvalue_ir.type.pointee, ir.ArrayType):
            # Array literal case: rvalue is pointer to array, need to copy contents
            if isinstance(target_llvm_type, ir.ArrayType):
                # Load the array value and store to lvalue
                array_value = self.builder.load(rvalue_ir)
                self.builder.store(array_value, ensure_ir(lvalue))
                return
        
        # Use safe_store for qualifier-aware storage (handles const check + volatile)
        safe_store(self.builder, ensure_ir(rvalue), ensure_ir(lvalue), target_pc_type)
    
    def _store_to_new_lvalue(self, node, var_name, pc_type, rvalue: ValueRef):
        """Create new lvalue for assignment"""
        # Create alloca
        llvm_type = pc_type.get_llvm_type(self.module.context)
        alloca = self._create_alloca_in_entry(llvm_type, f"{var_name}_addr")
        
        # Declare variable
        self.declare_variable(
            name=var_name,
            type_hint=pc_type,
            alloca=alloca,
            source="annotation",
            line_number=node.lineno
        )
        
        # Store value
        rvalue_ir = ensure_ir(rvalue)
        
        # Special handling for arrays: if rvalue is already a pointer to array,
        # we need to copy the array contents (load + store), not store the pointer
        if isinstance(rvalue_ir.type, ir.PointerType) and isinstance(rvalue_ir.type.pointee, ir.ArrayType):
            # Array literal case: rvalue is pointer to array, need to copy contents
            if isinstance(llvm_type, ir.ArrayType):
                # Load the array value and store to new alloca
                array_value = self.builder.load(rvalue_ir)
                self.builder.store(array_value, alloca)
            else:
                # Non-array target type, just store normally
                self.builder.store(rvalue_ir, alloca)
        else:
            # Normal case: store value directly
            self.builder.store(rvalue_ir, alloca)
    
    def visit_Assign(self, node: ast.Assign):
        """Handle assignment statements with automatic type inference
        
        Refactored to use lvalue computation for cleaner code.
        """
        # Evaluate rvalue once
        rvalue = self.visit_expression(node.value)
        
        # Check if rvalue is a linear type (for linear token registration)
        rvalue_is_linear = False
        if hasattr(rvalue, 'type_hint') and self._is_linear_type(rvalue.type_hint):
            rvalue_is_linear = True
        
        # Check if rvalue is a reference to a linear variable/field (cannot copy)
        # Use ValueRef's var_name and linear_path if available
        if hasattr(rvalue, 'var_name') and rvalue.var_name and hasattr(rvalue, 'linear_path') and rvalue.linear_path is not None:
            rvalue_var_name = rvalue.var_name
            rvalue_path = rvalue.linear_path
            rvalue_var_info = self.lookup_variable(rvalue_var_name)
            if rvalue_var_info:
                # Check if the specific path is in active state
                rvalue_state = self._get_linear_state(rvalue_var_info, rvalue_path)
                if rvalue_state == 'active':
                    # Format path for error message
                    if rvalue_path:
                        path_str = f"{rvalue_var_name}[{']['.join(map(str, rvalue_path))}]"
                    else:
                        path_str = rvalue_var_name
                    raise TypeError(
                        f"Cannot assign linear token '{path_str}' "
                        f"(use move() to transfer ownership, line {node.lineno})"
                    )
        
        # Note: Do NOT promote Python values here - keep them as Python values
        # Promotion happens when the value is used in PC context (operations, return, etc.)
        
        # Handle multiple targets
        for target in node.targets:
            # Handle tuple unpacking separately
            if isinstance(target, ast.Tuple):
                self._handle_tuple_unpacking(target, node.value, rvalue)
                continue
            
            # For ast.Name targets, try visit_lvalue_or_define first
            if isinstance(target, ast.Name):
                var_info = self.lookup_variable(target.id)
                if not var_info:
                    # New variable: use visit_lvalue_or_define for unified logic
                    lvalue = self.visit_lvalue_or_define(target, value_ref=rvalue, source="inference")
                    
                    # If Python constant was created (lvalue is None), skip to next target
                    if lvalue is None:
                        continue
                    
                    # PC value was created, store it
                    self._store_to_lvalue(lvalue, rvalue)
                    
                    # Handle linear token transfer for new variable
                    if rvalue_is_linear:
                        # Activate if:
                        # 1. rvalue is a variable reference (ownership transfer)
                        # 2. rvalue is an initialized value (not ir.Undefined)
                        from llvmlite import ir as llvm_ir
                        is_undefined = (
                            rvalue.kind == 'value' and 
                            isinstance(rvalue.value, llvm_ir.Constant) and
                            rvalue.value.constant == llvm_ir.Undefined
                        )
                        
                        if hasattr(rvalue, 'var_name') and rvalue.var_name:
                            # Variable reference - transfer ownership
                            self._register_linear_token(target.id, lvalue.type_hint, node, path=())
                            self._transfer_linear_ownership(rvalue, reason="assignment")
                        elif not is_undefined:
                            # Initialized value (function return, linear(), etc.)
                            self._register_linear_token(target.id, lvalue.type_hint, node, path=())
                    continue
                else:
                    # Variable exists - check const/linear constraints
                    # Check if variable is const
                    if hasattr(var_info.type_hint, 'is_const') and var_info.type_hint.is_const():
                        raise TypeError(
                            f"Cannot reassign to const variable '{target.id}' "
                            f"(declared at line {var_info.line_number}, line {node.lineno})"
                        )
                    
                    # Variable exists - check if it's a Python constant variable
                    if var_info.is_python_constant:
                        # This is a Python constant variable - update it
                        if rvalue.is_python_value():
                            # Update Python constant value_ref
                            var_info.value_ref = rvalue
                            continue
                        else:
                            # Trying to assign PC value to Python constant variable
                            # Need to convert variable to PC type with alloca
                            inferred_llvm_type = get_type(rvalue)
                            alloca = self._create_alloca_in_entry(inferred_llvm_type, f"{target.id}_addr")
                            self.builder.store(ensure_ir(rvalue), alloca)
                            
                            # Update variable info with alloca and new value_ref
                            inferred_pc_type = self.infer_pc_type_from_value(rvalue)
                            from ..valueref import ValueRef
                            var_info.alloca = alloca
                            var_info.value_ref = ValueRef(
                                kind='address',
                                value=alloca,
                                type_hint=inferred_pc_type,
                                address=alloca
                            )
                            continue
            
            # Check if reassigning to unconsumed linear token (for all lvalue types)
            # Use lvalue's var_name and linear_path from ValueRef
            lvalue = self.visit_lvalue(target)
            if hasattr(lvalue, 'var_name') and lvalue.var_name and hasattr(lvalue, 'linear_path') and lvalue.linear_path is not None:
                target_var_info = self.lookup_variable(lvalue.var_name)
                if target_var_info:
                    target_state = self._get_linear_state(target_var_info, lvalue.linear_path)
                    if target_state == 'active':
                        # Format path for error message
                        if lvalue.linear_path:
                            path_str = f"{lvalue.var_name}[{']['.join(map(str, lvalue.linear_path))}]"
                        else:
                            path_str = lvalue.var_name
                        raise TypeError(
                            f"Cannot reassign '{path_str}': linear token not consumed "
                            f"(declared at line {target_var_info.line_number}, line {node.lineno})"
                        )
            
            # For existing variables or complex lvalues, store the value
            self._store_to_lvalue(lvalue, rvalue)
            
            # After storing, register linear token if rvalue is linear type
            # This handles undefined/consumed -> active transition
            if rvalue_is_linear:
                if hasattr(lvalue, 'var_name') and lvalue.var_name and hasattr(lvalue, 'linear_path') and lvalue.linear_path is not None:
                    # Activate if:
                    # 1. rvalue is a variable reference (ownership transfer)
                    # 2. rvalue is an initialized value (not ir.Undefined)
                    from llvmlite import ir as llvm_ir
                    is_undefined = (
                        rvalue.kind == 'value' and 
                        isinstance(rvalue.value, llvm_ir.Constant) and
                        hasattr(rvalue.value, 'constant') and
                        rvalue.value.constant == llvm_ir.Undefined
                    )
                    
                    if hasattr(rvalue, 'var_name') and rvalue.var_name:
                        # Variable reference - transfer ownership
                        self._register_linear_token(lvalue.var_name, rvalue.type_hint, node, path=lvalue.linear_path)
                        self._transfer_linear_ownership(rvalue, reason="assignment")
                    elif not is_undefined:
                        # Initialized value (function return, linear(), etc.)
                        self._register_linear_token(lvalue.var_name, rvalue.type_hint, node, path=lvalue.linear_path)
    
    def _handle_tuple_unpacking(self, target: ast.Tuple, value_node: ast.AST, rvalue: ValueRef):
        """Handle tuple unpacking assignment"""
        if rvalue.is_python_value():
            # Python tuple unpacking: a, b = (1, 2) where (1, 2) is Python value
            tuple_value = rvalue.get_python_value()
            if len(tuple_value) != len(target.elts):
                raise TypeError(f"Unpacking mismatch: {len(target.elts)} variables, {len(tuple_value)} values")
            
            for py_val, elt in zip(tuple_value, target.elts):
                # Convert Python value to ValueRef
                from ..valueref import ValueRef
                from ..builtin_entities.python_type import PythonType
                val_ref = ValueRef(kind='python', value=py_val, 
                                 type_hint=PythonType.wrap(py_val, is_constant=True))
                
                # Get or create lvalue (may return None for Python constants)
                lvalue = self.visit_lvalue_or_define(elt, value_ref=val_ref)
                
                # Store the value (skip if Python constant was created)
                if lvalue is not None:
                    self._store_to_lvalue(lvalue, val_ref)
        elif hasattr(rvalue, 'type_hint') and hasattr(rvalue.type_hint, '_field_types'):
            # Struct unpacking: a, b = func() where func() returns struct
            struct_type = rvalue.type_hint
            field_types = struct_type._field_types
            
            if len(target.elts) != len(field_types):
                raise TypeError(f"Unpacking mismatch: {len(target.elts)} variables, {len(field_types)} fields")
            
            for i, elt in enumerate(target.elts):
                # Extract field value from struct
                field_value = self.builder.extract_value(ensure_ir(rvalue), i)
                field_pc_type = field_types[i]
                from ..valueref import ValueRef
                field_val_ref = ValueRef(kind='value', value=field_value, type_hint=field_pc_type)
                
                # Get or create lvalue (with explicit pc_type to force PC value)
                lvalue = self.visit_lvalue_or_define(elt, value_ref=field_val_ref, pc_type=field_pc_type)
                
                # Store the value
                self._store_to_lvalue(lvalue, field_val_ref)
                
                # Register linear token if needed
                if isinstance(elt, ast.Name) and self._is_linear_type(field_pc_type):
                    self._register_linear_token(elt.id, field_pc_type, target)
        else:
            # Cannot unpack from this type
            raise NotImplementedError(
                f"Cannot unpack from type {rvalue.type_hint}. "
                f"Expected struct type with _field_types, tuple literal, or Python tuple."
            )

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Handle annotated assignment statements (variable declarations with types)
        
        Now uses the new context system to track PC types alongside LLVM types.
        Supports static local variables (converted to global variables with internal linkage).
        """
        import ast as ast_module
        logger.debug(f"visit_AnnAssign: {ast_module.unparse(node)}")
        
        if not isinstance(node.target, ast.Name):
            raise RuntimeError("AnnAssign only supports simple names")
        var_name = node.target.id
        
        # Check if variable already exists in CURRENT scope - AnnAssign is declaration, not reassignment
        # Allow shadowing variables from outer scopes (C-like behavior)
        if self.ctx.var_registry.is_declared_in_current_scope(var_name):
            existing = self.ctx.var_registry.lookup(var_name)
            raise RuntimeError(
                f"Cannot redeclare variable '{var_name}': already declared in this scope at line {existing.line_number} "
                f"(attempting redeclaration at line {node.lineno})"
            )
        
        # Get PC type from annotation
        is_static_var = False
        if not hasattr(node, 'annotation'):
            raise RuntimeError("AnnAssign requires annotation")

        pc_type = self.get_pc_type_from_annotation(node.annotation)
        if pc_type is None:
            import ast as ast_module
            annotation_str = ast_module.unparse(node.annotation) if hasattr(ast_module, 'unparse') else str(node.annotation)
            raise RuntimeError(
                f"AnnAssign requires valid PC type annotation. "
                f"Line: {node.lineno}, annotation: {annotation_str}"
            )
        # Check if this is a static-qualified type
        # if pc_type and hasattr(pc_type, 'is_static') and pc_type.is_static():
        #     is_static_var = True
        #     # Get the underlying type
        #     if hasattr(pc_type, 'get_qualified_type'):
        #         pc_type = pc_type.get_qualified_type()

        # Now parse the RHS
        logger.debug("AnnAssign processing", has_value=(node.value is not None))
        if node is None or node.value is None:
            # No initialization value - create undefined value (matches C behavior)
            llvm_type = pc_type.get_llvm_type(self.module.context)
            undef_value = ir.Constant(llvm_type, ir.Undefined)
            rvalue = wrap_value(undef_value, kind="value", type_hint=pc_type)
        else:
            rvalue = self.visit_expression(node.value)
            logger.debug("AnnAssign rvalue", rvalue=rvalue, from_type=rvalue.type_hint, to_type=pc_type)

            # If the type of RHS does not match pc_type, convert it
            if rvalue.type_hint != pc_type:
                rvalue = self.type_converter.convert(rvalue, pc_type)
                logger.debug("AnnAssign converted", result_type=rvalue.type_hint)
            
        # Store the value
        logger.debug(f"visit_AnnAssign before store: var_name={var_name}, node={ast_module.unparse(node)}, rvalue={rvalue}, pc_type={pc_type}")
        self._store_to_new_lvalue(node, var_name, pc_type, rvalue)
    

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment statements (+=, -=, *=, etc.)"""
        # Don't process if current block is already terminated
        if self.builder.block.is_terminated:
            return
        
        # Get the lvalue (address) of the target
        target_addr = self.visit_lvalue(node.target)
        
        # Load current value
        current_value = self.builder.load(ensure_ir(target_addr))
        current_val_ref = wrap_value(current_value, kind="value", type_hint=target_addr.type_hint)
        
        # Evaluate the right-hand side
        rhs_value = self.visit_expression(node.value)
        
        # Create a fake BinOp node to reuse binary operation logic
        fake_binop = ast.BinOp(
            left=ast.Name(id='_dummy_'),
            op=node.op,
            right=ast.Name(id='_dummy_')
        )
        
        # Perform the operation using unified binary operation logic
        result = self._perform_binary_operation(fake_binop.op, current_val_ref, rhs_value)
        
        # Store the result back
        self.builder.store(ensure_ir(result), ensure_ir(target_addr))


    def _parse_type_annotation(self, annotation):
        """Parse type annotation and return LLVM type
        
        This is the main entry point for type annotation parsing.
        Uses TypeResolver for unified type parsing.
        """
        return self.type_resolver.annotation_to_llvm_type(annotation)
    

    def _parse_type_annotation_to_builtin(self, annotation):
        """Parse type annotation and return BuiltinType class or LLVM type
        
        This is the unified type annotation parser that returns intermediate representation.
        Uses TypeResolver for unified type parsing.
        
        Returns:
            - BuiltinType subclass for builtin types (i32, f64, ptr[T], array[T, N], tuple[T1, T2])
            - ir.Type for struct types
            - None if parsing fails
        """
        return self.type_resolver.parse_annotation(annotation)
    

    def _parse_array_type_annotation(self, slice_node):
        """Parse array type annotation and return LLVM type (backward compatibility)
        
        Delegates to TypeResolver.annotation_to_llvm_type.
        """
        # Create a fake Subscript node for TypeResolver
        fake_subscript = ast.Subscript(
            value=ast.Name(id='array'),
            slice=slice_node
        )
        return self.type_resolver.annotation_to_llvm_type(fake_subscript)
    

    def _parse_array_type_annotation_to_builtin(self, slice_node):
        """Parse array type annotation and return BuiltinType (backward compatibility)
        
        Delegates to TypeResolver.parse_annotation.
        """
        # Create a fake Subscript node for TypeResolver
        fake_subscript = ast.Subscript(
            value=ast.Name(id='array'),
            slice=slice_node
        )
        return self.type_resolver.parse_annotation(fake_subscript)
    

    def _parse_tuple_type_annotation(self, slice_node):
        """Parse tuple type annotation and return LLVM type (backward compatibility)
        
        Delegates to TypeResolver._parse_tuple_annotation.
        """
        # Create a fake Subscript node for TypeResolver
        fake_subscript = ast.Subscript(
            value=ast.Name(id='tuple'),
            slice=slice_node
        )
        return self.type_resolver.annotation_to_llvm_type(fake_subscript)
    

    def _parse_tuple_type_annotation_to_builtin(self, slice_node):
        """Parse tuple type annotation and return BuiltinType (backward compatibility)
        
        Delegates to TypeResolver._parse_tuple_annotation.
        """
        # Create a fake Subscript node for TypeResolver
        fake_subscript = ast.Subscript(
            value=ast.Name(id='tuple'),
            slice=slice_node
        )
        return self.type_resolver.parse_annotation(fake_subscript)
    

    def _initialize_array(self, alloca, pc_array_type, value_node):
        """Initialize an array from a list literal or function call
        
        Args:
            alloca: The array allocation
            pc_array_type: PC array type (array[T, N])
            value_node: AST node for initialization value
        """
        # Handle array[T, N]() zero-initialization
        if isinstance(value_node, ast.Call):
            # Check if it's array[T, N]() call
            if (isinstance(value_node.func, ast.Subscript) and 
                isinstance(value_node.func.value, ast.Name) and 
                value_node.func.value.id == 'array'):
                # Zero-initialize the array
                self._zero_initialize_array(alloca, pc_array_type)
                return
            else:
                # Other function calls - evaluate and store
                value = self.visit_expression(value_node)
                self.builder.store(ensure_ir(value), alloca)
                return
        
        # Handle list literal initialization
        if isinstance(value_node, ast.List):
            self._initialize_array_from_list(alloca, pc_array_type, value_node)
        else:
            # Fallback: evaluate expression and store
            value = self.visit_expression(value_node)
            self.builder.store(ensure_ir(value), alloca)


    def _get_array_element_type(self, array_type):
        """Get the innermost element type of an array (handles multi-dimensional arrays)"""
        current = array_type
        while isinstance(current, ir.ArrayType):
            current = current.element
        return current
    

    def _zero_initialize_array(self, alloca, pc_array_type):
        """Zero-initialize an array using PC type"""
        llvm_array_type = pc_array_type.get_llvm_type(self.module.context)
        zero_array = self.type_converter.create_zero_constant(llvm_array_type)
        self.builder.store(zero_array, alloca)
    

    def _initialize_array_from_list(self, alloca, pc_array_type, list_node):
        """Initialize an array from a list literal
        
        Args:
            alloca: The array allocation
            pc_array_type: PC array type (array[T, N])
            list_node: AST List node with initialization values
        """
        # Unwrap qualifiers (const, volatile, etc.) to get the actual array type
        actual_array_type = pc_array_type
        while hasattr(actual_array_type, 'get_qualified_type') and actual_array_type.get_qualified_type() is not None:
            actual_array_type = actual_array_type.get_qualified_type()
        
        # Get element PC type and dimensions from pythoc array type
        pc_elem_type = actual_array_type.element_type
        # Get dimensions - array types use 'dimensions' attribute (tuple or list)
        if hasattr(actual_array_type, 'dimensions'):
            dims = actual_array_type.dimensions
            dimensions = list(dims) if isinstance(dims, (list, tuple)) else [dims]
        else:
            # Fallback: try to extract from nested array types
            dimensions = []
            current_type = actual_array_type
            while hasattr(current_type, 'element_type'):
                if hasattr(current_type, 'count'):
                    dimensions.append(current_type.count)
                    current_type = current_type.element_type
                else:
                    break
        
        elem_llvm_type = pc_elem_type.get_llvm_type(self.module.context)
        
        # For 1D arrays
        if len(dimensions) == 1:
            size = dimensions[0]
            elements = list_node.elts
            
            # Store each element
            for i, elem_node in enumerate(elements):
                if i >= size:
                    break
                
                # Evaluate element
                elem_value = self.visit_expression(elem_node)
                
                # Type conversion if needed
                elem_value_type = get_type(elem_value)
                if elem_value_type != elem_llvm_type:
                    # Use PC element type for conversion
                    elem_value = self.type_converter.convert(elem_value, pc_elem_type)
                
                # Get pointer to array element
                zero = ir.Constant(ir.IntType(32), 0)
                idx = ir.Constant(ir.IntType(32), i)
                elem_ptr = self.builder.gep(alloca, [zero, idx])
                
                # Store element
                self.builder.store(ensure_ir(elem_value), elem_ptr)
            
            # Zero-initialize remaining elements if list is shorter than array
            if len(elements) < size:
                zero_val = self.type_converter.create_zero_constant(elem_llvm_type)
                for i in range(len(elements), size):
                    idx = ir.Constant(ir.IntType(32), i)
                    elem_ptr = self.builder.gep(alloca, [ir.Constant(ir.IntType(32), 0), idx])
                    self.builder.store(zero_val, elem_ptr)
        
        # For multi-dimensional arrays
        else:
            self._initialize_multidim_array(alloca, pc_array_type, pc_elem_type, list_node, dimensions)
    

    def _initialize_multidim_array(self, alloca, pc_array_type, pc_elem_type, list_node, dimensions):
        """Initialize a multi-dimensional array from nested lists
        
        Args:
            alloca: The array allocation
            pc_array_type: PC array type (array[array[T, M], N])
            pc_elem_type: PC element type (T)
            list_node: AST List node with nested lists
            dimensions: List of dimension sizes
        """
        elem_llvm_type = pc_elem_type.get_llvm_type(self.module.context)
        
        def store_nested_list(node, indices):
            """Recursively store nested list elements with zero-fill for missing items"""
            depth = len(indices)
            if isinstance(node, ast.List):
                # Determine max elements for this depth
                max_count = dimensions[depth] if depth < len(dimensions) else 0
                # Iterate provided elements
                for i, elem in enumerate(node.elts[:max_count]):
                    store_nested_list(elem, indices + [i])
                # Zero-fill remaining elements if provided list is shorter than max_count
                if depth < len(dimensions):
                    elem_zero = self.type_converter.create_zero_constant(elem_llvm_type)
                    for i in range(len(node.elts), max_count):
                        # Build GEP indices: [0, i0, i1, i2, ...]
                        gep_indices = [ir.Constant(ir.IntType(32), 0)]
                        for idx in indices + [i]:
                            gep_indices.append(ir.Constant(ir.IntType(32), idx))
                        elem_ptr = self.builder.gep(alloca, gep_indices)
                        self.builder.store(elem_zero, elem_ptr)
            else:
                # Leaf element - store it at the exact position
                elem_value = self.visit_expression(node)
                # Type conversion if needed
                elem_value_type = get_type(elem_value)
                if elem_value_type != elem_llvm_type:
                    elem_value = self.type_converter.convert(elem_value, pc_elem_type)
                # Build GEP indices: [0, i0, i1, i2, ...]
                gep_indices = [ir.Constant(ir.IntType(32), 0)]
                for idx in indices:
                    gep_indices.append(ir.Constant(ir.IntType(32), idx))
                elem_ptr = self.builder.gep(alloca, gep_indices)
                self.builder.store(ensure_ir(elem_value), elem_ptr)
        
        store_nested_list(list_node, [])
    

    def _initialize_tuple(self, alloca, tuple_type, value_node, type_hint=None):
        """Initialize a tuple/struct from a tuple literal
        
        Args:
            alloca: Pointer to struct
            tuple_type: LLVM struct type
            value_node: ast.Tuple node
            pc_type: PC type hint for the struct (optional, for type conversion)
        """
        # value_node should be ast.Tuple
        if not isinstance(value_node, ast.Tuple):
            raise TypeError(f"Expected tuple literal for tuple initialization, got {type(value_node)}")
        
        # Get tuple field types
        field_types = tuple_type.elements
        num_fields = len(field_types)
        
        # Get tuple elements
        elements = value_node.elts
        if len(elements) != num_fields:
            raise TypeError(f"Tuple literal has {len(elements)} elements, but type expects {num_fields}")
        
        # Store each element
        for i, (elem_node, field_type) in enumerate(zip(elements, field_types)):
            # Get pointer to field using GEP
            zero = ir.Constant(ir.IntType(32), 0)
            idx = ir.Constant(ir.IntType(32), i)
            field_ptr = self.builder.gep(alloca, [zero, idx])
            
            # Check if field is a union type (represented as {[N x i8]})
            is_union_field = (isinstance(field_type, ir.LiteralStructType) and 
                            len(field_type.elements) == 1 and 
                            isinstance(field_type.elements[0], ir.ArrayType) and
                            isinstance(field_type.elements[0].element, ir.IntType) and
                            field_type.elements[0].element.width == 8)
            
            if is_union_field and isinstance(elem_node, ast.Tuple):
                # Union field - only initialize first element
                if len(elem_node.elts) > 0:
                    first_elem = elem_node.elts[0]
                    first_value = self.visit_expression(first_elem)
                    first_type = get_type(first_value)
                    
                    # Bitcast union field pointer to first element type pointer
                    first_ptr = self.builder.bitcast(field_ptr, ir.PointerType(first_type))
                    self.builder.store(ensure_ir(first_value), first_ptr)
                continue
            
            # Evaluate element - handle special cases
            if isinstance(elem_node, ast.List):
                # Array literal - initialize array field
                if isinstance(field_type, ir.ArrayType):
                    # Initialize array field directly
                    self._initialize_array(field_ptr, field_type, elem_node)
                    continue
                else:
                    raise TypeError(f"List literal provided for non-array field type {field_type}")
            elif isinstance(elem_node, ast.Tuple):
                # Nested tuple - need to create a struct value
                # For now, create the nested struct inline
                nested_field_types = field_type.elements
                nested_struct = ir.Constant(field_type, ir.Undefined)
                
                # Try to get nested PC type
                field_types = _get_struct_field_types(pc_type)
                nested_pc_type = field_types[i] if (field_types and i < len(field_types)) else None
                
                for j, nested_elem_node in enumerate(elem_node.elts):
                    nested_elem_value = self.visit_expression(nested_elem_node)
                    
                    # Type conversion for nested tuple elements
                    nested_elem_type = get_type(nested_elem_value)
                    nested_field_type = nested_field_types[j]
                    if nested_elem_type != nested_field_type:
                        # Try conversion if we have nested PC type
                        nested_field_types = _get_struct_field_types(nested_pc_type)
                        nested_pc_field_type = nested_field_types[j] if (nested_field_types and j < len(nested_field_types)) else None
                        
                        if nested_pc_field_type:
                            nested_elem_value = self.type_converter.convert(nested_elem_value, nested_pc_field_type)
                        else:
                            raise TypeError(f"Nested tuple field {j} type mismatch: expected {nested_field_type}, got {nested_elem_type}")
                    
                    nested_struct = self.builder.insert_value(nested_struct, ensure_ir(nested_elem_value), j)
                
                elem_value = wrap_value(nested_struct, kind="value", type_hint=field_pc_type)
            else:
                elem_value = self.visit_expression(elem_node)
                
                # Type conversion if needed
                elem_type = get_type(elem_value)
                if elem_type != field_type:
                    # Try to get PC field type for conversion
                    field_types = _get_struct_field_types(pc_type)
                    pc_field_type = field_types[i] if (field_types and i < len(field_types)) else None
                    
                    if pc_field_type:
                        elem_value = self.type_converter.convert(elem_value, pc_field_type)
                    else:
                        raise TypeError(f"Struct field {i} type mismatch: expected {field_type}, got {elem_type}. Cannot convert without PC type hint.")
            
            # Store element (field_ptr already computed above)
            self.builder.store(ensure_ir(elem_value), field_ptr)
    

    def _initialize_union(self, alloca, union_type, value_node):
        """Initialize a union from a tuple literal
        
        For unions, only the first element in the tuple is used for initialization.
        The union storage is a byte array, so we need to bitcast and store.
        """
        if not isinstance(value_node, ast.Tuple):
            raise TypeError(f"Expected tuple literal for union initialization, got {type(value_node)}")
        
        # Get union field types
        if not hasattr(union_type, 'get_field_count') or union_type.get_field_count() == 0:
            raise TypeError("Union type has no fields")
        
        # Get tuple elements - only use the first one for initialization
        elements = value_node.elts
        if len(elements) == 0:
            # Empty initialization - just leave it uninitialized
            return
        
        # Use first element to initialize the union
        elem_node = elements[0]
        field_type = union_type.get_field_type(0)
        
        # Evaluate element value
        elem_value = self.visit_expression(elem_node)
        
        # Type conversion if needed
        elem_type = get_type(elem_value)
        if hasattr(field_type, 'get_llvm_type'):
            target_llvm_type = field_type.get_llvm_type(self.module.context)
            target_pc_type = field_type  # field_type is already a PC type
        else:
            target_llvm_type = ir.IntType(32)
            from ..builtin_entities import i32
            target_pc_type = i32
        
        if elem_type != target_llvm_type:
            # Use PC type for conversion
            elem_value = self.type_converter.convert(elem_value, target_pc_type)
        
        # Bitcast union pointer to field type pointer and store
        field_ptr = self.builder.bitcast(alloca, ir.PointerType(target_llvm_type))
        self.builder.store(ensure_ir(elem_value), field_ptr)


