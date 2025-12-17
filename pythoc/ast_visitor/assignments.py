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


class AssignmentsMixin:
    """Mixin containing assignments-related visitor methods"""
    
    def visit_lvalue(self, node: ast.AST) -> ValueRef:
        """Compute lvalue for assignment target, returns ValueRef with kind='address'
        
        Unified implementation: delegates to visit_expression which returns ValueRef
        with both loaded value and address, then extracts the address for lvalue context.
        """
        if isinstance(node, ast.Tuple):
            logger.error("Tuple unpacking should be handled by caller", node=node, exc_type=ValueError)
        
        result = self.visit_expression(node)
        return result.as_lvalue()

    def _apply_assign_decay(self, value_ref: ValueRef) -> ValueRef:
        """Apply assignment decay to rvalue if its type supports it.
        
        This implements C-like array-to-pointer decay for untyped assignments.
        Uses duck typing: if type_hint has handle_assign_decay method, call it.
        
        Args:
            value_ref: The rvalue to potentially decay
            
        Returns:
            Decayed ValueRef if applicable, otherwise original value_ref
        """
        type_hint = value_ref.get_pc_type()
        # Duck typing: check if type has handle_assign_decay method
        if hasattr(type_hint, 'handle_assign_decay'):
            return type_hint.handle_assign_decay(self, value_ref)
        
        return value_ref

    def _check_linear_rvalue_copy(self, rvalue: ValueRef, node) -> None:
        """Check if rvalue is an active linear token - forbid copy.
        
        Linear tokens cannot be copied; they must be moved explicitly.
        Raises error via logger if attempting to copy an active linear token.
        
        Args:
            rvalue: The rvalue being assigned
            node: AST node for error reporting (lineno)
        """
        if not (hasattr(rvalue, 'var_name') and rvalue.var_name and 
                hasattr(rvalue, 'linear_path') and rvalue.linear_path is not None):
            return
        
        rvalue_var_info = self.lookup_variable(rvalue.var_name)
        if not rvalue_var_info:
            return
        
        rvalue_state = self._get_linear_state(rvalue_var_info, rvalue.linear_path)
        if rvalue_state == 'active':
            # Format path for error message
            if rvalue.linear_path:
                path_str = f"{rvalue.var_name}[{']['.join(map(str, rvalue.linear_path))}]"
            else:
                path_str = rvalue.var_name
            logger.error(
                f"Cannot assign linear token '{path_str}' "
                f"(use move() to transfer ownership)",
                node
            )

    def _check_linear_lvalue_overwrite(self, lvalue: ValueRef, node) -> None:
        """Check if lvalue holds an unconsumed linear token - forbid overwrite.
        
        Cannot reassign to a location that holds an active linear token
        without first consuming it.
        Raises error via logger if attempting to overwrite an unconsumed linear token.
        
        Args:
            lvalue: The lvalue being assigned to
            node: AST node for error reporting (lineno)
        """
        if not (hasattr(lvalue, 'var_name') and lvalue.var_name and 
                hasattr(lvalue, 'linear_path') and lvalue.linear_path is not None):
            return
        
        target_var_info = self.lookup_variable(lvalue.var_name)
        if not target_var_info:
            return
        
        target_state = self._get_linear_state(target_var_info, lvalue.linear_path)
        if target_state == 'active':
            # Format path for error message
            if lvalue.linear_path:
                path_str = f"{lvalue.var_name}[{']['.join(map(str, lvalue.linear_path))}]"
            else:
                path_str = lvalue.var_name
            logger.error(
                f"Cannot reassign '{path_str}': linear token not consumed "
                f"(declared at line {target_var_info.line_number})",
                node
            )

    def visit_lvalue_or_define(self, node: ast.AST, value_ref: ValueRef, pc_type=None, source="inference") -> ValueRef:
        """Visit lvalue or define new variable if it doesn't exist
        
        Args:
            node: AST node (usually ast.Name)
            value_ref: ValueRef to infer type from (optional)
            pc_type: Explicit PC type (optional, overrides inference)
            source: Source for variable declaration
            
        Returns:
            ValueRef with kind='address' (lvalue)
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
            # Infer pc_type from value_ref if not provided
            # For Python values, this returns PythonType which has zero-sized LLVM type {}
            if pc_type is None and value_ref is not None:
                pc_type = self.infer_pc_type_from_value(value_ref)
            
            if pc_type is None:
                logger.error(f"Cannot determine type for new variable '{node.id}'", node=node, exc_type=TypeError)
            
            # Create alloca and declare variable
            # For pyconst/PythonType, this creates a zero-sized alloca {}
            llvm_type = pc_type.get_llvm_type(self.module.context)
            alloca = self._create_alloca_in_entry(llvm_type, f"{node.id}_addr")
            
            self.declare_variable(
                name=node.id,
                type_hint=pc_type,
                alloca=alloca,
                source=source,
                line_number=getattr(node, 'lineno', 0)
            )
            
            # Return lvalue for the new variable with linear tracking info
            from ..valueref import wrap_value
            return wrap_value(
                alloca,
                kind='address',
                type_hint=pc_type,
                address=alloca,
                var_name=node.id,
                linear_path=()
            )
    
    def _store_to_lvalue(self, lvalue: ValueRef, rvalue: ValueRef, node: ast.AST = None):
        """Store value to lvalue with type conversion and qualifier checks
        
        Special handling for pyconst fields (zero-sized, no actual store).
        """
        target_pc_type = lvalue.get_pc_type()
        
        # Special case: pyconst target - zero-sized, assignment is a no-op
        # Must check before convert() since Python values don't have ir_value
        from ..builtin_entities.python_type import PythonType
        if isinstance(target_pc_type, PythonType):
            # Type check: if target is pyconst[X], rvalue must be X
            if target_pc_type.is_constant():
                expected_value = target_pc_type.get_constant_value()
                if rvalue.is_python_value():
                    actual_value = rvalue.value
                else:
                    logger.error(f"Cannot store to pyconst target: {lvalue}={rvalue}", node=node, exc_type=TypeError)
                if actual_value != expected_value:
                    logger.error(f"Cannot store to pyconst target: {lvalue}={rvalue}", node=node, exc_type=TypeError)
            # pyconst fields are zero-sized, assignment is a no-op after type check
            return
        
        # Convert value to target type (type_converter will handle Python value promotion)
        rvalue = self.type_converter.convert(rvalue, target_pc_type)
        
        # Use safe_store for qualifier-aware storage (handles const check + volatile)
        safe_store(self.builder, ensure_ir(rvalue), ensure_ir(lvalue), target_pc_type)

    def _assign_to_target(self, target: ast.AST, rvalue: ValueRef, node, pc_type=None) -> None:
        """Unified single-target assignment: lvalue resolution, linear checks, store, and linear registration.
        
        Args:
            target: AST node for assignment target (ast.Name, ast.Attribute, ast.Subscript)
            rvalue: Value to assign
            node: AST node for error reporting
            pc_type: Explicit PC type (optional, overrides inference)
        """
        decayed_rvalue = self._apply_assign_decay(rvalue)
        
        # Get or create lvalue
        lvalue = self.visit_lvalue_or_define(target, value_ref=decayed_rvalue, pc_type=pc_type, source="inference")
        
        # Check if lvalue holds an unconsumed linear token (forbid overwrite)
        self._check_linear_lvalue_overwrite(lvalue, node)
        
        # Store value to lvalue
        self._store_to_lvalue(lvalue, decayed_rvalue, node)
        
        # Handle linear token registration
        rvalue_pc_type = rvalue.get_pc_type()
        if self._is_linear_type(rvalue_pc_type):
            lvalue_var_name = getattr(lvalue, 'var_name', None)
            lvalue_linear_path = getattr(lvalue, 'linear_path', None)
            
            if lvalue_var_name and lvalue_linear_path is not None:
                # Check if rvalue is undefined
                from llvmlite import ir as llvm_ir
                is_undefined = (
                    rvalue.kind == 'value' and 
                    isinstance(rvalue.value, llvm_ir.Constant) and
                    hasattr(rvalue.value, 'constant') and
                    rvalue.value.constant == llvm_ir.Undefined
                )
                
                if hasattr(rvalue, 'var_name') and rvalue.var_name:
                    # Variable reference - transfer ownership
                    self._register_linear_token(lvalue_var_name, lvalue.type_hint, node, path=lvalue_linear_path)
                    self._transfer_linear_ownership(rvalue, reason="assignment")
                elif not is_undefined:
                    # Initialized value (function return, linear(), etc.)
                    self._register_linear_token(lvalue_var_name, lvalue.type_hint, node, path=lvalue_linear_path)
    
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
        """Handle assignment statements with automatic type inference"""
        # Evaluate rvalue once
        rvalue = self.visit_expression(node.value)
        
        # Check if rvalue is an active linear token (forbid copy)
        self._check_linear_rvalue_copy(rvalue, node)
        
        # Handle multiple targets
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                self._handle_tuple_unpacking(target, node.value, rvalue, node)
            else:
                self._assign_to_target(target, rvalue, node)
    
    def _handle_tuple_unpacking(self, target: ast.Tuple, value_node: ast.AST, rvalue: ValueRef, node: ast.AST):
        """Handle tuple unpacking assignment"""
        if rvalue.is_python_value():
            # Python tuple unpacking: a, b = (1, 2) where (1, 2) is Python value
            tuple_value = rvalue.get_python_value()
            if len(tuple_value) != len(target.elts):
                logger.error(f"Unpacking mismatch: {len(target.elts)} variables, {len(tuple_value)} values",
                            node=target, exc_type=TypeError)
            
            for py_val, elt in zip(tuple_value, target.elts):
                # Convert Python value to ValueRef
                from ..valueref import wrap_value
                from ..builtin_entities.python_type import PythonType
                val_ref = wrap_value(py_val, kind='python',
                                 type_hint=PythonType.wrap(py_val, is_constant=True))
                self._assign_to_target(elt, val_ref, target)
        elif hasattr(rvalue, 'type_hint') and hasattr(rvalue.type_hint, '_field_types'):
            # Struct unpacking: a, b = func() where func() returns struct
            struct_type = rvalue.type_hint
            field_types = struct_type._field_types
            
            if len(target.elts) != len(field_types):
                logger.error(f"Unpacking mismatch: {len(target.elts)} variables, {len(field_types)} fields",
                            node=target, exc_type=TypeError)
            
            from ..valueref import wrap_value
            from ..builtin_entities.python_type import PythonType
            
            for i, elt in enumerate(target.elts):
                field_pc_type = field_types[i]
                
                # Special handling for pyconst fields (zero-sized, value is in type)
                if isinstance(field_pc_type, PythonType) and field_pc_type.is_constant():
                    # pyconst field: value is stored in the type itself, no LLVM extraction
                    const_value = field_pc_type.get_constant_value()
                    field_val_ref = wrap_value(const_value, kind='python', type_hint=field_pc_type)
                else:
                    # Regular field: extract from LLVM struct
                    # Use _get_llvm_field_index to handle pyconst fields (zero-sized)
                    llvm_index = struct_type._get_llvm_field_index(i)
                    if llvm_index == -1:
                        # This shouldn't happen since we already handled pyconst above
                        logger.error(f"Zero-sized field [{i}] has no LLVM representation", node=node, exc_type=RuntimeError)
                    field_value = self.builder.extract_value(ensure_ir(rvalue), llvm_index)
                    field_val_ref = wrap_value(field_value, type_hint=field_pc_type, node=node)
                
                self._assign_to_target(elt, field_val_ref, target, pc_type=field_pc_type)
        else:
            logger.error(f"Unsupported unpacking type: {rvalue.type_hint}.", node=value_node, exc_type=TypeError)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Handle annotated assignment statements (variable declarations with types)
        
        Now uses the new context system to track PC types alongside LLVM types.
        Supports static local variables (converted to global variables with internal linkage).
        """
        if not isinstance(node.target, ast.Name):
            logger.error("AnnAssign only supports simple names", node=node, exc_type=RuntimeError)
        var_name = node.target.id
        
        # Check if variable already exists in CURRENT scope - AnnAssign is declaration, not reassignment
        # Allow shadowing variables from outer scopes (C-like behavior)
        if self.ctx.var_registry.is_declared_in_current_scope(var_name):
            existing = self.ctx.var_registry.lookup(var_name)
            logger.error(
                f"Cannot redeclare variable '{var_name}': already declared in this scope at line {existing.line_number} "
                f"(attempting redeclaration at line {node.lineno})",
                node=node, exc_type=RuntimeError
            )
        
        # Get PC type from annotation
        is_static_var = False
        if not hasattr(node, 'annotation'):
            raise RuntimeError("AnnAssign requires annotation")

        pc_type = self.get_pc_type_from_annotation(node.annotation)
        if pc_type is None:
            import ast as ast_module
            annotation_str = ast_module.unparse(node.annotation) if hasattr(ast_module, 'unparse') else str(node.annotation)
            logger.error(
                f"AnnAssign requires valid PC type annotation. annotation: {annotation_str}", node)

        # Now parse the RHS
        if node is None or node.value is None:
            # No initialization value - create undefined value (matches C behavior)
            llvm_type = pc_type.get_llvm_type(self.module.context)
            undef_value = ir.Constant(llvm_type, ir.Undefined)
            rvalue = wrap_value(undef_value, kind="value", type_hint=pc_type)
        else:
            rvalue = self.visit_expression(node.value)

            # If the type of RHS does not match pc_type, convert it
            if rvalue.type_hint != pc_type:
                rvalue = self.type_converter.convert(rvalue, pc_type)
            
        # Store the value
        self._store_to_new_lvalue(node, var_name, pc_type, rvalue)
        
        # Handle linear token registration for the new variable
        if self._is_linear_type(pc_type):
            # Check if rvalue is undefined
            from llvmlite import ir as llvm_ir
            is_undefined = (
                rvalue.kind == 'value' and 
                isinstance(rvalue.value, llvm_ir.Constant) and
                rvalue.value.constant == llvm_ir.Undefined
            )
            
            if hasattr(rvalue, 'var_name') and rvalue.var_name:
                # Variable reference - transfer ownership
                self._register_linear_token(var_name, pc_type, node, path=())
                self._transfer_linear_ownership(rvalue, reason="assignment")
            elif not is_undefined:
                # Initialized value (function return, linear(), etc.)
                self._register_linear_token(var_name, pc_type, node, path=())
    

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
        result = self._perform_binary_operation(fake_binop.op, current_val_ref, rhs_value, node)
        
        # Store the result back
        self.builder.store(ensure_ir(result), ensure_ir(target_addr))
