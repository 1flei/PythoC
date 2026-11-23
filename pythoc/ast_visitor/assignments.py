"""
Assignments mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..logger import logger
from ..utils.ast_targets import is_rebinding_target, unpack_target_items
from ..ir_helpers import initialize_storage, safe_store, safe_load, is_const, is_volatile, is_static
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


def admit_uninitialized_local(node, pc_type):
    """Return whether a bare local declaration should allocate storage."""
    from ..type_descriptor import contains_refinement_storage
    if not contains_refinement_storage(pc_type):
        return True
    slot = getattr(node, '_inline_slot', None)
    if slot == 'yield_target':
        # Only the yield assignment can introduce a proof-bearing loop binding.
        return False
    if slot == 'return':
        # The inline return checker requires a value on every reachable exit.
        return True
    logger.error(f"Local '{node.target.id}' contains a refined value and requires an initializer",
                 node=node, exc_type=TypeError)


class AssignmentsMixin:
    """Mixin containing assignments-related visitor methods"""
    
    def visit_lvalue(self, node: ast.AST) -> ValueRef:
        """Resolve an expression that denotes writable storage."""
        if unpack_target_items(node) is not None:
            logger.error("Tuple unpacking should be handled by caller", node=node, exc_type=ValueError)

        return self.visit_expression(node).as_lvalue(node)

    def _check_linear_rvalue_copy(self, rvalue: ValueRef, node) -> None:
        """Check if rvalue is a linear token or embeds one in a carrier."""
        from ..literal_protocol import iter_value_ref_leaves

        logger.debug(
            f"_check_linear_rvalue_copy: rvalue={rvalue}, "
            f"var_name={getattr(rvalue, 'var_name', None)}, "
            f"linear_path={getattr(rvalue, 'linear_path', None)}"
        )

        for candidate in iter_value_ref_leaves(rvalue):
            candidate_var_name = getattr(candidate, 'var_name', None)
            candidate_linear_path = getattr(candidate, 'linear_path', None)
            if not candidate_var_name or candidate_linear_path is None:
                continue

            if not self._is_linear_type(candidate.type_hint):
                continue

            owner_id = candidate.linear_owner_id
            if owner_id is None:
                owner_id = self.lookup_variable(candidate_var_name, node).var_id
            self._get_cf_builder().record_linear_copy(
                owner_id, candidate_var_name, candidate_linear_path, node)

    def visit_lvalue_or_define(self, node: ast.AST, value_ref: ValueRef,
                               pc_type=None) -> tuple[ValueRef, bool]:
        """Return the target and whether this assignment initializes a new local."""
        if not isinstance(node, ast.Name):
            return self.visit_lvalue(node), False

        rebind = is_rebinding_target(node)
        binding = None if rebind else self.lookup_variable(node.id, node)
        # A surrounding Python name can be shadowed. A compiled global with
        # storage has a place of its own and is assigned through that place.
        if binding is not None and (not binding.is_global or binding.value_ref.has_place()):
            return self._bind_name_reference(binding, node.id).as_lvalue(node), False

        if pc_type is None:
            pc_type = value_ref.get_pc_type()
        llvm_type = pc_type.get_llvm_type(self.module.context)
        alloca = self._create_alloca_in_entry(llvm_type, f"{node.id}_addr")
        binding = self.declare_variable(
            name=node.id,
            type_hint=pc_type,
            node=node,
            alloca=alloca,
            value_ref=value_ref,
            source="inference",
            line_number=node.lineno,
            allow_redeclare=rebind,
        )
        return self._bind_name_reference(binding, node.id), True

    def _assignment_ir_value(self, rvalue: ValueRef, target_type: ir.Type):
        """Read pointer-backed array values when storing an array object."""
        value = ensure_ir(rvalue)
        if (isinstance(target_type, ir.ArrayType)
                and isinstance(value.type, ir.PointerType)
                and value.type.pointee == target_type):
            return safe_load(self.builder, value, rvalue.get_pc_type())
        return value

    def _store_to_lvalue(self, lvalue: ValueRef, rvalue: ValueRef, node: ast.AST,
                         *, initializing=False):
        """Store a value with type conversion and qualifier checks."""
        target_pc_type = lvalue.get_pc_type()
        rvalue = self.implicit_coercer.coerce(rvalue, target_pc_type, node)
        target_type = lvalue.get_storage_type()
        value_ir = self._assignment_ir_value(rvalue, target_type)
        
        if initializing:
            initialize_storage(self.builder, value_ir, lvalue.require_place(), target_pc_type)
        else:
            lvalue.store(self.builder, value_ir, node=node)

    def _assign_to_target(self, target: ast.AST, rvalue: ValueRef, node,
                          value_node, pc_type=None) -> None:
        """Unified single-target assignment: lvalue resolution, linear checks, store, and linear registration.
        
        Args:
            target: AST node for assignment target (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple)
            rvalue: Value to assign
            node: AST node for error reporting
            value_node: Original right-hand expression
            pc_type: Explicit PC type (optional, overrides inference)
        """
        # Handle nested tuple unpacking recursively
        items = unpack_target_items(target)
        if items is not None:
            self._handle_tuple_unpacking(target, items, rvalue, value_node)
            return
        
        # An unpacked field supplies its own type, including an array field.
        assigned_value = (self.value_dispatcher.prepare_assignment_rvalue(rvalue, node)
                          if pc_type is None else rvalue)
        lvalue, initializing = self.visit_lvalue_or_define(target, assigned_value, pc_type)
        self._store_to_lvalue(lvalue, assigned_value, node, initializing=initializing)

        # Every successful owned store hands responsibility to its destination,
        # including raw pointer storage outside the local CFG binding set.
        if self._is_linear_type(lvalue.type_hint) and not rvalue.is_undefined_pcvalue():
            owner_id = lvalue.linear_owner_id
            if owner_id is None and lvalue.var_name:
                owner_id = self.lookup_variable(lvalue.var_name, target).var_id
            if owner_id is not None:
                name = lvalue.var_name or 'temporary value'
                for path in self._get_linear_paths(lvalue.type_hint, lvalue.linear_path or ()):
                    self._get_cf_builder().record_linear_transition(
                        owner_id, name, path, 'invalid', 'valid', node)
            self._transfer_linear_ownership(rvalue, reason='assignment', node=node)

    def _store_to_new_lvalue(self, node, var_name, pc_type, rvalue: ValueRef):
        """Create new lvalue for assignment
        
        For static variables, creates a global variable with internal linkage
        and compile-time constant initialization (like C static local variables).
        For regular variables, creates a stack allocation (alloca).
        """
        llvm_type = pc_type.get_llvm_type(self.module.context)
        rvalue_ir = ensure_ir(rvalue)
        
        # Check if this is a static or thread-local variable.
        from ..ir_helpers import is_thread_local
        is_static_storage = is_static(pc_type) or is_thread_local(pc_type)
        if is_static_storage:
            # Create unique global variable name based on function name and variable name
            func_name = self.current_function.name if self.current_function else "global"
            global_name = f"{func_name}.{var_name}"
            
            # Static variable with compile-time constant initialization (like C).
            # Valid initializers are link-time constants: folded constants and
            # global values such as function or global variable references.
            from ..ir_helpers import is_link_time_constant
            if not is_link_time_constant(rvalue_ir):
                logger.error(
                    f"Static variable '{var_name}' requires compile-time constant initializer",
                    node=node
                )
            
            global_var = ir.GlobalVariable(self.module, llvm_type, global_name)
            global_var.linkage = 'internal'  # Internal linkage = static in C
            if is_thread_local(pc_type):
                global_var.storage_class = 'thread_local'
            # llvmlite rejects a scalar zero for aggregate static initializers.
            if isinstance(rvalue_ir, ir.Constant) and isinstance(rvalue_ir.type, (ir.ArrayType, ir.BaseStructType, ir.LiteralStructType, ir.IdentifiedStructType)):
                if isinstance(rvalue_ir.constant, int) and rvalue_ir.constant == 0:
                    rvalue_ir = self.type_converter.create_zero_constant(
                        rvalue_ir.type
                    )
            global_var.initializer = rvalue_ir
            global_var.global_constant = False
            
            var_addr = global_var
        else:
            # Regular variable: create stack allocation
            var_addr = self._create_alloca_in_entry(llvm_type, f"{var_name}_addr")
            
            if node.value is not None:
                value_ir = self._assignment_ir_value(rvalue, llvm_type)
                initialize_storage(self.builder, value_ir, var_addr, pc_type)
        
        # Declare variable (for both static and non-static)
        self.declare_variable(
            name=var_name,
            type_hint=pc_type,
            node=node,
            alloca=var_addr,
            value_ref=rvalue,
            source="annotation",
            line_number=node.lineno
        )
    
    def visit_Assign(self, node: ast.Assign):
        """Handle assignment statements with automatic type inference"""
        # Evaluate rvalue once
        rvalue = self.visit_rvalue_expression(node.value)
        
        # Check if rvalue is an active linear token (forbid copy)
        self._check_linear_rvalue_copy(rvalue, node)
        
        # Handle multiple targets
        for target in node.targets:
            self._assign_to_target(target, rvalue, node, node.value)
    
    def _handle_tuple_unpacking(self, target, items, rvalue: ValueRef, value_node):
        """Handle tuple unpacking assignment through the literal protocol."""
        from ..literal_protocol import get_unpack_values

        unpack_values = get_unpack_values(self, rvalue, value_node)
        if len(unpack_values) != len(items):
            logger.error(
                f"Unpacking mismatch: {len(items)} variables, {len(unpack_values)} values",
                node=target,
                exc_type=TypeError,
            )

        for unpack_value, elt in zip(unpack_values, items):
            self._assign_to_target(elt, unpack_value, elt, value_node,
                                   pc_type=unpack_value.get_pc_type())

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
        if self.scope_manager.is_declared_in_current_scope(var_name):
            existing = self.scope_manager.lookup_variable(var_name)
            logger.error(
                f"Cannot redeclare variable '{var_name}': already declared in this scope at line {existing.line_number} "
                f"(attempting redeclaration at line {node.lineno})",
                node=node, exc_type=RuntimeError
            )
        
        # Get PC type from annotation
        if not hasattr(node, 'annotation'):
            logger.error("AnnAssign requires annotation", node=node)

        pc_type = self.get_pc_type_from_annotation(node.annotation)
        if pc_type is None:
            import ast as ast_module
            annotation_str = ast_module.unparse(node.annotation) if hasattr(ast_module, 'unparse') else str(node.annotation)
            logger.error(
                f"AnnAssign requires valid PC type annotation. annotation: {annotation_str}", node)

        from ..type_descriptor import validate_static_storage
        from ..ir_helpers import is_thread_local
        if is_static(pc_type) or is_thread_local(pc_type):
            validate_static_storage(pc_type, node, zero_initialized=node.value is None)
        elif node.value is None and not admit_uninitialized_local(node, pc_type):
            return
        ownership_source = None
        # Now parse the RHS
        if node.value is None:
            llvm_type = pc_type.get_llvm_type(self.module.context)
            from ..ir_helpers import is_thread_local
            if is_static(pc_type) or is_thread_local(pc_type):
                # Static storage duration: an absent initializer is zero, not
                # indeterminate (C semantics, and the static seed must be a
                # constant anyway). create_zero_constant covers scalars,
                # pointers, arrays, and structs.
                seed_value = self.type_converter.create_zero_constant(llvm_type)
            else:
                # Automatic storage: leave the value indeterminate (matches C).
                seed_value = ir.Constant(llvm_type, ir.Undefined)
            rvalue = wrap_value(seed_value, kind="value", type_hint=pc_type)
        else:
            rvalue = self.visit_rvalue_expression(node.value)
            ownership_source = rvalue

            # A static aggregate initialized by a braced literal needs a
            # compile-time constant seed (C constant aggregate init), not the
            # runtime element stores the normal coercion would emit.
            from ..ir_helpers import is_thread_local
            const_seed = None
            if is_static(pc_type) or is_thread_local(pc_type):
                const_seed = self.type_converter.try_const_aggregate(rvalue, pc_type, node=node)
            if const_seed is not None:
                rvalue = wrap_value(const_seed, kind="value", type_hint=pc_type)
            # Always validate the semantic relation, including qualifiers and
            # refinements whose backend representations may be identical.
            else:
                rvalue = self.implicit_coercer.coerce(rvalue, pc_type, node)
            
        if ownership_source is not None and self._is_linear_type(pc_type):
            self._transfer_linear_ownership(ownership_source, reason="assignment", node=node)
        # Store the value
        self._store_to_new_lvalue(node, var_name, pc_type, rvalue)
        
        # Handle linear token registration for the new variable
        if self._is_linear_type(pc_type):
            is_undefined = rvalue.is_undefined_pcvalue()
            
            if not is_undefined:
                self._register_linear_token(var_name, pc_type, node, path=())

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment statements through the unified dispatcher."""
        if self.builder.is_terminated():
            return

        target_addr = self.visit_lvalue(node.target)
        current_val_ref = self.value_dispatcher.read_rvalue(
            target_addr,
            name=getattr(target_addr, "var_name", None),
        )
        rhs_value = self.visit_rvalue_expression(node.value)
        result = self.value_dispatcher.handle_binary_operation(
            node.op,
            current_val_ref,
            rhs_value,
            node,
            left_node=node.target,
            right_node=node.value,
        )
        self._store_to_lvalue(target_addr, result, node)
