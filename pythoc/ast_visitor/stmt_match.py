"""
Match statement visitor mixin
"""

import ast
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type
from ..logger import logger
from .control_flow_builder import ControlFlowBuilder


class MatchStatementMixin:
    """Mixin for match/case statement handling"""
    
    def _get_cf_builder(self) -> ControlFlowBuilder:
        """Get or create the ControlFlowBuilder for this visitor"""
        if not hasattr(self, '_cf_builder') or self._cf_builder is None:
            self._cf_builder = ControlFlowBuilder(self)
        return self._cf_builder
    
    def visit_Match(self, node: ast.Match):
        """Handle match/case statements (Python 3.10+)

        Translates match/case to if-elif-else chain for complex patterns,
        or optimized switch for simple integer literal patterns.

        Currently supports:
        - Single and multiple subjects: match x, y:
        - Literal patterns (integers, strings via hash)
        - Wildcard pattern (_)
        - Or patterns (|)

        Example:
            match x:
                case 1:
                    return 10
                case 2:
                    return 20
                case _:
                    return 0
        """
        cf = self._get_cf_builder()

        # Note: We do NOT skip processing when terminated because the match body
        # may contain label definitions that need to be registered for forward
        # goto resolution.

        # Handle multiple subjects: match x, y, z:
        # In Python AST, this becomes a Tuple node
        if isinstance(node.subject, ast.Tuple):
            # Multiple subjects - evaluate each one
            subjects = [self.visit_expression(elt) for elt in node.subject.elts]
        else:
            # Single subject
            subjects = [self.visit_expression(node.subject)]

        # Get subject types for exhaustiveness checking
        subject_types = [s.type_hint for s in subjects]

        # Check exhaustiveness before code generation
        is_exhaustive = self._check_match_exhaustiveness(node, subject_types)

        # For single subject, try switch optimization
        if len(subjects) == 1:
            subject_ir = ensure_ir(subjects[0])
            can_use_switch = self._can_use_switch_for_match(node)

            if can_use_switch:
                self._visit_match_as_switch(node, subjects[0], subject_ir, is_exhaustive)
                return

        # Multiple subjects or complex patterns - use if-chain
        self._visit_match_as_if_chain(node, subjects, is_exhaustive)
    
    def _can_use_switch_for_match(self, node: ast.Match) -> bool:
        """Check if match can be compiled to LLVM switch instruction"""
        for case in node.cases:
            pattern = case.pattern

            # Any guard clause disables switch optimization
            if case.guard is not None:
                return False

            # Wildcard is OK (becomes default)
            if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
                continue
            # Literal integers are OK
            if isinstance(pattern, ast.MatchValue):
                if isinstance(pattern.value, ast.Constant):
                    if isinstance(pattern.value.value, int):
                        continue
            # MatchOr with all integer literals is OK
            if isinstance(pattern, ast.MatchOr):
                all_int_literals = all(
                    isinstance(p, ast.MatchValue) and
                    isinstance(p.value, ast.Constant) and
                    isinstance(p.value.value, int)
                    for p in pattern.patterns
                )
                if all_int_literals:
                    continue
            # Any other pattern type requires if-chain
            return False
        return True

    def _check_match_exhaustiveness(self, node: ast.Match, subject_types: list) -> bool:
        """Check if match statement covers all possible cases.

        Returns True if the match is proven exhaustive, False otherwise.
        Raises CompileError for non-exhaustive matches.
        """
        from ..match_exhaustive import check_match_exhaustiveness

        # Skip check if any subject type is None (can't check without type info)
        if any(t is None for t in subject_types):
            return False

        try:
            check_match_exhaustiveness(node, subject_types, self)
            return True
        except Exception:
            # Error already logged by check_match_exhaustiveness
            # Re-raise to stop compilation
            raise
    
    def _visit_match_as_switch(self, node: ast.Match, subject_ref, subject_ir, is_exhaustive: bool = False):
        """Compile match to LLVM switch instruction (optimized path)

        Args:
            node: ast.Match node
            subject_ref: ValueRef for the subject (for variable bindings)
            subject_ir: LLVM IR value for the subject
            is_exhaustive: If True, no implicit fallthrough edge needed

        Linear state tracking is done by the AST visitor during execution.
        CFG-based linear checking is done at function end via CFG linear checker.
        """
        cf = self._get_cf_builder()
        
        # Capture linear states before match for branch restoration
        linear_states_before = cf.capture_linear_snapshot()
        
        # Create merge block
        merge_block = cf.create_block("match_merge")
        
        # Find default case (wildcard pattern)
        default_case_idx = None
        for idx, case in enumerate(node.cases):
            if isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                default_case_idx = idx
                break
        
        has_wildcard = default_case_idx is not None

        # Create default block
        if has_wildcard:
            default_block = cf.create_block("case_default")
        elif is_exhaustive:
            # Exhaustive without wildcard - default is unreachable
            default_block = cf.create_block("match_unreachable")
        else:
            # No default case - just jump to merge
            default_block = merge_block
        
        # Create case blocks and collect switch cases
        case_blocks = []
        switch_cases = []
        
        for idx, case in enumerate(node.cases):
            if idx == default_case_idx:
                case_blocks.append(default_block)
                continue
            
            case_block = cf.create_block(f"case_{idx}")
            case_blocks.append(case_block)
            
            # Collect integer values for this case
            pattern = case.pattern
            if isinstance(pattern, ast.MatchValue):
                value = pattern.value.value
                switch_cases.append((value, case_block))
            elif isinstance(pattern, ast.MatchOr):
                for p in pattern.patterns:
                    value = p.value.value
                    switch_cases.append((value, case_block))
        
        # Build switch instruction
        switch = cf.switch(subject_ir, default_block)
        for value, block in switch_cases:
            const_val = ir.Constant(subject_ir.type, value)
            cf.add_switch_case(switch, const_val, block)
        
        # Generate code for each case
        for idx, case in enumerate(node.cases):
            cf.position_at_end(case_blocks[idx])

            # Reset linear states to before match for this case
            cf.restore_linear_snapshot(linear_states_before)

            # Enter scope for case body
            self.ctx.var_registry.enter_scope()
            try:
                # Handle variable binding for MatchAs patterns (case x: or case _:)
                pattern = case.pattern
                if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
                    if pattern.name is not None:
                        # case x: - bind the subject to variable x
                        self._bind_match_variable(pattern.name, subject_ref)

                # Execute case body
                self._visit_stmt_list(case.body, add_to_cfg=True)
            finally:
                self.ctx.var_registry.exit_scope()

            # Branch to merge if not terminated
            if not cf.is_terminated():
                cf.branch(merge_block)

        # Handle unreachable default block for exhaustive matches
        if is_exhaustive and not has_wildcard and default_block != merge_block:
            cf.position_at_end(default_block)
            # Emit LLVM unreachable instruction - this path should never be taken
            self.builder.unreachable()

        # Linear state validation is done by CFG checker at function end

        # Continue at merge block
        cf.position_at_end(merge_block)
    
    def _visit_match_as_if_chain(self, node: ast.Match, subjects, is_exhaustive: bool = False):
        """Compile match to if-chain (general path)

        Args:
            subjects: list of ValueRef - one or more subject values
            is_exhaustive: If True, no implicit fallthrough edge needed

        Uses process_condition to reuse if statement logic for each case.
        All control flow is delegated to process_condition - no direct builder calls.

        Linear state handling:
        - Capture linear state before each case
        - Restore linear state when pattern fails (else branch)
        - This ensures each case starts with the same linear state
        """
        cf = self._get_cf_builder()
        
        # Create merge block for all cases to converge
        merge_block = cf.create_block("match_merge")

        # Process each case as an if-elif branch
        for idx, case in enumerate(node.cases):
            pattern = case.pattern
            
            # Capture linear state before this case
            # Each case should start with the same linear state
            linear_states_before_case = cf.capture_linear_snapshot()
            
            # Check if this is a wildcard (default case) without guard
            if isinstance(pattern, ast.MatchAs) and pattern.pattern is None and case.guard is None:
                # Wildcard always matches - execute body directly
                self._visit_stmt_list(case.body, add_to_cfg=True)
                if not cf.is_terminated():
                    cf.branch(merge_block)
                break
            
            # Generate condition and bindings for this pattern
            # For multiple subjects, pattern must be a tuple/sequence
            if len(subjects) > 1:
                # Multiple subjects - pattern should be a sequence matching all subjects
                pattern_cond, bindings = self._generate_multi_subject_pattern(pattern, subjects)
            else:
                # Single subject - use normal pattern matching
                pattern_cond, bindings = self._generate_match_pattern(pattern, subjects[0])
            
            # pattern_cond is already a ValueRef with bool type_hint
            condition_ref = pattern_cond
            
            # Create next case block (where else branch jumps to)
            next_case_block = cf.create_block(f"case_{idx}_next")
            
            # Capture for closure - need to copy to avoid late binding issues
            captured_linear_states = linear_states_before_case
            
            if case.guard is not None:
                # Guarded case: nested process_condition
                # 1. Check pattern: if matches -> check guard, else -> next case
                # 2. Check guard: if passes -> execute body, else -> next case
                
                def pattern_then_with_guard(bindings=bindings, case=case):
                    # Bind pattern variables (needed for guard evaluation)
                    for var_name, var_value in bindings:
                        self._bind_match_variable(var_name, var_value)
                    
                    # Evaluate guard
                    guard_result = self.visit_expression(case.guard)
                    
                    # Use process_condition for guard check
                    def guard_then():
                        # Execute case body
                        self._visit_stmt_list(case.body, add_to_cfg=True)
                        # Branch to merge
                        if not cf.is_terminated():
                            cf.branch(merge_block)
                    
                    def guard_else(states=captured_linear_states):
                        # Guard failed - restore linear state and go to next case
                        cf.restore_linear_snapshot(states)
                        if not cf.is_terminated():
                            cf.branch(next_case_block)
                    
                    self.process_condition(guard_result, guard_then, guard_else)
                
                def pattern_else(states=captured_linear_states):
                    # Pattern failed - restore linear state and go to next case
                    cf.restore_linear_snapshot(states)
                    if not cf.is_terminated():
                        cf.branch(next_case_block)
                
                # Use process_condition for pattern check
                self.process_condition(condition_ref, pattern_then_with_guard, pattern_else)
                
            else:
                # No guard - simple pattern match using process_condition
                
                def pattern_then(bindings=bindings, case=case):
                    # Bind pattern variables
                    for var_name, var_value in bindings:
                        self._bind_match_variable(var_name, var_value)
                    # Execute case body
                    self._visit_stmt_list(case.body, add_to_cfg=True)
                    # Branch to merge
                    if not cf.is_terminated():
                        cf.branch(merge_block)
                
                def pattern_else(states=captured_linear_states):
                    # Pattern failed - restore linear state and go to next case
                    cf.restore_linear_snapshot(states)
                    if not cf.is_terminated():
                        cf.branch(next_case_block)
                
                # Use process_condition for pattern check
                self.process_condition(condition_ref, pattern_then, pattern_else)
            
            # Position at next case block for the next iteration
            cf.position_at_end(next_case_block)

        # If we reach here, no case matched (and no wildcard)
        if not cf.is_terminated():
            if is_exhaustive:
                # Exhaustive - this path is unreachable
                self.builder.unreachable()
            else:
                cf.branch(merge_block)

        # Continue at merge block
        cf.position_at_end(merge_block)
    
    def _generate_match_pattern(self, pattern, subject):
        """Generate condition and bindings for a match pattern

        Returns:
            tuple: (condition_ir, bindings)
                condition_ir: LLVM IR value (i1) for pattern match condition
                bindings: list of (var_name, var_value) tuples for variable bindings
        """
        if isinstance(pattern, ast.MatchValue):
            # Literal value pattern: compare subject with literal
            value = self.visit_expression(pattern.value)
            
            # Special case: if subject is enum and value is a tag constant,
            # only compare the tag field (subject[0]) instead of the whole enum
            subject_type = subject.type_hint
            is_enum = hasattr(subject_type, '_is_enum') and subject_type._is_enum
            
            if is_enum and isinstance(value, ValueRef) and value.kind == "python":
                # Comparing enum with a tag constant - only compare tag field
                tag_field = self._subscript_access(subject, 0)
                condition = self._compare_values(tag_field, value, ast.Eq())
            else:
                # Normal comparison
                condition = self._compare_values(subject, value, ast.Eq())
            
            return condition, []
        
        elif isinstance(pattern, ast.MatchAs):
            # Binding pattern: case x: or case _:
            if pattern.pattern is None:
                # Wildcard: always matches
                from ..valueref import wrap_value
                from ..builtin_entities import bool as pc_bool
                true_val = wrap_value(ir.Constant(ir.IntType(1), 1), kind="value", type_hint=pc_bool)
                if pattern.name is not None:
                    # case _ as x: (bind wildcard)
                    return true_val, [(pattern.name, subject)]
                else:
                    # case _: (no binding)
                    return true_val, []
            else:
                # case <pattern> as name: (match pattern and bind)
                condition, bindings = self._generate_match_pattern(pattern.pattern, subject)
                if pattern.name is not None:
                    bindings.append((pattern.name, subject))
                return condition, bindings
        
        elif isinstance(pattern, ast.MatchOr):
            # OR pattern: match any of the alternatives
            # Note: OR patterns cannot have bindings in Python
            from ..valueref import wrap_value
            from ..builtin_entities import bool as pc_bool
            
            conditions = []
            for p in pattern.patterns:
                cond, _ = self._generate_match_pattern(p, subject)
                conditions.append(ensure_ir(cond))
            
            # Combine with OR
            result_ir = conditions[0]
            for cond_ir in conditions[1:]:
                result_ir = self.builder.or_(result_ir, cond_ir)
            return wrap_value(result_ir, kind="value", type_hint=pc_bool), []
        
        elif isinstance(pattern, ast.MatchClass):
            # Struct destructuring pattern
            return self._generate_struct_pattern(pattern, subject)
        
        elif isinstance(pattern, ast.MatchSequence):
            # Array/sequence pattern
            return self._generate_sequence_pattern(pattern, subject)
        
        else:
            logger.error(
                f"Match pattern type {type(pattern).__name__} not yet supported",
                node=pattern if hasattr(pattern, 'lineno') else None, exc_type=NotImplementedError
            )
    
    def _generate_multi_subject_pattern(self, pattern, subjects):
        """Generate condition and bindings for multiple subjects
        
        When matching multiple subjects: match x, y, z:
        The pattern must be a tuple/sequence: case (pat_x, pat_y, pat_z):
        
        Args:
            pattern: AST pattern node
            subjects: list of ValueRef - the subject values
        
        Returns:
            tuple: (condition_ir, bindings)
        """
        # Pattern must be a sequence (tuple) for multiple subjects
        if not isinstance(pattern, ast.MatchSequence):
            logger.error(
                f"Multiple subjects require sequence pattern, got {type(pattern).__name__}",
                node=pattern if hasattr(pattern, 'lineno') else None, exc_type=TypeError
            )
        
        # Number of patterns must match number of subjects
        if len(pattern.patterns) != len(subjects):
            logger.error(
                f"Pattern count {len(pattern.patterns)} != subject count {len(subjects)}",
                node=pattern if hasattr(pattern, 'lineno') else None, exc_type=ValueError
            )
        
        # Match each subject against corresponding pattern
        conditions = []
        bindings = []
        
        for sub_pattern, subject in zip(pattern.patterns, subjects):
            cond, sub_bindings = self._generate_match_pattern(sub_pattern, subject)
            if cond is not None:
                conditions.append(ensure_ir(cond))
            bindings.extend(sub_bindings)
        
        # Combine all conditions with AND
        from ..valueref import wrap_value
        from ..builtin_entities import bool as pc_bool
        
        if conditions:
            result_ir = conditions[0]
            for cond_ir in conditions[1:]:
                result_ir = self.builder.and_(result_ir, cond_ir)
        else:
            # No conditions - always matches
            result_ir = ir.Constant(ir.IntType(1), 1)
        
        return wrap_value(result_ir, kind="value", type_hint=pc_bool), bindings
    
    def _generate_struct_pattern(self, pattern, subject):
        """Generate condition and bindings for struct pattern
        
        Example: case Point(x=0, y=y):
            - Check if subject.x == 0
            - Bind y to subject.y
        """
        from ..registry import get_unified_registry
        
        # Get struct type from subject
        if not isinstance(subject, ValueRef):
            logger.error(f"Expected ValueRef for struct pattern, got {type(subject)}",
                        node=pattern.cls if hasattr(pattern, 'cls') else None, exc_type=TypeError)
        
        # Extract struct class name
        if isinstance(pattern.cls, ast.Name):
            struct_name = pattern.cls.id
        else:
            logger.error(f"Complex struct class patterns not supported: {pattern.cls}",
                        node=pattern.cls, exc_type=NotImplementedError)
        
        # Verify subject type matches pattern type  
        registry = get_unified_registry()
        struct_info = registry.get_struct(struct_name)
        if struct_info is None:
            logger.error(f"Unknown struct type: {struct_name}",
                        node=pattern.cls, exc_type=TypeError)
        
        # Build conditions and bindings from keyword patterns
        conditions = []
        bindings = []
        
        for field_name, field_pattern in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            # Access struct field using type's handle_attribute
            subject_type = subject.type_hint
            if subject_type and hasattr(subject_type, 'handle_attribute'):
                field_value = subject_type.handle_attribute(self, subject, field_name, None)
            else:
                logger.error(f"Struct type does not support attribute access: {subject_type}",
                            node=None, exc_type=TypeError)
            
            # Recursively match field pattern
            field_cond, field_bindings = self._generate_match_pattern(field_pattern, field_value)
            
            if field_cond is not None:
                conditions.append(ensure_ir(field_cond))
            bindings.extend(field_bindings)
        
        # Combine all conditions with AND
        from ..valueref import wrap_value
        from ..builtin_entities import bool as pc_bool
        
        if conditions:
            result_ir = conditions[0]
            for cond_ir in conditions[1:]:
                result_ir = self.builder.and_(result_ir, cond_ir)
        else:
            # No conditions - always matches
            result_ir = ir.Constant(ir.IntType(1), 1)
        
        return wrap_value(result_ir, kind="value", type_hint=pc_bool), bindings
    
    def _generate_sequence_pattern(self, pattern, subject):
        """Generate condition and bindings for sequence pattern
        
        UNIFIED SEMANTICS: All sequence patterns use subscript access
        - case (a, b)        => subject[0] matches a, subject[1] matches b
        - case (a, (b, c))   => subject[0] matches a, subject[1][0] matches b, subject[1][1] matches c
        
        This works uniformly for:
        - Arrays: subject[i] => array element access
        - Structs: subject[i] => i-th field (in definition order)
        - Enums: subject[0] => tag, subject[1] => payload
        - Tuples: subject[i] => i-th element
        
        SPECIAL CASE FOR ENUM MATCHING:
        When pattern is (EnumClass.Variant, x) and subject is an enum:
        - pattern[0] is a Python value (the variant tag constant)
        - Check if subject[0] (tag) == EnumClass.Variant
        - Bind x to the specific variant's payload type (not the full union)
        """
        if not isinstance(subject, ValueRef):
            logger.error(f"Expected ValueRef for sequence pattern, got {type(subject)}",
                        node=None, exc_type=TypeError)
        
        # Check if this is enum pattern matching: (EnumClass.Variant, x)
        subject_type = subject.type_hint
        is_enum = hasattr(subject_type, '_is_enum') and subject_type._is_enum
        
        if is_enum and len(pattern.patterns) == 2:
            # Check if first pattern is a Python value (constant tag)
            first_pattern = pattern.patterns[0]
            second_pattern = pattern.patterns[1]
            
            # Try to detect if first_pattern is a constant (like Status.Ok)
            # For enum matching, the first pattern should be MatchValue with a constant
            is_constant_tag = isinstance(first_pattern, ast.MatchValue)
            
            if is_constant_tag:
                # Enum pattern: (EnumClass.Variant, payload_var)
                # 1. Check tag matches
                tag_value = self._subscript_access(subject, 0)
                tag_cond, _ = self._generate_match_pattern(first_pattern, tag_value)
                
                # 2. Get the variant info from the constant tag value
                # Evaluate the tag constant to get its value
                if isinstance(first_pattern, ast.MatchValue):
                    tag_const_node = first_pattern.value
                else:
                    tag_const_node = first_pattern
                
                tag_const_val = self.visit_expression(tag_const_node)
                
                # Try to find which variant this tag corresponds to
                variant_idx = None
                variant_name = None
                
                # Extract the integer tag value
                tag_int_val = None
                if isinstance(tag_const_val, ValueRef):
                    if tag_const_val.kind == "python":
                        # Python value - directly get the integer
                        tag_int_val = tag_const_val.value
                    elif isinstance(tag_const_val.value, ir.Constant):
                        # LLVM constant
                        tag_int_val = tag_const_val.value.constant
                
                if tag_int_val is not None:
                    if hasattr(subject_type, '_tag_values') and hasattr(subject_type, '_variant_names'):
                        tag_values = subject_type._tag_values
                        variant_names = subject_type._variant_names
                        
                        # Find the variant name matching this tag value
                        if isinstance(tag_values, dict):
                            for vname, vtag in tag_values.items():
                                if vtag == tag_int_val:
                                    variant_name = vname
                                    variant_idx = variant_names.index(vname)
                                    break
                
                # 3. Handle payload pattern (can be binding or literal match)
                payload_bindings = []
                payload_cond = None
                
                # Get the full payload (union)
                payload_union = self._subscript_access(subject, 1)
                
                # Extract the specific variant's payload if we know the variant
                if variant_idx is not None and hasattr(subject_type, '_union_payload'):
                    # Use subscript to access the specific variant field: union[variant_idx]
                    variant_payload = self._subscript_access(payload_union, variant_idx)
                else:
                    # Fallback: use the whole union
                    variant_payload = payload_union
                
                # Now handle the second pattern against the variant payload
                if isinstance(second_pattern, ast.MatchAs):
                    # Variable binding: case (Status.Ok, code):
                    if second_pattern.name is not None:
                        payload_bindings.append((second_pattern.name, variant_payload))
                elif isinstance(second_pattern, ast.MatchValue):
                    # Literal match: case (Status.Ok, 0):
                    # Recursively generate condition for the payload literal
                    payload_cond, _ = self._generate_match_pattern(second_pattern, variant_payload)
                else:
                    # Other pattern types (wildcard, etc.)
                    # Recursively handle them
                    payload_cond, payload_sub_bindings = self._generate_match_pattern(second_pattern, variant_payload)
                    payload_bindings.extend(payload_sub_bindings)
                
                # Combine tag condition with payload condition
                if payload_cond is not None:
                    tag_cond_ir = ensure_ir(tag_cond)
                    payload_cond_ir = ensure_ir(payload_cond)
                    final_cond_ir = self.builder.and_(tag_cond_ir, payload_cond_ir)
                    from ..valueref import wrap_value
                    from ..builtin_entities import bool as pc_bool
                    final_cond = wrap_value(final_cond_ir, kind="value", type_hint=pc_bool)
                else:
                    final_cond = tag_cond
                
                return final_cond, payload_bindings
        
        # Standard sequence pattern matching (non-enum or non-constant tag)
        # Build conditions and bindings by matching each element via subscript
        conditions = []
        bindings = []
        
        for idx, elem_pattern in enumerate(pattern.patterns):
            # Unified subscript access: subject[idx]
            elem_value = self._subscript_access(subject, idx)
            
            # Recursively match the element pattern
            elem_cond, elem_bindings = self._generate_match_pattern(elem_pattern, elem_value)
            
            if elem_cond is not None:
                conditions.append(ensure_ir(elem_cond))
            bindings.extend(elem_bindings)
        
        # Combine all conditions with AND
        from ..valueref import wrap_value
        from ..builtin_entities import bool as pc_bool
        
        if conditions:
            result_ir = conditions[0]
            for cond_ir in conditions[1:]:
                result_ir = self.builder.and_(result_ir, cond_ir)
        else:
            # No conditions - always matches
            result_ir = ir.Constant(ir.IntType(1), 1)
        
        return wrap_value(result_ir, kind="value", type_hint=pc_bool), bindings
    
    def _subscript_access(self, subject, index):
        """Unified subscript access for match patterns: subject[index]
        
        This method provides uniform indexing semantics for all types:
        - Arrays: GEP + load element
        - Structs: extractvalue i-th field (in definition order)
        - Enums: extractvalue (0=tag, 1=payload)
        - Tuples: extractvalue i-th element
        
        The type's handle_subscript method handles the details.
        """
        from llvmlite import ir as llvm_ir
        from ..builtin_entities import i32
        
        if not isinstance(subject, ValueRef):
            logger.error(f"Expected ValueRef for subscript access, got {type(subject)}",
                        node=None, exc_type=TypeError)
        
        # Create constant index
        index_ir = llvm_ir.Constant(llvm_ir.IntType(32), index)
        index_const = wrap_value(index_ir, kind="value", type_hint=i32)
        
        # Create a fake AST node for subscript (needed by handle_subscript)
        fake_node = ast.Subscript(
            value=ast.Name(id='_subject', ctx=ast.Load()),
            slice=ast.Constant(value=index),
            ctx=ast.Load()
        )
        
        # Delegate to type's handle_subscript
        subject_type = subject.type_hint
        
        # For arrays, need to decay to pointer first
        if hasattr(subject_type, 'get_decay_pointer_type'):
            ptr_type = subject_type.get_decay_pointer_type()
            ptr_subject = self.type_converter.convert(subject, ptr_type)
            result = ptr_type.handle_subscript(self, ptr_subject, index_const, fake_node)
            return result
        
        # For other types (struct, enum, tuple), use handle_subscript directly
        if hasattr(subject_type, 'handle_subscript'):
            result = subject_type.handle_subscript(self, subject, index_const, fake_node)
            return result
        
        logger.error(f"Type {subject_type} does not support subscript access",
                    node=None, exc_type=TypeError)
    
    def _get_array_element(self, array_value, index):
        """Legacy method - now uses _subscript_access"""
        return self._subscript_access(array_value, index)
    
    def _generate_match_condition(self, pattern, subject):
        """Legacy method - now uses _generate_match_pattern"""
        condition, _ = self._generate_match_pattern(pattern, subject)
        return condition
    
    def _compare_values(self, left, right, op):
        """Helper to compare two values (used by match)
        
        Returns:
            ValueRef with bool type_hint
        """
        from ..valueref import wrap_value
        from ..builtin_entities import bool as pc_bool
        
        # Type unification - returns ValueRef objects
        left_unified, right_unified, is_float = self.type_converter.unify_binop_types(
            left, right
        )
        
        # Extract LLVM IR values
        left_ir = ensure_ir(left_unified)
        right_ir = ensure_ir(right_unified)
        
        # Generate comparison
        if isinstance(op, ast.Eq):
            if is_float:
                cmp_ir = self.builder.fcmp_ordered('==', left_ir, right_ir)
            else:
                cmp_ir = self.builder.icmp_signed('==', left_ir, right_ir)
            return wrap_value(cmp_ir, kind="value", type_hint=pc_bool)
        else:
            logger.error(f"Comparison operator {type(op).__name__}",
                        node=None, exc_type=NotImplementedError)
    
    def _bind_match_variable(self, var_name, var_value):
        """Bind a variable from match pattern
        
        Creates a local variable and stores the pattern-matched value.
        Similar to variable declaration but without type annotation.
        """
        if not isinstance(var_value, ValueRef):
            logger.error(f"Expected ValueRef for match binding, got {type(var_value)}",
                        node=None, exc_type=TypeError)
        
        # Get PC type from the value
        pc_type = var_value.type_hint
        if pc_type is None:
            logger.error(f"Cannot bind variable '{var_name}' - value has no type",
                        node=None, exc_type=TypeError)
        
        # Create alloca for the variable
        llvm_type = pc_type.get_llvm_type(self.module.context)
        alloca = self._create_alloca_in_entry(llvm_type, f"{var_name}_addr")
        
        # Declare variable in symbol table
        self.declare_variable(
            name=var_name,
            type_hint=pc_type,
            alloca=alloca,
            source="match_pattern",
            line_number=None
        )
        
        # Store the matched value
        var_value_ir = ensure_ir(var_value)
        self.builder.store(var_value_ir, alloca)
