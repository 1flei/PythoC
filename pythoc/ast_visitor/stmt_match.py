"""
Match statement visitor mixin
"""

import ast
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value
from ..logger import logger
from ..scope_manager import ScopeType
from ..match_exhaustive import PatternKind, PatternNormalizer


class MatchStatementMixin:
    """Mixin for match/case statement handling"""

    def visit_Match(self, node: ast.Match):
        """Handle match/case statements (Python 3.10+)

        Generate ordered comparisons for every pattern. LLVM chooses
        switch and jump-table lowering when the comparisons permit it.

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
            subjects = [self.visit_rvalue_expression(elt) for elt in node.subject.elts]
        else:
            # Single subject
            subjects = [self.visit_rvalue_expression(node.subject)]

        self._visit_match_as_if_chain(node, subjects)

    def _check_match_exhaustiveness(self, node, subject_types, patterns) -> bool:
        """Check if match statement covers all possible cases.

        Returns True if the match is proven exhaustive, False otherwise.
        Raises CompileError for non-exhaustive matches.
        """
        from ..match_exhaustive import check_match_exhaustiveness

        # Skip check if any subject type is None (can't check without type info)
        if any(t is None for t in subject_types):
            return False

        check_match_exhaustiveness(node, subject_types, patterns)
        return True
    
    def _visit_match_as_if_chain(self, node, subjects):
        """Compile match to if-chain (general path)

        Args:
            subjects: list of ValueRef - one or more subject values

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

        subject_types = [subject.type_hint for subject in subjects]
        patterns = []
        for idx, case in enumerate(node.cases):
            pattern = PatternNormalizer(self).normalize_case(case.pattern, subject_types)
            patterns.append(pattern)

            if pattern.is_wildcard() and not pattern.pointer_capture and case.guard is None:
                # Wildcard always matches - execute body directly
                with self.scope_manager.scope(ScopeType.MATCH, cf, node=case):
                    # Bind variable if present (case x:)
                    for name in pattern.bindings:
                        self._bind_match_variable(name, subjects[0])
                    # Consume linear subjects before executing body
                    self._consume_linear_subjects(subjects, case)
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

            if case.guard is not None:
                # Guarded case: nested process_condition
                # 1. Check pattern: if matches -> check guard, else -> next case
                # 2. Check guard: if passes -> execute body, else -> next case

                def pattern_then_with_guard(bindings=bindings, case=case):
                    # Enter MATCH scope for bindings, guard, and body
                    with self.scope_manager.scope(ScopeType.MATCH, cf, node=case):
                        # Bind pattern variables (needed for guard evaluation)
                        for var_name, var_value in bindings:
                            self._bind_match_variable(var_name, var_value)

                        # Evaluate guard
                        guard_result = self.visit_rvalue_expression(case.guard)

                        # Use process_condition for guard check
                        def guard_then():
                            # Guard passed: consume linear subjects before body
                            self._consume_linear_subjects(subjects, case)
                            # Execute case body
                            self._visit_stmt_list(case.body, add_to_cfg=True)
                            # Branch to merge
                            if not cf.is_terminated():
                                cf.branch(merge_block)

                        def guard_else():
                            # Guard failed - go to next case
                            if not cf.is_terminated():
                                cf.branch(next_case_block)

                        self.process_condition(guard_result, guard_then, guard_else)
                
                def pattern_else():
                    # Pattern failed - go to next case
                    if not cf.is_terminated():
                        cf.branch(next_case_block)

                # Use process_condition for pattern check
                self.process_condition(condition_ref, pattern_then_with_guard, pattern_else)

            else:
                # No guard - simple pattern match using process_condition
                
                def pattern_then(bindings=bindings, case=case):
                    # Enter MATCH scope for bindings and body
                    with self.scope_manager.scope(ScopeType.MATCH, cf, node=case):
                        # Bind pattern variables
                        for var_name, var_value in bindings:
                            self._bind_match_variable(var_name, var_value)
                        # Consume linear subjects before executing body
                        self._consume_linear_subjects(subjects, case)
                        # Execute case body
                        self._visit_stmt_list(case.body, add_to_cfg=True)
                    # Branch to merge
                    if not cf.is_terminated():
                        cf.branch(merge_block)
                
                def pattern_else():
                    # Pattern failed - go to next case
                    if not cf.is_terminated():
                        cf.branch(next_case_block)
                
                # Use process_condition for pattern check
                self.process_condition(condition_ref, pattern_then, pattern_else)
            
            # Position at next case block for the next iteration
            cf.position_at_end(next_case_block)

        is_exhaustive = self._check_match_exhaustiveness(node, subject_types, patterns)

        # If we reach here, no case matched (and no wildcard)
        if not cf.is_terminated():
            if is_exhaustive:
                # Exhaustive - this path is unreachable
                self.builder.unreachable()
            else:
                cf.branch(merge_block)

        # Continue at merge block
        cf.position_at_end(merge_block)
    
    def _match_true(self):
        from ..builtin_entities import bool as pc_bool
        return wrap_value(ir.Constant(ir.IntType(1), 1), kind='value', type_hint=pc_bool)

    def _combine_match_conditions(self, conditions, operation='and_'):
        from ..builtin_entities import bool as pc_bool
        if not conditions:
            return self._match_true()
        value = ensure_ir(conditions[0])
        for condition in conditions[1:]:
            value = getattr(self.builder, operation)(value, ensure_ir(condition))
        return wrap_value(value, kind='value', type_hint=pc_bool)

    def _generate_match_pattern(self, pattern, subject):
        bindings = []
        if pattern.pointer_capture:
            from ..builtin_entities import ptr
            if not subject.has_place():
                logger.error('ptr(...) pattern requires an addressable matched value',
                             node=pattern.ast_node, exc_type=TypeError)
            captured = wrap_value(subject.require_place(), kind='value',
                                  type_hint=ptr[subject.type_hint])
            condition = self._match_true()
        else:
            captured = subject
            if pattern.kind == PatternKind.WILDCARD:
                condition = self._match_true()
            elif pattern.kind == PatternKind.LITERAL:
                compared = (self._subscript_access(subject, 0)
                            if getattr(subject.type_hint, '_is_enum', False) else subject)
                condition = self._compare_values(compared, pattern.value_ref, ast.Eq())
            elif pattern.kind == PatternKind.OR:
                conditions = []
                for alternative in pattern.alternatives:
                    value, alternative_bindings = self._generate_match_pattern(
                        alternative, subject)
                    if alternative_bindings:
                        raise TypeError('Captures in OR patterns are unsupported')
                    conditions.append(value)
                condition = self._combine_match_conditions(conditions, 'or_')
            elif pattern.kind == PatternKind.CONSTRUCTOR:
                conditions = []
                if pattern.payload_index is not None:
                    tag = self._subscript_access(subject, 0)
                    conditions.append(self._compare_values(tag, pattern.value_ref, ast.Eq()))
                    if pattern.field_indices:
                        payload = self._subscript_access(subject, 1)
                        payload = self._subscript_access(payload, pattern.payload_index)
                        value, values = self._generate_match_pattern(
                            pattern.sub_patterns[0], payload)
                        conditions.append(value)
                        bindings.extend(values)
                else:
                    for index in pattern.field_indices:
                        item = self._subscript_access(subject, index)
                        value, values = self._generate_match_pattern(
                            pattern.sub_patterns[index], item)
                        conditions.append(value)
                        bindings.extend(values)
                condition = self._combine_match_conditions(conditions)
            else:
                raise NotImplementedError(f'Unknown normalized pattern: {pattern.kind}')
        bindings.extend((name, captured) for name in pattern.bindings)
        return condition, bindings

    def _generate_multi_subject_pattern(self, pattern, subjects):
        if pattern.kind == PatternKind.OR:
            conditions = []
            for alternative in pattern.alternatives:
                value, bindings = self._generate_multi_subject_pattern(alternative, subjects)
                if bindings:
                    raise TypeError('Captures in OR patterns are unsupported')
                conditions.append(value)
            return self._combine_match_conditions(conditions, 'or_'), []
        conditions, bindings = [], []
        for item, subject in zip(pattern.sub_patterns, subjects):
            value, values = self._generate_match_pattern(item, subject)
            conditions.append(value)
            bindings.extend(values)
        return self._combine_match_conditions(conditions), bindings

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
        
        # Delegate through dispatcher so match patterns share the same protocol
        # lookup path as regular expression evaluation.
        subject_type = subject.type_hint

        # For arrays, keep the explicit decay step, then use the same dispatcher
        # path as normal pointer subscripting.
        from ..type_converter import get_base_type
        from ..type_descriptor import array_decay_type
        if getattr(get_base_type(subject_type), 'dimensions', None) is not None:
            ptr_type = array_decay_type(subject_type)
            ptr_subject = self.type_converter.convert(subject, ptr_type)
            element = self.value_dispatcher.handle_subscript(ptr_subject, index_const, fake_node)
        else:
            element = self.value_dispatcher.handle_subscript(subject, index_const, fake_node)
        return self.value_dispatcher.read_rvalue(element)
    
    def _compare_values(self, left, right, op):
        """Helper to compare two values (used by match)."""
        return self.value_dispatcher.handle_compare(op, left, right, None)
    
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
        var_value_ir = ensure_ir(self.value_dispatcher.read_rvalue(var_value))
        self.builder.store(var_value_ir, alloca)

        # Register the newly bound variable as the holder of the linear token.
        # When a match extracts a linear field, ownership moves from the scrutinee
        # to the bound variable; this transitions that variable from invalid to valid.
        if pc_type is not None and self._is_linear_type(pc_type):
            var_info = self.lookup_variable(var_name)
            if var_info is not None:
                self._init_linear_states(var_info, pc_type, initial_state='active')

    def _consume_linear_subjects(self, subjects: list, case_node):
        """Consume linear match subjects when a case is selected.

        Matching a linear value consumes the scrutinee. For each subject whose type
        is linear (including enums with linear variants), emit a LinearTransition
        valid -> invalid.

        This must be called *after* the guard has been verified to pass, so that a
        failing guard does not consume the subject before falling through to the
        next case.
        """
        for subject in subjects:
            if not isinstance(subject, ValueRef):
                continue
            if subject.type_hint is None:
                continue
            if self._is_linear_type(subject.type_hint):
                self._transfer_linear_ownership(
                    subject, reason="match_consumption", node=case_node
                )
