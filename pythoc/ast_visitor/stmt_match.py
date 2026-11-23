"""
Match statement visitor mixin
"""

import ast
from llvmlite import ir
from ..valueref import ValueRef, wrap_value
from ..ir_helpers import initialize_storage
from ..literal_protocol import get_unpack_values, project_subscript
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
                        self._bind_match_variable(name, subjects[0], case.pattern)
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
                            self._bind_match_variable(var_name, var_value, case.pattern)

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

                        self.process_condition(guard_result, guard_then, case.guard, guard_else)
                
                def pattern_else():
                    # Pattern failed - go to next case
                    if not cf.is_terminated():
                        cf.branch(next_case_block)

                # Use process_condition for pattern check
                self.process_condition(condition_ref, pattern_then_with_guard, case.pattern, pattern_else)

            else:
                # No guard - simple pattern match using process_condition
                
                def pattern_then(bindings=bindings, case=case):
                    # Enter MATCH scope for bindings and body
                    with self.scope_manager.scope(ScopeType.MATCH, cf, node=case):
                        # Bind pattern variables
                        for var_name, var_value in bindings:
                            self._bind_match_variable(var_name, var_value, case.pattern)
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
                self.process_condition(condition_ref, pattern_then, case.pattern, pattern_else)
            
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

    def _combine_match_conditions(self, conditions, node, operation='and_'):
        from ..builtin_entities import bool as pc_bool
        if not conditions:
            return self._match_true()
        to_boolean = self.value_dispatcher.to_boolean
        value = to_boolean(conditions[0], node)
        for condition in conditions[1:]:
            value = getattr(self.builder, operation)(value, to_boolean(condition, node))
        return wrap_value(value, kind='value', type_hint=pc_bool)

    def _generate_match_pattern(self, pattern, subject):
        bindings = []
        node = pattern.ast_node
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
                compared = (project_subscript(self, subject, 0, node)
                            if getattr(subject.type_hint, '_is_enum', False) else subject)
                condition = self.value_dispatcher.handle_compare(
                    ast.Eq(), compared, pattern.value_ref, node)
            elif pattern.kind == PatternKind.OR:
                conditions = []
                for alternative in pattern.alternatives:
                    value, alternative_bindings = self._generate_match_pattern(
                        alternative, subject)
                    if alternative_bindings:
                        logger.error('Captures in OR patterns are unsupported',
                                     node=alternative.ast_node, exc_type=TypeError)
                    conditions.append(value)
                condition = self._combine_match_conditions(conditions, node, 'or_')
            elif pattern.kind == PatternKind.CONSTRUCTOR:
                conditions = []
                if pattern.payload_index is not None:
                    tag = project_subscript(self, subject, 0, node)
                    conditions.append(self.value_dispatcher.handle_compare(
                        ast.Eq(), tag, pattern.value_ref, node))
                    fields = ()
                    if pattern.field_indices:
                        payload = project_subscript(self, subject, 1, node)
                        item = project_subscript(self, payload, pattern.payload_index, node)
                        fields = ((pattern.sub_patterns[0], item),)
                elif pattern.constructor_name == 'sequence':
                    fields = zip(pattern.sub_patterns, get_unpack_values(self, subject, node))
                else:
                    fields = ((pattern.sub_patterns[index],
                               project_subscript(self, subject, index, node))
                              for index in pattern.field_indices)
                for sub_pattern, item in fields:
                    value, values = self._generate_match_pattern(sub_pattern, item)
                    conditions.append(value)
                    bindings.extend(values)
                condition = self._combine_match_conditions(conditions, node)
            else:
                logger.error(f'Unknown normalized pattern: {pattern.kind}',
                             node=pattern.ast_node, exc_type=NotImplementedError)
        bindings.extend((name, captured) for name in pattern.bindings)
        return condition, bindings

    def _generate_multi_subject_pattern(self, pattern, subjects):
        if pattern.kind == PatternKind.OR:
            conditions = []
            for alternative in pattern.alternatives:
                value, bindings = self._generate_multi_subject_pattern(alternative, subjects)
                if bindings:
                    logger.error('Captures in OR patterns are unsupported',
                                 node=alternative.ast_node, exc_type=TypeError)
                conditions.append(value)
            return self._combine_match_conditions(conditions, pattern.ast_node, 'or_'), []
        conditions, bindings = [], []
        for item, subject in zip(pattern.sub_patterns, subjects):
            value, values = self._generate_match_pattern(item, subject)
            conditions.append(value)
            bindings.extend(values)
        return self._combine_match_conditions(conditions, pattern.ast_node), bindings
    
    def _bind_match_variable(self, var_name, var_value, node):
        """Bind a variable from match pattern
        
        Creates a local variable and stores the pattern-matched value.
        Similar to variable declaration but without type annotation.
        """
        if not isinstance(var_value, ValueRef):
            logger.error(f"Expected ValueRef for match binding, got {type(var_value)}",
                        node=node, exc_type=TypeError)
        
        # Get PC type from the value
        pc_type = var_value.type_hint
        if pc_type is None:
            logger.error(f"Cannot bind variable '{var_name}' - value has no type",
                        node=node, exc_type=TypeError)

        # Create alloca for the variable
        llvm_type = pc_type.get_llvm_type(self.module.context)
        alloca = self._create_alloca_in_entry(llvm_type, f"{var_name}_addr")
        
        # Declare variable in symbol table
        var_info = self.declare_variable(
            name=var_name,
            type_hint=pc_type,
            node=node,
            alloca=alloca,
            source="match_pattern",
            line_number=None
        )
        
        # Store the matched value
        var_value_ir = self._assignment_ir_value(self.read_rvalue(var_value), llvm_type)
        initialize_storage(self.builder, var_value_ir, alloca, pc_type)

        # Register the newly bound variable as the holder of the linear token.
        # When a match extracts a linear field, ownership moves from the scrutinee
        # to the bound variable; this transitions that variable from invalid to valid.
        if self._is_linear_type(pc_type):
            self._init_linear_states(var_info, pc_type, node, 'active')

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
            self._transfer_linear_ownership(subject, reason="match_consumption", node=case_node)
