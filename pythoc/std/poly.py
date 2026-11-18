"""Exact-type overloading with enum dispatch expanded in the caller.

    operation = Poly(integer_impl, floating_impl)
    operation(i32(1))   # direct native target
    operation(value)    # enum payloads select a target through ordinary match
"""
from __future__ import annotations

import ast
from typing import Any, List

from pythoc import typeof, inline, meta
from pythoc.literal_protocol import get_sequence_elements
from pythoc.type_converter import TypeConverter


class Poly:
    """Polymorphism dispatcher for @compile functions.
    
    Supports:
    - Static polymorphism: dispatch by argument PC types
    - Dynamic polymorphism: if arg is Enum, inspect tag at runtime and dispatch
    """
    
    def __init__(self, *compiled_funcs):
        self._funcs: List[Any] = list(compiled_funcs)
        self._signatures()

        @inline
        def _call(*args):
            return self._select(args)(*args)

        self._call = _call
        self.handle_call = _call.handle_call
        self.defer_linear_transfer = _call.defer_linear_transfer

    def append(self, compiled_func):
        """Add an overload; the latest exact signature takes precedence."""
        self._signature(compiled_func)
        self._funcs.append(compiled_func)
        return self

    @staticmethod
    def _signature(function):
        function_type = typeof(function)
        if not hasattr(function_type, 'param_types'):
            raise TypeError('Poly requires a compiled function with a concrete signature')
        return tuple(function_type.param_types), function_type.return_type

    def _signatures(self):
        """Project current authoritative type objects without maintaining a second index."""
        from pythoc.type_id import get_type_id
        signatures = {}
        for function in self._funcs:
            parameters, _ = self._signature(function)
            key = tuple(get_type_id(parameter) for parameter in parameters)
            signatures[key] = (function, parameters)
        return signatures

    def _select(self, argument_pack):
        """Inspect evaluated argument references without transferring their owners."""
        return self.resolve(*(
            TypeConverter.infer_default_pc_type_from_python(value.get_python_value())
            if value.is_python_value() else value.type_hint
            for value in get_sequence_elements(argument_pack)))

    # The selected ordinary call, rather than this metadata query, moves owners.
    _select.defer_linear_transfer = True

    def resolve(self, *arg_types):
        """Select an existing function or construct a compile-only inline body."""
        signatures = self._signatures()
        arg_types = tuple(arg_types)
        from pythoc.type_id import get_type_id
        sig_key = tuple(get_type_id(t) for t in arg_types)

        target_info = signatures.get(sig_key)
        if target_info is not None:
            target_func, _ = target_info
            return target_func
        
        # Find all enum arguments
        enum_indices = []
        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and getattr(arg_type, 'is_enum_type', lambda: False)():
                enum_indices.append((i, arg_type))
        
        if not enum_indices:
            raise TypeError(f'Poly: no matching signature for {arg_types} '
                            'and no Enum argument for dynamic dispatch')
        
        return self._expand_enum(arg_types, enum_indices, signatures)

    def _expand_enum(self, arg_types, enum_indices, signatures):
        """Build ordinary match/call syntax for expansion in the caller's CFG."""
        import itertools
        from pythoc.type_id import get_type_id

        # Step 1: Validate all combinations are covered
        enum_classes = [enum_cls for _, enum_cls in enum_indices]
        
        # Get all variant types for each enum
        variants_per_enum = []
        for enum_cls in enum_classes:
            variants = []
            for vname, vtype, _ in enum_cls.get_variants():
                if vtype is None:
                    raise TypeError('Poly dynamic dispatch requires a payload for every '
                                    f'variant; {enum_cls.__name__}.{vname} has none')
                variants.append((vname, vtype))
            variants_per_enum.append(variants)
        
        # Check all combinations and collect their native return types.
        missing_combinations = []
        return_types = []
        variant_combos = []
        
        for combo in itertools.product(*variants_per_enum):
            # Build expected signature
            expected_types = list(arg_types)
            for i, (enum_idx, _) in enumerate(enum_indices):
                _, variant_type = combo[i]
                expected_types[enum_idx] = variant_type
            
            sig_key = tuple(get_type_id(t) for t in expected_types)
            target_info = signatures.get(sig_key)
            
            if target_info is None:
                variant_names = [vname for vname, _ in combo]
                missing_combinations.append((variant_names, expected_types))
            else:
                variant_combos.append((combo, expected_types, target_info))
                # Inspect the selected native callable through its public PC type.
                target_func, _ = target_info
                return_types.append(self._signature(target_func)[1])
        
        if missing_combinations:
            error_msg = "Poly: cannot expand dynamic dispatch - missing function overloads:\n"
            for variant_names, sig in missing_combinations:
                error_msg += f"  - Variants {variant_names}: {sig}\n"
            raise TypeError(error_msg)
        
        # Step 2: Validate all return types are the same
        if return_types:
            first_return_type = return_types[0]
            for rt in return_types[1:]:
                if get_type_id(rt) != get_type_id(first_return_type):
                    error_msg = (
                        f"Poly: cannot expand dynamic dispatch - inconsistent return types:\n"
                        f"  Expected all overloads to return {first_return_type}, "
                        f"but found {rt}\n"
                    )
                    raise TypeError(error_msg)

        # Specialize ordinary native match code. Pattern extraction transfers
        # linear payloads through the same rules as handwritten ADT consumers.
        enum_by_index = dict(enum_indices)
        globals_dict = {}
        cases = []
        for case_index, (combo, _, target_info) in enumerate(variant_combos):
            patterns = []
            arguments = []
            variant_by_index = dict(zip(enum_by_index, combo))
            for index in range(len(arg_types)):
                if index not in enum_by_index:
                    arguments.append(ast.Name(id=f'arg_{index}', ctx=ast.Load()))
                    continue
                enum_name = f'_enum_{index}'
                globals_dict[enum_name] = enum_by_index[index]
                variant_name, _ = variant_by_index[index]
                payload_name = f'payload_{index}'
                patterns.append(ast.MatchSequence(patterns=[
                    ast.MatchValue(value=ast.Attribute(
                        value=ast.Name(id=enum_name, ctx=ast.Load()),
                        attr=variant_name, ctx=ast.Load())),
                    ast.MatchAs(name=payload_name),
                ]))
                arguments.append(ast.Name(id=payload_name, ctx=ast.Load()))

            target_name = f'_target_{case_index}'
            globals_dict[target_name] = target_info[0]
            pattern = (patterns[0] if len(patterns) == 1
                       else ast.MatchSequence(patterns=patterns))
            cases.append(ast.match_case(pattern=pattern, guard=None, body=[
                ast.Return(value=ast.Call(
                    func=ast.Name(id=target_name, ctx=ast.Load()),
                    args=arguments, keywords=[])),
            ]))

        subjects = [ast.Name(id=f'arg_{index}', ctx=ast.Load())
                    for index in enum_by_index]
        subject = (subjects[0] if len(subjects) == 1
                   else ast.Tuple(elts=subjects, ctx=ast.Load()))
        body = [ast.Match(subject=subject, cases=cases)]
        generated = meta.func(
            name='dynamic_dispatch',
            params=[(f'arg_{index}', pc_type)
                    for index, pc_type in enumerate(arg_types)],
            return_type=first_return_type,
            body=body,
            required_globals=globals_dict,
            source_file=__file__,
        )
        return meta.inline_generated(generated)
