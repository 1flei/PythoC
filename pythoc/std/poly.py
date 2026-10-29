"""
Poly standard library for PC

Provides polymorphism dispatcher for @compile functions.

Usage:
    add = Poly(add_i32, add_f64)
    add.append(add_str)
    add(1, 2)  # static dispatch by type
    add(my_enum_val, other_enum_val)  # dynamic dispatch by enum tag

"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

from ..logger import logger
from ..builtin_entities import i64, f64, ptr, i8, bool as pc_bool

class Poly:
    """Polymorphism dispatcher for @compile functions.
    
    Supports:
    - Static polymorphism: dispatch by argument PC types
    - Dynamic polymorphism: if arg is Enum, inspect tag at runtime and dispatch
    """
    
    def __init__(self, *compiled_funcs):
        self._funcs: List[Any] = list(compiled_funcs)
        self._sig_map: Dict[Tuple[type, ...], Any] = {}
        self._dynamic_dispatch_cache = {}
        for f in self._funcs:
            self._register_function(f)
    
    def append(self, compiled_func):
        """Add another function overload to the dispatcher"""
        self._funcs.append(compiled_func)
        self._register_function(compiled_func)
        return self
    
    def _register_function(self, f):
        """Extract function signature and register in map"""
        # Get func_info directly from wrapper instead of registry lookup
        func_info = getattr(f, '_func_info', None)
        if not func_info:
            raise NameError(f"Function '{getattr(f, '__name__', f)}' not found - missing _func_info attribute")
        
        param_types = tuple(
            func_info.param_type_hints[name] 
            for name in func_info.param_names
        )
        # Use get_type_id() as key for semantic equivalence
        from ..type_id import get_type_id
        sig_key = tuple(get_type_id(t) for t in param_types)
        self._sig_map[sig_key] = (f, param_types)
    
    def can_be_called(self) -> bool:
        return True

    def _infer_default_pc_type(self, python_val):
        if isinstance(python_val, str):
            return ptr[i8]
        elif isinstance(python_val, bool):
            return pc_bool
        elif isinstance(python_val, int):
            return i64
        elif isinstance(python_val, float):
            return f64
        else:
            raise TypeError(
                f"Poly: unsupported Python type {type(python_val).__name__}"
            )

    def _get_type_hint(self, arg):
        """Get type hint for argument"""
        if arg.is_python_value():
            return self._infer_default_pc_type(arg.get_python_value())
        else:
            return arg.type_hint
    
    def handle_call(self, visitor, func_ref, args, node: ast.Call):
        """PC callable protocol: dispatch to correct compiled function
        
        Strategy:
        1. Try static dispatch by argument type hints
        2. If args contain Enum types, generate dynamic dispatch function
        """
        # Try static dispatch first
        arg_types = tuple(self._get_type_hint(a) for a in args)
        from ..type_id import get_type_id
        sig_key = tuple(get_type_id(t) for t in arg_types)

        logger.debug(f"Poly handle_call: sig_map={self._sig_map}, sig_key={sig_key}")
        
        target_info = self._sig_map.get(sig_key)
        if target_info is not None:
            target_func, _ = target_info
            return target_func.handle_call(visitor, target_func, args, node)
        
        # Find all enum arguments
        enum_indices = []
        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and getattr(arg_type, '_is_enum', False):
                enum_indices.append((i, arg_type))
        
        if not enum_indices:
            logger.error(
                f"Poly: no matching signature for {arg_types} "
                f"and no Enum argument for dynamic dispatch",
                node
            )
        
        # Generate or reuse dynamic dispatch for enum arguments.
        dynamic_dispatch = self._get_dynamic_dispatch_func(arg_types, enum_indices, node)
        return dynamic_dispatch.handle_call(visitor, dynamic_dispatch, args, node)

    def prepare_dynamic_dispatch(self, *arg_types):
        """Optionally pre-generate a wrapper for an argument-type combination.

        Ordinary calls infer these types and generate the wrapper on demand.
        Explicit preparation can move generation before object building when
        the caller enables the guard against build-time compilation queueing.
        """
        enum_indices = []
        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and getattr(arg_type, '_is_enum', False):
                enum_indices.append((i, arg_type))
        if not enum_indices:
            raise TypeError("Poly.prepare_dynamic_dispatch requires at least one enum argument type")
        return self._get_dynamic_dispatch_func(tuple(arg_types), enum_indices, None)
    
    def _get_dynamic_dispatch_func(self, arg_types, enum_indices, node):
        """Generate dynamic dispatch logic for enum arguments
        
        Strategy:
        1. Validate that ALL variant combinations have matching function overloads
        2. Validate that ALL overloads have the SAME return type
        3. Generate and compile a dispatch function using match/case
        4. Call the compiled dispatch function
        """
        import itertools

        logger.debug(f"Poly _get_dynamic_dispatch_func: arg_types={arg_types}, enum_indices={enum_indices}")
        from ..type_id import get_type_id
        cache_key = tuple(get_type_id(t) for t in arg_types)
        cached = self._dynamic_dispatch_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Step 1: Validate all combinations are covered
        enum_classes = [enum_cls for _, enum_cls in enum_indices]
        
        # Get all variant types for each enum
        variants_per_enum = []
        for enum_cls in enum_classes:
            variants = []
            for vname, vtype in zip(enum_cls._variant_names, enum_cls._variant_types):
                if vtype is None:
                    logger.error(
                        "Poly dynamic dispatch requires a payload for every "
                        f"variant; {enum_cls.__name__}.{vname} has none",
                        node, exc_type=TypeError)
                variants.append((vname, vtype))
            variants_per_enum.append(variants)
        
        # Check all combinations exist in _sig_map and collect return types
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
            logger.debug(f"Checking signature: {sig_key}")
            target_info = self._sig_map.get(sig_key)
            
            if target_info is None:
                variant_names = [vname for vname, _ in combo]
                missing_combinations.append((variant_names, expected_types))
            else:
                variant_combos.append((combo, expected_types, target_info))
                # Get return type from function info directly from wrapper
                target_func, _ = target_info
                func_info = getattr(target_func, '_func_info', None)
                return_types.append(func_info.return_type_hint)
        
        if missing_combinations:
            error_msg = "Poly: cannot generate dynamic dispatch - missing function overloads:\n"
            for variant_names, sig in missing_combinations:
                error_msg += f"  - Variants {variant_names}: {sig}\n"
            logger.error(error_msg, node)
        
        # Step 2: Validate all return types are the same
        if return_types:
            first_return_type = return_types[0]
            for rt in return_types[1:]:
                if get_type_id(rt) != get_type_id(first_return_type):
                    error_msg = (
                        f"Poly: cannot generate dynamic dispatch - inconsistent return types:\n"
                        f"  Expected all overloads to return {first_return_type}, "
                        f"but found {rt}\n"
                    )
                    logger.error(error_msg, node)

        # Specialize ordinary native match code. Pattern extraction transfers
        # linear payloads through the same rules as handwritten ADT consumers.
        from .. import meta

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
        # Two dispatchers over the same input types may select different
        # overloads. Include native target identities in the specialization.
        targets = tuple((target[0]._func_info.source_file,
                         target[0]._func_info.mangled_name
                         or target[0]._func_info.name)
                        for _, _, target in variant_combos)
        dynamic_dispatch = meta.compile_generated(
            generated, suffix=('poly_match', cache_key, targets))
        self._dynamic_dispatch_cache[cache_key] = dynamic_dispatch
        return dynamic_dispatch
