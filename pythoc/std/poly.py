"""
Poly standard library for PC

Provides polymorphism dispatcher for @compile functions.

Usage:
    add = Poly(add_i32, add_f64)
    add.append(add_str)
    add(1, 2)  # static dispatch by type
    add(my_enum_val, other_enum_val)  # dynamic dispatch by enum tag

See test/integration/test_poly.py for examples.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

from ..registry import get_unified_registry
from ..logger import logger
from ..decorators import compile
from ..builtin_entities import struct, enum

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
        registry = get_unified_registry()
        func_info = registry.get_function_info(f.__name__)
        if not func_info:
            lookup = getattr(f, '_mangled_name', None)
            if lookup:
                func_info = registry.get_function_info_by_mangled(lookup)
        if not func_info:
            raise NameError(f"Function '{getattr(f, '__name__', f)}' not found in registry")
        
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
        from ..builtin_entities import i32, i64, f64, ptr, i8, bool as pc_bool
        
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
        from ..valueref import ensure_ir, wrap_value
        from llvmlite import ir
        
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
            raise TypeError(
                f"Poly: no matching signature for {arg_types} "
                f"and no Enum argument for dynamic dispatch"
            )
        
        # Generate dynamic dispatch for enum arguments
        return self._generate_dynamic_dispatch(visitor, args, arg_types, enum_indices, node)
    
    def _generate_dynamic_dispatch(self, visitor, args, arg_types, enum_indices, node):
        """Generate dynamic dispatch logic for enum arguments
        
        Strategy:
        1. Validate that ALL variant combinations have matching function overloads
        2. Validate that ALL overloads have the SAME return type
        3. Generate and compile a dispatch function using match/case
        4. Call the compiled dispatch function
        """
        import itertools

        logger.debug(f"Poly _generate_dynamic_dispatch: args={args}, arg_types={arg_types}, enum_indices={enum_indices}")
        
        # Step 1: Validate all combinations are covered
        enum_classes = [enum_cls for _, enum_cls in enum_indices]
        
        # Get all variant types for each enum
        variants_per_enum = []
        for enum_cls in enum_classes:
            variants = []
            for vname, vtype in zip(enum_cls._variant_names, enum_cls._variant_types):
                if vtype is not None:  # Skip None payload variants
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
            
            from ..type_id import get_type_id
            sig_key = tuple(get_type_id(t) for t in expected_types)
            logger.debug(f"Checking signature: {sig_key}")
            target_info = self._sig_map.get(sig_key)
            
            if target_info is None:
                variant_names = [vname for vname, _ in combo]
                missing_combinations.append((variant_names, expected_types))
            else:
                variant_combos.append((combo, expected_types, target_info))
                # Get return type from function info
                target_func, _ = target_info
                from ..registry import get_unified_registry
                registry = get_unified_registry()
                func_info = registry.get_function_info(target_func.__name__)
                if not func_info:
                    lookup = getattr(target_func, '_mangled_name', None)
                    if lookup:
                        func_info = registry.get_function_info_by_mangled(lookup)
                if func_info and func_info.return_type_hint:
                    return_types.append(func_info.return_type_hint)
        
        if missing_combinations:
            error_msg = "Poly: cannot generate dynamic dispatch - missing function overloads:\n"
            for variant_names, sig in missing_combinations:
                error_msg += f"  - Variants {variant_names}: {sig}\n"
            raise TypeError(error_msg)
        
        # Step 2: Validate all return types are the same
        if return_types:
            first_return_type = return_types[0]
            for rt in return_types[1:]:
                if repr(rt) != repr(first_return_type):
                    error_msg = (
                        f"Poly: cannot generate dynamic dispatch - inconsistent return types:\n"
                        f"  Expected all overloads to return {first_return_type}, "
                        f"but found {rt}\n"
                    )
                    raise TypeError(error_msg)

        # Step 3: Generate and compile dispatch function
        # print(f"all_sigs={all_sigs}")
        # all_sigs_key = tuple(all_sigs.keys())
        # if all_sigs_key in self._dynamic_dispatch_cache:
        #     return self._dynamic_dispatch_cache[all_sigs_key].handle_call(visitor, args, node)

        def gen_tag_combinations(enum_classes):
            """Generate all combinations of tag values from multiple enum classes"""
            import itertools
            
            # Get all tag values for each enum class
            tag_lists = []
            for enum_cls in enum_classes:
                # Get all tag values from _variant_names
                tags = [getattr(enum_cls, vname) for vname in enum_cls._variant_names]
                tag_lists.append(tags)
            
            # Generate all combinations
            for tag_combo in itertools.product(*tag_lists):
                yield tag_combo

        tag_to_func = {}
        tag_to_struct = {}
        tag_to_param_map = {}
        tag_to_variant_indices = {}  # Map tag_tuple -> list of variant indices for each enum arg
        
        # Build enum position set for quick lookup
        enum_positions = set(pos for pos, _ in enum_indices)
        
        for combo, expected_types, target_info in variant_combos:
            tag_tuple = tuple(
                enum_cls._tag_values[vname] 
                for (vname, vtype), enum_cls in zip(combo, enum_classes)
            )
            
            # Build list of variant indices for this tag combination
            variant_idx_list = []
            for (vname, vtype), enum_cls in zip(combo, enum_classes):
                # Find the index of this variant in the enum's variant list
                variant_idx = enum_cls._variant_names.index(vname)
                variant_idx_list.append(variant_idx)
            tag_to_variant_indices[tag_tuple] = variant_idx_list
            
            target_func, param_types = target_info
            tag_to_func[tag_tuple] = target_func
            tag_to_struct[tag_tuple] = struct[param_types]
            
            # Build parameter extraction map: (arg_idx, extract_payload, enum_arg_index)
            # extract_payload=True means take args[arg_idx][1][variant_idx] (enum payload from union)
            # extract_payload=False means take args[arg_idx] directly
            # enum_arg_index is the index in enum_indices list (to get variant_idx)
            param_extract = []
            enum_arg_counter = 0
            for i in range(len(arg_types)):
                if i in enum_positions:
                    param_extract.append((i, True, enum_arg_counter))
                    enum_arg_counter += 1
                else:
                    param_extract.append((i, False, -1))
            tag_to_param_map[tag_tuple] = param_extract
        
        ArgTypesPack = struct[arg_types]
        
        # Create new dynamic dispatch function
        @compile(anonymous=True)
        def dynamic_dispatch(*args : ArgTypesPack) -> first_return_type:
            for tags in gen_tag_combinations(enum_classes):
                all_match: i32 = 1
                for i in range(len(enum_indices)):
                    if args[enum_indices[i][0]][0] != tags[i]:
                        all_match = 0
                        break
                if all_match != 0:
                    params: tag_to_struct[tags]
                    
                    # CRITICAL: Fully inline all dictionary accesses to avoid variable sharing issues.
                    # When constant loop unrolling occurs, intermediate variables like target_func,
                    # variant_indices, etc. can share storage across iterations, causing later
                    # assignments to overwrite earlier ones. By directly accessing dictionaries inline,
                    # we ensure each iteration gets the correct values without variable conflicts.
                    for param_idx in range(len(tag_to_param_map[tags])):
                        arg_idx, extract_payload, enum_arg_idx = tag_to_param_map[tags][param_idx]
                        if extract_payload:
                            params[param_idx] = args[arg_idx][1][tag_to_variant_indices[tags][enum_arg_idx]]
                        else:
                            params[param_idx] = args[arg_idx]
                        pass
                    
                    return tag_to_func[tags](*params)

        return dynamic_dispatch.handle_call(visitor, dynamic_dispatch, args, node)
        