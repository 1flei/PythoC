from llvmlite import ir
from typing import Optional
from .base import BuiltinFunction, BuiltinEntity, _get_unified_registry
from ..valueref import ensure_ir, wrap_value, get_type
import ast
import ctypes


class typeof(BuiltinFunction):
    """typeof(x) - Get the type of a value or return a type
    
    Works in both Python preprocessing and @compile contexts:
    - Python preprocessing: typeof(5) -> pyconst[5], typeof(i32) -> i32
    - In @compile: typeof(x) -> type of x
    
    Returns:
        - Python value -> pyconst[value]
        - PC value -> its type_hint  
        - Type -> the type itself
    
    Examples:
        typeof(5)        # pyconst[5]
        typeof(100)      # pyconst[100]
        typeof(x)        # i32 (if x: i32)
        typeof(i32)      # i32
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'typeof'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    def __new__(cls, value_or_type):
        """Support direct Python-level calls: typeof(5), typeof(i32)"""
        # Check if it's already a type (BuiltinEntity subclass or instance)
        if isinstance(value_or_type, type) and issubclass(value_or_type, BuiltinEntity):
            # It's a type class - return it
            return value_or_type
        elif isinstance(value_or_type, BuiltinEntity):
            # It's a type instance - return it
            return value_or_type
        else:
            # It's a Python value - wrap as pyconst
            from .python_type import PythonType
            return PythonType.wrap(value_or_type, is_constant=True)
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle typeof(x) call in @compile context
        
        Strategy:
        1. If arg is a type annotation (Name/Attribute/Subscript) -> parse as type
        2. If arg evaluates to ValueRef with Python value -> wrap as pyconst
        3. If arg evaluates to ValueRef with LLVM value -> return its type_hint
        """
        if len(node.args) != 1:
            raise TypeError(f"typeof() takes exactly 1 argument ({len(node.args)} given)")
        
        arg_node = node.args[0]
        
        # Strategy 1: Try to parse as type annotation first
        # This handles: typeof(i32), typeof(ptr[i32]), etc.
        try:
            pc_type = visitor.type_resolver.parse_annotation(arg_node)
            if pc_type is not None:
                # It's a type - return it as-is
                from .python_type import PythonType
                return wrap_value(pc_type, kind="python", type_hint=PythonType.wrap(pc_type))
        except:
            pass
        
        # Strategy 2 & 3: Evaluate the expression
        value_ref = visitor.visit_expression(arg_node)
        
        # Check if it's a Python value (constant)
        if value_ref.is_python_value():
            # Python value -> pyconst[value]
            from .python_type import PythonType
            py_value = value_ref.value
            pyconst_type = PythonType.wrap(py_value, is_constant=True)
            return wrap_value(pyconst_type, kind="python", type_hint=PythonType.wrap(pyconst_type))
        
        # PC value -> return its type_hint
        if value_ref.type_hint is not None:
            from .python_type import PythonType
            return wrap_value(value_ref.type_hint, kind="python", type_hint=PythonType.wrap(value_ref.type_hint))
        
        raise TypeError(f"typeof() cannot determine type of {ast.dump(arg_node)}")


class sizeof(BuiltinFunction):
    """sizeof(type) - Get size of a type in bytes"""
    
    @classmethod
    def get_name(cls) -> str:
        return 'sizeof'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call) -> ir.Value:
        """Handle sizeof(type) call
        
        Unified implementation using TypeResolver to parse all types.
        This eliminates ~150 lines of duplicate type parsing logic.
        """
        if len(node.args) != 1:
            raise TypeError(f"sizeof() takes exactly 1 argument ({len(node.args)} given)")
        
        arg = node.args[0]
        
        # Use TypeResolver to parse the type uniformly
        pc_type = visitor.type_resolver.parse_annotation(arg)
        if pc_type is None:
            raise TypeError(f"sizeof() argument must be a type, got: {ast.dump(arg)}")
        
        # Calculate type size
        size = cls._get_type_size(pc_type, visitor)
        # Return as Python int (will be promoted when needed)
        from .python_type import PythonType
        python_type = PythonType.wrap(size, is_constant=True)
        return wrap_value(size, kind="python", type_hint=python_type)
    
    @classmethod
    def _get_type_size(cls, pc_type, visitor) -> int:
        """Get size of a PC type
        
        Args:
            pc_type: BuiltinEntity type (i32, f64, ptr[T], etc.) or Python struct class
            visitor: AST visitor for context
        
        Returns:
            int: Size in bytes
        """
        # 1. Check if it's a BuiltinType with get_size_bytes method
        if hasattr(pc_type, 'get_size_bytes'):
            return pc_type.get_size_bytes()
        
        # 2. Check if it's a Python struct class (from @compile decorator)
        if isinstance(pc_type, type) and hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            registry = _get_unified_registry()
            struct_name = pc_type.__name__
            if registry.has_struct(struct_name):
                struct_info = registry.get_struct(struct_name)
                return cls._calculate_struct_size(struct_info, registry)
            raise TypeError(f"sizeof(): struct '{struct_name}' not found in registry")
        
        # 3. This is a bug - sizeof should only receive PC types
        if isinstance(pc_type, ir.Type):
            raise TypeError(
                f"sizeof() received ir.Type ({pc_type}). This is a bug - "
                "use BuiltinEntity (i32, f64, ptr[T], etc.) instead."
            )
        
        # Unknown/unsupported type for sizeof
        raise TypeError(f"sizeof(): unknown or unsupported type {pc_type}")
    
    @classmethod
    def _align_to(cls, size: int, alignment: int) -> int:
        """Align size to the specified alignment boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
    

    
    @classmethod
    def _get_field_size(cls, field_type, struct_registry) -> int:
        """Get size of a field type"""
        # Handle builtin types
        if isinstance(field_type, type) and issubclass(field_type, BuiltinEntity):
            if field_type.can_be_type():
                return field_type.get_size_bytes()
        
        # Handle string type names
        if isinstance(field_type, str):
            # Check builtin types
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(field_type)
            if entity_cls and entity_cls.can_be_type():
                return entity_cls.get_size_bytes()
            # Check struct types
            if struct_registry.has_struct(field_type):
                struct_info = struct_registry.get_struct(field_type)
                return cls._calculate_struct_size(struct_info, struct_registry)
        
        # Handle type classes directly
        if hasattr(field_type, '__name__'):
            type_name = field_type.__name__
            # Check builtin types
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(type_name)
            if entity_cls and entity_cls.can_be_type():
                return entity_cls.get_size_bytes()
            # Check struct types
            if struct_registry.has_struct(type_name):
                struct_info = struct_registry.get_struct(type_name)
                return cls._calculate_struct_size(struct_info, struct_registry)
        
        # Handle ptr types
        if hasattr(field_type, 'pointee_type') or (hasattr(field_type, '__name__') and field_type.__name__ == 'ptr'):
            return ctypes.sizeof(ctypes.c_void_p)
        
        # Unknown field type size
        raise TypeError("sizeof(): unknown field type for size calculation")
    
    @classmethod
    def _get_field_alignment(cls, field_type, struct_registry) -> int:
        """Get alignment of a field type"""
        # Handle builtin types
        if isinstance(field_type, type) and issubclass(field_type, BuiltinEntity):
            if field_type.can_be_type():
                return min(field_type.get_size_bytes(), 8)
        
        # Handle string type names
        if isinstance(field_type, str):
            # Check builtin types
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(field_type)
            if entity_cls and entity_cls.can_be_type():
                return min(entity_cls.get_size_bytes(), 8)
            # Check struct types
            if struct_registry.has_struct(field_type):
                struct_info = struct_registry.get_struct(field_type)
                return cls._calculate_struct_alignment(struct_info, struct_registry)
        
        # Handle type classes directly
        if hasattr(field_type, '__name__'):
            type_name = field_type.__name__
            # Check builtin types
            registry = _get_unified_registry()
            entity_cls = registry.get_builtin_entity(type_name)
            if entity_cls and entity_cls.can_be_type():
                return min(entity_cls.get_size_bytes(), 8)
            # Check struct types
            if struct_registry.has_struct(type_name):
                struct_info = struct_registry.get_struct(type_name)
                return cls._calculate_struct_alignment(struct_info, struct_registry)
        
        # Handle ptr types
        if hasattr(field_type, 'pointee_type') or (hasattr(field_type, '__name__') and field_type.__name__ == 'ptr'):
            return ctypes.sizeof(ctypes.c_void_p)
        
        # Unknown field type size
        raise TypeError("sizeof(): unknown field type for size calculation")
    
    @classmethod
    def _calculate_struct_alignment(cls, struct_info, struct_registry) -> int:
        """Calculate the alignment requirement for a struct"""
        max_alignment = 1
        for field_name, field_type in struct_info.fields:
            field_alignment = cls._get_field_alignment(field_type, struct_registry)
            max_alignment = max(max_alignment, field_alignment)
        return max_alignment
    
    @classmethod
    def _calculate_struct_size(cls, struct_info, struct_registry) -> int:
        """Calculate struct size with proper alignment"""
        current_offset = 0
        max_alignment = 1
        
        for field_name, field_type in struct_info.fields:
            field_size = cls._get_field_size(field_type, struct_registry)
            field_alignment = cls._get_field_alignment(field_type, struct_registry)
            
            # Update maximum alignment
            max_alignment = max(max_alignment, field_alignment)
            
            # Align current offset to field alignment
            current_offset = cls._align_to(current_offset, field_alignment)
            
            # Add field size
            current_offset += field_size
        
        # Align total size to struct alignment
        total_size = cls._align_to(current_offset, max_alignment)
        
        return total_size


from .types import ptr, i8
nullptr = wrap_value(ir.Constant(ir.PointerType(ir.IntType(8)), None), kind="value", type_hint=ptr[i8])


class char(BuiltinFunction):
    """char(value) - Convert string or int to i8 character
    
    Converts Python values to i8 type:
    - char("abc") -> i8(ord('a')) - first character of string
    - char("s") -> i8(ord('s')) - single character
    - char("") -> i8(0) - empty string returns null terminator
    - char(48) -> i8(48) - int directly converted to i8
    
    Only accepts int or str Python values.
    Raises TypeError for other types.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'char'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle char(value) call
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs
            node: ast.Call node
            
        Returns:
            ValueRef containing i8 value
        """
        from .python_type import PythonType
        
        if len(args) != 1:
            raise TypeError(f"char() takes exactly 1 argument ({len(args)} given)")
        
        arg = args[0]
        
        # Only accept Python values (int or str)
        if not arg.is_python_value():
            raise TypeError(f"char() only accepts Python int or str values, got {arg.type_hint}")
        
        py_value = arg.get_python_value()
        
        # Handle str type
        if isinstance(py_value, str):
            if len(py_value) == 0:
                # Empty string -> '\0' (null terminator)
                char_value = 0
            else:
                # Non-empty string -> first character
                char_value = ord(py_value[0])
        # Handle int type
        elif isinstance(py_value, int):
            # Int directly converted to i8
            char_value = py_value
        else:
            # Reject other types
            raise TypeError(f"char() only accepts int or str, got {type(py_value).__name__}")

        from .python_type import PythonType
        python_type = PythonType.wrap(char_value, is_constant=True)
        return wrap_value(char_value, kind="python", type_hint=python_type)


class seq(BuiltinFunction):
    """seq(end) or seq(start, end) - Iterator for integer sequences
    
    Uses yield-based implementation for efficient inlining.
    Returns Python range for constant iteration (compile-time unrolling).
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'seq'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle seq(end) or seq(start, end) call
        
        Returns:
        - Python range for constant arguments (enables compile-time unrolling)
        - Calls yield-based counter_yield or seq_yield for runtime iteration
        
        Note: step parameter is not supported yet for performance reasons.
        """
        from .python_type import PythonType
        from ..builtin_entities import get_builtin_entity
        from ..std.seq import counter, counter_range, counter_range_step

        def all_python(args):
            return all(arg.is_python_value() or arg is None for arg in args)

        if all_python(args):
            vals = [arg.get_python_value() for arg in args]
            py_range = range(*vals)
            return wrap_value(
                py_range,
                kind="python",
                type_hint=PythonType.wrap(py_range, is_constant=True)
            )
        
        # Parse arguments
        if len(args) == 1:
            return counter.handle_call(visitor, args, node)
        elif len(args) == 2:
            return counter_range.handle_call(visitor, args, node)
        elif len(args) == 3:
            return counter_range_step.handle_call(visitor, args, node)
            raise NotImplementedError("Runtime seq(start, end, step) not yet supported")
        else:
            raise ValueError("seq() takes 1 to 3 arguments")


class consume(BuiltinFunction):
    """consume(t: linear) -> void
    
    Consume a linear token, marking it as destroyed.
    The token variable becomes invalid after consumption.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'consume'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle consume(token) call
        
        consume() is a no-op at IR level. The actual consumption happens in visit_Call
        when it calls _transfer_linear_ownership on the argument.
        
        We just validate the argument and return void.
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (ownership already transferred)
            node: ast.Call node
        
        Returns:
            void
        """
        from .types import void
        
        if len(args) != 1:
            raise TypeError("consume() takes exactly 1 argument")
        
        # Validate argument is a linear type
        arg_value = args[0]
        if not hasattr(arg_value, 'type_hint') or not arg_value.type_hint:
            raise TypeError(f"consume() argument must have type information (line {node.lineno})")
        
        # Check if it's a linear type (optional validation)
        if hasattr(arg_value.type_hint, 'is_linear') and not arg_value.type_hint.is_linear():
            raise TypeError(f"consume() requires a linear type argument (line {node.lineno})")
        
        # Return void - no LLVM code generated, ownership already transferred
        return wrap_value(None, kind='python', type_hint=void)
    
    @classmethod
    def _parse_linear_path(cls, node: ast.AST):
        """Parse variable name and index path from AST node
        
        Examples:
            t -> ('t', ())
            t[0] -> ('t', (0,))
            t[1][0] -> ('t', (1, 0))
        
        Returns:
            (var_name, path_tuple)
        """
        path = []
        current = node
        
        # Walk backwards through subscript chain to build path
        while isinstance(current, ast.Subscript):
            # Extract index (must be constant integer)
            if isinstance(current.slice, ast.Constant):
                idx = current.slice.value
                if not isinstance(idx, int):
                    raise TypeError(f"consume() requires integer index, got {type(idx).__name__}")
                path.insert(0, idx)
            elif isinstance(current.slice, ast.Index):  # Python < 3.9 compatibility
                if isinstance(current.slice.value, ast.Constant):
                    idx = current.slice.value.value
                    if not isinstance(idx, int):
                        raise TypeError(f"consume() requires integer index, got {type(idx).__name__}")
                    path.insert(0, idx)
                else:
                    raise TypeError("consume() requires constant integer index")
            else:
                raise TypeError("consume() requires constant integer index")
            
            current = current.value
        
        # Base must be a variable name
        if not isinstance(current, ast.Name):
            raise TypeError(f"consume() requires variable name, got {type(current).__name__}")
        
        return current.id, tuple(path)


class assume(BuiltinFunction):
    """assume(value, pred1, pred2, "tag1", "tag2", ...) -> refined[T, pred1, pred2, "tag1", "tag2"]
    
    Create a refined type instance without checking the predicates.
    Supports multiple predicates and tags.
    
    Use when:
    - You know the value satisfies the predicates
    - Performance is critical and validation is unnecessary
    - The predicates have already been checked elsewhere
    
    Example:
        def is_positive(x: i32) -> bool:
            return x > 0
        
        x = assume(5, is_positive)  # refined[i32, is_positive]
        ptr = assume(p, "owned")    # refined[ptr[T], "owned"]
        y = assume(10, is_positive, "validated")  # refined[i32, is_positive, "validated"]
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'assume'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle assume(value, pred1, pred2, "tag1", ...) call
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs
            node: ast.Call node
            
        Returns:
            ValueRef containing refined type value
        """
        from .refined import RefinedType
        from ..valueref import ValueRef
        
        if len(args) < 2:
            raise TypeError("assume() requires at least 2 arguments")
        if len(node.args) < 2:
            raise TypeError("assume() requires at least 2 arguments")
        
        # Parse predicates and tags from AST nodes
        user_globals = {}
        if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
            user_globals = visitor.ctx.user_globals
        elif hasattr(visitor, 'user_globals'):
            user_globals = visitor.user_globals
        
        # Check if last argument is a predicate (multi-param refined form)
        # But only if there are no string tags mixed in
        last_arg_node = node.args[-1]
        is_multi_param_form = False
        multi_param_predicate = None
        
        # First check: no string constants in args (except potentially in multi-param case)
        has_string_tag = any(
            isinstance(node.args[i], ast.Constant) and isinstance(node.args[i].value, str)
            for i in range(1, len(node.args))
        )
        
        # Only consider multi-param form if:
        # 1. Last arg is a callable predicate
        # 2. No string tags are present
        # 3. Predicate param count matches value count
        if not has_string_tag and isinstance(last_arg_node, ast.Name):
            func_name = last_arg_node.id
            if func_name in user_globals:
                maybe_pred = user_globals[func_name]
                if hasattr(maybe_pred, '_original_func'):
                    maybe_pred = maybe_pred._original_func
                if callable(maybe_pred):
                    # Check if it's a multi-param predicate
                    import inspect
                    sig = inspect.signature(maybe_pred)
                    params = list(sig.parameters.values())
                    # Multi-param if: predicate has N>1 params AND we have N values
                    if len(params) > 1 and len(args) == len(params) + 1:
                        is_multi_param_form = True
                        multi_param_predicate = maybe_pred
        
        if is_multi_param_form:
            # Form: assume(val1, val2, ..., pred)
            # All args except last are values
            value_args = args[:-1]
            predicates = [multi_param_predicate]
            tags = []
            
            # Get base types from predicate signature
            import inspect
            sig = inspect.signature(multi_param_predicate)
            params = list(sig.parameters.values())
            
            from ..type_resolver import TypeResolver
            type_resolver = TypeResolver(visitor.ctx)
            param_types = []
            for param in params:
                pc_type = type_resolver.parse_annotation(param.annotation)
                param_types.append(pc_type)
            
            # Promote Python values if needed
            promoted_args = []
            for i, (value_arg, target_type) in enumerate(zip(value_args, param_types)):
                if isinstance(value_arg, ValueRef) and value_arg.is_python_value():
                    value_arg = visitor.type_converter._promote_python_to_pc(
                        value_arg.get_python_value(), target_type)
                promoted_args.append(value_arg)
            
            # Create refined type from multi-param predicate
            refined_type = cls._create_refined_type_from_predicate(
                multi_param_predicate, param_types, visitor)
            
            # Call constructor with all values
            return refined_type.handle_call(visitor, promoted_args, node)
        
        else:
            # Form: assume(value, pred1, pred2, "tag1", "tag2", ...)
            value_arg = args[0]
            
            predicates = []
            tags = []
            
            for i in range(1, len(node.args)):
                arg_node = node.args[i]
                
                # Check if it's a string literal (tag)
                if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                    tags.append(arg_node.value)
                # Check if it's a name (predicate function)
                elif isinstance(arg_node, ast.Name):
                    func_name = arg_node.id
                    
                    if func_name in user_globals:
                        predicate_func = user_globals[func_name]
                        if hasattr(predicate_func, '_original_func'):
                            predicate_func = predicate_func._original_func
                        if callable(predicate_func):
                            predicates.append(predicate_func)
                        else:
                            raise TypeError(f"assume() constraint must be a callable predicate, got {type(predicate_func)}")
                    else:
                        raise TypeError(f"assume() constraint '{func_name}' not found in globals")
                else:
                    raise TypeError(f"assume() constraint must be a predicate function or string tag, got {ast.dump(arg_node)}")
            
            if len(predicates) == 0 and len(tags) == 0:
                raise TypeError("assume() requires at least one predicate or tag")
            
            # Get base type from value
            if not isinstance(value_arg, ValueRef):
                raise TypeError("assume() value must be a ValueRef")
            
            # If value is Python value, we need to infer its type
            if value_arg.is_python_value():
                py_value = value_arg.get_python_value()
                
                if len(predicates) > 0:
                    # Get type from first predicate's first parameter
                    import inspect
                    sig = inspect.signature(predicates[0])
                    params = list(sig.parameters.values())
                    if len(params) > 0:
                        from ..type_resolver import TypeResolver
                        type_resolver = TypeResolver(visitor.ctx)
                        base_type = type_resolver.parse_annotation(params[0].annotation)
                    else:
                        raise TypeError(f"Predicate {predicates[0].__name__} has no parameters")
                else:
                    # No predicates, only tags - use default type inference
                    # int -> i32, float -> f64, bool -> i32
                    if isinstance(py_value, bool):
                        from .types import i32
                        base_type = i32
                    elif isinstance(py_value, int):
                        from .types import i32
                        base_type = i32
                    elif isinstance(py_value, float):
                        from .types import f64
                        base_type = f64
                    else:
                        raise TypeError(f"assume() with Python value and only tags: cannot infer type for {type(py_value).__name__}")
                
                # Promote Python value to PC type
                value_arg = visitor.type_converter._promote_python_to_pc(py_value, base_type)
            else:
                if value_arg.type_hint is None:
                    raise TypeError("assume() value must have type information")
                base_type = value_arg.type_hint
            
            # Create refined type: refined[base_type, pred1, pred2, "tag1", "tag2"]
            from .refined import refined
            
            # Build the refined type
            refined_type = cls._create_refined_type(base_type, predicates, tags, visitor)
            
            # Call constructor with value
            return refined_type.handle_call(visitor, [value_arg], node)
    
    @classmethod
    def _create_refined_type_from_predicate(cls, predicate, param_types, visitor):
        """Create refined type from multi-param predicate"""
        from .refined import RefinedType
        from .struct import create_struct_type
        import inspect
        
        sig = inspect.signature(predicate)
        params = list(sig.parameters.values())
        param_names = [p.name for p in params]
        
        # Create struct type for multi-param
        struct_type = create_struct_type(param_types, param_names)
        
        # Create refined type class
        class_name = f"RefinedType_{predicate.__name__}"
        new_refined_type = type(class_name, (RefinedType,), {
            '_base_type': None,  # Multi-param has no single base type
            '_predicates': [predicate],
            '_tags': [],
            '_struct_type': struct_type,
            '_field_types': param_types,
            '_field_names': param_names,
            '_param_types': param_types,
            '_param_names': param_names,
            '_is_refined': True,
            '_is_single_param': False,
        })
        
        return new_refined_type
    
    @classmethod
    def _create_refined_type(cls, base_type, predicates, tags, visitor):
        """Create refined type from base + predicates + tags"""
        from .refined import RefinedType
        
        # Create name
        base_name = base_type.get_name() if hasattr(base_type, 'get_name') else str(base_type)
        pred_names = [p.__name__ for p in predicates]
        tag_names = [f'"{t}"' for t in tags]
        all_names = [base_name] + pred_names + tag_names
        class_name = f"RefinedType_{'_'.join(str(n).replace('[', '_').replace(']', '_').replace(',', '_').replace(' ', '').replace('"', '') for n in all_names)}"
        
        new_refined_type = type(class_name, (RefinedType,), {
            '_base_type': base_type,
            '_predicates': predicates,
            '_tags': tags,
            '_struct_type': None,
            '_field_types': [base_type],
            '_field_names': ['value'],
            '_param_types': [base_type],
            '_param_names': ['value'],
            '_is_refined': True,
            '_is_single_param': True,
        })
        
        return new_refined_type


class refine(BuiltinFunction):
    """refine(value, pred1, pred2, "tag1", "tag2", ...) -> yield refined[T, pred1, pred2, "tag1", "tag2"]
    
    Check all predicates and yield refined type if all satisfied.
    Must be used in for-else loop.
    
    Example:
        for x in refine(5, is_positive):
            # x is refined[i32, is_positive] type
            use(x)
        else:
            # Predicate failed
            handle_error()
        
        for p in refine(ptr, "owned"):
            # p is refined[ptr[T], "owned"] type
            use(p)
        else:
            # never happens (tags are always true)
            pass
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'refine'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle refine(value, pred1, "tag1", ...) call
        
        This function is special - it's a yield function that should be
        handled by the yield inlining system.
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs (NOT used - we need AST)
            node: ast.Call node
            
        Returns:
            ValueRef with yield inline info for for-loop processing
        """
        # Create the inline function AST
        func_ast = cls._create_refine_inline_ast(node, visitor)
        
        # Return a placeholder with yield inline info
        from ..valueref import wrap_value, ValueRef
        from .python_type import PythonType
        
        def refine_placeholder(*args, **kwargs):
            raise RuntimeError("refine() must be used in a for loop and will be inlined")
        
        refine_placeholder._is_yield_generated = True
        refine_placeholder.__name__ = 'refine'
        refine_placeholder._original_ast = func_ast
        
        # Create ValueRef with yield inline info
        result = ValueRef('python', refine_placeholder, PythonType(refine_placeholder))
        result._yield_inline_info = {
            'func_obj': refine_placeholder,
            'original_ast': func_ast,
            'call_node': node,
            'call_args': node.args
        }
        
        return result
    
    @classmethod
    def _create_refine_inline_ast(cls, call_node: ast.Call, visitor) -> ast.FunctionDef:
        """Create AST for inline refine function
        
        Supports two forms:
        1. Single-param: refine(value, pred1, pred2, "tag1", "tag2")
           -> if pred1(value) and pred2(value): yield assume(value, pred1, pred2, "tag1", "tag2")
        
        2. Multi-param: refine(val1, val2, pred)
           -> if pred(val1, val2): yield assume(val1, val2, pred)
        
        Args:
            call_node: The refine() call node
            visitor: AST visitor for context
            
        Returns:
            ast.FunctionDef for the inline function
        """
        import copy
        
        if len(call_node.args) < 2:
            raise TypeError("refine() requires at least 2 arguments")
        
        # Check if last arg is a predicate (multi-param form detection)
        user_globals = {}
        if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
            user_globals = visitor.ctx.user_globals
        elif hasattr(visitor, 'user_globals'):
            user_globals = visitor.user_globals
        
        last_arg_node = call_node.args[-1]
        is_multi_param_form = False
        
        # Check for string tags
        has_string_tag = any(
            isinstance(call_node.args[i], ast.Constant) and isinstance(call_node.args[i].value, str)
            for i in range(1, len(call_node.args))
        )
        
        # Only consider multi-param if no string tags and last arg is callable
        if not has_string_tag and isinstance(last_arg_node, ast.Name):
            func_name = last_arg_node.id
            if func_name in user_globals:
                maybe_pred = user_globals[func_name]
                if hasattr(maybe_pred, '_original_func'):
                    maybe_pred = maybe_pred._original_func
                if callable(maybe_pred):
                    import inspect
                    sig = inspect.signature(maybe_pred)
                    params = list(sig.parameters.values())
                    # Multi-param if predicate has N>1 params and we have N+1 args
                    if len(params) > 1 and len(call_node.args) == len(params) + 1:
                        is_multi_param_form = True
        
        if is_multi_param_form:
            # Multi-param form: refine(val1, val2, ..., pred)
            # Generate: if pred(val1, val2, ...): yield assume(val1, val2, ..., pred)
            value_args = call_node.args[:-1]
            pred_node = call_node.args[-1]
            
            # Build predicate call: pred(val1, val2, ...)
            pred_call = ast.Call(
                func=copy.deepcopy(pred_node),
                args=[copy.deepcopy(arg) for arg in value_args],
                keywords=[]
            )
            
            # Build assume call with all args
            assume_call = ast.Call(
                func=ast.Name(id='assume', ctx=ast.Load()),
                args=[copy.deepcopy(arg) for arg in call_node.args],
                keywords=[]
            )
            
            yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
            
            # Build if statement
            if_stmt = ast.If(
                test=pred_call,
                body=[yield_stmt],
                orelse=[]
            )
            
            func_def = ast.FunctionDef(
                name='__refine_inline__',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None
                ),
                body=[if_stmt],
                decorator_list=[],
                returns=None,
                lineno=call_node.lineno,
                col_offset=call_node.col_offset
            )
        
        else:
            # Single-param form: refine(value, pred1, pred2, "tag1", "tag2")
            value_arg = call_node.args[0]
            
            # Rest are predicates/tags
            predicate_nodes = []
            all_constraint_nodes = []
            
            for i in range(1, len(call_node.args)):
                arg_node = call_node.args[i]
                all_constraint_nodes.append(arg_node)
                
                # Only add to predicate checks if it's NOT a string constant
                if not (isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str)):
                    predicate_nodes.append(arg_node)
            
            # Build predicate checks: pred1(value) and pred2(value) and ...
            if len(predicate_nodes) == 0:
                # No predicates, only tags - always true
                # Just yield assume(value, "tag1", "tag2")
                assume_call = ast.Call(
                    func=ast.Name(id='assume', ctx=ast.Load()),
                    args=[copy.deepcopy(arg) for arg in call_node.args],
                    keywords=[]
                )
                
                yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
                
                func_def = ast.FunctionDef(
                    name='__refine_inline__',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None
                    ),
                    body=[yield_stmt],
                    decorator_list=[],
                    returns=None,
                    lineno=call_node.lineno,
                    col_offset=call_node.col_offset
                )
            else:
                # Build predicate calls
                predicate_calls = []
                for pred_node in predicate_nodes:
                    pred_call = ast.Call(
                        func=copy.deepcopy(pred_node),
                        args=[copy.deepcopy(value_arg)],
                        keywords=[]
                    )
                    predicate_calls.append(pred_call)
                
                # Combine with 'and'
                if len(predicate_calls) == 1:
                    combined_test = predicate_calls[0]
                else:
                    combined_test = ast.BoolOp(
                        op=ast.And(),
                        values=predicate_calls
                    )
                
                # Build assume call with ALL constraints (predicates + tags)
                assume_call = ast.Call(
                    func=ast.Name(id='assume', ctx=ast.Load()),
                    args=[copy.deepcopy(arg) for arg in call_node.args],
                    keywords=[]
                )
                
                yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
                
                # Build if statement
                if_stmt = ast.If(
                    test=combined_test,
                    body=[yield_stmt],
                    orelse=[]
                )
                
                func_def = ast.FunctionDef(
                    name='__refine_inline__',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None
                    ),
                    body=[if_stmt],
                    decorator_list=[],
                    returns=None,
                    lineno=call_node.lineno,
                    col_offset=call_node.col_offset
                )
        
        ast.fix_missing_locations(func_def)
        return func_def



