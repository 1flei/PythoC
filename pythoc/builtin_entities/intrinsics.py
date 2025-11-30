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
    """assume(a, b, ..., Pred) -> refined[Pred]
    
    Create a refined type instance without checking the predicate.
    This is equivalent to refined[Pred](a, b, ...) constructor.
    
    Use when:
    - You know the values satisfy the predicate
    - Performance is critical and validation is unnecessary
    - The predicate has already been checked elsewhere
    
    Example:
        def is_positive(x: i32) -> bool:
            return x > 0
        
        x = assume(5, is_positive)  # Creates refined type without checking
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'assume'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle assume(a, b, ..., predicate) call
        
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
            raise TypeError("assume() requires at least 2 arguments (values and predicate)")
        if len(node.args) < 2:
            raise TypeError("assume() requires at least 2 arguments (values and predicate)")
        
        # Value arguments (all except last)
        value_args = args[:-1]
        
        # Get predicate function from AST node (not from evaluated args)
        # This allows us to get the original Python function
        predicate_node = node.args[-1]
        predicate_func = None
        
        if isinstance(predicate_node, ast.Name):
            # Simple name - look up in user_globals
            func_name = predicate_node.id
            # Try visitor.ctx.user_globals first, then visitor.user_globals
            user_globals = {}
            if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
                user_globals = visitor.ctx.user_globals
            elif hasattr(visitor, 'user_globals'):
                user_globals = visitor.user_globals
            
            if func_name in user_globals:
                predicate_func = user_globals[func_name]
                # Get original function if it's a compiled wrapper
                if hasattr(predicate_func, '_original_func'):
                    predicate_func = predicate_func._original_func
        
        if predicate_func is None or not callable(predicate_func):
            raise TypeError(f"assume() last argument must be a function (got {ast.unparse(predicate_node)})")
        
        # Create refined type from predicate
        from .refined import refined
        refined_factory = refined[predicate_func]
        
        if isinstance(refined_factory, type) and issubclass(refined_factory, RefinedType):
            refined_type = refined_factory
        else:
            raise TypeError(f"Failed to create refined type from predicate {predicate_func}")
        
        # Call the refined type constructor with the value arguments
        return refined_type.handle_call(visitor, value_args, node)


class refine(BuiltinFunction):
    """refine(a, b, ..., Pred) -> yield refined[Pred]
    
    Check predicate and yield refined type if satisfied.
    Must be used in for-else loop.
    
    Example:
        for x in refine(5, is_positive):
            # x is refined[is_positive] type
            use(x)
        else:
            # Predicate failed
            handle_error()
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'refine'
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call):
        """Handle refine(a, b, ..., predicate) call
        
        This function is special - it's a yield function that should be
        handled by the yield inlining system. We create a Python-level
        generator function that will be inlined.
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs (NOT used - we need AST)
            node: ast.Call node
            
        Returns:
            ValueRef with yield inline info for for-loop processing
        """
        # For refine, we need to create a yield function dynamically
        # that checks the predicate and yields the refined value
        
        # Since refine must be inlined, we create a placeholder that
        # triggers inline expansion in the for loop visitor
        
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
        
        Translates:
            refine(a, b, predicate)
        
        Into:
            def __refine_inline__():
                if predicate(a, b):
                    yield assume(a, b, predicate)
        
        Args:
            call_node: The refine() call node
            visitor: AST visitor for context
            
        Returns:
            ast.FunctionDef for the inline function
        """
        import copy
        
        if len(call_node.args) < 2:
            raise TypeError("refine() requires at least 2 arguments (values and predicate)")
        
        # Extract arguments: values and predicate
        value_args = call_node.args[:-1]
        predicate_arg = call_node.args[-1]
        
        # Create the inline function:
        # def __refine_inline__(<inferred return type>):
        #     if predicate(values...):
        #         yield assume(values..., predicate)
        
        # Build predicate call: predicate(a, b, ...)
        predicate_call = ast.Call(
            func=copy.deepcopy(predicate_arg),
            args=[copy.deepcopy(arg) for arg in value_args],
            keywords=[]
        )
        
        # Build assume call: assume(a, b, ..., predicate)
        assume_call = ast.Call(
            func=ast.Name(id='assume', ctx=ast.Load()),
            args=[copy.deepcopy(arg) for arg in call_node.args],  # All args including predicate
            keywords=[]
        )
        
        # Build yield statement: yield assume(...)
        yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
        
        # Build if statement: if predicate(...): yield assume(...)
        if_stmt = ast.If(
            test=predicate_call,
            body=[yield_stmt],
            orelse=[]
        )
        
        # Build function def
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
            returns=None,  # Will be inferred from yield type
            lineno=call_node.lineno,
            col_offset=call_node.col_offset
        )
        
        # Fix missing locations
        ast.fix_missing_locations(func_def)
        
        return func_def




