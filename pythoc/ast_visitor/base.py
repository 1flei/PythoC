"""
Base visitor class with initialization and core helper methods
"""

import ast
import builtins
from typing import Optional, Any, List, Tuple
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..type_converter import TypeConverter
from ..context import (
    VariableInfo, VariableRegistry,
    CompilationContext, PC_TYPE_MAP
)
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
    TYPE_REGISTRY,
    SIGNED_INT_TYPES,
    UNSIGNED_INT_TYPES,
    INT_TYPES,
    FLOAT_TYPES,
    NUMERIC_TYPES,
    is_signed_int,
    is_unsigned_int,
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..type_resolver import TypeResolver
from ..logger import logger


class LLVMIRVisitor(ast.NodeVisitor):
    """Enhanced AST visitor that generates LLVM IR using llvmlite
    
    Now includes context management for better type tracking and scope handling.
    """
    
    def __init__(self, module: ir.Module, builder: ir.IRBuilder, func_type_hints: dict, struct_types: dict = None, source_globals: dict = None, compiler=None, user_globals: dict = None):
        self.module = module
        self.builder = builder
        self.func_type_hints = func_type_hints
        
        # New context management system
        self.ctx = CompilationContext(module, builder, user_globals=user_globals)
        self.ctx.struct_types = struct_types or {}
        self.ctx.source_globals = source_globals or {}
        
        # Type resolver for unified type annotation parsing
        # Pass visitor=self to allow type resolver to access local Python type variables
        self.type_resolver = TypeResolver(module.context, user_globals=user_globals, visitor=self)
        
        # Type converter for centralized type conversion
        self.type_converter = TypeConverter(self)
        
        # Linear type tracking - now integrated with VariableInfo
        self.scope_depth = 0  # Track scope depth for loop restrictions
        
        # Backward compatibility aliases
        self.current_function = None
        self.label_counter = 0
        self.struct_types = self.ctx.struct_types
        self.source_globals = self.ctx.source_globals
        self.compiler = compiler
        self.loop_stack = self.ctx.loop_stack
        
    def get_next_label(self, prefix="label"):
        """Generate unique label names"""
        label = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return label
    
    def get_llvm_type(self, type_hint):
        """Convert type hint to LLVM type"""
        from ..builtin_entities import get_llvm_type_by_name, BuiltinEntity
        
        if isinstance(type_hint, type) and issubclass(type_hint, BuiltinEntity):
            if type_hint.can_be_type():
                return type_hint.get_llvm_type(self.module.context)
        
        if isinstance(type_hint, str):
            llvm_type = get_llvm_type_by_name(type_hint)
            if llvm_type is not None:
                return llvm_type
        
        if type_hint in TYPE_MAP:
            type_name = TYPE_MAP[type_hint]
            llvm_type = get_llvm_type_by_name(type_name)
            if llvm_type is not None:
                return llvm_type
        
        if isinstance(type_hint, ptr):
            pointee_type = self.get_llvm_type(type_hint.pointee_type)
            return ir.PointerType(pointee_type)
        
        return ir.IntType(32)
    
    def get_pc_type_from_annotation(self, annotation) -> Optional[Any]:
        """Convert type annotation to builtin type class"""
        return self.type_resolver.parse_annotation(annotation)
    
    def infer_pc_type_from_value(self, value: ValueRef) -> Optional[Any]:
        """Infer PC type from a ValueRef
        
        For Python values, returns the default promote type.
        For PC values, returns the type_hint if available.
        """
        if hasattr(value, 'type_hint') and value.type_hint:
            return value.type_hint
        
        # For Python values without type_hint, infer from Python type
        if value.is_python_value():
            try:
                from ..type_converter import TypeConverter
                python_val = value.get_python_value()
                return TypeConverter.infer_default_pc_type_from_python(python_val)
            except TypeError:
                # For Python type objects (like i32, ptr[T]), return None
                return None
        
        # Do not infer from LLVM; require explicit type hints
        return None
    
    def _infer_pc_type_from_llvm_type(self, llvm_type: ir.Type) -> Optional[Any]:
        """No longer supported: LLVM->PC inference disabled"""
        return None
    
    def declare_variable(self, name: str, type_hint: Any, alloca: Optional[ir.AllocaInstr] = None,
                        source: str = "unknown", line_number: Optional[int] = None,
                        is_parameter: bool = False, allow_redeclare: bool = False,
                        value_ref: Optional[Any] = None) -> VariableInfo:
        """Declare a variable with PC type information
        
        Args:
            name: Variable name
            type_hint: PC type hint
            alloca: Storage location (None for Python constants)
            source: Source of declaration
            line_number: Line number for debugging
            is_parameter: Whether this is a function parameter
            allow_redeclare: Allow redeclaration in same scope
            value_ref: Optional ValueRef to store directly
        """
        from ..valueref import wrap_value

        pc_type = type_hint
        # Create ValueRef if not provided
        if value_ref is None and alloca is not None:
            # Check if this is a function pointer type
            from ..builtin_entities.func import func
            if isinstance(pc_type, type) and issubclass(pc_type, func):
                # Store alloca directly, func.handle_call will load it when needed
                value_ref = wrap_value(
                    kind='address',
                    value=alloca,
                    type_hint=pc_type,
                    address=alloca
                )
            else:
                # For regular PC variables with alloca, create an address ValueRef
                value_ref = wrap_value(
                    kind='address',
                    value=alloca,  # Store alloca as value for now
                    type_hint=pc_type,
                    address=alloca
                )
        
        var_info = VariableInfo(
            name=name,
            value_ref=value_ref,
            alloca=alloca,
            source=source,
            line_number=line_number,
            is_parameter=is_parameter,
        )
        
        if self.ctx.var_registry.is_declared_in_current_scope(name):
            existing = self.ctx.var_registry.lookup(name)
            if existing:
                existing.alloca = alloca
                existing.value_ref = value_ref
                return existing
        
        self.ctx.var_registry.declare(var_info, allow_shadow=True)
        
        # Initialize linear states for all linear paths in the type
        # For parameters, initialize as 'active' (passed in with ownership)
        # For other variables, initialize as 'undefined' (not yet assigned)
        if self._is_linear_type(type_hint):
            initial_state = 'active' if is_parameter else 'undefined'
            self._init_linear_states(var_info, type_hint, initial_state=initial_state)
        
        return var_info

    def lookup_variable(self, name: str) -> Optional[VariableInfo]:
        """Look up variable or function in context registry (unified)"""
        # 1. First check variable registry
        var_info = self.ctx.var_registry.lookup(name)
        if var_info:
            return var_info
        
        # 2. Check user globals for objects with handle_call (like ExternFunctionWrapper)
        # This must come BEFORE checking global functions, because extern functions
        # may be declared as LLVM functions but should still use their wrapper's handle_call
        if self.ctx.user_globals and name in self.ctx.user_globals:
            python_obj = self.ctx.user_globals[name]
            if hasattr(python_obj, 'handle_call') and callable(python_obj.handle_call):
                from ..valueref import wrap_value
                from ..builtin_entities.python_type import PythonType
                
                # For @compile functions, get type hints from function annotations
                type_hint = PythonType.wrap(python_obj, is_constant=True)  # Wrap with PythonType for unified subscript handling
                if hasattr(python_obj, '_is_compiled') and python_obj._is_compiled:
                    # Get type hints from function annotations
                    if hasattr(python_obj, '__annotations__'):
                        annotations = python_obj.__annotations__
                        param_type_hints = {}
                        return_type_hint = None
                        
                        for key, value in annotations.items():
                            if key == 'return':
                                return_type_hint = value
                            else:
                                param_type_hints[key] = value
                        
                        # Create func type if we have return type
                        # Otherwise keep the wrapper as type_hint (it has handle_call)
                        if return_type_hint is not None:
                            from ..builtin_entities import func
                            # Build param types list in order
                            param_types = list(param_type_hints.values())
                            if param_types:
                                type_hint = func[param_types, return_type_hint]
                            else:
                                type_hint = func[[], return_type_hint]
                
                return VariableInfo(
                    name=name,
                    value_ref = wrap_value(value=python_obj, kind='python', type_hint=type_hint),
                    alloca=None,
                    source="python_global",
                    is_global=True,
                    is_mutable=False,
                )
        
        # 3. Then check if it's a global function
        try:
            func = self.module.get_global(name)
            if isinstance(func, ir.Function):
                # Get function type hints from unified registry
                from ..registry import get_unified_registry
                func_type_hints = get_unified_registry().get_function_type_hints(name)
                
                # Create a wrapper object with handle_call for unified call handling
                class LLVMFunctionWrapper:
                    def __init__(self, func_name, func_type_hints):
                        self.func_name = func_name
                        self.func_type_hints = func_type_hints
                    
                    def handle_call(self, visitor, args, node):
                        """Handle calls to LLVM functions"""
                        # Get the function from the module
                        try:
                            ir_func = visitor.module.get_global(self.func_name)
                        except KeyError:
                            raise NameError(f"Function '{self.func_name}' not found in module")
                        
                        # Extract parameter types
                        if not self.func_type_hints or 'params' not in self.func_type_hints:
                            raise TypeError(f"No parameter type hints found for function '{self.func_name}'")
                        
                        param_llvm_types = []
                        for param_name, pc_param_type in self.func_type_hints['params'].items():
                            if hasattr(pc_param_type, 'get_llvm_type'):
                                # Try with module_context first (for struct types), fallback to no-arg
                                try:
                                    param_llvm_types.append(pc_param_type.get_llvm_type(visitor.module.context))
                                except TypeError:
                                    param_llvm_types.append(pc_param_type.get_llvm_type())
                            else:
                                raise TypeError(f"Invalid parameter type hint for '{param_name}' in function '{self.func_name}'")
                        
                        return_type_hint = self.func_type_hints.get('return', None)
                        
                        # Use visitor's _perform_call with pre-evaluated args
                        return visitor._perform_call(node, ir_func, param_llvm_types, return_type_hint, evaluated_args=args)
                
                wrapper = LLVMFunctionWrapper(name, func_type_hints)
                
                # Create a pseudo VariableInfo for the function
                from ..valueref import wrap_value
                return VariableInfo(
                    name=name,
                    value_ref=wrap_value(
                        kind='python',
                        value=wrapper,
                        type_hint=func_type_hints,
                    ),
                    alloca=func,  # Store function as "alloca" for backward compat
                    source="function",
                    is_global=True,
                    is_mutable=False,
                )
        except KeyError:
            pass
        
        # 3. Check user globals for Python objects
        if self.ctx.user_globals and name in self.ctx.user_globals:
            python_obj = self.ctx.user_globals[name]
            
            # Skip PC builtin entities and struct classes - they should be handled by type system, not as Python objects
            from ..builtin_entities import BuiltinEntity, BuiltinType, BuiltinFunction
            if isinstance(python_obj, type):
                try:
                    if issubclass(python_obj, BuiltinEntity):
                        # BuiltinType and BuiltinFunction have their own handle_call/handle_subscript
                        # Return them directly as type_hint (not wrapped in PythonType)
                        if issubclass(python_obj, (BuiltinType, BuiltinFunction)):
                            # Create a ValueRef to hold the type_hint
                            from ..valueref import wrap_value
                            return VariableInfo(
                                name=name,
                                value_ref=wrap_value(
                                    kind='python',
                                    value=python_obj,
                                    type_hint=python_obj,  # The class itself as type_hint
                                ),
                                alloca=None,
                                source="builtin_entity",
                                is_global=True,
                                is_mutable=False,
                            )
                        # Other BuiltinEntity subclasses can be wrapped
                except TypeError:
                    # Not a class, continue
                    pass
                
                # Check if it's a struct class (marked by @compile decorator)
                if hasattr(python_obj, '_is_struct') and python_obj._is_struct:
                    # This is a PC struct type - don't wrap it as Python object
                    return None
            
            # Wrap Python object
            from ..valueref import ValueRef, wrap_value
            
            # If python_obj is already a ValueRef (like nullptr), return it directly
            if isinstance(python_obj, ValueRef):
                return VariableInfo(
                    name=name,
                    value_ref=python_obj,
                    alloca=None,
                    source="python_global",
                    is_global=True,
                    is_mutable=False,
                )
            
            from ..builtin_entities.python_type import PythonType
            python_type = PythonType.wrap(python_obj)
            
            # Create pseudo VariableInfo for Python object
            return VariableInfo(
                name=name,
                value_ref=wrap_value(
                    kind='python',
                    value=python_obj,
                    type_hint=python_type,
                ),
                alloca=None,  # No LLVM alloca for Python objects
                source="python_global",
                is_global=True,
                is_mutable=False,
            )
        
        return None
    
    def get_variable_alloca(self, name: str) -> Optional[ir.AllocaInstr]:
        """Get variable alloca from registry"""
        var_info = self.ctx.var_registry.lookup(name)
        return var_info.alloca if var_info else None
    
    def has_variable(self, name: str) -> bool:
        """Check if variable exists in registry"""
        return self.ctx.var_registry.lookup(name) is not None
    
    def store_variable(self, name: str, value: ir.Value):
        """Store value to variable"""
        var_info = self.ctx.var_registry.lookup(name)
        if var_info:
            self.builder.store(value, var_info.alloca)
        else:
            raise NameError(f"Variable '{name}' not defined")
    
    def _store_python_constant(self, name: str, value: ValueRef):
        """Store Python constant value in VariableInfo.value_ref"""
        var_info = self.lookup_variable(name)
        if var_info:
            var_info.value_ref = value
    
    def _get_python_constant(self, name: str) -> Optional[ValueRef]:
        """Retrieve Python constant value from VariableInfo.value_ref"""
        var_info = self.lookup_variable(name)
        if var_info and var_info.is_python_constant:
            return var_info.value_ref
        return None
    
    def wrap_value_with_pc_type(self, ir_value: ir.Value, pc_type: Optional[Any] = None,
                                kind: str = "value", source_node: Optional[ast.AST] = None) -> ValueRef:
        """Wrap IR value with PC type information"""
        # Do not infer PC type from LLVM; preserve None to surface errors upstream
        
        return wrap_value(ir_value, kind=kind, type_hint=pc_type, source_node=source_node)
    
    def visit_expression(self, expr):
        """Visit an expression and return a ValueRef preserving type hints"""
        result = self.visit(expr)
        if result is None:
            raise ValueError(f"Expression {ast.dump(expr)} returned None")
        
        # Return the result directly without tracking
        # Linear expressions will be checked at the statement level (visit_Expr)
        if isinstance(result, ValueRef):
            return result
        
        # Handle list results (from Tuple expressions)
        if isinstance(result, list):
            return result
        
        # Handle type objects (from type expressions like array[i32, 5])
        # Type objects don't have .type attribute, they ARE types
        if isinstance(result, type):
            return result

        logger.debug("Expression result", result=result)
        return result
    
    # ========================================================================
    # Linear Token Tracking
    # ========================================================================
    
    def _is_linear_type(self, type_hint) -> bool:
        """Check if a type is linear
        
        A type is linear if:
        1. It's the linear token type itself
        2. It has _is_linear attribute set to True
        3. It's a struct containing linear fields (recursively)
        """
        from ..builtin_entities import linear
        if type_hint is linear:
            return True
        # Check if it's a class with _is_linear attribute
        if isinstance(type_hint, type) and hasattr(type_hint, '_is_linear'):
            return type_hint._is_linear
        
        # Check if it's a struct with linear fields
        if isinstance(type_hint, type) and hasattr(type_hint, '_field_types'):
            field_types = type_hint._field_types
            if field_types:
                for field_type in field_types:
                    # Recursively check if any field is linear
                    if self._is_linear_type(field_type):
                        return True
        
        return False
    
    def _get_linear_paths(self, type_hint, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
        """Get all linear token paths in a type
        
        Returns list of index paths where linear tokens exist.
        
        Examples:
            linear -> [()]
            struct[ptr, linear] -> [(1,)]
            struct[struct[ptr, linear], linear] -> [(0, 1), (1,)]
        """
        from ..builtin_entities import linear
        
        if type_hint is linear:
            return [prefix]
        
        if isinstance(type_hint, type) and hasattr(type_hint, '_is_linear') and type_hint._is_linear:
            return [prefix]
        
        # Check if it's a struct with linear fields
        if isinstance(type_hint, type) and hasattr(type_hint, '_field_types'):
            field_types = type_hint._field_types
            if field_types:
                paths = []
                for i, field_type in enumerate(field_types):
                    # Recursively get paths from each field
                    field_paths = self._get_linear_paths(field_type, prefix + (i,))
                    paths.extend(field_paths)
                return paths
        
        return []
    
    def _get_linear_state(self, var_info, path: Tuple[int, ...] = ()) -> Optional[str]:
        """Get linear state at a specific path"""
        return var_info.linear_states.get(path, None)
    
    def _set_linear_state(self, var_info, path: Tuple[int, ...], state: str):
        """Set linear state at a specific path"""
        logger.debug(f"Set linear state, var_info={var_info.name}, path={path}, state={state}")
        var_info.linear_states[path] = state
    
    def _init_linear_states(self, var_info, type_hint, initial_state: str = 'undefined'):
        """Initialize linear states for all linear paths in a type"""
        paths = self._get_linear_paths(type_hint)
        for path in paths:
            self._set_linear_state(var_info, path, initial_state)
        if paths:
            var_info.linear_scope_depth = self.scope_depth
            logger.debug(f"Initialized linear states for '{var_info.name}': {paths} -> {initial_state}")
    
    def _parse_subscript_path(self, node: ast.AST) -> Tuple[str, Tuple[int, ...]]:
        """Parse variable name and index path from subscript AST node
        
        Similar to consume._parse_linear_path but for assignment targets.
        
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
                    raise TypeError(f"Subscript assignment requires integer index, got {type(idx).__name__}")
                path.insert(0, idx)
            elif hasattr(ast, 'Index') and isinstance(current.slice, ast.Index):  # Python < 3.9 compatibility
                if isinstance(current.slice.value, ast.Constant):
                    idx = current.slice.value.value
                    if not isinstance(idx, int):
                        raise TypeError(f"Subscript assignment requires integer index, got {type(idx).__name__}")
                    path.insert(0, idx)
                else:
                    raise TypeError("Subscript assignment requires constant integer index")
            else:
                raise TypeError("Subscript assignment requires constant integer index")
            
            current = current.value
        
        # Base must be a variable name
        if not isinstance(current, ast.Name):
            raise TypeError(f"Subscript assignment requires variable name, got {type(current).__name__}")
        
        return current.id, tuple(path)
    
    def _parse_lvalue_path(self, node: ast.AST) -> Tuple[str, Tuple[int, ...]]:
        """Parse variable name and index path from any lvalue AST node
        
        Supports:
        - ast.Name: direct variable access
        - ast.Subscript: array/struct subscript (with constant index)
        - ast.Attribute: struct field access (mapped to index internally)
        
        Examples:
            t -> ('t', ())
            t[0] -> ('t', (0,))
            t[1][0] -> ('t', (1, 0))
            t.field -> ('t', (field_index,))  # if type info available
        
        Returns:
            (var_name, path_tuple)
        """
        if isinstance(node, ast.Name):
            return node.id, ()
        
        if isinstance(node, ast.Subscript):
            return self._parse_subscript_path(node)
        
        if isinstance(node, ast.Attribute):
            # For attribute access, we need to map field name to index
            # First, get the base variable
            base_node = node.value
            if isinstance(base_node, ast.Name):
                var_name = base_node.id
                var_info = self.lookup_variable(var_name)
                if var_info and hasattr(var_info.type_hint, '_field_names'):
                    field_names = var_info.type_hint._field_names
                    field_name = node.attr
                    if field_name in field_names:
                        field_idx = field_names.index(field_name)
                        return var_name, (field_idx,)
                    else:
                        raise TypeError(f"Field '{field_name}' not found in type {var_info.type_hint}")
                else:
                    # Cannot determine field index without type info
                    raise TypeError(f"Cannot determine field index for attribute access without type info")
            elif isinstance(base_node, ast.Subscript):
                # Nested access like cs.fp[0]
                base_var, base_path = self._parse_subscript_path(base_node)
                var_info = self.lookup_variable(base_var)
                if var_info:
                    # Walk through the path to get the type at that position
                    current_type = var_info.type_hint
                    for idx in base_path:
                        if hasattr(current_type, '_field_types'):
                            current_type = current_type._field_types[idx]
                        else:
                            raise TypeError(f"Cannot navigate path {base_path} in type {var_info.type_hint}")
                    
                    # Now get the field index from current_type
                    if hasattr(current_type, '_field_names'):
                        field_names = current_type._field_names
                        field_name = node.attr
                        if field_name in field_names:
                            field_idx = field_names.index(field_name)
                            return base_var, base_path + (field_idx,)
                        else:
                            raise TypeError(f"Field '{field_name}' not found in type {current_type}")
                raise TypeError(f"Cannot determine type for nested attribute access")
            else:
                raise TypeError(f"Unsupported attribute base: {type(base_node).__name__}")
        
        raise TypeError(f"Unsupported lvalue type: {type(node).__name__}")
    
    def _register_linear_token(self, var_name: str, type_hint, node: ast.AST, path: Tuple[int, ...] = ()):
        """Register/update linear token states when value is assigned
        
        Transitions all linear paths from undefined/consumed -> active
        
        Args:
            var_name: Variable name
            type_hint: Type of the value being assigned
            node: AST node for error reporting
            path: Index path prefix (for nested assignments)
        """
        if self._is_linear_type(type_hint):
            var_info = self.lookup_variable(var_name)
            if var_info:
                # Get all linear paths in the assigned value
                linear_paths = self._get_linear_paths(type_hint, path)
                for lin_path in linear_paths:
                    # Transition to active
                    self._set_linear_state(var_info, lin_path, 'active')
                var_info.linear_scope_depth = self.scope_depth
                logger.debug(f"Linear token '{var_name}' paths {linear_paths} transitioned to active")
    
    def _consume_linear(self, var_name: str, node: ast.AST, path: Tuple[int, ...] = ()):
        """Mark a linear token as consumed
        
        Can only consume tokens in 'active' state.
        undefined/consumed states cannot be consumed.
        
        Args:
            var_name: Variable name
            node: AST node for error reporting
            path: Index path to the linear token
        """
        var_info = self.lookup_variable(var_name)
        if not var_info:
            raise TypeError(f"Variable '{var_name}' not found (line {getattr(node, 'lineno', '?')})")
        
        state = self._get_linear_state(var_info, path)
        path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
        
        if state is None:
            raise TypeError(f"Variable '{path_str}' is not a linear token (line {getattr(node, 'lineno', '?')})")
        
        if state == 'consumed':
            raise TypeError(
                f"Linear token '{path_str}' already consumed "
                f"(declared at line {var_info.line_number}, "
                f"attempting to consume again at line {getattr(node, 'lineno', '?')})"
            )
        
        if state == 'undefined':
            raise TypeError(
                f"Cannot consume undefined linear token '{path_str}' "
                f"(declared at line {var_info.line_number}, "
                f"line {getattr(node, 'lineno', '?')})"
            )
        
        if state == 'moved':
            raise TypeError(
                f"Linear token '{path_str}' was already moved "
                f"(cannot use at line {getattr(node, 'lineno', '?')})"
            )
        
        # Check loop scope restriction: cannot consume external token inside loop
        if self.scope_depth > var_info.linear_scope_depth:
            raise TypeError(
                f"Cannot consume external linear token '{path_str}' inside loop "
                f"(token declared at scope depth {var_info.linear_scope_depth}, "
                f"attempting to consume at depth {self.scope_depth}, "
                f"line {getattr(node, 'lineno', '?')})"
            )
        
        self._set_linear_state(var_info, path, 'consumed')
        logger.debug(f"Consumed linear token '{path_str}'")
    
    def _transfer_linear_ownership(self, value_ref: ValueRef, reason: str = "transfer"):
        """Transfer ownership of linear tokens from a ValueRef
        
        This is the unified method for transferring linear ownership in:
        - Function calls (arguments)
        - Returns (return value)
        - Move operations
        
        Args:
            value_ref: ValueRef carrying linear tracking info (via var_name)
            reason: Description of why ownership is being transferred
        """
        logger.debug(f"_transfer_linear_ownership: value_ref={value_ref}, reason={reason}")
        
        # Handle Python tuple containing ValueRefs (e.g., return statements)
        # Check this BEFORE checking if it's a linear type, because the tuple itself
        # is not linear but may contain linear elements
        if value_ref.is_python_value():
            py_val = value_ref.get_python_value()
            if isinstance(py_val, tuple):
                # Transfer ownership for each element in the tuple
                for elem in py_val:
                    if isinstance(elem, ValueRef):
                        self._transfer_linear_ownership(elem, reason)
                return
        
        # Skip if not a linear type
        if not self._is_linear_type(value_ref.type_hint):
            return
        
        # Check if ValueRef carries variable tracking info
        if not hasattr(value_ref, 'var_name') or not value_ref.var_name:
            logger.debug(f"_transfer_linear_ownership: No var_name, skipping ({reason})")
            return  # No tracking info, nothing to transfer
        
        if not hasattr(value_ref, 'linear_path') or value_ref.linear_path is None:
            logger.debug(f"_transfer_linear_ownership: No linear_path for {value_ref.var_name}, skipping ({reason})")
            return  # No linear path, nothing to transfer
        
        var_name = value_ref.var_name
        base_path = value_ref.linear_path
        var_info = self.lookup_variable(var_name)
        
        if not var_info:
            logger.debug(f"_transfer_linear_ownership: Variable '{var_name}' not found, skipping ({reason})")
            return  # Variable not found, skip
        
        # Get all linear paths in this value (starting from base_path)
        linear_paths = self._get_linear_paths(value_ref.type_hint, base_path)
        logger.debug(f"_transfer_linear_ownership: {var_name} has {len(linear_paths)} linear paths: {linear_paths}")
        
        # Transfer each linear path
        for path in linear_paths:
            state = self._get_linear_state(var_info, path)
            logger.debug(f"_transfer_linear_ownership: {var_name}{path} state={state}")
            
            path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
            
            if state == 'undefined':
                raise TypeError(
                    f"Cannot {reason} undefined linear token '{path_str}' "
                    f"(declared at line {var_info.line_number})"
                )
            
            if state == 'consumed':
                raise TypeError(
                    f"Cannot {reason} already consumed linear token '{path_str}' "
                    f"(declared at line {var_info.line_number})"
                )
            
            if state == 'active':
                # Check loop scope restriction: cannot consume external token inside loop
                if self.scope_depth > var_info.linear_scope_depth:
                    raise TypeError(
                        f"Cannot {reason} external linear token '{path_str}' inside loop "
                        f"(token declared at scope depth {var_info.linear_scope_depth}, "
                        f"attempting to use at depth {self.scope_depth})"
                    )
                
                # Mark as consumed (ownership transferred)
                self._set_linear_state(var_info, path, 'consumed')
                logger.debug(f"Transferred ownership of '{path_str}' ({reason})")

    
    def _check_linear_tokens_consumed(self):
        """Check that all active linear tokens have been consumed before scope exit"""
        unconsumed = []
        logger.debug(f"Checking linear tokens at scope depth {self.scope_depth}")
        
        # Check variables in current scope
        for var_info in self.ctx.var_registry.get_all_in_current_scope():
            if var_info.linear_scope_depth == self.scope_depth:
                logger.debug(f"Checking variable {var_info}")
                for path, state in var_info.linear_states.items():
                    if state == 'active':
                        path_str = f"{var_info.name}[{']['.join(map(str, path))}]" if path else var_info.name
                        unconsumed.append(f"'{path_str}' (declared at line {var_info.line_number})")
        
        if unconsumed:
            raise TypeError(
                f"Linear tokens not consumed before scope exit: {', '.join(unconsumed)}"
            )




