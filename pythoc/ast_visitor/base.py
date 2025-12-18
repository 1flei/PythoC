"""
Base visitor class with initialization and core helper methods
"""

import ast
import builtins
from typing import Optional, Any, List, Tuple, TYPE_CHECKING
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

if TYPE_CHECKING:
    from ..backend import AbstractBackend


class LLVMIRVisitor(ast.NodeVisitor):
    """Enhanced AST visitor that generates LLVM IR using llvmlite
    
    Now includes context management for better type tracking and scope handling.
    Supports both legacy (module/builder) and new (backend) initialization.
    """
    
    def __init__(self, module: ir.Module = None, builder: ir.IRBuilder = None,
                 func_type_hints: dict = None, struct_types: dict = None,
                 source_globals: dict = None, compiler=None, user_globals: dict = None,
                 backend: "AbstractBackend" = None):
        # Support both legacy (module/builder) and new (backend) initialization
        if backend is not None:
            self._backend = backend
            self.module = backend.get_module()
            self.builder = backend.get_llvm_builder() if hasattr(backend, 'get_llvm_builder') else None
            # Create context with backend
            self.ctx = CompilationContext(backend=backend, user_globals=user_globals)
        else:
            self._backend = None
            self.module = module
            self.builder = builder
            # Legacy context initialization
            self.ctx = CompilationContext(module, builder, user_globals=user_globals)
        
        self.func_type_hints = func_type_hints or {}
        self.ctx.struct_types = struct_types or {}
        self.ctx.source_globals = source_globals or {}
        
        # Type resolver for unified type annotation parsing
        # Pass visitor=self to allow type resolver to access local Python type variables
        module_context = self.module.context if self.module else None
        self.type_resolver = TypeResolver(module_context, user_globals=user_globals, visitor=self)
        
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
    
    @property
    def backend(self) -> Optional["AbstractBackend"]:
        """Get the backend (if initialized with one)"""
        return self._backend
    
    def is_constexpr(self) -> bool:
        """Check if this visitor is for constexpr evaluation"""
        if self._backend is not None:
            return self._backend.is_constexpr()
        return self.module is None
        
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
        # For other variables, initialize as 'consumed' (not yet assigned, no ownership)
        if self._is_linear_type(type_hint):
            initial_state = 'active' if is_parameter else 'consumed'
            self._init_linear_states(var_info, type_hint, initial_state=initial_state)
        
        return var_info

    def lookup_variable(self, name: str) -> Optional[VariableInfo]:
        """Look up variable or function in context registry (unified)"""
        # 1. First check variable registry
        var_info = self.ctx.var_registry.lookup(name)
        if var_info:
            return var_info
        
        # 2. Check user globals
        if self.ctx.user_globals and name in self.ctx.user_globals:
            python_obj = self.ctx.user_globals[name]
            from ..valueref import ValueRef, wrap_value
            
            # 2a. Objects with handle_call (ExternFunctionWrapper, @compile wrapper, etc.)
            if hasattr(python_obj, 'handle_call') and callable(python_obj.handle_call):
                from ..builtin_entities.python_type import PythonType
                
                # For @compile functions, get type hints from function annotations
                type_hint = PythonType.wrap(python_obj, is_constant=True)
                if hasattr(python_obj, '_is_compiled') and python_obj._is_compiled:
                    if hasattr(python_obj, '__annotations__'):
                        annotations = python_obj.__annotations__
                        param_type_hints = {}
                        return_type_hint = None
                        
                        for key, value in annotations.items():
                            if key == 'return':
                                return_type_hint = value
                            else:
                                param_type_hints[key] = value
                        
                        if return_type_hint is not None:
                            from ..builtin_entities import func
                            param_types = list(param_type_hints.values())
                            type_hint = func[param_types, return_type_hint] if param_types else func[[], return_type_hint]
                
                return VariableInfo(
                    name=name,
                    value_ref=wrap_value(value=python_obj, kind='python', type_hint=type_hint),
                    alloca=None,
                    source="python_global",
                    is_global=True,
                    is_mutable=False,
                )
            
            # 2b. BuiltinEntity types (i32, ptr, etc.)
            from ..builtin_entities import BuiltinEntity, BuiltinType, BuiltinFunction
            if isinstance(python_obj, type):
                try:
                    if issubclass(python_obj, BuiltinEntity):
                        if issubclass(python_obj, (BuiltinType, BuiltinFunction)):
                            return VariableInfo(
                                name=name,
                                value_ref=wrap_value(
                                    kind='python',
                                    value=python_obj,
                                    type_hint=python_obj,
                                ),
                                alloca=None,
                                source="builtin_entity",
                                is_global=True,
                                is_mutable=False,
                            )
                except TypeError:
                    pass
                
                # 2c. Struct class - return None to let type system handle it
                if hasattr(python_obj, '_is_struct') and python_obj._is_struct:
                    return None
            
            # 2d. Already a ValueRef (like nullptr)
            if isinstance(python_obj, ValueRef):
                return VariableInfo(
                    name=name,
                    value_ref=python_obj,
                    alloca=None,
                    source="python_global",
                    is_global=True,
                    is_mutable=False,
                )
            
            # 2e. Generic Python object
            from ..builtin_entities.python_type import PythonType
            python_type = PythonType.wrap(python_obj)
            return VariableInfo(
                name=name,
                value_ref=wrap_value(
                    kind='python',
                    value=python_obj,
                    type_hint=python_type,
                ),
                alloca=None,
                source="python_global",
                is_global=True,
                is_mutable=False,
            )
        
        # 4. Check unified registry for @compile functions (supports mutual recursion)
        # This handles the case where function B calls function A, but A's wrapper
        # is not yet in user_globals when B's decorator runs (deferred compilation)
        from ..registry import get_unified_registry
        registry = get_unified_registry()
        func_info = registry.get_function_info(name)
        if func_info:
            # Create a callable wrapper that uses registry info
            class RegistryFunctionWrapper:
                def __init__(self, func_name, func_info):
                    self.func_name = func_name
                    self.func_info = func_info
                
                def handle_call(self, visitor, func_ref, args, node):
                    """Handle calls to functions registered in unified registry"""
                    from llvmlite import ir as llvm_ir
                    
                    actual_func_name = self.func_info.mangled_name or self.func_name
                    
                    # Get or declare the function in the module
                    try:
                        ir_func = visitor.module.get_global(actual_func_name)
                    except KeyError:
                        # Declare the function with proper ABI handling
                        param_llvm_types = []
                        for param_name in self.func_info.param_names:
                            pc_param_type = self.func_info.param_type_hints.get(param_name)
                            if pc_param_type and hasattr(pc_param_type, 'get_llvm_type'):
                                param_llvm_types.append(pc_param_type.get_llvm_type(visitor.module.context))
                            else:
                                logger.error(f"Invalid parameter type hint for '{param_name}' in function '{actual_func_name}'",
                                            node=node, exc_type=TypeError)
                        
                        if self.func_info.return_type_hint and hasattr(self.func_info.return_type_hint, 'get_llvm_type'):
                            return_llvm_type = self.func_info.return_type_hint.get_llvm_type(visitor.module.context)
                        else:
                            return_llvm_type = llvm_ir.VoidType()
                        
                        # Use LLVMBuilder to declare function with C ABI
                        from ..builder import LLVMBuilder
                        temp_builder = LLVMBuilder()
                        func_wrapper = temp_builder.declare_function(
                            visitor.module, actual_func_name,
                            param_llvm_types, return_llvm_type
                        )
                        ir_func = func_wrapper.ir_function
                    
                    # Build parameter LLVM types
                    param_llvm_types = []
                    for p in self.func_info.param_names:
                        param_type = self.func_info.param_type_hints[p]
                        param_llvm_types.append(param_type.get_llvm_type(visitor.module.context))
                    
                    return visitor._perform_call(
                        node, ir_func, param_llvm_types,
                        self.func_info.return_type_hint,
                        evaluated_args=args
                    )
            
            wrapper = RegistryFunctionWrapper(name, func_info)
            from ..valueref import wrap_value
            return VariableInfo(
                name=name,
                value_ref=wrap_value(
                    kind='python',
                    value=wrapper,
                    type_hint=func_info,
                ),
                alloca=None,
                source="registry_function",
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
    
    def visit_expression(self, expr):
        """Visit an expression and return a ValueRef preserving type hints"""
        result = self.visit(expr)
        if result is None:
            logger.error(f"Expression {ast.dump(expr)} returned None", node=expr, exc_type=ValueError)
        
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

        return result
    
    # ========================================================================
    # Linear Token Tracking
    # ========================================================================
    
    def _is_linear_type(self, type_hint) -> bool:
        """Check if a type is linear
        
        A type is linear if:
        1. It's the linear token type itself
        2. It has _is_linear attribute set to True
        3. It's a RefinedType wrapping a linear type
        4. It's a struct containing linear fields (recursively)
        """
        from ..builtin_entities import linear
        from ..builtin_entities.refined import RefinedType
        
        if type_hint is linear:
            return True
        # Check if it's a class with _is_linear attribute
        if isinstance(type_hint, type) and hasattr(type_hint, '_is_linear'):
            return type_hint._is_linear
        
        # Check if it's a RefinedType - delegate to base type
        if isinstance(type_hint, type) and issubclass(type_hint, RefinedType):
            base_type = getattr(type_hint, '_base_type', None)
            if base_type is not None:
                return self._is_linear_type(base_type)
        
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
            refined[linear, "tag"] -> [()] (refined wrapping linear is linear itself)
            struct[ptr, linear] -> [(1,)]
            struct[struct[ptr, linear], linear] -> [(0, 1), (1,)]
        """
        from ..builtin_entities import linear
        from ..builtin_entities.refined import RefinedType
        
        if type_hint is linear:
            return [prefix]
        
        if isinstance(type_hint, type) and hasattr(type_hint, '_is_linear') and type_hint._is_linear:
            return [prefix]
        
        # Check if it's a RefinedType - delegate to base type
        # A refined[linear, "tag"] is linear at the current path, not nested
        if isinstance(type_hint, type) and issubclass(type_hint, RefinedType):
            base_type = getattr(type_hint, '_base_type', None)
            if base_type is not None:
                # Delegate to base type at the same prefix (not nested)
                return self._get_linear_paths(base_type, prefix)
        
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
    
    def _init_linear_states(self, var_info, type_hint, initial_state: str = 'consumed'):
        """Initialize linear states for all linear paths in a type"""
        paths = self._get_linear_paths(type_hint)
        for path in paths:
            self._set_linear_state(var_info, path, initial_state)
        if paths:
            var_info.linear_scope_depth = self.scope_depth
            logger.debug(f"Initialized linear states for '{var_info.name}': {paths} -> {initial_state}")
    
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
            return  # No tracking info, nothing to transfer
        
        if not hasattr(value_ref, 'linear_path') or value_ref.linear_path is None:
            return  # No linear path, nothing to transfer
        
        var_name = value_ref.var_name
        base_path = value_ref.linear_path
        var_info = self.lookup_variable(var_name)
        
        if not var_info:
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
                logger.error(
                    f"Cannot {reason} undefined linear token '{path_str}' "
                    f"(declared at line {var_info.line_number})"
                )
            
            if state == 'consumed':
                logger.error(
                    f"Cannot {reason} already consumed linear token '{path_str}' "
                    f"(declared at line {var_info.line_number})"
                )
            
            if state == 'active':
                # Check loop scope restriction: cannot consume external token inside loop
                if self.scope_depth > var_info.linear_scope_depth:
                    logger.error(
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
            logger.error(
                f"Linear tokens not consumed before scope exit: {', '.join(unconsumed)}"
            )




