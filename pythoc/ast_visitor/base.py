"""
Base visitor class with initialization and core helper methods
"""

import ast
import builtins
from typing import Optional, Any, List, Tuple, TYPE_CHECKING
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..ir_helpers import safe_load
from ..type_converter import TypeConverter, ImplicitCoercer
from ..context import (
    VariableInfo,
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
from ..builtin_entities.base import BuiltinType
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..type_resolver import TypeResolver
from ..logger import logger
from ..scope_manager import ScopeManager, ScopeType

if TYPE_CHECKING:
    from ..backend import AbstractBackend


class LLVMIRVisitor(ast.NodeVisitor):
    """Enhanced AST visitor that generates LLVM IR using llvmlite
    
    Now includes context management for better type tracking and scope handling.
    Supports both legacy (module/builder) and new (backend) initialization.
    """
    
    def __init__(self, module: ir.Module = None, builder: ir.IRBuilder = None,
                 struct_types: dict = None,
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
        
        self.ctx.struct_types = struct_types or {}
        self.ctx.source_globals = source_globals or {}
        
        # Type resolver for unified type annotation parsing
        # Pass visitor=self to allow type resolver to access local Python type variables
        module_context = self.module.context if self.module else None
        self.type_resolver = TypeResolver(module_context, user_globals=user_globals, visitor=self)
        
        # Type converter for centralized type conversion
        self.type_converter = TypeConverter(self)

        # Implicit coercer for policy-checked implicit conversions
        self.implicit_coercer = ImplicitCoercer(self.type_converter)
        
        # Unified scope manager for defer, linear types, and variable lifetime
        self.scope_manager = self.ctx.scope_manager
        self.scope_manager.set_visitor(self)

        # Backward compatibility aliases
        self.func_state = None  # ActiveCompileFrame, set by compiler.py
        self.binding_state = None  # FunctionBindingState, set by compiler.py
        self._current_function = None
        self.label_counter = 0
        self.struct_types = self.ctx.struct_types
        self.source_globals = self.ctx.source_globals
        self.compiler = compiler
    
    @property
    def current_function(self):
        if self.func_state is not None:
            return self.func_state.current_function
        return self._current_function

    @current_function.setter
    def current_function(self, value):
        if self.func_state is not None:
            self.func_state.current_function = value
        else:
            self._current_function = value

    @property
    def current_group_key(self):
        if self.binding_state is not None:
            return self.binding_state.group_key
        return getattr(self, '_current_group_key', None)

    @current_group_key.setter
    def current_group_key(self, value):
        if self.binding_state is not None:
            self.binding_state.group_key = value
        else:
            self._current_group_key = value

    @property
    def current_function_name(self):
        if self.func_state is not None:
            return self.func_state.function_name
        return None

    @property
    def current_return_type_hint(self):
        if self.func_state is not None:
            return self.func_state.return_type_hint
        return None

    @property
    def current_param_type_hints(self):
        if self.func_state is not None:
            return self.func_state.param_type_hints
        return {}

    @property
    def current_sret_info(self):
        if self.func_state is not None:
            return self.func_state.sret_info
        return None

    @property
    def current_param_coercion_info(self):
        if self.func_state is not None:
            return self.func_state.param_coercion_info
        return {}

    @property
    def current_varargs_info(self):
        if self.func_state is not None:
            return self.func_state.varargs_info
        return None

    @current_varargs_info.setter
    def current_varargs_info(self, value):
        if self.func_state is not None:
            self.func_state.varargs_info = value
        else:
            self._current_varargs_info = value

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
        
        if self.scope_manager.is_declared_in_current_scope(name):
            existing = self.scope_manager.lookup_variable(name)
            if existing:
                existing.alloca = alloca
                existing.value_ref = value_ref
                return existing
        
        self.scope_manager.declare_variable(var_info, allow_shadow=True)

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
        var_info = self.scope_manager.lookup_variable(name)
        if var_info:
            return var_info
        
        # 2. Check user globals
        if self.ctx.user_globals and name in self.ctx.user_globals:
            python_obj = self.ctx.user_globals[name]
            from ..valueref import ValueRef, wrap_value

            # 2a. BuiltinEntity types (i32, ptr, etc.)
            from ..builtin_entities import BuiltinEntity, BuiltinType, BuiltinFunction
            from ..builtin_entities.python_type import PythonType
            if isinstance(python_obj, type):
                try:
                    if issubclass(python_obj, BuiltinEntity):
                        if issubclass(python_obj, (BuiltinType, BuiltinFunction)):
                            python_type = PythonType.wrap(python_obj, is_constant=True)
                            return VariableInfo(
                                name=name,
                                value_ref=wrap_value(
                                    kind='python',
                                    value=python_obj,
                                    type_hint=python_type,
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
            
            # 2b. Already a ValueRef (like nullptr)
            if isinstance(python_obj, ValueRef):
                return VariableInfo(
                    name=name,
                    value_ref=python_obj,
                    alloca=None,
                    source="python_global",
                    is_global=True,
                    is_mutable=False,
                )
            
            # 2c. Generic Python object
            from ..builtin_entities.python_type import PythonType
            python_type = PythonType.wrap(python_obj, is_constant=True)
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
        
        # Note: Self/mutual recursion is now handled via group scope injection
        # in output_manager._compile_pending_for_group(). All functions in the
        # same compilation group have their wrappers injected into each other's
        # compilation_globals before compilation starts. This eliminates the need
        # for name-based registry lookup here.
        
        return None
    
    def get_variable_alloca(self, name: str) -> Optional[ir.AllocaInstr]:
        """Get variable alloca from registry"""
        var_info = self.scope_manager.lookup_variable(name)
        return var_info.alloca if var_info else None
    
    def has_variable(self, name: str) -> bool:
        """Check if variable exists in registry"""
        return self.scope_manager.lookup_variable(name) is not None
    
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

    def _bind_name_reference(self, value_ref: ValueRef, name: str) -> ValueRef:
        """Attach top-level variable tracking metadata to a binding reference."""
        if value_ref.is_python_value():
            return value_ref

        if value_ref.has_place():
            place = value_ref.require_place()
            if value_ref.value is place:
                return value_ref.clone(var_name=name, linear_path=())

        return value_ref

    def read_rvalue(self, value_ref: ValueRef, *, name: Optional[str] = None) -> ValueRef:
        """Explicitly materialize a binding into its rvalue form when needed."""
        if value_ref.is_python_value() or not value_ref.has_place():
            return value_ref

        place = value_ref.require_place()
        if value_ref.value is not place:
            return value_ref

        from ..type_converter import get_base_type

        base_type = get_base_type(value_ref.type_hint)
        if (
            base_type is not None
            and isinstance(base_type, type)
            and issubclass(base_type, BuiltinType)
            and base_type.is_array()
        ):
            return value_ref

        loaded_val = safe_load(
            self.builder,
            ensure_ir(place),
            value_ref.type_hint,
            name=name or getattr(value_ref, "var_name", "") or "",
        )
        return wrap_value(
            loaded_val,
            kind="address",
            type_hint=value_ref.type_hint,
            address=place,
            source_node=value_ref.source_node,
            var_name=value_ref.var_name,
            linear_path=value_ref.linear_path,
            vref_id=value_ref.vref_id,
        )

    def visit_rvalue_expression(self, expr):
        """Visit an expression in value context."""
        result = self.visit_expression(expr)
        if isinstance(result, ValueRef):
            name = expr.id if isinstance(expr, ast.Name) else None
            return self.read_rvalue(result, name=name)
        return result

    def prepare_protocol_base(self, value_ref: ValueRef) -> ValueRef:
        """Prepare a base ValueRef for attribute/subscript protocol dispatch."""
        if value_ref.is_python_value():
            return value_ref

        from ..type_converter import get_base_type

        base_type = get_base_type(value_ref.type_hint)
        if (
            base_type is not None
            and isinstance(base_type, type)
            and issubclass(base_type, BuiltinType)
            and base_type.is_pointer()
        ):
            return self.read_rvalue(value_ref, name=getattr(value_ref, "var_name", None))
        return value_ref
    
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
    
    def _init_linear_states(self, var_info, type_hint, initial_state: str = 'consumed'):
        """Initialize linear states for all linear paths in a type
        
        Emits LinearRegister events to CFG for path-sensitive analysis.
        """
        paths = self._get_linear_paths(type_hint)
        for path in paths:
            # Emit LinearRegister event to CFG
            # Map 'active' -> 'valid', 'consumed' -> 'invalid'
            cfg_state = 'valid' if initial_state == 'active' else 'invalid'
            cf = self._get_cf_builder()
            if cf is not None:
                cf.record_linear_register(
                    var_info.var_id, var_info.name, path, cfg_state,
                    line_number=var_info.line_number, node=None
                )
        if paths:
            logger.debug(f"Initialized linear states for '{var_info.name}': {paths} -> {initial_state}")
    
    def _format_linear_path(self, var_name: str, path: Tuple[int, ...], type_hint) -> str:
        """Format a linear path as a human-readable string with field names
        
        Args:
            var_name: Variable name
            path: Index path tuple (e.g., (1,) or (0, 2))
            type_hint: Type hint of the variable (to get field names)
            
        Returns:
            Human-readable path string like 'var.field' or 'var.field.subfield'
        """
        if not path:
            return var_name
        
        parts = [var_name]
        current_type = type_hint
        
        for idx in path:
            # Try to get field name from current type
            field_name = None
            if hasattr(current_type, '_struct_fields') and current_type._struct_fields:
                # Named struct with _struct_fields
                if idx < len(current_type._struct_fields):
                    field_name, field_type = current_type._struct_fields[idx]
                    current_type = field_type
            elif hasattr(current_type, '_field_types') and current_type._field_types:
                # Anonymous struct with _field_types only
                if idx < len(current_type._field_types):
                    current_type = current_type._field_types[idx]
            
            if field_name:
                parts.append(f".{field_name}")
            else:
                parts.append(f"[{idx}]")
        
        return ''.join(parts)
    
    def _get_actual_line_number(self, line_number: Optional[int]) -> Optional[int]:
        """Get actual line number by adding logger's line offset
        
        Args:
            line_number: AST relative line number
            
        Returns:
            Actual line number in source file
        """
        if line_number is None:
            return None
        return line_number + logger.current_line_offset
    
    def _register_linear_token(self, var_name: str, type_hint, node: ast.AST, path: Tuple[int, ...] = ()):
        """Register/update linear token states when value is assigned
        
        Transitions all linear paths from undefined/consumed -> active.
        Emits LinearTransition events to CFG for path-sensitive analysis.
        
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
                    # Emit LinearTransition event: invalid -> valid (token created)
                    cf = self._get_cf_builder()
                    if cf is not None:
                        line_num = node.lineno if node and hasattr(node, 'lineno') else var_info.line_number
                        cf.record_linear_transition(
                            var_info.var_id, var_name, lin_path, 'invalid', 'valid',
                            line_number=line_num, node=node
                        )
                logger.debug(f"Linear token '{var_name}' paths {linear_paths} transitioned to active")
    
    def _get_cf_builder(self):
        """Get the ControlFlowBuilder if builder is one, else None"""
        from .control_flow_builder import ControlFlowBuilder
        if isinstance(self.builder, ControlFlowBuilder):
            return self.builder
        return None

    def _transfer_linear_ownership(self, value_ref: ValueRef, reason: str = "transfer", node=None):
        """Transfer ownership of linear tokens from a ValueRef
        
        This is the unified method for transferring linear ownership in:
        - Function calls (arguments)
        - Returns (return value)
        - Move operations
        
        Emits LinearTransition events to CFG for path-sensitive analysis.
        CFG checker validates state transitions; AST visitor just emits events.
        
        Args:
            value_ref: ValueRef carrying linear tracking info (via var_name)
            reason: Description of why ownership is being transferred
            node: AST node for error reporting
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
                        self._transfer_linear_ownership(elem, reason, node)
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
            # Emit LinearTransition event: valid -> invalid (token consumed)
            # CFG checker will validate that current state is 'valid'
            cf = self._get_cf_builder()
            if cf is not None:
                line_num = node.lineno if node and hasattr(node, 'lineno') else var_info.line_number
                cf.record_linear_transition(
                    var_info.var_id, var_name, path, 'valid', 'invalid',
                    line_number=line_num, node=node
                )

