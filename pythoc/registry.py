"""
Unified Registry System for PC Compiler

This module provides a centralized registry for all compilation artifacts:
- Variables (scope-aware)
- Functions (compiled, extern, runtime-generated)
- Types (builtin, structs)
- Compilers and modules

This replaces the scattered global dictionaries across multiple files.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from llvmlite import ir
import ast


@dataclass
class VariableInfo:
    """Information about a variable
    
    Stores variable metadata and value reference. The value_ref field contains
    the actual value (LLVM or Python) and type information.
    """
    name: str
    value_ref: Optional[Any] = None  # ValueRef - unified value storage
    alloca: Optional[ir.AllocaInstr] = None  # Storage location for PC variables
    
    # Symbol table metadata
    source: str = "unknown"  # "annotation", "inference", "parameter"
    line_number: Optional[int] = None
    is_parameter: bool = False
    is_mutable: bool = True
    is_global: bool = False
    scope_level: int = 0
    column: Optional[int] = None
    
    # Linear type tracking - path-based for composite types
    # Maps index path (tuple of ints) to linear state
    # Examples:
    #   Simple linear: {(): 'active'}
    #   Struct field: {(0, 1): 'active', (1, 0): 'undefined'}
    linear_states: Dict[Tuple[int, ...], str] = field(default_factory=dict)
    linear_scope_depth: int = 0  # Scope depth where linear token was created
    
    # Backward compatibility: linear_state property for simple linear variables
    @property
    def linear_state(self) -> Optional[str]:
        """Get linear state for simple (non-composite) linear variables"""
        if not self.linear_states:
            return None
        # For simple linear variables, use path ()
        return self.linear_states.get((), None)
    
    @linear_state.setter
    def linear_state(self, value: Optional[str]):
        """Set linear state for simple linear variables"""
        if value is None:
            self.linear_states.pop((), None)
        else:
            self.linear_states[()] = value
    
    # Type information accessed via value_ref
    @property
    def type_hint(self):
        """Get type hint from value_ref"""
        return self.value_ref.type_hint if self.value_ref else None
    
    @property
    def llvm_type(self):
        """Get LLVM type from value_ref or alloca"""
        if self.value_ref and not self.value_ref.is_python_value():
            # If value is an alloca (pointer), return the pointee type
            if hasattr(self.value_ref.value, 'type'):
                val_type = self.value_ref.value.type
                if isinstance(val_type, ir.PointerType):
                    return val_type.pointee
                return val_type
        # Fallback to alloca pointee type
        if self.alloca and isinstance(self.alloca.type, ir.PointerType):
            return self.alloca.type.pointee
        return None
    
    @property
    def is_python_constant(self):
        """Check if this is a Python constant variable"""
        return self.alloca is None and self.value_ref and self.value_ref.is_python_value()
    
    def __repr__(self) -> str:
        type_name = self.type_hint.get_name() if self.type_hint and hasattr(self.type_hint, 'get_name') else str(self.type_hint)
        storage = "python_const" if self.is_python_constant else "alloca" if self.alloca else "no_storage"
        return f"VariableInfo({self.name}: {type_name}, source={self.source}, storage={storage})"


@dataclass
class FunctionInfo:
    """Information about a compiled function"""
    name: str
    source_file: str
    ast_node: Optional[ast.FunctionDef] = None
    llvm_function: Optional[ir.Function] = None
    return_type_hint: Optional[Any] = None
    param_type_hints: Dict[str, Any] = field(default_factory=dict)
    param_names: List[str] = field(default_factory=list)  # Ordered parameter names
    source_code: Optional[str] = None
    is_compiled: bool = False
    mangled_name: Optional[str] = None  # For function overloading
    overload_enabled: bool = False  # Whether overloading is enabled for this function
    so_file: Optional[str] = None  # Path to the .so file for this function
    # Effect system: track which effects this function uses (e.g., {'rng', 'd_impl'})
    # Used for transitive effect propagation - when a function with suffix calls
    # another function that uses an overridden effect, we generate a suffix version
    effect_dependencies: Set[str] = field(default_factory=set)
    # The wrapper object for this function (used for on-demand suffix generation)
    wrapper: Optional[Any] = None


@dataclass
class ExternFunctionInfo:
    """Information about an external function"""
    name: str
    lib: str = "libc"
    calling_convention: str = "cdecl"
    return_type: Optional[Any] = None
    param_types: List[Any] = field(default_factory=list)
    llvm_function: Optional[ir.Function] = None
    signature: Optional[Any] = None  # inspect.Signature
    function: Optional[Any] = None  # Python function object


@dataclass
class RuntimeFunctionInfo:
    """Information about a runtime-generated function"""
    name: str
    ast_node: Optional[ast.FunctionDef] = None
    compiler: Optional[Any] = None  # LLVMCompiler instance
    source_code: Optional[str] = None
    parent_function: Optional[str] = None
    function: Optional[Any] = None  # Python function object


@dataclass
class StructInfo:
    """Information about a struct type
    
    This combines the functionality of the old StructMetadata and StructInfo classes.
    It stores both metadata and LLVM type information for user-defined struct types.
    """
    name: str
    fields: List[Tuple[str, Any]]  # List of (field_name, field_type) tuples
    field_indices: Dict[str, int] = field(default_factory=dict)
    llvm_type: Optional[ir.Type] = None
    python_class: Optional[type] = None
    
    def get_field_index(self, field_name: str) -> Optional[int]:
        """Get the index of a field by name"""
        return self.field_indices.get(field_name)
    
    def get_field_count(self) -> int:
        """Get the total number of fields"""
        return len(self.fields)
    
    def get_field_names(self) -> List[str]:
        """Get all field names"""
        return [field_name for field_name, _ in self.fields]
    
    def has_field(self, field_name: str) -> bool:
        """Check if a field exists"""
        return field_name in self.field_indices
    
    def get_field_type_hint(self, field_name: str, type_resolver=None):
        """Get the resolved type hint for a field
        
        Args:
            field_name: Name of the field
            type_resolver: Optional TypeResolver to parse annotations
            
        Returns:
            The resolved BuiltinEntity type, or None if not resolvable
        """
        field_index = self.get_field_index(field_name)
        if field_index is None:
            return None
        
        field_type_annotation = self.fields[field_index][1]
        
        # If already a BuiltinEntity class, return it directly
        from .builtin_entities.types import BuiltinType
        if isinstance(field_type_annotation, type) and issubclass(field_type_annotation, BuiltinType):
            return field_type_annotation
        
        # If it's a BuiltinType instance, return it
        if isinstance(field_type_annotation, BuiltinType):
            return field_type_annotation
        
        # Try to parse the annotation if type_resolver is provided
        if type_resolver is not None:
            parsed_type = type_resolver.parse_annotation(field_type_annotation)
            if parsed_type is not None:
                return parsed_type
        
        # For self-referential types like ptr[TreeNode], try to construct it
        # This handles the case where TreeNode is now registered
        if type_resolver is not None:
            try:
                parsed_type = type_resolver.parse_annotation(field_type_annotation)
                if parsed_type is not None:
                    return parsed_type
            except:
                pass
        
        return None
    
    def infer_from_llvm_type(self, llvm_type: ir.Type) -> bool:
        """Try to match this struct with an LLVM type
        
        Returns True if the LLVM type matches this struct's structure.
        """
        if isinstance(llvm_type, ir.IdentifiedStructType):
            # Match by name
            struct_name = llvm_type.name.strip('"')
            return struct_name == self.name or struct_name.startswith(f"{self.name}.")
        elif isinstance(llvm_type, ir.LiteralStructType):
            # Match by field count (fallback)
            return len(llvm_type.elements) == self.get_field_count()
        return False


class ScopeContext:
    """Context manager for automatic scope management
    
    Usage:
        with var_registry.scope():
            # Declare variables in new scope
            ...
        # Scope automatically exits here
    """
    
    def __init__(self, registry: 'VariableRegistry'):
        self.registry = registry
        self.exited_vars: Optional[Dict[str, VariableInfo]] = None
    
    def __enter__(self):
        self.registry.enter_scope()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited_vars = self.registry.exit_scope()
        return False


class VariableRegistry:
    """Scope-aware variable registry
    
    Manages variables with proper scope handling, supporting nested scopes,
    variable shadowing, and type inference integration.
    """
    
    def __init__(self):
        # Scope stack: each scope is a dict of variable name -> VariableInfo
        self.scopes: List[Dict[str, VariableInfo]] = [{}]
        
        # Global variables (module-level)
        self.global_vars: Dict[str, VariableInfo] = {}
        
        # Type inference context (optional integration)
        self.type_inference_ctx: Optional[Any] = None
        
        # Current scope level
        self._scope_level = 0
    
    def enter_scope(self):
        """Enter a new scope (function, block, etc.)"""
        self.scopes.append({})
        self._scope_level += 1
    
    def exit_scope(self) -> Dict[str, VariableInfo]:
        """Exit current scope and return variables in that scope"""
        if len(self.scopes) > 1:
            scope_vars = self.scopes.pop()
            self._scope_level -= 1
            return scope_vars
        return {}
    
    def scope(self) -> ScopeContext:
        """Create a scope context manager for automatic scope management
        
        Usage:
            with var_registry.scope():
                var_registry.declare(VariableInfo(...))
                # Variables declared here
            # Scope automatically exits
        
        Returns:
            ScopeContext that handles enter/exit
        """
        return ScopeContext(self)
    
    def declare(self, var_info: VariableInfo, allow_shadow: bool = False):
        """Declare a variable in the current scope
        
        Args:
            var_info: Variable information
            allow_shadow: If True, allow shadowing variables from outer scopes
        
        Raises:
            NameError: If variable already exists in current scope
        """
        current_scope = self.scopes[-1]
        
        # Check if already declared in current scope
        if var_info.name in current_scope and not allow_shadow:
            existing = current_scope[var_info.name]
            raise NameError(
                f"Variable '{var_info.name}' already declared in this scope "
                f"(line {existing.line_number})"
            )
        
        # Set scope level
        var_info.scope_level = self._scope_level
        
        # Add to current scope
        current_scope[var_info.name] = var_info
        
        # Sync with type inference context if available
        if self.type_inference_ctx and var_info.type_hint:
            self.type_inference_ctx.set_var_type(var_info.name, var_info.type_hint)
    
    def lookup(self, name: str) -> Optional[VariableInfo]:
        """Look up a variable in the scope chain
        
        Searches from innermost to outermost scope.
        Returns None if variable is not found (caller will try other namespaces).
        """
        # Search scopes from inner to outer
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        
        # Check global variables
        if name in self.global_vars:
            return self.global_vars[name]
        
        # Variable not found - return None (caller will check other namespaces)
        return None
    
    def update_type(self, name: str, new_type_hint: Any):
        """Update the type of an existing variable"""
        var_info = self.lookup(name)
        if var_info:
            var_info.type_hint = new_type_hint
            if self.type_inference_ctx:
                self.type_inference_ctx.set_var_type(name, new_type_hint)
    
    def list_current_scope(self) -> Dict[str, VariableInfo]:
        """List all variables in the current scope"""
        return self.scopes[-1].copy()
    
    def list_all_visible(self) -> Dict[str, VariableInfo]:
        """List all variables visible in current scope"""
        result = {}
        # Start from global, then add each scope (inner scopes override)
        result.update(self.global_vars)
        for scope in self.scopes:
            result.update(scope)
        return result
    
    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if variable is declared in the current scope"""
        return name in self.scopes[-1]
    
    def get_all_in_current_scope(self) -> List[VariableInfo]:
        """Get all variables in the current scope"""
        return list(self.scopes[-1].values())
    
    def get_type_hint(self, name: str) -> Optional[Any]:
        """Get type hint of a variable"""
        var_info = self.lookup(name)
        return var_info.type_hint if var_info else None
    
    def get_llvm_type(self, name: str) -> Optional[ir.Type]:
        """Get LLVM type of a variable (pointee type of alloca)"""
        var_info = self.lookup(name)
        if var_info and var_info.alloca:
            return var_info.alloca.type.pointee
        return None
    
    def get_alloca(self, name: str) -> Optional[ir.AllocaInstr]:
        """Get alloca instruction for a variable"""
        var_info = self.lookup(name)
        return var_info.alloca if var_info else None
    
    def get_all_in_scope(self, scope_level: Optional[int] = None) -> Dict[str, VariableInfo]:
        """Get all variables in a specific scope level
        
        Args:
            scope_level: Scope level to query (None = current scope)
        """
        if scope_level is None:
            scope_level = self._scope_level
        
        if 0 <= scope_level < len(self.scopes):
            return self.scopes[scope_level].copy()
        return {}
    
    def get_all_visible(self) -> Dict[str, VariableInfo]:
        """Get all currently visible variables (from all scopes)"""
        visible = {}
        
        # Add globals first
        visible.update(self.global_vars)
        
        # Add from outer to inner scopes (inner shadows outer)
        for scope in self.scopes:
            visible.update(scope)
        
        return visible
    
    def clear(self):
        """Clear all scopes (useful for testing)"""
        self.scopes = [{}]
        self.global_vars.clear()
        self._scope_level = 0
    
    def __repr__(self) -> str:
        return f"VariableRegistry(scopes={len(self.scopes)}, level={self._scope_level})"


class UnifiedCompilationRegistry:
    """Unified registry for all compilation artifacts
    
    This centralizes all registration information that was previously scattered
    across multiple global dictionaries in different files.
    """
    
    def __init__(self):
        # ===== Variable Registry =====
        # Per-compiler variable registry (each compiler has its own scope)
        self._variable_registries: Dict[str, VariableRegistry] = {}
        
        # ===== Function Registry =====
        # Compiled functions: source_file -> list of function names
        self._compiled_functions: Dict[str, List[str]] = {}
        
        # Function details: name -> FunctionInfo (original name)
        self._function_info: Dict[str, FunctionInfo] = {}
        # Function details by mangled name (for overloads)
        self._function_info_by_mangled: Dict[str, FunctionInfo] = {}
        
        # Function type hints: func_name -> {'return': type, 'params': {...}}
        self._function_type_hints: Dict[str, Dict[str, Any]] = {}
        
        # ===== Extern Functions =====
        self._extern_functions: Dict[str, ExternFunctionInfo] = {}
        
        # ===== Runtime Functions =====
        self._runtime_functions: Dict[str, RuntimeFunctionInfo] = {}
        self._runtime_function_counter: int = 0
        
        # ===== Source Code Registry =====
        # Source files: source_file -> full source code
        self._source_files: Dict[str, str] = {}
        
        # Individual function sources: "source_file:func_name" -> source code
        self._function_sources: Dict[str, str] = {}
        
        # ===== Compiler Registry =====
        # Compilers: source_file -> LLVMCompiler instance
        self._compilers: Dict[str, Any] = {}
        
        # Shared libraries: source_file -> .so file path
        self._shared_libraries: Dict[str, str] = {}
        
        # ===== Type Registry =====
        # Struct types: struct_name -> StructInfo
        self._structs: Dict[str, StructInfo] = {}
        
        # ===== Builtin Entity Registry =====
        # Builtin entities (types, functions): name -> entity class
        self._builtin_entities: Dict[str, type] = {}
        
        # ===== Link Libraries Registry =====
        # Libraries to link against (collected from extern functions)
        self._link_libraries: Set[str] = set()
    
    # ========== Variable Registry Methods ==========
    
    def get_variable_registry(self, compiler_id: str = "default") -> VariableRegistry:
        """Get or create a variable registry for a compiler instance
        
        Args:
            compiler_id: Identifier for the compiler (usually source file path)
        
        Returns:
            VariableRegistry instance
        """
        if compiler_id not in self._variable_registries:
            self._variable_registries[compiler_id] = VariableRegistry()
        return self._variable_registries[compiler_id]
    
    def clear_variable_registry(self, compiler_id: str = "default"):
        """Clear variable registry for a specific compiler"""
        if compiler_id in self._variable_registries:
            self._variable_registries[compiler_id].clear()
    
    # ========== Function Registry Methods ==========
    
    def register_function(self, func_info: FunctionInfo):
        """Register a compiled function"""
        # Add to compiled functions list (by original name)
        if func_info.source_file not in self._compiled_functions:
            self._compiled_functions[func_info.source_file] = []
        if func_info.name not in self._compiled_functions[func_info.source_file]:
            self._compiled_functions[func_info.source_file].append(func_info.name)
        
        # Store function info under original name
        # IMPORTANT: Only store under original name if this is the base version (no suffix)
        # or if no base version exists yet. This prevents suffix versions from
        # overwriting the base version in the registry.
        if func_info.mangled_name:
            # This is a suffix/overload version
            # Only store under original name if no base version exists
            if func_info.name not in self._function_info:
                self._function_info[func_info.name] = func_info
            # Always store by mangled name
            self._function_info_by_mangled[func_info.mangled_name] = func_info
        else:
            # This is the base version (no suffix)
            self._function_info[func_info.name] = func_info
        
        # Store type hints if available
        if func_info.return_type_hint or func_info.param_type_hints:
            type_hints = {
                'return': func_info.return_type_hint,
                'params': func_info.param_type_hints
            }
            self._function_type_hints[func_info.name] = type_hints
            # If has mangled name (overload), also store type hints by mangled name
            if func_info.mangled_name:
                self._function_type_hints[func_info.mangled_name] = type_hints
    
    def get_function_info(self, func_name: str) -> Optional[FunctionInfo]:
        """Get function information by original name"""
        return self._function_info.get(func_name)
    
    def get_function_info_by_mangled(self, mangled_name: str) -> Optional[FunctionInfo]:
        """Get function information by mangled name (for overloads)"""
        return self._function_info_by_mangled.get(mangled_name)
    
    def get_function_type_hints(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get function type hints"""
        return self._function_type_hints.get(func_name)
    
    def list_compiled_functions(self, source_file: Optional[str] = None) -> Dict[str, List[str]]:
        """List compiled functions
        
        Args:
            source_file: If provided, return only functions from this file
        
        Returns:
            Dict mapping source files to function name lists
        """
        if source_file:
            return {source_file: self._compiled_functions.get(source_file, [])}
        return self._compiled_functions.copy()
    
    def find_function_source_file(self, func_name: str) -> Optional[str]:
        """Find which source file contains a function"""
        # First check original names
        for source_file, func_list in self._compiled_functions.items():
            if func_name in func_list:
                return source_file
        
        # Then check mangled names
        func_info = self._function_info_by_mangled.get(func_name)
        if func_info:
            return func_info.source_file
        
        return None
    
    # ========== Extern Function Methods ==========
    
    def register_extern_function(self, extern_info: ExternFunctionInfo):
        """Register an external function"""
        self._extern_functions[extern_info.name] = extern_info
    
    def get_extern_function(self, func_name: str) -> Optional[ExternFunctionInfo]:
        """Get extern function information"""
        return self._extern_functions.get(func_name)
    
    def is_extern_function(self, func_name: str) -> bool:
        """Check if a function is registered as extern"""
        return func_name in self._extern_functions
    
    def list_extern_functions(self) -> List[str]:
        """List all extern function names"""
        return list(self._extern_functions.keys())
    
    # ========== Runtime Function Methods ==========
    
    def register_runtime_function(self, runtime_info: RuntimeFunctionInfo) -> str:
        """Register a runtime-generated function
        
        Returns:
            The function name (may be auto-generated)
        """
        if not runtime_info.name:
            # Generate unique name
            self._runtime_function_counter += 1
            runtime_info.name = f"_runtime_func_{self._runtime_function_counter}"
        
        self._runtime_functions[runtime_info.name] = runtime_info
        return runtime_info.name
    
    def get_runtime_function(self, func_name: str) -> Optional[RuntimeFunctionInfo]:
        """Get runtime function information"""
        return self._runtime_functions.get(func_name)
    
    def list_runtime_functions(self) -> List[str]:
        """List all runtime function names"""
        return list(self._runtime_functions.keys())
    
    # ========== Source Code Methods ==========
    
    def register_source_file(self, source_file: str, source_code: str):
        """Register source file content"""
        self._source_files[source_file] = source_code
    
    def get_source_file(self, source_file: str) -> Optional[str]:
        """Get source file content"""
        return self._source_files.get(source_file)
    
    def register_function_source(self, source_file: str, func_name: str, source_code: str):
        """Register individual function source code"""
        key = f"{source_file}:{func_name}"
        self._function_sources[key] = source_code
    
    def get_function_source(self, func_name: str, source_file: Optional[str] = None) -> Optional[str]:
        """Get function source code"""
        if source_file:
            key = f"{source_file}:{func_name}"
            return self._function_sources.get(key)
        
        # Search all source files
        for key, source in self._function_sources.items():
            if key.endswith(f":{func_name}"):
                return source
        return None
    
    def list_function_sources(self) -> Dict[str, str]:
        """List all function sources"""
        return self._function_sources.copy()
    
    def list_source_files(self) -> List[str]:
        """List all source files that have compiled code
        
        Returns:
            List of source file paths
        """
        return list(self._source_files.keys())
    
    # ========== Compiler Methods ==========
    
    def register_compiler(self, source_file: str, compiler: Any):
        """Register a compiler instance for a source file"""
        self._compilers[source_file] = compiler
    
    def get_compiler(self, source_file: str) -> Optional[Any]:
        """Get compiler instance for a source file"""
        return self._compilers.get(source_file)
    
    def register_shared_library(self, source_file: str, lib_path: str):
        """Register compiled shared library path"""
        self._shared_libraries[source_file] = lib_path
    
    def get_shared_library(self, source_file: str) -> Optional[str]:
        """Get shared library path for a source file"""
        return self._shared_libraries.get(source_file)
    
    # ========== Struct Type Methods ==========
    
    def register_struct(self, struct_info: StructInfo):
        """Register a struct type"""
        # Build field indices if not provided
        if not struct_info.field_indices:
            struct_info.field_indices = {
                field_name: idx 
                for idx, (field_name, _) in enumerate(struct_info.fields)
            }
        
        self._structs[struct_info.name] = struct_info
    
    def register_struct_from_fields(self, name: str, fields: List[Tuple[str, Any]], 
                                   python_class: Optional[type] = None) -> StructInfo:
        """Register a struct type from field list
        
        This is a convenience method that creates a StructInfo and registers it.
        Compatible with the old struct_metadata.register_struct() API.
        
        Args:
            name: Struct name
            fields: List of (field_name, field_type) tuples
            python_class: Optional Python class that this struct represents
        
        Returns:
            The registered StructInfo
        """
        # Check if already registered - update if so
        if name in self._structs:
            struct_info = self._structs[name]
            struct_info.fields = fields
            struct_info.field_indices = {
                field_name: idx 
                for idx, (field_name, _) in enumerate(fields)
            }
            if python_class is not None:
                struct_info.python_class = python_class
            return struct_info
        
        # Create new StructInfo
        struct_info = StructInfo(
            name=name,
            fields=fields,
            python_class=python_class
        )
        self.register_struct(struct_info)
        return struct_info
    
    def get_struct(self, struct_name: str) -> Optional[StructInfo]:
        """Get struct information"""
        return self._structs.get(struct_name)
    
    def has_struct(self, struct_name: str) -> bool:
        """Check if a struct is registered"""
        return struct_name in self._structs
    
    def list_structs(self) -> List[str]:
        """List all struct names"""
        return list(self._structs.keys())
    
    def infer_struct_from_llvm_type(self, llvm_type: ir.Type) -> Optional[StructInfo]:
        """Try to infer struct metadata from LLVM type
        
        Args:
            llvm_type: LLVM type to match
        
        Returns:
            StructInfo if a match is found, None otherwise
        """
        if isinstance(llvm_type, ir.IdentifiedStructType):
            # Look for struct by name
            struct_name = llvm_type.name.strip('"')
            
            # Try exact match first
            struct_info = self.get_struct(struct_name)
            if struct_info:
                return struct_info
            
            # Try matching without suffix (e.g., "TestStruct.3" -> "TestStruct")
            if '.' in struct_name:
                base_name = struct_name.split('.')[0]
                struct_info = self.get_struct(base_name)
                if struct_info:
                    return struct_info
        
        elif isinstance(llvm_type, ir.LiteralStructType):
            # For literal struct types, match by field count
            field_count = len(llvm_type.elements)
            for struct_info in self._structs.values():
                if struct_info.get_field_count() == field_count:
                    return struct_info
        
        return None
    
    def infer_struct_from_access(self, llvm_type: ir.Type, field_name: str) -> Optional[StructInfo]:
        """Infer struct type from field access pattern
        
        Args:
            llvm_type: LLVM type being accessed
            field_name: Name of the field being accessed
        
        Returns:
            StructInfo if a match is found, None otherwise
        """
        # If we have an IdentifiedStructType, try to match by name
        if isinstance(llvm_type, ir.IdentifiedStructType):
            struct_name = llvm_type.name.strip('"')
            
            # Try exact match first
            if struct_name in self._structs:
                struct_info = self._structs[struct_name]
                if struct_info.has_field(field_name):
                    return struct_info
            
            # Try matching without suffix (e.g., "TestStruct.3" -> "TestStruct")
            if '.' in struct_name:
                base_name = struct_name.split('.')[0]
                if base_name in self._structs:
                    struct_info = self._structs[base_name]
                    if struct_info.has_field(field_name):
                        return struct_info
        
        # Try to find a struct that has this field
        for struct_info in self._structs.values():
            if struct_info.has_field(field_name):
                # Could add more sophisticated matching here
                return struct_info
        
        return None
    
    def clear_structs(self):
        """Clear all registered structs"""
        self._structs.clear()
    
    # ========== Builtin Entity Methods ==========
    
    def register_builtin_entity(self, name: str, entity_class: type):
        """Register a builtin entity (type or function)"""
        self._builtin_entities[name.lower()] = entity_class
    
    def get_builtin_entity(self, name: str) -> Optional[type]:
        """Get a builtin entity class by name"""
        return self._builtin_entities.get(name.lower())
    
    def has_builtin_entity(self, name: str) -> bool:
        """Check if a builtin entity exists"""
        return name.lower() in self._builtin_entities
    
    def list_builtin_entities(self) -> List[str]:
        """List all registered builtin entity names"""
        return list(self._builtin_entities.keys())
    
    def list_builtin_types(self) -> List[str]:
        """List all builtin types"""
        return [
            name for name, entity in self._builtin_entities.items() 
            if hasattr(entity, 'can_be_type') and entity.can_be_type()
        ]
    
    def list_builtin_functions(self) -> List[str]:
        """List all builtin functions"""
        return [
            name for name, entity in self._builtin_entities.items() 
            if hasattr(entity, 'can_be_called') and entity.can_be_called()
        ]
    
    # ========== Link Libraries Methods ==========
    
    def add_link_library(self, lib: str):
        """Add a library to link against
        
        Args:
            lib: Library name without 'lib' prefix or extension
                 Examples: 'c', 'm', 'pthread', 'gcc'
        """
        self._link_libraries.add(lib)
    
    def get_link_libraries(self) -> List[str]:
        """Get all libraries to link against
        
        Collects from:
        1. Explicitly added via add_link_library()
        2. Automatically from extern functions
        
        Returns:
            Sorted list of library names
        """
        libs = set(self._link_libraries)
        
        # Collect from extern functions
        for extern_info in self._extern_functions.values():
            if extern_info.lib:
                libs.add(extern_info.lib)
        
        return sorted(libs)
    
    # ========== Utility Methods ==========
    
    def clear_all(self):
        """Clear all registries (useful for testing)"""
        self._variable_registries.clear()
        self._compiled_functions.clear()
        self._function_info.clear()
        self._function_info_by_mangled.clear()
        self._function_type_hints.clear()
        self._extern_functions.clear()
        self._runtime_functions.clear()
        self._runtime_function_counter = 0
        self._source_files.clear()
        self._function_sources.clear()
        self._compilers.clear()
        self._shared_libraries.clear()
        self._structs.clear()
        self._builtin_entities.clear()
        self._link_libraries.clear()
    
    def clear_functions(self):
        """Clear only function-related registries"""
        self._compiled_functions.clear()
        self._function_info.clear()
        self._function_info_by_mangled.clear()
        self._function_type_hints.clear()
        self._extern_functions.clear()
        self._runtime_functions.clear()
        self._function_sources.clear()
        self._link_libraries.clear()
    
    def dump_state(self, verbose: bool = False):
        """Debug: dump registry state"""
        print("=" * 60)
        print("UNIFIED COMPILATION REGISTRY STATE")
        print("=" * 60)
        
        print(f"\n[Variables] {len(self._variable_registries)} registries")
        for compiler_id, var_reg in self._variable_registries.items():
            visible = var_reg.list_all_visible()
            print(f"  {compiler_id}: {len(visible)} variables")
            if verbose:
                for name, info in visible.items():
                    print(f"    - {name}: {info.type_hint}")
        
        print(f"\n[Functions] {len(self._function_info)} total")
        print(f"  Compiled: {sum(len(funcs) for funcs in self._compiled_functions.values())}")
        print(f"  Extern: {len(self._extern_functions)}")
        print(f"  Runtime: {len(self._runtime_functions)}")
        if verbose:
            for name, info in self._function_info.items():
                print(f"    - {name} ({info.source_file})")
        
        print(f"\n[Source Files] {len(self._source_files)}")
        for source_file in self._source_files.keys():
            funcs = self._compiled_functions.get(source_file, [])
            print(f"  {source_file}: {len(funcs)} functions")
        
        print(f"\n[Compilers] {len(self._compilers)}")
        for source_file in self._compilers.keys():
            lib = self._shared_libraries.get(source_file, "not compiled")
            print(f"  {source_file} -> {lib}")
        
        print(f"\n[Structs] {len(self._structs)}")
        if verbose:
            for name, info in self._structs.items():
                print(f"  {name}: {len(info.fields)} fields")
        
        print(f"\n[Builtin Entities] {len(self._builtin_entities)}")
        if verbose:
            types = self.list_builtin_types()
            funcs = self.list_builtin_functions()
            print(f"  Types: {', '.join(types)}")
            print(f"  Functions: {', '.join(funcs)}")
        
        print("=" * 60)


# Global unified registry instance
_unified_registry = UnifiedCompilationRegistry()


def get_unified_registry() -> UnifiedCompilationRegistry:
    """Get the global unified registry instance"""
    return _unified_registry


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def register_extern_function(name: str, return_type: Any = None, 
                            param_types: List[Any] = None, 
                            lib: str = "libc", 
                            calling_convention: str = "cdecl",
                            **kwargs) -> None:
    """Register an external function (backward compatibility)
    
    DEPRECATED: This is kept for backward compatibility with old code.
    New code should use UnifiedCompilationRegistry.register_extern_function()
    
    Note: Extra kwargs (like 'signature', 'function') are ignored in the new system.
    """
    extern_info = ExternFunctionInfo(
        name=name,
        lib=lib,
        calling_convention=calling_convention,
        return_type=return_type,
        param_types=param_types or []
    )
    _unified_registry.register_extern_function(extern_info)


# ============================================================================
# Struct Registry Compatibility Layer
# ============================================================================


def register_struct_from_class(cls) -> Optional[StructInfo]:
    """Register a struct from a Python class decorated with @compile
    
    Backward compatibility wrapper for struct_metadata.register_struct_from_class()
    """
    if not hasattr(cls, '_is_struct') or not cls._is_struct:
        return None
    
    if not hasattr(cls, '_struct_fields'):
        return None
    
    struct_name = cls.__name__
    fields = cls._struct_fields
    
    return _unified_registry.register_struct_from_fields(struct_name, fields, python_class=cls)


def get_field_index(struct_name: str, field_name: str) -> Optional[int]:
    """Get field index for a struct field
    
    Backward compatibility wrapper for struct_metadata.get_field_index()
    """
    struct_info = _unified_registry.get_struct(struct_name)
    if struct_info:
        return struct_info.get_field_index(field_name)
    return None


def get_struct_field_count(struct_name: str) -> int:
    """Get the number of fields in a struct
    
    Backward compatibility wrapper for struct_metadata.get_struct_field_count()
    """
    struct_info = _unified_registry.get_struct(struct_name)
    if struct_info:
        return struct_info.get_field_count()
    return 0


def infer_struct_from_access(llvm_type: ir.Type, field_name: str) -> Optional[StructInfo]:
    """Infer struct type from field access pattern
    
    Backward compatibility wrapper for struct_metadata.infer_struct_from_access()
    """
    return _unified_registry.infer_struct_from_access(llvm_type, field_name)
