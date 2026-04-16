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


# Global counter for unique variable IDs
_next_var_id: int = 0

def _get_next_var_id() -> int:
    """Get next unique variable ID"""
    global _next_var_id
    _next_var_id += 1
    return _next_var_id


@dataclass
class VariableInfo:
    """Information about a variable
    
    Stores variable metadata and value reference. The value_ref field contains
    the actual value (LLVM or Python) and type information.
    
    Each VariableInfo has a unique ID for CFG linear tracking, which allows
    distinguishing shadowed variables with the same name.
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
    
    # Unique ID for CFG linear tracking (distinguishes shadowed variables)
    var_id: int = field(default_factory=_get_next_var_id)
    
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
    """Static semantic record for a compiled function wrapper."""
    name: str
    source_file: str
    ast_node: Optional[ast.FunctionDef] = None
    llvm_function: Optional[ir.Function] = None
    return_type_hint: Optional[Any] = None
    param_type_hints: Dict[str, Any] = field(default_factory=dict)
    param_names: List[str] = field(default_factory=list)  # Ordered parameter names
    source_code: Optional[str] = None
    is_compiled: bool = False
    overload_enabled: bool = False  # Whether overloading is enabled for this function
    # Effect system: track which effects this function uses (e.g., {'rng', 'd_impl'})
    # Used for transitive effect propagation - when a function with suffix calls
    # another function that uses an overridden effect, we generate a suffix version
    effect_dependencies: Set[str] = field(default_factory=set)
    # Whether this function uses LLVM-level varargs (union/enum/none varargs).
    # Struct varargs are expanded at compile time and do not set this flag.
    has_llvm_varargs: bool = False
    # Whether this function was declared with *args: T (typed varargs).
    has_varargs: bool = False
    # Whether this function was declared with **kwargs: T.
    has_kwargs: bool = False
    # LLVM function-level attributes (e.g. {'readnone', 'nounwind'}).
    # Applied to cross-module `declare` so the optimizer can treat calls as pure, etc.
    fn_attrs: Set[str] = field(default_factory=set)
    binding_state: Optional[Any] = None

    @property
    def wrapper(self):
        return self.binding_state.wrapper if self.binding_state else None

    @property
    def compilation_globals(self):
        return self.binding_state.compilation_globals if self.binding_state else None

    @property
    def mangled_name(self):
        return self.binding_state.mangled_name if self.binding_state else None

    @property
    def so_file(self):
        return self.binding_state.so_file if self.binding_state else None

    @property
    def callable_pc_type(self):
        """Expose the public callable PC type for this function."""
        from .builtin_entities.func import func as func_type_cls

        param_types = tuple(self.param_type_hints[name] for name in self.param_names)
        # Build named items so that func type preserves param_names
        # (needed for keyword argument support at call sites).
        items = tuple(
            (name, self.param_type_hints[name]) for name in self.param_names
        ) + ((None, self.return_type_hint),)
        result = func_type_cls.handle_type_subscript(items)
        # Propagate *args/**kwargs flags so call sites can detect them
        if self.has_varargs:
            result.has_varargs = True
        if self.has_kwargs:
            result.has_kwargs = True
        return result


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

    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if variable is declared in the current scope"""
        return name in self.scopes[-1]

    def get_all_in_current_scope(self) -> List[VariableInfo]:
        """Get all variables in the current scope"""
        return list(self.scopes[-1].values())

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

        # ===== Link Objects Registry =====
        # Object files to link (from cimport compiled sources)
        self._link_objects: Set[str] = set()
    
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

        Returns libraries explicitly added via add_link_library().

        Returns:
            Sorted list of library names
        """
        return sorted(self._link_libraries)
    
    # ========== Link Objects Methods ==========
    
    def add_link_object(self, path: str):
        """Add an object file to link against
        
        Args:
            path: Path to .o object file
        """
        self._link_objects.add(path)
    
    def get_link_objects(self) -> List[str]:
        """Get all object files to link
        
        Returns:
            Sorted list of object file paths
        """
        return sorted(self._link_objects)
    
    def clear_link_objects(self):
        """Clear all registered link objects"""
        self._link_objects.clear()
    
    # ========== Utility Methods ==========

    def clear_all(self):
        """Clear all registries (useful for testing)"""
        self._source_files.clear()
        self._function_sources.clear()
        self._compilers.clear()
        self._shared_libraries.clear()
        self._structs.clear()
        self._builtin_entities.clear()
        self._link_libraries.clear()
        self._link_objects.clear()

    def dump_state(self, verbose: bool = False):
        """Debug: dump registry state"""
        print("=" * 60)
        print("UNIFIED COMPILATION REGISTRY STATE")
        print("=" * 60)

        print(f"\n[Source Files] {len(self._source_files)}")
        for source_file in self._source_files.keys():
            print(f"  {source_file}")

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
