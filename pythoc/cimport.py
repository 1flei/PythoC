"""
C Import (cimport) - Import C headers/sources as pythoc modules

This module provides the cimport() function to:
1. Parse C header/source files
2. Generate pythoc bindings
3. Import the bindings as a Python module
4. Optionally compile C sources and register for linking

Usage:
    from pythoc.cimport import cimport
    
    # Header-only import with library
    libc = cimport('stdio.h', lib='c')
    
    # Import with source compilation
    mylib = cimport('mylib.h', sources=['mylib.c'], compile_sources=True)
    
    # Direct C source import
    mod = cimport('helper.c', lib='helper', compile_sources=True)
"""

import os
import sys
import hashlib
import importlib.util
from typing import Optional, List, Any
from types import ModuleType

from .registry import get_unified_registry
from .utils.cc_utils import compile_c_to_object, compile_c_sources


def _compute_cache_key(path: str, lib: str, sources: Optional[List[str]] = None,
                       objects: Optional[List[str]] = None) -> str:
    """Compute a cache key for the bindings module.
    
    Args:
        path: Path to C header/source file
        lib: Library name
        sources: Additional source files
        objects: Object files
    
    Returns:
        Hex hash string for caching
    """
    hasher = hashlib.sha256()
    
    # Include main file path and mtime
    hasher.update(path.encode())
    if os.path.exists(path):
        hasher.update(str(os.path.getmtime(path)).encode())
    
    # Include lib name
    hasher.update((lib or '').encode())
    
    # Include source files
    for src in sorted(sources or []):
        hasher.update(src.encode())
        if os.path.exists(src):
            hasher.update(str(os.path.getmtime(src)).encode())
    
    # Include object files
    for obj in sorted(objects or []):
        hasher.update(obj.encode())
        if os.path.exists(obj):
            hasher.update(str(os.path.getmtime(obj)).encode())
    
    return hasher.hexdigest()[:16]


def _get_cache_dir(cache_key: str) -> str:
    """Get the cache directory for a given cache key."""
    cache_dir = os.path.join('build', 'cimport', cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _read_file_content(path: str) -> str:
    """Read file content as string."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _generate_bindings_pure_python(source_text: str, lib: str, output_path: str) -> None:
    """Generate bindings using pure Python code generation.
    
    This approach parses C declarations and generates pythoc bindings
    without requiring the compiled bindgen (which has runtime constraints).
    
    Uses a simplified approach that generates valid pythoc code.
    """
    import re
    
    lines = []
    lines.append('"""Auto-generated pythoc bindings"""\n')
    lines.append('')
    lines.append('from pythoc import (')
    lines.append('    compile, extern, enum, i8, i16, i32, i64,')
    lines.append('    u8, u16, u32, u64, f32, f64, ptr, array,')
    lines.append('    void, char, nullptr, sizeof, struct, union')
    lines.append(')')
    lines.append('')
    
    # Simple C type to pythoc type mapping
    type_map = {
        'void': 'void',
        'char': 'char',
        'signed char': 'i8',
        'unsigned char': 'u8',
        'short': 'i16',
        'short int': 'i16',
        'unsigned short': 'u16',
        'unsigned short int': 'u16',
        'int': 'i32',
        'unsigned': 'u32',
        'unsigned int': 'u32',
        'long': 'i64',
        'long int': 'i64',
        'unsigned long': 'u64',
        'unsigned long int': 'u64',
        'long long': 'i64',
        'long long int': 'i64',
        'unsigned long long': 'u64',
        'unsigned long long int': 'u64',
        'float': 'f32',
        'double': 'f64',
        'long double': 'f64',
        'size_t': 'u64',
        'ssize_t': 'i64',
        'ptrdiff_t': 'i64',
        'intptr_t': 'i64',
        'uintptr_t': 'u64',
    }
    
    def parse_c_type(type_str: str) -> str:
        """Convert C type string to pythoc type."""
        type_str = type_str.strip()
        
        # Remove const/volatile qualifiers
        type_str = re.sub(r'\b(const|volatile)\b', '', type_str).strip()
        type_str = re.sub(r'\s+', ' ', type_str)
        
        # Count and remove pointer stars
        ptr_count = type_str.count('*')
        type_str = type_str.replace('*', '').strip()
        
        # Look up base type
        base_type = type_map.get(type_str, type_str)
        
        # Wrap in ptr[] for pointers
        result = base_type
        for _ in range(ptr_count):
            result = f'ptr[{result}]'
        
        return result
    
    # Preprocess: remove preprocessor directives and comments
    # Remove single-line comments
    source_text = re.sub(r'//[^\n]*', '', source_text)
    # Remove multi-line comments
    source_text = re.sub(r'/\*.*?\*/', '', source_text, flags=re.DOTALL)
    # Remove preprocessor directives
    source_text = re.sub(r'^\s*#[^\n]*', '', source_text, flags=re.MULTILINE)
    
    # Remove function bodies (content between { and } after function signature)
    # This is a simplified approach that handles nested braces
    def remove_function_bodies(text: str) -> str:
        result = []
        depth = 0
        i = 0
        in_function = False
        func_depth = 0  # Track depth when function body started
        while i < len(text):
            c = text[i]
            if c == '{':
                depth += 1
                if not in_function:
                    # Check if this is a function body (preceded by ')')
                    # Look back for ')'
                    j = i - 1
                    while j >= 0 and text[j] in ' \t\n':
                        j -= 1
                    if j >= 0 and text[j] == ')':
                        in_function = True
                        func_depth = depth
                        result.append(';')  # Replace function body with semicolon
                        i += 1
                        continue
                if not in_function:
                    result.append(c)
            elif c == '}':
                if in_function and depth == func_depth:
                    in_function = False
                    func_depth = 0
                    depth -= 1
                    i += 1
                    continue
                depth -= 1
                if not in_function:
                    result.append(c)
            else:
                if not in_function:
                    result.append(c)
            i += 1
        return ''.join(result)
    
    source_text = remove_function_bodies(source_text)
    
    # Parse function declarations: return_type name(params);
    func_pattern = re.compile(
        r'^\s*(?:extern\s+)?(?:static\s+)?'
        r'((?:const\s+|volatile\s+|unsigned\s+|signed\s+|long\s+|short\s+)*'
        r'(?:void|char|int|float|double|long|short|unsigned|signed|[a-zA-Z_][a-zA-Z0-9_]*)'
        r'(?:\s*\*)*)\s+'
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*'
        r'\(([^)]*)\)\s*;',
        re.MULTILINE
    )
    
    # Parse struct declarations
    struct_pattern = re.compile(
        r'(?:typedef\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{([^}]*)\}\s*'
        r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\s*)?;',
        re.MULTILINE | re.DOTALL
    )
    
    # Parse typedef declarations
    typedef_pattern = re.compile(
        r'typedef\s+((?:const\s+|volatile\s+|unsigned\s+|signed\s+|long\s+|short\s+)*'
        r'(?:void|char|int|float|double|long|short|unsigned|signed|[a-zA-Z_][a-zA-Z0-9_]*)'
        r'(?:\s*\*)*)\s+'
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*;',
        re.MULTILINE
    )
    
    # Parse enum declarations
    enum_pattern = re.compile(
        r'(?:typedef\s+)?enum\s+(?:([a-zA-Z_][a-zA-Z0-9_]*)\s*)?\{([^}]*)\}\s*'
        r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\s*)?;',
        re.MULTILINE | re.DOTALL
    )
    
    # Process structs
    for match in struct_pattern.finditer(source_text):
        struct_name = match.group(1) or match.group(3)
        if not struct_name:
            continue
        fields_text = match.group(2)
        
        lines.append('')
        lines.append('@compile')
        lines.append(f'class {struct_name}:')
        
        # Parse fields
        field_lines = fields_text.strip().split(';')
        has_fields = False
        for field_line in field_lines:
            field_line = field_line.strip()
            if not field_line:
                continue
            # Simple field parsing: type name
            parts = field_line.rsplit(None, 1)
            if len(parts) == 2:
                field_type, field_name = parts
                field_name = field_name.strip()
                # Handle array notation
                if '[' in field_name:
                    field_name = field_name.split('[')[0]
                pythoc_type = parse_c_type(field_type)
                lines.append(f'    {field_name}: {pythoc_type}')
                has_fields = True
        
        if not has_fields:
            lines.append('    pass')
        lines.append('')
    
    # Process enums
    for match in enum_pattern.finditer(source_text):
        enum_name = match.group(1) or match.group(3)
        if not enum_name:
            continue
        values_text = match.group(2)
        
        lines.append('')
        lines.append('@enum(i32)')
        lines.append(f'class {enum_name}:')
        
        # Parse enum values
        values = [v.strip() for v in values_text.split(',') if v.strip()]
        if values:
            for val in values:
                if '=' in val:
                    name, value = val.split('=', 1)
                    lines.append(f'    {name.strip()} = {value.strip()}')
                else:
                    lines.append(f'    {val}: None')
        else:
            lines.append('    pass')
        lines.append('')
    
    # Process typedefs
    for match in typedef_pattern.finditer(source_text):
        orig_type = match.group(1)
        new_name = match.group(2)
        pythoc_type = parse_c_type(orig_type)
        lines.append(f'{new_name} = {pythoc_type}')
        lines.append('')
    
    # Process functions
    for match in func_pattern.finditer(source_text):
        return_type = match.group(1)
        func_name = match.group(2)
        params_text = match.group(3).strip()
        
        pythoc_return = parse_c_type(return_type)
        
        # Parse parameters
        params = []
        if params_text and params_text != 'void':
            param_list = params_text.split(',')
            for i, param in enumerate(param_list):
                param = param.strip()
                if param == '...':
                    params.append('*args')
                elif param:
                    # Parse param: type name or just type
                    parts = param.rsplit(None, 1)
                    if len(parts) == 2:
                        param_type, param_name = parts
                        # Handle array notation in param name
                        if '[' in param_name:
                            param_name = param_name.split('[')[0]
                        param_name = param_name.strip('*')
                    else:
                        param_type = parts[0]
                        param_name = f'arg{i}'
                    pythoc_param_type = parse_c_type(param_type)
                    params.append(f'{param_name}: {pythoc_param_type}')
        
        params_str = ', '.join(params)
        
        # Generate @extern decorator
        # If lib is empty, omit the lib parameter (symbols from .o files)
        if lib:
            lines.append(f"@extern(lib='{lib}')")
        else:
            lines.append('@extern')
        lines.append(f'def {func_name}({params_str}) -> {pythoc_return}:')
        lines.append('    pass')
        lines.append('')
    
    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _import_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """Import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cimport(path: str, *,
            kind: str = 'auto',
            lib: Optional[str] = None,
            sources: Optional[List[str]] = None,
            objects: Optional[List[str]] = None,
            compile_sources: bool = False,
            cc: Optional[str] = None,
            cflags: Optional[List[str]] = None,
            include_dirs: Optional[List[str]] = None,
            defines: Optional[List[str]] = None,
            export: Optional[List[str]] = None,
            export_all: bool = False,
            prefix: Optional[str] = None) -> ModuleType:
    """Import C header/source and return a pythoc bindings module.
    
    Args:
        path: Path to .h or .c file
        kind: 'auto' (infer from extension), 'header', or 'source'
        lib: Library name for @extern(lib='...'). If contains '/' treated as path.
        sources: Additional .c sources to compile
        objects: Explicit .o files to register for linking
        compile_sources: If True, compile .c sources to .o
        cc: C compiler to use (auto-detect if None)
        cflags: Additional compiler flags
        include_dirs: Include directories for compilation
        defines: Preprocessor defines
        export: Symbol names to export to caller globals (explicit opt-in)
        export_all: If True, export all symbols to caller globals
        prefix: Optional symbol prefix
    
    Returns:
        Module object containing the generated bindings
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If parsing or compilation fails
    """
    # Resolve path
    if not os.path.isabs(path):
        # Try relative to caller's directory first
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get('__file__')
            if caller_file:
                caller_dir = os.path.dirname(os.path.abspath(caller_file))
                candidate = os.path.join(caller_dir, path)
                if os.path.exists(candidate):
                    path = candidate
    
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"C file not found: {path}")
    
    # Determine kind
    if kind == 'auto':
        ext = os.path.splitext(path)[1].lower()
        if ext == '.h':
            kind = 'header'
        elif ext == '.c':
            kind = 'source'
        else:
            kind = 'header'  # Default to header for unknown extensions
    
    # Default lib name
    # When compile_sources=True and lib is not specified, use empty string
    # to indicate symbols come from directly linked object files
    if lib is None:
        if compile_sources:
            # Symbols will be resolved from .o files, no library needed
            lib = ''
        else:
            basename = os.path.splitext(os.path.basename(path))[0]
            lib = basename
    
    # Initialize lists
    sources = list(sources or [])
    objects = list(objects or [])
    
    # For source files, add to sources list for compilation
    if kind == 'source' and compile_sources:
        if path not in sources:
            sources.insert(0, path)
    
    # Compute cache key
    cache_key = _compute_cache_key(path, lib, sources, objects)
    cache_dir = _get_cache_dir(cache_key)
    
    # Generate bindings module path
    basename = os.path.splitext(os.path.basename(path))[0]
    if prefix:
        module_name = f"_cimport_{prefix}_{basename}_{cache_key}"
    else:
        module_name = f"_cimport_{basename}_{cache_key}"
    bindings_path = os.path.join(cache_dir, f"bindings_{basename}.py")
    
    # Check if bindings need regeneration
    needs_regen = not os.path.exists(bindings_path)
    if not needs_regen:
        bindings_mtime = os.path.getmtime(bindings_path)
        source_mtime = os.path.getmtime(path)
        if source_mtime > bindings_mtime:
            needs_regen = True
    
    # Generate bindings if needed
    if needs_regen:
        source_text = _read_file_content(path)
        _generate_bindings_pure_python(source_text, lib, bindings_path)
    
    # Compile sources if requested
    if compile_sources and sources:
        compiled_objects = compile_c_sources(
            sources, cc=cc, cflags=cflags,
            include_dirs=include_dirs, defines=defines,
            cache_dir=cache_dir
        )
        objects.extend(compiled_objects)
    
    # Register objects for linking
    registry = get_unified_registry()
    for obj in objects:
        registry.add_link_object(obj)
    
    # Import the bindings module
    module = _import_module_from_file(module_name, bindings_path)
    
    # Handle exports to caller globals
    if export or export_all:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_globals = frame.f_back.f_globals
            
            if export_all:
                # Export all public symbols
                for name in dir(module):
                    if not name.startswith('_'):
                        caller_globals[name] = getattr(module, name)
            elif export:
                # Export only specified symbols
                for name in export:
                    if hasattr(module, name):
                        caller_globals[name] = getattr(module, name)
                    else:
                        raise AttributeError(
                            f"Symbol '{name}' not found in generated bindings"
                        )
    
    return module


# Convenience alias
def cimport_header(path: str, lib: str, **kwargs) -> ModuleType:
    """Import a C header file.
    
    Convenience wrapper for cimport(..., kind='header').
    """
    return cimport(path, kind='header', lib=lib, **kwargs)


def cimport_source(path: str, lib: Optional[str] = None,
                   compile_sources: bool = True, **kwargs) -> ModuleType:
    """Import a C source file.
    
    Convenience wrapper for cimport(..., kind='source', compile_sources=True).
    """
    return cimport(path, kind='source', lib=lib, compile_sources=compile_sources, **kwargs)
