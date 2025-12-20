
"""
PythoC: Python DSL to LLVM IR Compiler
A Python DSL compiler that maps statically-typed Python subset to LLVM IR,
providing C-equivalent capabilities with Python syntax.
"""

# Automatically enable future annotations for better type handling
from __future__ import annotations

from .builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64, 
    f16, bf16, f32, f64, f128, bool,
    ptr, array, struct, union, func, enum,
    const, static, volatile,
    TYPE_MAP,
    sizeof, nullptr, typeof, char,
    seq, linear, consume, void,
    refined, assume, refine,
    pyconst
)

# Provide lowercase alias for convenience
from .decorators import compile, jit, extern, inline, get_compiler, list_compiled_functions, clear_registry
from .effect import effect
from .decorators.compile import flush_all_pending_outputs
from .compiler import LLVMCompiler
from .utils import (
    analyze_function, 
    get_llvm_version, 
    print_module_info, 
    validate_ir,
    compare_performance,
    disassemble_to_native,
    create_build_info
)
from .utils.build_utils import compile_to_executable

# Import libc module for convenient access
from . import libc
import builtins as _py

# Version information
__version__ = "0.3.0"
__author__ = "PythoC Compiler Team"

# Export public API
__all__ = [
    # Integer types
    'i8', 'i16', 'i32', 'i64',
    'u8', 'u16', 'u32', 'u64',
    # Floating point types
    'f16', 'bf16', 'f32', 'f64', 'f128',
    # Other types
    'bool', 'ptr', 'array', 'struct', 'union', 'func', 'enum',
    # Refined types
    'refined', 'assume', 'refine',
    # Type qualifiers
    'const', 'static', 'volatile',
    # Type utilities
    'TYPE_MAP',
    'sizeof',
    'typeof',
    'char',
    'pyconst',
    
    # Decorators
    'compile', 'jit', 'extern', 'inline',
    
    # Effect system
    'effect',
    
    # C Library
    'libc',
    
    # Core compiler
    'LLVMCompiler',
    
    # Utilities
    'analyze_function',
    'get_llvm_version',
    'print_module_info',
    'validate_ir',
    'compare_performance',
    'disassemble_to_native',
    'create_build_info',
    'get_compiler',
    'list_compiled_functions',
    'clear_registry',
    'nullptr',
    'sizeof',
    'typeof',
    'char',
    'pyconst',
    'seq',
    'linear',
    'consume',
    'void',
    'compile_to_executable',
    
    # Metadata
    '__version__',
    '__author__'
]

# Auto-export dynamic iN/uN types from unified registry
from .builtin_entities import get_builtin_entity
for _w in _py.range(1, 65):
    for _p in ('i', 'u'):
        _n = f'{_p}{_w}'
        _ent = get_builtin_entity(_n)
        if _ent is not None:
            globals()[_n] = _ent
            if _n not in __all__:
                __all__.append(_n)


def info():
    """Print information about the PythoC compiler"""
    build_info = create_build_info()
    print("PythoC Compiler v{}".format(__version__))
    print("   LLVM Version: {}".format(build_info['llvm_version']))
    print("   Target Triple: {}".format(build_info['target_triple']))
    print("   Host CPU: {}".format(build_info['host_cpu']))
    print("   Features: Enhanced AST visitor, Multi-function compilation, Optimization")
    print("   Backend: llvmlite")

def hello():
    """Print a welcome message"""
    print("Welcome to PythoC Compiler v{}!".format(__version__))
    print("   A Python DSL compiler that maps statically-typed Python subset to LLVM IR")
    print("   Use @compile decorator to compile your functions to LLVM IR")
    print("   Use @jit decorator for Just-In-Time compilation")
    print("   Call pythoc.info() for more details")
