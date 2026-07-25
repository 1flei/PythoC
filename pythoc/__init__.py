
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
    const, static, thread_local, volatile,
    TYPE_MAP,
    sizeof, offsetof, nullptr, typeof, char,
    seq, linear, consume, move, void,
    refined, assume, refine,
    pyconst, defer, label, goto, goto_end,
    va_start, va_arg, va_end,
    pc_literal,
    llvm_asm,
    param,
)

# Provide lowercase alias for convenience
from .decorators import compile, jit, extern, inline, get_compiler, clear_registry
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
from .utils.build_utils import (
    compile_to_executable,
    compile_to_static_library,
    compile_to_dynamic_library,
    export_c_headers,
)
from .cimport import cimport, cimport_header, cimport_source
from .config import config
from .forward_ref import mark_type_defined, register_forward_ref_callback
from .session import CompileSession

import builtins as _py

# Version information
__version__ = "0.5.0"
__author__ = "PythoC Compiler Team"


def init() -> CompileSession:
    """Ensure a compile session is active in this context and return it.

    Every process entry point that compiles PC code should call this once
    before any @compile decoration executes.  If the current context
    already has an active session it is returned unchanged; otherwise a
    new session is created and activated for the rest of the process.
    """
    session = CompileSession.active()
    if session is None:
        session = CompileSession()
        session.activate()
    return session


# libc/std/meta are imported lazily on first attribute access: importing
# them executes @compile/@extern/mark_type_defined at module level, which
# requires an active compile session.  Keeping them out of the eager
# import chain above keeps 'import pythoc' itself light.
_LAZY_SUBMODULES = ('libc', 'std', 'meta')


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module('.' + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name))


# Export public API
__all__ = [
    # Integer types
    'i8', 'i16', 'i32', 'i64',
    'u8', 'u16', 'u32', 'u64',
    # Floating point types
    'f16', 'bf16', 'f32', 'f64', 'f128',
    # Other types
    'bool', 'ptr', 'array', 'struct', 'union', 'func', 'enum',
    # Parametric parameter type
    'param',
    # Refined types
    'refined', 'assume', 'refine',
    # Type qualifiers
    'const', 'static', 'thread_local', 'volatile',
    # Type utilities
    'TYPE_MAP',
    'sizeof',
    'offsetof',
    'typeof',
    'char',
    'pyconst',
    
    # Decorators
    'compile', 'jit', 'extern', 'inline',
    
    # Effect system
    'effect',

    # Compile session
    'CompileSession',
    'init',
    
    # C Library
    'libc',
    'std',

    # Meta module
    'meta',
    
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
    'clear_registry',
    'nullptr',
    'sizeof',
    'offsetof',
    'typeof',
    'char',
    'pyconst',
    'seq',
    'linear',
    'move',
    'consume',
    'void',
    'defer',
    'label',
    'goto',
    'goto_end',
    'va_start',
    'va_arg',
    'va_end',

    # PC literal carrier
    'pc_literal',

    # Forward reference resolution
    'mark_type_defined',
    'register_forward_ref_callback',

    # Centralised runtime configuration
    'config',

    'compile_to_executable',
    'compile_to_static_library',
    'compile_to_dynamic_library',
    'export_c_headers',
    
    # C Import
    'cimport',
    'cimport_header',
    'cimport_source',
    
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


# Convenience default: importing pythoc installs a session in this context
# so plain scripts and libraries work without boilerplate.  Internals never
# rely on this being the only session: the session is captured explicitly
# at decoration time, and tests/embedders can shadow or replace it with
# 'with CompileSession():' or CompileSession().activate().  Note that a
# threading.Thread starts with an empty context and must re-activate a
# session itself.
init()
