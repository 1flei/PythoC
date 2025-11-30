"""
Linear wrapper for resource management functions

Provides metaprogramming utilities to wrap non-linear functions 
with linear token enforcement for resource safety.

Usage:
    from pythoc.std.linear_wrapper import linear_wrap
    from pythoc.libc.stdlib import malloc, free
    
    # Generate linear-safe malloc/free pair
    lmalloc, lfree = linear_wrap(malloc, free)
    
    # Use with custom struct name
    from pythoc.libc.stdio import fopen, fclose
    lfopen, lfclose = linear_wrap(
        fopen, fclose, 
        resource_struct_name="FileHandle"
    )
"""

from ..decorators import compile
from ..builtin_entities import linear, struct, consume
from ..registry import get_unified_registry


def _extract_function_info(func):
    """Extract function info from either compiled functions or extern wrappers"""
    # Handle ExternFunctionWrapper
    if hasattr(func, 'func_name') and hasattr(func, 'param_types'):
        func_name = func.func_name
        param_types = [ptype for _, ptype in func.param_types if _ != 'args']  # Filter out varargs
        return_type = func.return_type
        return func_name, param_types, return_type
    
    # Handle @compile functions through registry
    func_name = getattr(func, '__name__', str(func))
    registry = get_unified_registry()
    
    func_info = registry.get_function_info(func_name)
    if not func_info:
        lookup = getattr(func, '_mangled_name', None)
        if lookup:
            func_info = registry.get_function_info_by_mangled(lookup)
    
    if not func_info:
        raise NameError(f"Function '{func_name}' not found in registry")
    
    param_types = [
        func_info.param_type_hints[name] 
        for name in func_info.param_names
    ]
    return_type = func_info.return_type_hint
    
    return func_name, param_types, return_type


def linear_wrap(acquire_func, release_func, struct_name=None):
    """
    Wrap a pair of resource acquire/release functions with linear token enforcement.

    Args:
        acquire_func: Function that acquires a resource (e.g., malloc, fopen)
        release_func: Function that releases a resource (e.g., free, fclose)
        struct_name: Optional custom name for the proof struct type.
                    If provided, returns (ProofStruct, wrapped_acquire, wrapped_release).
                    If None, returns (wrapped_acquire, wrapped_release) with raw linear.
    
    Returns:
        If struct_name is None:
            (wrapped_acquire, wrapped_release)
            - acquire returns struct[resource, linear]
            - release takes (resource_params..., linear)
        If struct_name is provided:
            (ProofStruct, wrapped_acquire, wrapped_release)
            - ProofStruct is a named struct[linear] type
            - acquire returns struct[resource, ProofStruct]
            - release takes (resource_params..., ProofStruct)
    """
    
    # Extract function information
    acquire_name, acquire_param_types, acquire_return_type = _extract_function_info(acquire_func)
    release_name, release_param_types, _ = _extract_function_info(release_func)
    
    # Create proof struct type
    if struct_name:
        ProofStruct = struct[linear]
        ProofStruct.__name__ = struct_name
        proof_type = ProofStruct
        
        # Create a helper function to make proof struct
        @compile(anonymous=True)
        def make_proof() -> proof_type:
            prf: proof_type
            prf[0] = linear()
            return prf

        @compile(anonymous=True)
        def release_proof(prf: proof_type):
            consume(prf[0])
    else:
        proof_type = linear
        @compile(anonymous=True)
        def make_proof() -> proof_type:
            return linear()
        
        @compile(anonymous=True)
        def release_proof(prf: proof_type):
            consume(prf)
    
    # Build return struct type
    ReturnStruct = struct[proof_type, acquire_return_type]
    
    # Build acquire parameter struct
    AcquireParamsStruct = struct[tuple(acquire_param_types)]
    
    # Build wrapped acquire function
    @compile(anonymous=True)
    def wrapped_acquire(*args: AcquireParamsStruct) -> ReturnStruct:
        ret: ReturnStruct
        ret[0] = make_proof()
        ret[1] = acquire_func(*args)
        return ret
    
    # Build release function
    ReleaseParams = struct[tuple(release_param_types)]

    # Raw linear: consume directly
    @compile(anonymous=True)
    def wrapped_release(prf: proof_type, *args: ReleaseParams):
        release_func(*args)
        release_proof(prf)
    
    if struct_name:
        return ProofStruct, wrapped_acquire, wrapped_release
    else:
        return wrapped_acquire, wrapped_release

from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose
lmalloc, lfree = linear_wrap(malloc, free)
lfopen, lfclose = linear_wrap(fopen, fclose)
