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
        struct_name: Optional custom name for the resource struct type.
                    If provided, returns (ResourceStruct, acquire, release) tuple.
                    If None, returns (acquire, release) tuple.
    
    Returns:
        (wrapped_acquire, wrapped_release) if struct_name is None
        (ResourceStruct, wrapped_acquire, wrapped_release) if struct_name is provided
        
    Example:
        from pythoc.libc.stdlib import malloc, free
        lmalloc, lfree = linear_wrap(malloc, free)
        
        # With named struct
        FileHandle, lfopen, lfclose = linear_wrap(fopen, fclose, 
                                                   struct_name="FileHandle")
        
        # Generated signatures:
        # lmalloc(size: i64) -> struct[ptr[i8], linear]
        # lfree(ptr: ptr[i8], token: linear) -> void
    """
    
    # Extract function information
    acquire_name, acquire_param_types, acquire_return_type = _extract_function_info(acquire_func)
    release_name, release_param_types, _ = _extract_function_info(release_func)
    
    # Create resource struct type
    ResourceStruct = struct[acquire_return_type, linear]
    if struct_name:
        ResourceStruct.__name__ = struct_name
    
    # Build acquire parameter struct
    AcquireParamsStruct = struct[tuple(acquire_param_types)] if acquire_param_types else None
    
    # Build wrapped acquire function - use anonymous compilation
    if AcquireParamsStruct:
        @compile(anonymous=True)
        def wrapped_acquire(*args: AcquireParamsStruct) -> ResourceStruct:
            resource = acquire_func(*args)
            return resource, linear()
    else:
        @compile(anonymous=True)
        def wrapped_acquire() -> ResourceStruct:
            resource = acquire_func()
            return resource, linear()
    
    # Build release function
    ReleaseParams = struct[tuple(release_param_types)] if release_param_types else None
    ReleaseParamsWithToken = struct[tuple(release_param_types + [linear])]
    
    # Build wrapped release function using struct unpacking
    if ReleaseParams:
        @compile(anonymous=True)
        def wrapped_release(*args: ReleaseParamsWithToken):
            release_params: ReleaseParams
            for i in range(len(release_param_types)):
                release_params[i] = args[i]
            release_func(*release_params)
            consume(args[len(release_param_types)])
    else:
        @compile(anonymous=True)
        def wrapped_release(*args: ReleaseParamsWithToken):
            release_func()
            consume(args[0])
    
    if struct_name:
        return ResourceStruct, wrapped_acquire, wrapped_release
    else:
        return wrapped_acquire, wrapped_release

from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose
lmalloc, lfree = linear_wrap(malloc, free)
lfopen, lfclose = linear_wrap(fopen, fclose)
