# varargs ABI v2
"""
Linearize resource management functions.
# *args: T refactored to single-param ABI

Provides metaprogramming utilities to transform a pair of acquire/release
functions into linear-token enforced APIs for resource safety.

Usage:
    from pythoc.std.linearize import linearize
    from pythoc.libc.stdlib import malloc, free

    # Generate linear-safe malloc/free pair
    lmalloc, lfree = linearize(malloc, free)

    # Use with custom proof struct name
    from pythoc.libc.stdio import fopen, fclose
    FileHandle, lfopen, lfclose = linearize(
        fopen,
        fclose,
        struct_name="FileHandle",
    )
"""

from ..decorators import compile
from ..builtin_entities import linear, struct, consume, refined, assume


def _extract_function_info(func):
    """Extract function info from either compiled functions or extern wrappers."""
    # Handle ExternFunctionWrapper
    if hasattr(func, 'func_name') and hasattr(func, 'param_types'):
        func_name = func.func_name
        param_types = [ptype for name, ptype in func.param_types if name != 'args']
        return_type = func.return_type
        return func_name, param_types, return_type

    # Handle @compile functions - get func_info directly from wrapper
    func_info = getattr(func, '_func_info', None)
    if not func_info:
        func_name = getattr(func, '__name__', str(func))
        raise NameError(f"Function '{func_name}' not found - missing _func_info attribute")

    func_name = getattr(func, '__name__', str(func))
    param_types = [
        func_info.param_type_hints[name]
        for name in func_info.param_names
    ]
    return_type = func_info.return_type_hint

    return func_name, param_types, return_type


def linearize(acquire_func, release_func, struct_name=None):
    """Linearize an acquire/release pair with a linear proof token.

    Args:
        acquire_func: Function that acquires a resource (e.g., malloc, fopen)
        release_func: Function that releases a resource (e.g., free, fclose)
        struct_name: Optional custom name for the proof struct type.
            If provided, returns (ProofType, wrapped_acquire, wrapped_release).
            If None, returns (wrapped_acquire, wrapped_release) with raw linear.

    Returns:
        If struct_name is None:
            (wrapped_acquire, wrapped_release)
            - acquire returns struct[resource, linear]
            - release takes (resource_params..., linear)
        If struct_name is provided:
            (ProofType, wrapped_acquire, wrapped_release)
            - ProofType is refined[linear, struct_name]
            - acquire returns struct[resource, ProofType]
            - release takes (resource_params..., ProofType)

    Note:
        The proof is always the SECOND field in the acquire return struct.
        This matches the natural resource-first calling convention.
    """

    acquire_name, acquire_param_types, acquire_return_type = _extract_function_info(acquire_func)
    release_name, release_param_types, _ = _extract_function_info(release_func)

    # Deterministic suffix ensures stable symbol names across processes.
    # Format: lz_<acquire>_<release>[_<struct_name>] (lz = linearize)
    if struct_name:
        deterministic_suffix = f"lz_{acquire_name}_{release_name}_{struct_name}"
    else:
        deterministic_suffix = f"lz_{acquire_name}_{release_name}"

    # Suppress the effect suffix so @compile inside linearize always produces
    # the default (non-suffixed) version. Without this, if linearize() is called
    # during module init inside `with effect(...)`, the wrappers would pick up
    # the effect suffix and permanently contaminate the module-level bindings.
    # Effect specialization is handled later by the import hook via
    # get_effect_specialized().
    from ..effect import suppress_effect_suffix

    with suppress_effect_suffix():
        # Create proof type
        if struct_name:
            ProofType = refined[linear, struct_name]

            @compile(suffix=deterministic_suffix)
            def make_proof() -> ProofType:
                return assume(linear(), struct_name)
        else:
            ProofType = linear

            @compile(suffix=deterministic_suffix)
            def make_proof() -> ProofType:
                return linear()

        @compile(suffix=deterministic_suffix)
        def release_proof(prf: ProofType):
            consume(prf)

        # acquire returns (resource, proof)
        ReturnStruct = struct[acquire_return_type, ProofType]

        AcquireParamsStruct = struct[tuple(acquire_param_types)]

        @compile(suffix=deterministic_suffix)
        def wrapped_acquire(*args: AcquireParamsStruct) -> ReturnStruct:
            ret: ReturnStruct
            ret[0] = acquire_func(*args)
            ret[1] = make_proof()
            return ret

        # release takes (resource_params..., proof)
        ReleaseParamsStruct = struct[tuple(release_param_types)]
        ReleaseWithProofStruct = struct[tuple(release_param_types + [ProofType])]
        num_release_params = len(release_param_types)

        @compile(suffix=deterministic_suffix)
        def wrapped_release(*args: ReleaseWithProofStruct):
            params: ReleaseParamsStruct
            for i in range(num_release_params):
                params[i] = args[i]

            prf: ProofType = args[num_release_params]
            release_func(*params)
            release_proof(prf)

    if struct_name:
        return ProofType, wrapped_acquire, wrapped_release
    else:
        return wrapped_acquire, wrapped_release
