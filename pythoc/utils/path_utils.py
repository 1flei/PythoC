import os

from .link_utils import get_shared_lib_extension


def sanitize_filename(name):
    """
    Convert string to safe filename component.
    
    Args:
        name: String to sanitize
    
    Returns:
        Safe filename string, or None if input is None
    """
    if name is None:
        return None
    
    # Replace special chars
    safe = name.replace('[', '_').replace(']', '_').replace('*', 'ptr_')
    safe = safe.replace('<', '_').replace('>', '_').replace(':', '_')
    safe = safe.replace('/', '_').replace('\\', '_').replace('|', '_')
    safe = safe.replace('?', '_').replace('"', '_').replace(' ', '_')
    
    # If too long or still has problematic chars, use hash
    if len(safe) > 50:
        import hashlib
        hash_val = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"hash_{hash_val}"
    
    return safe


def get_build_paths(source_file, suffix=None, scope=None):
    """
    Calculate build output paths (ir_file, obj_file, so_file).
    
    Args:
        source_file: Source file path
        suffix: Optional suffix for specialized versions
        scope: Optional scope name (e.g., factory function name)
    
    Returns:
        tuple: (build_dir, ir_file, obj_file, so_file)
    """
    cwd = os.getcwd()
    
    # Calculate relative path for build directory
    if source_file.startswith(cwd):
        rel_path = os.path.relpath(source_file, cwd)
    else:
        rel_path = source_file
    
    build_dir = os.path.join('build', os.path.dirname(rel_path))
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    
    # Build filename with optional scope and suffix
    if suffix and scope:
        safe_suffix = sanitize_filename(suffix)
        file_base = f"{base_name}.{scope}.{safe_suffix}"
    elif suffix:
        safe_suffix = sanitize_filename(suffix)
        file_base = f"{base_name}.{safe_suffix}"
    else:
        file_base = base_name
    
    # Ensure build directory exists
    if build_dir and not os.path.exists(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    
    # Calculate output file paths
    lib_ext = get_shared_lib_extension()
    if build_dir:
        ir_file = os.path.join(build_dir, file_base + '.ll')
        obj_file = os.path.join(build_dir, file_base + '.o')
        so_file = os.path.join(build_dir, file_base + lib_ext)
    else:
        # Fallback to build/ root
        ir_file = os.path.join('build', file_base + '.ll')
        obj_file = os.path.join('build', file_base + '.o')
        so_file = os.path.join('build', file_base + lib_ext)
        if not os.path.exists('build'):
            os.makedirs('build', exist_ok=True)
    
    return build_dir, ir_file, obj_file, so_file
