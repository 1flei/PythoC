"""Build utilities for compiling PC programs to native artifacts."""

import inspect
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def get_source_file_from_caller(offset: int = 0) -> str:
    """Get the source file path from the calling frame."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        target_frame = frame.f_back
        for _ in range(offset + 1):
            if target_frame and target_frame.f_back:
                target_frame = target_frame.f_back
            else:
                break

        if target_frame:
            return target_frame.f_code.co_filename

    raise RuntimeError("Cannot detect source file. Please provide source_file parameter.")


def _rel_path_for_source(source_file: str) -> str:
    cwd = os.getcwd()
    if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
        return os.path.relpath(source_file, cwd)
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    return f"external/{base_name}"


def determine_output_path(source_file: str, output_path: Optional[str] = None) -> str:
    """Determine the output executable path."""
    if output_path is not None:
        if sys.platform == 'win32' and not output_path.endswith('.exe'):
            return output_path + '.exe'
        return output_path

    rel_path = _rel_path_for_source(source_file)
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    output_dir = os.path.join('build', os.path.dirname(rel_path))
    exe_path = os.path.join(output_dir, base_name)

    if sys.platform == 'win32':
        exe_path += '.exe'

    return exe_path


def determine_library_output_path(
    source_file: str,
    output_path: Optional[str],
    extension: str,
    *,
    use_lib_prefix: bool,
) -> str:
    """Determine output path for a library artifact."""
    if output_path is not None:
        _, ext = os.path.splitext(output_path)
        if ext.lower() != extension.lower():
            return output_path + extension
        return output_path

    rel_path = _rel_path_for_source(source_file)
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    if use_lib_prefix and not base_name.startswith('lib'):
        base_name = 'lib' + base_name
    output_dir = os.path.join('build', os.path.dirname(rel_path))
    return os.path.join(output_dir, base_name + extension)


def determine_header_output_path(
    source_file: str,
    output_path: Optional[str] = None,
) -> str:
    """Determine output path for a generated C header."""
    if output_path is not None:
        return output_path if output_path.endswith('.h') else output_path + '.h'

    rel_path = _rel_path_for_source(source_file)
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    output_dir = os.path.join('build', os.path.dirname(rel_path))
    return os.path.join(output_dir, base_name + '.h')


def get_object_file_path(source_file: str) -> str:
    """Convert source file path to corresponding object file path."""
    rel_path = _rel_path_for_source(source_file)
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    return os.path.join('build', os.path.dirname(rel_path), base_name + '.o')


def collect_object_files(source_files: List[str]) -> List[str]:
    """Collect all object files from source files."""
    obj_files = []
    for src_file in source_files:
        obj_file = get_object_file_path(src_file)
        if os.path.exists(obj_file):
            obj_files.append(obj_file)

    if not obj_files:
        raise RuntimeError(
            "No object files found. "
            "Make sure you have @compile functions before calling compile_to_executable()."
        )

    return obj_files


def _caller_frame():
    frame = inspect.currentframe()
    return frame.f_back.f_back if frame and frame.f_back and frame.f_back.f_back else None


def _source_file_from_frame(frame, source_file: Optional[str]) -> str:
    if source_file is not None:
        return os.path.abspath(source_file)
    if frame is not None:
        return os.path.abspath(frame.f_code.co_filename)
    return os.path.abspath(get_source_file_from_caller(offset=1))


def _iter_visible_symbols(frame) -> Iterable[Tuple[str, Any]]:
    if frame is None:
        return []

    merged: Dict[str, Any] = {}
    merged.update(frame.f_globals)
    merged.update(frame.f_locals)
    return merged.items()


def _is_compiled_symbol(value: Any) -> bool:
    return hasattr(value, '_func_info') and hasattr(value, '_binding')


def _visible_compiled_symbols(frame, include_private: bool) -> List[Any]:
    result = []
    seen_ids = set()
    for name, value in _iter_visible_symbols(frame):
        if not include_private and name.startswith('_'):
            continue
        if _is_compiled_symbol(value) and id(value) not in seen_ids:
            seen_ids.add(id(value))
            result.append(value)
    return result


def _function_names(symbol: Any) -> Set[str]:
    func_info = getattr(symbol, '_func_info', None)
    if func_info is None:
        return set()
    names = {func_info.name}
    binding = getattr(func_info, 'binding_state', None)
    if binding is not None:
        for attr in ('original_name', 'actual_func_name', 'mangled_name'):
            value = getattr(binding, attr, None)
            if value:
                names.add(value)
    mangled = getattr(func_info, 'mangled_name', None)
    if mangled:
        names.add(mangled)
    return names


def _resolve_symbol_name(name: str, frame) -> Any:
    visible = list(_visible_compiled_symbols(frame, include_private=True))
    for visible_name, value in _iter_visible_symbols(frame):
        if visible_name == name and _is_compiled_symbol(value):
            return value
    for symbol in visible:
        if name in _function_names(symbol):
            return symbol
    raise NameError(f"Compiled symbol '{name}' is not visible from this call site.")


def _resolve_export_symbols(symbols: Sequence[Any], frame) -> List[Any]:
    if symbols:
        resolved = []
        seen_ids = set()
        for symbol in symbols:
            if isinstance(symbol, str):
                symbol = _resolve_symbol_name(symbol, frame)
            if not _is_compiled_symbol(symbol):
                raise TypeError(f"Expected a @compile function symbol, got {symbol!r}")
            if id(symbol) not in seen_ids:
                seen_ids.add(id(symbol))
                resolved.append(symbol)
        return resolved

    resolved = _visible_compiled_symbols(frame, include_private=False)
    if not resolved:
        raise RuntimeError(
            "No visible public @compile symbols found. "
            "Pass symbols explicitly or call from a scope where they are visible."
        )
    return resolved


def _collect_group_link_plan(group_keys: Sequence[Tuple]):
    from ..artifact import LinkPlan

    return LinkPlan.from_group_roots(group_keys, flush=False)


def _build_selected_link_plan(symbols: Sequence[Any]):
    from ..artifact import LinkPlan

    return LinkPlan.from_compiled_symbols(symbols)


def link_executable(obj_files: List[str], output_path: str) -> str:
    """Link object files into a native executable."""
    from ..artifact import (
        ArtifactKind,
        ArtifactPlan,
        LinkPlan,
        LinkScope,
        build_artifact,
    )

    link = LinkPlan(
        roots=(),
        obj_files=tuple(obj_files),
        scope=LinkScope.OWN_GROUP,
    )
    result = build_artifact(ArtifactPlan(
        kind=ArtifactKind.EXECUTABLE,
        link=link,
        output_path=output_path,
    )).path
    print(f"Successfully compiled to executable: {output_path}")
    print(f"Linked {len(obj_files)} object file(s)")
    return result


def link_dynamic_library(
    obj_files: List[str],
    output_path: str,
    link_libraries: Optional[List[str]] = None,
) -> str:
    """Link object files into a dynamic library."""
    from ..artifact import (
        ArtifactKind,
        ArtifactPlan,
        LinkPlan,
        LinkScope,
        build_artifact,
    )

    link = LinkPlan(
        roots=(),
        obj_files=tuple(obj_files),
        link_libraries=tuple(link_libraries or ()),
        scope=LinkScope.OWN_GROUP,
    )
    result = build_artifact(ArtifactPlan(
        kind=ArtifactKind.SHARED_LIBRARY,
        link=link,
        output_path=output_path,
    )).path
    print(f"Successfully compiled to dynamic library: {output_path}")
    print(f"Linked {len(obj_files)} object file(s)")
    return result


def link_static_library(obj_files: List[str], output_path: str) -> str:
    """Archive object files into a static library."""
    from ..artifact import (
        ArtifactKind,
        ArtifactPlan,
        LinkPlan,
        LinkScope,
        build_artifact,
    )

    link = LinkPlan(
        roots=(),
        obj_files=tuple(obj_files),
        scope=LinkScope.OWN_GROUP,
    )
    result = build_artifact(ArtifactPlan(
        kind=ArtifactKind.STATIC_LIBRARY,
        link=link,
        output_path=output_path,
    )).path
    print(f"Successfully compiled to static library: {output_path}")
    print(f"Archived {len(obj_files)} object file(s)")
    return result


def compile_to_executable(output_path: Optional[str] = None,
                          source_file: Optional[str] = None) -> str:
    """Compile all @compile decorated functions to a native executable."""
    from ..artifact import LinkPlan

    if source_file is None:
        source_file = get_source_file_from_caller(offset=0)
    source_file = os.path.abspath(source_file)
    output_path = determine_output_path(source_file, output_path)

    plan = LinkPlan.from_all_groups()
    return link_executable(list(plan.obj_files), output_path)


def compile_to_static_library(
    *symbols: Any,
    output_path: Optional[str] = None,
    source_file: Optional[str] = None,
) -> str:
    """Compile visible or selected @compile symbols to a static library."""
    from .link_utils import get_static_lib_extension

    frame = _caller_frame()
    source_file = _source_file_from_frame(frame, source_file)
    selected_symbols = _resolve_export_symbols(symbols, frame)
    plan = _build_selected_link_plan(selected_symbols)
    output_path = determine_library_output_path(
        source_file,
        output_path,
        get_static_lib_extension(),
        use_lib_prefix=sys.platform != 'win32',
    )
    return link_static_library(plan.obj_files, output_path)


def compile_to_dynamic_library(
    *symbols: Any,
    output_path: Optional[str] = None,
    source_file: Optional[str] = None,
) -> str:
    """Compile visible or selected @compile symbols to a dynamic library."""
    from .link_utils import get_shared_lib_extension

    frame = _caller_frame()
    source_file = _source_file_from_frame(frame, source_file)
    selected_symbols = _resolve_export_symbols(symbols, frame)
    plan = _build_selected_link_plan(selected_symbols)
    output_path = determine_library_output_path(
        source_file,
        output_path,
        get_shared_lib_extension(),
        use_lib_prefix=sys.platform != 'win32',
    )
    return link_dynamic_library(plan.obj_files, output_path, plan.link_libraries)


def compile_to_python_extension(
    *symbols: Any,
    output_path: Optional[str] = None,
    module_name: str = '_pythoc_native',
    source_file: Optional[str] = None,
) -> str:
    """Compile selected @compile symbols into one CPython extension.

    This target adds Python adapters around the existing kernel objects.
    Executable, static-library, and dynamic-library targets do not use it.
    """
    from ..python_adapter import compile_python_extension

    frame = _caller_frame()
    source_file = _source_file_from_frame(frame, source_file)
    selected_symbols = _resolve_export_symbols(symbols, frame)
    if output_path is None:
        output_path = determine_library_output_path(
            source_file,
            None,
            '',
            use_lib_prefix=False,
        )
        output_path = os.path.join(
            os.path.dirname(output_path),
            module_name.split('.')[-1],
        )
    return compile_python_extension(selected_symbols, output_path, module_name)


def export_c_headers(
    *symbols: Any,
    output_path: Optional[str] = None,
    source_file: Optional[str] = None,
) -> str:
    """Export visible or selected @compile symbols as a C header."""
    from ..decorators.compile import flush_all_pending_outputs
    from .header_utils import export_c_header

    frame = _caller_frame()
    source_file = _source_file_from_frame(frame, source_file)
    selected_symbols = _resolve_export_symbols(symbols, frame)
    flush_all_pending_outputs()
    output_path = determine_header_output_path(source_file, output_path)
    return export_c_header(selected_symbols, output_path)
