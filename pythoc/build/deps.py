# -*- coding: utf-8 -*-
"""
Group-Level Dependency Tracking System for pythoc.

This module provides a simplified dependency tracking system that operates at
the compilation group level instead of individual callable level. This aligns
with C's linking model where dependencies are resolved at the file/object level.

Core concepts:
- GroupKey: 4-tuple (file, scope, compile_suffix, effect_suffix)
- GroupDependency: A compilation group that this group depends on
- Link dependencies: -l libraries and .o files needed for linking
- Effect dependencies: Effects used by this group (for effect propagation)

Benefits:
- 98% reduction in storage size
- Simpler dependency resolution
- Aligns with C linking model
- Maintains effect system compatibility
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from ..logger import logger

# Version for .deps file format
DEPS_VERSION = 11  # Increment when scheduler/cache/effect planning semantics change


def _type_source_file(pc_type: Any) -> Optional[str]:
    """Best-effort source file where a @compile aggregate/enum type is defined."""
    import inspect
    try:
        src = inspect.getsourcefile(pc_type) or inspect.getfile(pc_type)
    except (TypeError, OSError):
        return None
    return os.path.abspath(src) if src else None


@dataclass
class GroupKey:
    """
    Identifier for a compilation group.
    
    A compilation group is a set of functions compiled into the same .ll/.o/.so.
    """
    file: str                           # Source file path
    scope: Optional[str]                # Scope name (e.g., "GenericType.<locals>")
    compile_suffix: Optional[str]       # From @compile(suffix=T)
    effect_suffix: Optional[str]        # From with effect(suffix="mock")
    
    def to_tuple(self) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Convert to tuple for dict keys and hashing."""
        return (self.file, self.scope, self.compile_suffix, self.effect_suffix)
    
    def to_list(self) -> List[Optional[str]]:
        """Convert to list for JSON serialization."""
        return [self.file, self.scope, self.compile_suffix, self.effect_suffix]
    
    @classmethod
    def from_tuple(cls, t: Tuple) -> 'GroupKey':
        """Create from tuple (supports both 3-tuple and 4-tuple for compatibility)."""
        if len(t) == 3:
            # Old format: (file, scope, suffix)
            # Treat suffix as compile_suffix for backward compatibility
            return cls(file=t[0], scope=t[1], compile_suffix=t[2], effect_suffix=None)
        elif len(t) == 4:
            return cls(file=t[0], scope=t[1], compile_suffix=t[2], effect_suffix=t[3])
        else:
            raise ValueError(f"Invalid group key tuple length: {len(t)}")
    
    @classmethod
    def from_list(cls, lst: List) -> 'GroupKey':
        """Create from list (JSON deserialization)."""
        return cls.from_tuple(tuple(lst))
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        if isinstance(other, GroupKey):
            return self.to_tuple() == other.to_tuple()
        if isinstance(other, tuple):
            return self.to_tuple() == other
        return False
    
    def get_mangled_suffix(self) -> Optional[str]:
        """Get combined suffix for mangled names."""
        parts = []
        if self.compile_suffix:
            parts.append(self.compile_suffix)
        if self.effect_suffix:
            parts.append(self.effect_suffix)
        return '_'.join(parts) if parts else None
    
    def get_file_suffix(self) -> str:
        """Get suffix string for output file naming.

        IMPORTANT: This is used to construct on-disk filenames (e.g. `.dll` on
        Windows). Group keys persisted in older `.deps` files may contain
        characters that are invalid in filenames (e.g. `<`, `>` from
        `.<locals>`). Always sanitize each component here so we can still derive
        usable paths even when loading legacy deps.
        """
        # Local import to avoid import cycles during early startup.
        from ..utils.path_utils import sanitize_filename

        parts = []
        if self.scope:
            parts.append(sanitize_filename(self.scope))
        if self.compile_suffix:
            parts.append(sanitize_filename(self.compile_suffix))
        if self.effect_suffix:
            parts.append(sanitize_filename(self.effect_suffix))
        return '.'.join(parts) if parts else ''


@dataclass
class GroupDependency:
    """Dependency on another compilation group.

    dependency_type semantics:
    - "function_call"/"function_ref"/"effect"/"import": link-time references.
      The caller does not embed the target's code by value, so a change to the
      target's *body* only requires a relink, not a caller recompile. The cache
      only checks that the target's output artifacts still exist.
    - "source_embed": the caller's object code embeds, by value, code or a type
      layout whose source of truth lives in the target group's source file
      (e.g. a cross-module yield generator inlined into the caller, which also
      bakes in the layout/size of structs it allocates). Such a caller MUST be
      recompiled when the target's source file changes, otherwise the caller
      keeps a stale, wrong-sized frame -> silent memory corruption. The cache
      therefore mtime-invalidates the caller against these targets' sources.
    """
    target_group: GroupKey              # The group we depend on
    dependency_type: str = "function_call"  # See class docstring for values
    
    def to_dict(self, group_key_index: Optional[int] = None) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {'dependency_type': self.dependency_type}
        if group_key_index is not None:
            d['target_group_idx'] = group_key_index
        else:
            d['target_group'] = self.target_group.to_list()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], group_keys: Optional[List[List]] = None) -> 'GroupDependency':
        """Create from dict (JSON deserialization)."""
        target_group = None
        if 'target_group_idx' in d and group_keys:
            # New compressed format
            group_idx = d['target_group_idx']
            if 0 <= group_idx < len(group_keys):
                target_group = GroupKey.from_list(group_keys[group_idx])
        elif 'target_group' in d:
            # Legacy or uncompressed format
            target_group = GroupKey.from_list(d['target_group'])
        
        if target_group is None:
            raise ValueError("Invalid GroupDependency: missing target_group")
        
        return cls(
            target_group=target_group,
            dependency_type=d.get('dependency_type', 'function_call')
        )


@dataclass
class GroupDeps:
    """
    Complete dependency information for a compilation group (Group-Level).
    
    This simplified version tracks dependencies at the group level instead of
    individual callable level, providing massive storage savings and simpler
    dependency resolution.
    """
    version: int = DEPS_VERSION
    group_key: Optional[GroupKey] = None
    source_mtime: float = 0.0
    
    # Group-level dependencies
    group_dependencies: List[GroupDependency] = field(default_factory=list)
    
    # Link dependencies (aggregated from all functions in group)
    link_objects: List[str] = field(default_factory=list)
    link_libraries: List[str] = field(default_factory=list)

    # Symbols materialized into the current object file for this group.
    # Used to decide whether an up-to-date .o fully covers current pending defs.
    compiled_symbols: List[str] = field(default_factory=list)

    # Content hash of the AST bodies compiled into this group.
    # Used to detect stale cache entries when source_file stays the same but
    # the generated AST changes (common in meta-generated code).
    ast_content_hash: Optional[str] = None

    # Whether this object was built with debug info enabled.  Changing the
    # flag must invalidate the cached object.
    debug_info: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization with compressed group keys."""
        # Collect all unique group keys
        unique_group_keys = []
        group_key_map = {}
        
        # Add this group's key first
        if self.group_key:
            group_key_list = self.group_key.to_list()
            unique_group_keys.append(group_key_list)
            group_key_map[self.group_key.to_tuple()] = 0
        
        # Collect group keys from all dependencies
        for dep in self.group_dependencies:
            key_tuple = dep.target_group.to_tuple()
            if key_tuple not in group_key_map:
                group_key_map[key_tuple] = len(unique_group_keys)
                unique_group_keys.append(dep.target_group.to_list())
        
        # Build the result dict
        result = {
            'version': self.version,
            'source_mtime': self.source_mtime,
            'link_objects': self.link_objects,
            'link_libraries': self.link_libraries,
        }

        if self.compiled_symbols:
            result['compiled_symbols'] = sorted(self.compiled_symbols)

        if self.ast_content_hash:
            result['ast_content_hash'] = self.ast_content_hash

        if self.debug_info:
            result['debug_info'] = True

        # Add group keys table if we have any
        if unique_group_keys:
            result['group_keys'] = unique_group_keys
            result['main_group_idx'] = 0 if self.group_key else None
        else:
            result['group_key'] = self.group_key.to_list() if self.group_key else None
        
        # Serialize dependencies with compressed group references
        result['group_dependencies'] = []
        for dep in self.group_dependencies:
            group_idx = group_key_map.get(dep.target_group.to_tuple())
            result['group_dependencies'].append(dep.to_dict(group_idx))
        
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GroupDeps':
        """Create from dict (JSON deserialization) with support for compressed format."""
        # Handle group key(s)
        group_key = None
        group_keys = d.get('group_keys')
        
        if group_keys:
            # New compressed format
            main_group_idx = d.get('main_group_idx')
            if main_group_idx is not None and 0 <= main_group_idx < len(group_keys):
                group_key = GroupKey.from_list(group_keys[main_group_idx])
        elif d.get('group_key'):
            # Legacy format
            group_key = GroupKey.from_list(d['group_key'])
        
        # Parse group dependencies
        group_dependencies = []
        for dep_data in d.get('group_dependencies', []):
            group_dependencies.append(GroupDependency.from_dict(dep_data, group_keys))
        
        compiled_symbols = d.get('compiled_symbols', [])
        ast_content_hash = d.get('ast_content_hash')
        debug_info = d.get('debug_info', False)

        return cls(
            version=d.get('version', DEPS_VERSION),
            group_key=group_key,
            source_mtime=d.get('source_mtime', 0.0),
            group_dependencies=group_dependencies,
            link_objects=d.get('link_objects', []),
            link_libraries=d.get('link_libraries', []),
            compiled_symbols=compiled_symbols,
            ast_content_hash=ast_content_hash,
            debug_info=debug_info,
        )

    
    def add_group_dependency(self, target_group: GroupKey, dependency_type: str = "function_call"):
        """Add a dependency on another group."""
        # Check if already exists
        for dep in self.group_dependencies:
            if dep.target_group == target_group and dep.dependency_type == dependency_type:
                return
        
        self.group_dependencies.append(GroupDependency(target_group, dependency_type))
    
    def add_link_library(self, library: str):
        """Add a link library (if not already present)."""
        if library not in self.link_libraries:
            self.link_libraries.append(library)
    
    def add_link_object(self, obj_file: str):
        """Add a link object (if not already present)."""
        if obj_file not in self.link_objects:
            self.link_objects.append(obj_file)

    def get_all_dependent_groups(self) -> Set[Tuple]:
        """Get all group keys this group depends on."""
        return {dep.target_group.to_tuple() for dep in self.group_dependencies}


class DependencyTracker:
    """
    Group-Level Dependency Tracking System.
    
    This simplified tracker operates at the compilation group level,
    providing massive performance improvements and storage savings.
    """
    
    def __init__(self):
        # Current compilation's dependency info: group_key -> GroupDeps
        self._group_deps: Dict[Tuple, GroupDeps] = {}
        
        # Loaded deps from files: group_key -> GroupDeps
        self._loaded_deps: Dict[Tuple, GroupDeps] = {}

        # Build tasks can record/load deps concurrently when object workers > 1.
        self._lock = threading.RLock()

    def get_or_create_group_deps(self, group_key: Tuple) -> GroupDeps:
        """Get or create GroupDeps for a group key."""
        with self._lock:
            if group_key not in self._group_deps:
                gk = GroupKey.from_tuple(group_key)
                self._group_deps[group_key] = GroupDeps(group_key=gk)
            return self._group_deps[group_key]
    
    def record_group_dependency(self, caller_group_key: Tuple, target_group_key: Tuple, 
                               dependency_type: str = "function_call"):
        """
        Record that one group depends on another group.
        
        Args:
            caller_group_key: Group key of the caller
            target_group_key: Group key of the target
            dependency_type: Type of dependency ("function_call", "effect", "import")
        """
        with self._lock:
            caller_deps = self.get_or_create_group_deps(caller_group_key)
            target_group = GroupKey.from_tuple(target_group_key)
            caller_deps.add_group_dependency(target_group, dependency_type)
    
    def record_type_layout_deps_from_globals(self, group_key: Tuple, globals_dict: Dict[str, Any]):
        """Record source_embed deps for cross-module @compile types in globals.

        A group that imports a @compile struct/union/enum bakes that type's
        layout (size, field offsets) into its own object code -- e.g. via
        sizeof(T), an array[T, N] local, a by-value T field/parameter, or a
        stack slot of type T. The incremental cache only tracks each group's
        own source mtime, so a layout change in the *defining* module would
        otherwise be served from this group's stale .o whose baked-in frame
        size no longer matches, causing silent memory corruption.

        Recording a source_embed edge to the defining file forces a rebuild of
        this group whenever that file changes. Plain function references stay
        link-time (function_ref) and are intentionally not covered here.
        """
        if not isinstance(globals_dict, dict) or not group_key:
            return
        import sys
        stdlib = getattr(sys, 'stdlib_module_names', frozenset())
        caller_file = group_key[0] if len(group_key) else None
        recorded: Set[str] = set()
        for value in globals_dict.values():
            if not isinstance(value, type):
                continue
            # @compile struct/union/enum types carry a `_field_types` schema and
            # a sized LLVM layout; builtin scalars (i32, ptr, ...) and function
            # wrappers do not, so their ABI is fixed and needs no source edge.
            if getattr(value, '_field_types', None) is None:
                continue
            if not hasattr(value, 'get_size_bytes'):
                continue
            # Skip stdlib-defined classes (e.g. abc.ABC bases) that happen to
            # expose these attributes; their layout is not pythoc-owned.
            top_mod = (getattr(value, '__module__', '') or '').split('.')[0]
            if top_mod in stdlib:
                continue
            src = _type_source_file(value)
            if not src or src == caller_file or src in recorded:
                continue
            recorded.add(src)
            self.record_group_dependency(
                tuple(group_key), (src, None, None, None), "source_embed",
            )

    def record_extern_dependency(self, group_key: Tuple, libraries: List[str]):
        """Record dependency on external libraries."""
        with self._lock:
            group_deps = self.get_or_create_group_deps(group_key)
            for lib in libraries:
                group_deps.add_link_library(lib)
    
    def record_cimport_dependency(self, group_key: Tuple, obj_files: List[str]):
        """Record dependency on cimport object files."""
        with self._lock:
            group_deps = self.get_or_create_group_deps(group_key)
            for obj in obj_files:
                group_deps.add_link_object(obj)

    def get_deps_file_path(self, obj_file: str) -> str:
        """Get .deps file path corresponding to an .o file."""
        return obj_file.replace('.o', '.deps')
    
    def save_deps(self, group_key: Tuple, obj_file: str):
        """
        Save dependency info to .deps file.
        
        Args:
            group_key: Group key
            obj_file: Path to .o file (deps file will be derived from this)
        """
        with self._lock:
            if group_key not in self._group_deps:
                return
            
            deps = self._group_deps[group_key]
            deps_file = self.get_deps_file_path(obj_file)
            
            try:
                with open(deps_file, 'w') as f:
                    json.dump(deps.to_dict(), f, indent=2)
                logger.debug(f"Saved group-level deps to {deps_file}")
            except Exception as e:
                logger.debug(f"Failed to save deps to {deps_file}: {e}")
    
    def load_deps(self, obj_file: str) -> Optional[GroupDeps]:
        """
        Load dependency info from .deps file.
        
        Args:
            obj_file: Path to .o file
            
        Returns:
            GroupDeps or None if not found/invalid
        """
        deps_file = self.get_deps_file_path(obj_file)
        
        if not os.path.exists(deps_file):
            return None
        
        try:
            with open(deps_file, 'r') as f:
                data = json.load(f)

            if data.get('version') != DEPS_VERSION:
                logger.debug(
                    f"Ignoring stale deps format in {deps_file}: "
                    f"version={data.get('version')} expected={DEPS_VERSION}"
                )
                return None
            
            deps = GroupDeps.from_dict(data)
            
            # Cache loaded deps
            with self._lock:
                if deps.group_key:
                    self._loaded_deps[deps.group_key.to_tuple()] = deps
            
            logger.debug(f"Loaded group-level deps from {deps_file}")
            return deps
        except Exception as e:
            logger.debug(f"Failed to load deps from {deps_file}: {e}")
            return None
    
    def get_deps(self, group_key: Tuple, obj_file: str = None) -> Optional[GroupDeps]:
        """
        Get dependencies for a group (from memory or file).
        
        Args:
            group_key: Group key
            obj_file: Path to .o file (for loading from disk)
            
        Returns:
            GroupDeps or None
        """
        with self._lock:
            # First check in-memory
            if group_key in self._group_deps:
                return self._group_deps[group_key]
            
            # Then check loaded cache
            if group_key in self._loaded_deps:
                return self._loaded_deps[group_key]
        
        # Try loading from file
        if obj_file:
            return self.load_deps(obj_file)
        
        return None

    def derive_obj_file_from_group_key(self, group_key: Tuple) -> Optional[str]:
        """Derive the object file path for a group key."""
        if isinstance(group_key, GroupKey):
            gk = group_key
        else:
            gk = GroupKey.from_tuple(group_key)

        source_file = gk.file
        if not source_file:
            return None

        cwd = os.getcwd()
        if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            rel_path = f"external/{base_name}.py"

        build_dir = os.path.join('build', os.path.dirname(rel_path))
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        file_suffix = gk.get_file_suffix()
        file_base = f"{base_name}.{file_suffix}" if file_suffix else base_name
        return os.path.join(build_dir, f"{file_base}.o")

    def get_deps_for_group(self, group_key: Tuple) -> Optional[GroupDeps]:
        """Get dependencies for a group key, loading persisted deps if needed."""
        if isinstance(group_key, GroupKey):
            group_key = group_key.to_tuple()

        obj_file = self.derive_obj_file_from_group_key(group_key)
        return self.get_deps(group_key, obj_file=obj_file)

    def get_link_libraries(self, group_key: Tuple, obj_file: str = None) -> List[str]:
        """Get all link libraries for a group."""
        deps = self.get_deps(group_key, obj_file)
        if deps:
            return deps.link_libraries
        return []
    
    def get_link_objects(self, group_key: Tuple, obj_file: str = None) -> List[str]:
        """Get all link objects for a group."""
        deps = self.get_deps(group_key, obj_file)
        if deps:
            return deps.link_objects
        return []
    
    def get_dependent_groups(self, group_key: Tuple, obj_file: str = None) -> Set[Tuple]:
        """Get all groups this group depends on."""
        deps = self.get_deps(group_key, obj_file)
        if deps:
            return deps.get_all_dependent_groups()
        return set()
    
    def clear_group(self, group_key: Tuple):
        """Clear in-memory deps for a group (for recompilation)."""
        if group_key in self._group_deps:
            del self._group_deps[group_key]
        if group_key in self._loaded_deps:
            del self._loaded_deps[group_key]
    
    def clear_all(self):
        """Clear all in-memory state."""
        self._group_deps.clear()
        self._loaded_deps.clear()





# Global singleton
_dependency_tracker = DependencyTracker()


def get_dependency_tracker() -> DependencyTracker:
    """Get the global dependency tracker instance."""
    return _dependency_tracker


# Backward compatibility
def get_group_level_dependency_tracker() -> DependencyTracker:
    """Get the global group-level dependency tracker instance (alias)."""
    return _dependency_tracker