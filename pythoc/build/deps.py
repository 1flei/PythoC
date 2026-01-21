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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from ..logger import logger

# Version for .deps file format
DEPS_VERSION = 2  # Increment version for group-level format


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
        """Get suffix string for output file naming."""
        parts = []
        if self.scope:
            parts.append(self.scope)
        if self.compile_suffix:
            parts.append(self.compile_suffix)
        if self.effect_suffix:
            parts.append(self.effect_suffix)
        return '.'.join(parts) if parts else ''


@dataclass
class GroupDependency:
    """Dependency on another compilation group."""
    target_group: GroupKey              # The group we depend on
    dependency_type: str = "function_call"  # Type: "function_call", "effect", "import"
    
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
    
    # Effect system support (group-level)
    effects_used: Set[str] = field(default_factory=set)  # All effects used by this group
    
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
        
        # Add effects if any
        if self.effects_used:
            result['effects_used'] = list(self.effects_used)
        
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
        
        # Parse effects
        effects_used = set(d.get('effects_used', []))
        
        return cls(
            version=d.get('version', DEPS_VERSION),
            group_key=group_key,
            source_mtime=d.get('source_mtime', 0.0),
            group_dependencies=group_dependencies,
            link_objects=d.get('link_objects', []),
            link_libraries=d.get('link_libraries', []),
            effects_used=effects_used,
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
    
    def add_effect(self, effect_name: str):
        """Add an effect used by this group."""
        self.effects_used.add(effect_name)
    
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
    
    def get_or_create_group_deps(self, group_key: Tuple) -> GroupDeps:
        """Get or create GroupDeps for a group key."""
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
        caller_deps = self.get_or_create_group_deps(caller_group_key)
        target_group = GroupKey.from_tuple(target_group_key)
        caller_deps.add_group_dependency(target_group, dependency_type)
    
    def record_extern_dependency(self, group_key: Tuple, libraries: List[str]):
        """Record dependency on external libraries."""
        group_deps = self.get_or_create_group_deps(group_key)
        for lib in libraries:
            group_deps.add_link_library(lib)
    
    def record_cimport_dependency(self, group_key: Tuple, obj_files: List[str]):
        """Record dependency on cimport object files."""
        group_deps = self.get_or_create_group_deps(group_key)
        for obj in obj_files:
            group_deps.add_link_object(obj)
    
    def record_effect_usage(self, group_key: Tuple, effect_name: str):
        """Record that a group uses an effect."""
        group_deps = self.get_or_create_group_deps(group_key)
        group_deps.add_effect(effect_name)
    
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
            
            deps = GroupDeps.from_dict(data)
            
            # Cache loaded deps
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
    
    def get_effects_used(self, group_key: Tuple, obj_file: str = None) -> Set[str]:
        """Get all effects used by this group."""
        deps = self.get_deps(group_key, obj_file)
        if deps:
            return deps.effects_used
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