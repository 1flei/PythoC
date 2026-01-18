# -*- coding: utf-8 -*-
"""
Unified Dependency Tracking System for pythoc.

This module provides a centralized system for tracking dependencies between
compilation groups, managing cache invalidation, and persisting dependency
information to .deps files.

Core concepts:
- GroupKey: 4-tuple (file, scope, compile_suffix, effect_suffix)
- Dependency: A callable that this group depends on (other group or extern)
- Link dependencies: -l libraries and .o files needed for linking

Layered invalidation:
    Source (.py) -> IR (.ll) -> Object (.o) -> Shared Lib (.so) -> dlopen
    Each layer updates ONLY when its input changes.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from ..logger import logger

# Version for .deps file format
DEPS_VERSION = 1


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
class CallableDep:
    """Dependency on another callable."""
    name: str                                    # Mangled name
    group_key: Optional[GroupKey] = None         # Group key (None for extern)
    extern: bool = False                         # Is @extern declaration
    link_libraries: List[str] = field(default_factory=list)  # Libraries from extern
    link_objects: List[str] = field(default_factory=list)    # Objects from cimport
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {'name': self.name}
        if self.group_key:
            d['group_key'] = self.group_key.to_list()
        if self.extern:
            d['extern'] = True
        if self.link_libraries:
            d['link_libraries'] = self.link_libraries
        if self.link_objects:
            d['link_objects'] = self.link_objects
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CallableDep':
        """Create from dict (JSON deserialization)."""
        group_key = None
        if 'group_key' in d and d['group_key']:
            group_key = GroupKey.from_list(d['group_key'])
        return cls(
            name=d['name'],
            group_key=group_key,
            extern=d.get('extern', False),
            link_libraries=d.get('link_libraries', []),
            link_objects=d.get('link_objects', [])
        )


@dataclass
class CallableInfo:
    """Dependency info for a single callable in a group."""
    deps: List[CallableDep] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'deps': [d.to_dict() for d in self.deps]}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CallableInfo':
        return cls(deps=[CallableDep.from_dict(dep) for dep in d.get('deps', [])])


@dataclass
class GroupDeps:
    """
    Complete dependency information for a compilation group.
    
    This is persisted to .deps files and loaded on cache hit.
    """
    version: int = DEPS_VERSION
    group_key: Optional[GroupKey] = None
    source_mtime: float = 0.0
    callables: Dict[str, CallableInfo] = field(default_factory=dict)
    link_objects: List[str] = field(default_factory=list)
    link_libraries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'version': self.version,
            'group_key': self.group_key.to_list() if self.group_key else None,
            'source_mtime': self.source_mtime,
            'callables': {k: v.to_dict() for k, v in self.callables.items()},
            'link_objects': self.link_objects,
            'link_libraries': self.link_libraries,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GroupDeps':
        """Create from dict (JSON deserialization)."""
        group_key = None
        if d.get('group_key'):
            group_key = GroupKey.from_list(d['group_key'])
        return cls(
            version=d.get('version', 1),
            group_key=group_key,
            source_mtime=d.get('source_mtime', 0.0),
            callables={k: CallableInfo.from_dict(v) for k, v in d.get('callables', {}).items()},
            link_objects=d.get('link_objects', []),
            link_libraries=d.get('link_libraries', []),
        )
    
    def add_callable(self, name: str, deps: List[CallableDep] = None):
        """Add or update a callable's dependencies."""
        if deps is None:
            deps = []
        self.callables[name] = CallableInfo(deps=deps)
        
        # Aggregate link dependencies
        for dep in deps:
            for lib in dep.link_libraries:
                if lib not in self.link_libraries:
                    self.link_libraries.append(lib)
            for obj in dep.link_objects:
                if obj not in self.link_objects:
                    self.link_objects.append(obj)
    
    def get_all_dependent_groups(self) -> Set[Tuple]:
        """Get all group keys this group depends on."""
        groups = set()
        for callable_info in self.callables.values():
            for dep in callable_info.deps:
                if dep.group_key:
                    groups.add(dep.group_key.to_tuple())
        return groups


class DependencyTracker:
    """
    Central dependency tracking system.
    
    Responsibilities:
    - Track dependencies during compilation
    - Persist dependencies to .deps files
    - Load dependencies on cache hit
    - Determine what needs recompilation
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
    
    def record_dependency(self, caller_group_key: Tuple, caller_name: str, 
                         dep: CallableDep):
        """
        Record that a callable depends on another callable.
        
        Args:
            caller_group_key: Group key of the caller
            caller_name: Mangled name of the caller
            dep: Dependency information
        """
        group_deps = self.get_or_create_group_deps(caller_group_key)
        if caller_name not in group_deps.callables:
            group_deps.callables[caller_name] = CallableInfo()
        
        # Check if already recorded
        for existing in group_deps.callables[caller_name].deps:
            if existing.name == dep.name:
                return
        
        group_deps.callables[caller_name].deps.append(dep)
        
        # Aggregate link dependencies
        for lib in dep.link_libraries:
            if lib not in group_deps.link_libraries:
                group_deps.link_libraries.append(lib)
        for obj in dep.link_objects:
            if obj not in group_deps.link_objects:
                group_deps.link_objects.append(obj)
    
    def record_extern_dependency(self, caller_group_key: Tuple, caller_name: str,
                                 extern_name: str, libraries: List[str]):
        """Record dependency on an @extern function."""
        dep = CallableDep(
            name=extern_name,
            extern=True,
            link_libraries=libraries
        )
        self.record_dependency(caller_group_key, caller_name, dep)
    
    def record_user_function_dependency(self, caller_group_key: Tuple, caller_name: str,
                                        dep_name: str, dep_group_key: Tuple):
        """Record dependency on another @compile function."""
        dep = CallableDep(
            name=dep_name,
            group_key=GroupKey.from_tuple(dep_group_key)
        )
        self.record_dependency(caller_group_key, caller_name, dep)
    
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
            logger.debug(f"Saved deps to {deps_file}")
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
            
            logger.debug(f"Loaded deps from {deps_file}")
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
        """Get all link libraries for a group (including transitive)."""
        deps = self.get_deps(group_key, obj_file)
        if deps:
            return deps.link_libraries
        return []
    
    def get_link_objects(self, group_key: Tuple, obj_file: str = None) -> List[str]:
        """Get all link objects for a group (including transitive)."""
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
