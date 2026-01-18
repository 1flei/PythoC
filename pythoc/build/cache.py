# -*- coding: utf-8 -*-
"""
Layered build cache for pythoc.

Implements the layered invalidation model:
    Source (.py) -> IR (.ll) -> Object (.o) -> Shared Lib (.so) -> dlopen

Each layer updates ONLY when its input (previous layer) changes.
"""

import os
from typing import Optional, Tuple


class BuildCache:
    """
    Manages build cache and timestamp checking for incremental compilation.
    
    Layered invalidation rules:
    - .ll regenerated when: source file changes
    - .o recompiled when: .ll changes
    - .so re-linked when: any dependent .o changes
    - dlopen reloaded when: .so changes
    """
    
    @staticmethod
    def check_timestamp_skip(ir_file: str, obj_file: str, source_file: str) -> bool:
        """
        Check if .ll and .o files are up-to-date based on timestamps.
        
        If files are outdated, delete them to force recompilation.
        
        Args:
            ir_file: Path to .ll file
            obj_file: Path to .o file
            source_file: Path to source .py file
        
        Returns:
            bool: True if can skip compilation, False if must compile
        """
        if not os.path.exists(source_file):
            return False
        
        # If output files don't exist, must compile
        if not (os.path.exists(ir_file) and os.path.exists(obj_file)):
            return False
        
        source_mtime = os.path.getmtime(source_file)
        ir_mtime = os.path.getmtime(ir_file)
        obj_mtime = os.path.getmtime(obj_file)
        
        # Layer 1: Source -> IR
        # If source is newer than IR, need to regenerate
        if source_mtime > ir_mtime:
            BuildCache._delete_files(ir_file, obj_file)
            return False
        
        # Layer 2: IR -> Object
        # If IR is newer than object, need to recompile
        if ir_mtime > obj_mtime:
            BuildCache._delete_files(obj_file)
            return False
        
        # Both layers up-to-date
        return True
    
    @staticmethod
    def check_so_needs_relink(so_file: str, obj_files: list) -> bool:
        """
        Check if .so file needs to be re-linked.
        
        Layer 3: Objects -> Shared Library
        
        Args:
            so_file: Path to .so file
            obj_files: List of .o files this .so depends on
            
        Returns:
            bool: True if .so needs re-linking, False if up-to-date
        """
        if not os.path.exists(so_file):
            return True
        
        so_mtime = os.path.getmtime(so_file)
        
        for obj_file in obj_files:
            if not os.path.exists(obj_file):
                # Missing dependency - needs relink
                return True
            if os.path.getmtime(obj_file) > so_mtime:
                # Object newer than so - needs relink
                return True
        
        return False
    
    @staticmethod
    def check_deps_changed(deps_file: str, source_file: str) -> bool:
        """
        Check if .deps file is still valid for the source.
        
        Args:
            deps_file: Path to .deps file
            source_file: Path to source .py file
            
        Returns:
            bool: True if deps are outdated and need regeneration
        """
        if not os.path.exists(deps_file):
            return True
        
        if not os.path.exists(source_file):
            return True
        
        # Deps file should be newer than source
        return os.path.getmtime(source_file) > os.path.getmtime(deps_file)
    
    @staticmethod
    def _delete_files(*files):
        """Delete files if they exist, ignoring errors."""
        for f in files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass  # Ignore deletion errors
    
    @staticmethod
    def invalidate(source_file: str, ir_file: str = None, obj_file: str = None, 
                   so_file: str = None):
        """
        Invalidate cached files for a source file.
        
        Deletes outdated files to force recompilation.
        
        Args:
            source_file: Path to source file
            ir_file: Path to .ll file (optional)
            obj_file: Path to .o file (optional)
            so_file: Path to .so file (optional)
        """
        BuildCache._delete_files(ir_file, obj_file, so_file)
        
        # Also delete .deps file if obj_file is specified
        if obj_file:
            deps_file = obj_file.replace('.o', '.deps')
            BuildCache._delete_files(deps_file)
    
    @staticmethod
    def get_layer_status(source_file: str, ir_file: str, obj_file: str, 
                         so_file: str = None) -> Tuple[bool, bool, bool, bool]:
        """
        Get the status of each cache layer.
        
        Returns:
            Tuple of (source_exists, ir_uptodate, obj_uptodate, so_uptodate)
        """
        if not os.path.exists(source_file):
            return (False, False, False, False)
        
        source_mtime = os.path.getmtime(source_file)
        
        # Check IR
        ir_uptodate = False
        ir_mtime = 0
        if os.path.exists(ir_file):
            ir_mtime = os.path.getmtime(ir_file)
            ir_uptodate = ir_mtime >= source_mtime
        
        # Check Object
        obj_uptodate = False
        obj_mtime = 0
        if ir_uptodate and os.path.exists(obj_file):
            obj_mtime = os.path.getmtime(obj_file)
            obj_uptodate = obj_mtime >= ir_mtime
        
        # Check Shared Library
        so_uptodate = False
        if obj_uptodate and so_file and os.path.exists(so_file):
            so_mtime = os.path.getmtime(so_file)
            so_uptodate = so_mtime >= obj_mtime
        
        return (True, ir_uptodate, obj_uptodate, so_uptodate)
