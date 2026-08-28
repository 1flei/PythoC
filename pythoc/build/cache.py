# -*- coding: utf-8 -*-
"""
Layered build cache for pythoc.

Implements the layered invalidation model:
    Source (.py) -> Object (.o) -> Shared Lib (.so) -> dlopen

Each layer updates ONLY when its input (previous layer) changes.
.ll is just an intermediate artifact, not a cache layer.
"""

import ast
import hashlib
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

_MISSING = object()


class FunctionContentFingerprint(NamedTuple):
    digest: Optional[str]
    captured: Tuple[str, ...]


def fingerprint_function_content(
    fn_ast, user_globals: Optional[Dict[str, Any]] = None
) -> FunctionContentFingerprint:
    """Fingerprint a function AST plus bakeable captured constants.

    Python scalars referenced by name (for example a ctypes.addressof
    result assigned to a local and then used as ``ptr[T](addr)``) are
    folded into IR at compile time.  The source file mtime does not
    change when those values change across processes, so they have to
    participate in the cache key; otherwise a cached .o embeds a
    stale, process-local address.

    ``captured`` is non-empty when the digest includes such scalars.
    Nested @compile created during a parent's codegen must fold that
    digest into the parent: the parent is the artefact that dlopens the
    nested .so, and a parent cache hit would skip re-decoration of the
    nested function entirely.
    """
    user_globals = user_globals or {}
    try:
        dumped = ast.dump(fn_ast)
    except Exception:
        return FunctionContentFingerprint(None, ())

    params = set()
    if isinstance(fn_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = fn_ast.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            params.add(arg.arg)
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)

    captured = []
    for node in ast.walk(fn_ast):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id in params:
            continue
        value = user_globals.get(node.id, _MISSING)
        if value is _MISSING:
            continue
        if isinstance(value, (int, float, bytes, str)) or value is None:
            captured.append(f"{node.id}={value!r}")

    captured_entries = tuple(sorted(set(captured)))
    payload = dumped
    if captured_entries:
        payload = dumped + '\n' + '|'.join(captured_entries)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]
    return FunctionContentFingerprint(digest, captured_entries)


def function_content_hash(fn_ast, user_globals: Optional[Dict[str, Any]] = None):
    """Hash a function AST plus bakeable captured constants."""
    return fingerprint_function_content(fn_ast, user_globals).digest


class BuildCache:
    """
    Manages build cache and timestamp checking for incremental compilation.
    
    Layered invalidation rules:
    - .o recompiled when: source file changes
    - .so re-linked when: any dependent .o changes
    - dlopen reloaded when: .so changes
    """
    
    @staticmethod
    def check_obj_uptodate(obj_file: str, source_file: str) -> bool:
        """Check if `.o` file is up-to-date."""
        if not os.path.exists(source_file):
            return False
        if not os.path.exists(obj_file):
            return False
        source_mtime = os.path.getmtime(source_file)
        obj_mtime = os.path.getmtime(obj_file)
        return obj_mtime >= source_mtime

    
    @staticmethod
    def check_so_needs_relink(so_file: str, obj_files: List[str]) -> bool:
        """
        Check if .so file needs to be re-linked.
        
        Args:
            so_file: Path to .so file
            obj_files: List of .o files this .so depends on
            
        Returns:
            bool: True if .so needs re-linking, False if up-to-date
        """
        if not os.path.exists(so_file):
            return True

        # Windows-specific: after linker changes, older DLLs may lack a generated
        # exports definition and therefore end up with an empty export table.
        # This breaks downstream links that rely on the import library (`.lib`).
        # Treat missing sidecar artifacts as requiring a relink.
        if so_file.lower().endswith('.dll'):
            exports_def = os.path.splitext(so_file)[0] + '.exports.def'
            implib = os.path.splitext(so_file)[0] + '.lib'
            if not os.path.exists(exports_def):
                return True
            if not os.path.exists(implib):
                return True

        so_mtime = os.path.getmtime(so_file)

        for obj_file in obj_files:
            if not os.path.exists(obj_file):
                return True
            if os.path.getmtime(obj_file) > so_mtime:
                return True
        
        return False

    
    @staticmethod
    def _delete_files(*files):
        """Delete files if they exist, ignoring errors."""
        for f in files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
    
    @staticmethod
    def invalidate_obj(obj_file: str):
        """
        Invalidate .o and related files.
        
        Args:
            obj_file: Path to .o file
        """
        if obj_file:
            ir_file = obj_file.replace('.o', '.ll')
            deps_file = obj_file.replace('.o', '.deps')
            BuildCache._delete_files(obj_file, ir_file, deps_file)
