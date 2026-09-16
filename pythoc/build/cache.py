# -*- coding: utf-8 -*-
"""
Layered build cache for pythoc.

Implements the layered invalidation model:
    Source (.py) -> Object (.o) -> Shared Lib (.so) -> dlopen

Each layer updates ONLY when its input (previous layer) changes.
.ll is just an intermediate artifact, not a cache layer.

Additionally, every cached artifact must be newer than the pythoc
compiler itself (see utils.compiler_stamp): an artifact built by an
older compiler is stale even when its source is untouched.
"""

import ast
import hashlib
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

_MISSING = object()

# Bounds for _stable_capture_repr: deep or huge containers fall back to
# "not fingerprintable" (the capture is then simply absent, same as
# non-constant values).
_CAPTURE_MAX_DEPTH = 4
_CAPTURE_MAX_LEN = 65536


def _stable_capture_repr(value: Any, _depth: int = 0) -> Optional[str]:
    """Deterministic repr for fingerprintable constants, else None.

    Scalars use repr directly.  Constant containers are rendered
    recursively; unordered types (set/frozenset/dict) are sorted by key
    rendering so the result is independent of iteration order.  Anything
    else (arbitrary objects, modules, functions, ...) is not
    fingerprintable and yields None -- such captures stay outside the
    digest, exactly as non-captured names always have.
    """
    if value is None or isinstance(value, (int, float, bytes, str)):
        return repr(value)
    if _depth >= _CAPTURE_MAX_DEPTH:
        return None
    if isinstance(value, (tuple, list)):
        parts = [_stable_capture_repr(item, _depth + 1) for item in value]
        if any(part is None for part in parts):
            return None
        open_, close_ = ('(', ')') if isinstance(value, tuple) else ('[', ']')
        result = open_ + ','.join(parts) + close_
    elif isinstance(value, (set, frozenset)):
        parts = [_stable_capture_repr(item, _depth + 1) for item in value]
        if any(part is None for part in parts):
            return None
        result = '{' + ','.join(sorted(parts)) + '}'
    elif isinstance(value, dict):
        items = []
        for k, v in value.items():
            key_repr = _stable_capture_repr(k, _depth + 1)
            val_repr = _stable_capture_repr(v, _depth + 1)
            if key_repr is None or val_repr is None:
                return None
            items.append((key_repr, val_repr))
        items.sort()
        result = '{' + ','.join(f'{k}:{v}' for k, v in items) + '}'
    else:
        return None
    if len(result) > _CAPTURE_MAX_LEN:
        return None
    return result


class FunctionContentFingerprint(NamedTuple):
    digest: Optional[str]
    captured: Tuple[str, ...]


def fingerprint_function_content(
    fn_ast, user_globals: Optional[Dict[str, Any]] = None,
    include_ast: bool = True,
) -> FunctionContentFingerprint:
    """Fingerprint a function AST plus bakeable captured constants.

    Python constants referenced by name (for example a ctypes.addressof
    result assigned to a module global and then used as ``ptr[T](addr)``,
    or a ``from config import SIZE`` scalar) are folded into IR at compile
    time.  The source file mtime does not change when those values change
    across processes, so they have to participate in the cache key;
    otherwise a cached .o embeds a stale, process-local address.

    ``include_ast=False`` is for functions whose AST is provably derived
    from the group's source file text (the plain @compile path): the file
    mtime already keys the AST, so the fingerprint only needs to cover
    captured environment values.  Meta/generated functions (compile_ast,
    pre-stored ``__pc_source__`` wrappers, ...) must keep the AST
    component -- their source file does not determine their body.

    ``captured`` is non-empty when the digest includes such constants.
    Nested @compile created during a parent's codegen must fold that
    digest into the parent: the parent is the artefact that dlopens the
    nested .so, and a parent cache hit would skip re-decoration of the
    nested function entirely.
    """
    user_globals = user_globals or {}
    if include_ast:
        try:
            dumped = ast.dump(fn_ast)
        except Exception:
            return FunctionContentFingerprint(None, ())
    else:
        dumped = ''

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
        value_repr = _stable_capture_repr(value)
        if value_repr is not None:
            captured.append(f"{node.id}={value_repr}")

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
        """Check if `.o` file is up-to-date.

        The artifact must be newer than both its source and the pythoc
        compiler itself: an .o built by an older compiler is stale even
        when the source is untouched.
        """
        if not os.path.exists(source_file):
            return False
        if not os.path.exists(obj_file):
            return False
        from ..utils.compiler_stamp import get_compiler_mtime
        source_mtime = os.path.getmtime(source_file)
        obj_mtime = os.path.getmtime(obj_file)
        return (obj_mtime >= source_mtime
                and obj_mtime >= get_compiler_mtime())

    
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

        # Shared-link schema versioning: outputs linked before the current
        # schema (e.g. with registry link objects statically copied in) must
        # be relinked once.  No-op unless the process registered link
        # objects, so non-cimport sessions never pay for this.
        from ..utils.link_utils import shared_link_schema_stale
        if shared_link_schema_stale(so_file):
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
