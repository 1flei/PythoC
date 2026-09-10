"""Compiler-version stamp for build caches.

Cached artifacts (.o files, cimport bindings/objects) are keyed on the
mtimes of the user sources they were built from, but their content also
depends on pythoc's own compiler code.  Without a compiler stamp, editing
pythoc itself leaves stale artifacts behind: the cache serves code built
by an older compiler, silently masking (or faking) regressions.

This module provides a single process-wide stamp: the latest mtime across
the pythoc package's own .py files.  Cache layers compare artifact mtimes
against it (or fold it into cache keys) so any compiler edit invalidates
previously cached outputs exactly once.
"""

import os

_COMPILER_MTIME = None


def get_compiler_mtime() -> float:
    """Latest mtime across the pythoc package's own source files.

    Computed once per process.  Returns 0.0 when no .py sources are found
    (e.g. a pyc-only installation), which disables stamp-based
    invalidation rather than breaking caching entirely.
    """
    global _COMPILER_MTIME
    if _COMPILER_MTIME is None:
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        latest = 0.0
        for dirpath, dirnames, filenames in os.walk(pkg_root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for name in filenames:
                if not name.endswith('.py'):
                    continue
                try:
                    mtime = os.path.getmtime(os.path.join(dirpath, name))
                except OSError:
                    continue
                if mtime > latest:
                    latest = mtime
        _COMPILER_MTIME = latest
    return _COMPILER_MTIME
