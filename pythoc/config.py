"""Centralised configuration for pythoc.

All process-wide configuration knobs that used to be scattered as raw
``os.environ.get`` calls live here behind a single, typed registry.
The goal is three-fold:

1. **One canonical name + default + docstring per knob** -- no more
   hunting through ``logger.py`` / ``output_manager.py`` /
   ``ast_debug.py`` to find what a flag does.
2. **Backwards-compatible env override** -- every knob still picks up
   the historical ``PC_*`` environment variable at process start, so
   existing scripts and CI invocations keep working unchanged.
3. **Python-level API for toggling at runtime** -- callers may flip a
   knob via ``config.save_ir = True``, ``config.set(save_ir=True)``,
   or ``with config.override(save_ir=True): ...``.  Each knob's
   metadata declares **when** changes actually take effect (e.g.
   ``logger`` knobs are sampled once at logger construction time and
   require ``logger.reload_config()`` after a change).

Usage
-----

::

    from pythoc import config

    # Read
    if config.save_ir:
        ...

    # Write (process-wide)
    config.save_ir = True
    config.set(opt_level=0, save_ir=True)

    # Scoped override
    with config.override(opt_level=3):
        compile_things()

Defining new knobs
------------------

Add an entry to ``_SCHEMA`` below.  Each entry declares the typed
default, the historical env-var name, and a free-form ``effective``
string for documentation.  Then ``from .config import config`` from
the consumer module and read ``config.<name>``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional


# ---------------------------------------------------------------------------
# Type coercers: env vars are always strings; coerce into the typed
# default's domain.  Each coercer must be total over its string input
# (raise ValueError on bad input rather than silently misbehaving).
# ---------------------------------------------------------------------------


def _to_bool(s: str) -> bool:
    """Permissive truthiness for env strings.

    Matches the historical pythoc convention: any non-empty value
    counts as true unless it is one of the explicit "off" strings.
    This is intentional -- existing scripts setting flags like
    ``PC_ENABLE_CLANG_CIMPORT=yes`` should keep working unchanged.
    """
    return s.strip().lower() not in {'', '0', 'false', 'no', 'off'}


def _to_int(s: str) -> int:
    return int(s.strip())


def _to_str(s: str) -> str:
    return s


# ---------------------------------------------------------------------------
# Schema: single source of truth for every PC_* knob.
#
# Each ConfigKnob carries:
#   - name:       canonical Python attribute name (snake_case)
#   - env:        historical env var name (kept verbatim for compatibility)
#   - default:    typed default used when env var is unset
#   - coerce:     str -> typed-value callable applied to env var content
#   - doc:        short description shown in repr / help
#   - effective:  free-form note about when changes take effect
# ---------------------------------------------------------------------------


class ConfigKnob:
    """One typed configuration entry."""

    __slots__ = ('name', 'env', 'default', 'coerce', 'doc', 'effective')

    def __init__(self, name: str, env: str, default: Any,
                 coerce: Callable[[str], Any], doc: str, effective: str):
        self.name = name
        self.env = env
        self.default = default
        self.coerce = coerce
        self.doc = doc
        self.effective = effective

    def read_env(self) -> Any:
        """Pick up the typed value from the environment, falling back
        to the schema default when the env var is unset or empty."""
        raw = os.environ.get(self.env)
        if raw is None or raw == '':
            return self.default
        try:
            return self.coerce(raw)
        except (TypeError, ValueError):
            # Fall back to default rather than crashing the import; the
            # logger may not yet be initialised so we deliberately do
            # not call logger.error here.
            return self.default


_SCHEMA: Dict[str, ConfigKnob] = {}


def _register(name: str, env: str, default: Any,
              coerce: Callable[[str], Any], doc: str,
              effective: str) -> None:
    _SCHEMA[name] = ConfigKnob(name, env, default, coerce, doc, effective)


# --- Logging --------------------------------------------------------------
_register(
    'log_level', 'PC_LOG_LEVEL', 1, _to_int,
    'Minimum log level emitted by pythoc.logger '
    '(0=debug, 1=info, 2=warn, 3=error).',
    'Sampled once when pythoc.logger.Logger() is constructed. '
    'Call logger.reload_config() to re-sample after a change.',
)
_register(
    'log_modules', 'PC_LOG_MODULES', '', _to_str,
    'Comma-separated module-prefix filter for debug logs. '
    'Prefix with "-" to exclude (e.g. "build,-build.cache").',
    'Sampled once at logger construction. See log_level note.',
)
_register(
    'raise_on_error', 'PC_RAISE_ON_ERROR', False, _to_bool,
    'When true, logger.error raises the specified exception type '
    'instead of calling sys.exit.  Used by unit tests.',
    'Sampled once at logger construction. See log_level note.',
)


# --- AST debugging --------------------------------------------------------
_register(
    'debug_ast', 'PC_DEBUG_AST', False, _to_bool,
    'Dump the AST before/after each transform pass for inspection.',
    'Read on every call into ASTDebugger.is_enabled().',
)
_register(
    'debug_ast_format', 'PC_DEBUG_AST_FORMAT', 'all', _to_str,
    'AST dump format: "all", "text", "json".',
    'Read by ASTDebugger when dumping.',
)
_register(
    'debug_ast_diff', 'PC_DEBUG_AST_DIFF', False, _to_bool,
    'Show per-pass AST diffs in addition to full dumps.',
    'Read by ASTDebugger when dumping.',
)


# --- Build / codegen ------------------------------------------------------
_register(
    'save_ir', 'PC_SAVE_IR', False, _to_bool,
    'Persist optimised LLVM IR as .ll alongside each .o.  Default off '
    'because the .ll is not consumed by the toolchain; .o is the only '
    'cache-relevant artefact.',
    'Sampled when each compile group is flushed.  Setting this *before* '
    'a @compile-decorated function is first invoked is what matters; '
    'flipping it mid-compile may give mixed results across groups.',
)
_register(
    'save_unopt_ir', 'PC_SAVE_UNOPT_IR', False, _to_bool,
    'Persist the pre-optimisation LLVM IR as .unopt.ll for debugging.',
    'Same timing as save_ir.',
)
_register(
    'opt_level', 'PC_OPT_LEVEL', 2, _to_int,
    'LLVM optimisation level (0..3) applied before object emission.',
    'Sampled when each compile group is flushed.',
)
_register(
    'debug_info', 'PC_DEBUG_INFO', False, _to_bool,
    'Emit DWARF debug info (line tables for functions/source lines) into '
    'object files and linked binaries.',
    'Sampled when each compile group is flushed.',
)


# --- cimport / external toolchain ----------------------------------------
_register(
    'cimport_backend', 'PC_CIMPORT_BACKEND', None,
    lambda s: s.strip().lower() or None,
    'Preferred cimport backend: "auto" or "clang".  None '
    'defaults to "auto" (resolves to clang).',
    'Read on every cimport call.',
)
_register(
    'cimport_target', 'PC_CIMPORT_TARGET', None, _to_str,
    'Target triple passed to the cimport backend (e.g. when '
    'cross-parsing system headers).',
    'Read on every cimport call.',
)
_register(
    'cimport_sysroot', 'PC_CIMPORT_SYSROOT', None, _to_str,
    'Sysroot passed to the cimport backend.',
    'Read on every cimport call.',
)
_register(
    'libclang_path', 'PC_LIBCLANG_PATH', None, _to_str,
    'Override path to the libclang shared library used by the clang '
    'cimport backend.',
    'Read once when the clang backend resolves libclang.',
)
_register(
    'cimport_clang_args', 'PC_CIMPORT_CLANG_ARGS', '', _to_str,
    'Extra command-line flags forwarded to the clang cimport backend. '
    'Shell-tokenised via shlex; e.g. "-I/usr/local/include -DFOO=1".',
    'Read on every clang-backed cimport call.',
)


# ---------------------------------------------------------------------------
# Config: typed accessor object.  Attribute access (read/write) and
# `set()`/`override()` helpers all funnel through `_SCHEMA`.
# ---------------------------------------------------------------------------


class _Config:
    """Process-wide configuration store with env-var fallback.

    Semantics:
    - **Read**: if a knob has been explicitly set in Python (via
      attribute assignment / ``set()`` / ``override()``), that
      in-memory value wins.  Otherwise the corresponding env var is
      re-sampled on every access -- so tests that temporarily patch
      ``os.environ`` continue to work, and existing call sites that
      relied on ``os.environ.get`` semantics see no behavioural
      change.
    - **Write**: stores the typed value in-memory.  Subsequent reads
      return that value until ``reset(name)`` (or ``reset()`` for all
      knobs) is called.
    """

    def __init__(self) -> None:
        object.__setattr__(self, '_values', {})

    # ------------------------------------------------------------------
    # Attribute protocol
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, so we
        # never accidentally shadow real attributes (e.g. _values).
        if name not in _SCHEMA:
            raise AttributeError(
                f"pythoc.config has no knob '{name}'. "
                f"Known knobs: {sorted(_SCHEMA)}")
        store = object.__getattribute__(self, '_values')
        if name in store:
            return store[name]
        # Not explicitly set -- re-sample from env every read.
        return _SCHEMA[name].read_env()

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in _SCHEMA:
            raise AttributeError(
                f"pythoc.config has no knob '{name}'. "
                f"Known knobs: {sorted(_SCHEMA)}")
        # No coercion on programmatic writes -- callers pass typed
        # values directly (config.opt_level = 3, not "3").
        object.__getattribute__(self, '_values')[name] = value

    # ------------------------------------------------------------------
    # Bulk / scoped APIs
    # ------------------------------------------------------------------

    def set(self, **kwargs: Any) -> None:
        """Set one or more knobs at once.

        ::

            config.set(save_ir=True, opt_level=0)
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    @contextmanager
    def override(self, **kwargs: Any) -> Iterator[None]:
        """Temporarily set knobs for the duration of a ``with`` block.

        Restores prior state on exit (including knobs that were not
        explicitly set before -- they revert to env-fallback), even on
        exception.

        ::

            with config.override(save_ir=True):
                some_compile_call()
        """
        store = object.__getattribute__(self, '_values')
        # Snapshot which knobs had explicit values *before* the override
        # so we can restore the "env fallback" state precisely.
        was_set: Dict[str, Any] = {}
        was_unset: list = []
        try:
            for name, value in kwargs.items():
                if name not in _SCHEMA:
                    raise AttributeError(
                        f"pythoc.config has no knob '{name}'.")
                if name in store:
                    was_set[name] = store[name]
                else:
                    was_unset.append(name)
                store[name] = value
            yield
        finally:
            for name, value in was_set.items():
                store[name] = value
            for name in was_unset:
                store.pop(name, None)

    def reset(self, *names: str) -> None:
        """Forget any in-memory overrides for the given knobs (or all
        knobs if called with no arguments) and re-sample them from the
        environment on the next read.
        """
        store = object.__getattribute__(self, '_values')
        if not names:
            store.clear()
            return
        for name in names:
            store.pop(name, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self, name: Optional[str] = None) -> str:
        """Return a human-readable description of one or all knobs."""
        if name is not None:
            knob = _SCHEMA[name]
            return (
                f"{knob.name} (env: {knob.env}) = {getattr(self, name)!r}\n"
                f"  default:   {knob.default!r}\n"
                f"  effective: {knob.effective}\n"
                f"  doc:       {knob.doc}"
            )
        return '\n\n'.join(self.describe(n) for n in sorted(_SCHEMA))

    def __repr__(self) -> str:
        items = ', '.join(
            f"{n}={getattr(self, n)!r}" for n in sorted(_SCHEMA)
        )
        return f"<pythoc.config {items}>"


# Module-level singleton -- the documented public surface.
config = _Config()


__all__ = ['config']
