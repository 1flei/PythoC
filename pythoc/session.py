"""Compile session container.

A CompileSession holds the shared mutable state of one compilation
session: the effect configuration, the forward reference namespace,
the build-layer singletons (output manager, dependency tracker,
multi-SO executor), and the unified registry of mutable compilation
artifacts.

There is no implicit default session.  A session must be created and
activated explicitly -- either via ``pythoc.init()`` (once per process
entry point) or via ``with CompileSession():``.  Code that needs the
active session resolves it through ``CompileSession.current()``, which
raises RuntimeError when no session is active.

Activation is tracked with a ContextVar (not threading.local) so the
active session propagates through ``contextvars.copy_context()``, which
the build scheduler uses when submitting worker tasks.

This module is intentionally dependency-free so that any pythoc module
can import it without creating import cycles.  Fields are plain Optional
attributes; owning modules attach their singletons to the active
session when they create them.
"""

from contextvars import ContextVar, Token
from typing import Any, List, Optional


NO_ACTIVE_SESSION_MSG = (
    "no active compile session: call pythoc.init() once at the process "
    "entry point, or run inside 'with CompileSession():'"
)


class CompileSession:
    """Shared mutable state for one compilation session."""

    def __init__(self) -> None:
        # pythoc.effect.Effect instance; lazily created and attached by
        # the effect proxy on first access.
        self.effects: Optional[Any] = None
        # pythoc.build.output_manager.OutputManager instance; attached by
        # build.output_manager.get_output_manager() on first access.
        self.output_manager: Optional[Any] = None
        # pythoc.build.deps.DependencyTracker instance; attached by
        # build.deps.get_dependency_tracker() on first access.
        self.dependency_tracker: Optional[Any] = None
        # pythoc.native_executor.MultiSOExecutor instance; attached by
        # native_executor.get_multi_so_executor() on first access.
        self.multi_so_executor: Optional[Any] = None
        # Per-session unified registry for mutable compilation artifacts
        # (structs, compilers, source files, link libraries/objects).
        # The process-level frozen builtin entity table is shared and
        # lives in pythoc.registry itself.
        from .registry import UnifiedCompilationRegistry
        self.registry: Any = UnifiedCompilationRegistry()
        # pythoc.forward_ref.ForwardRefNamespace instance; created here via
        # a local import to keep this module import-cycle-free.
        from .forward_ref import ForwardRefNamespace
        self.forward_refs: ForwardRefNamespace = ForwardRefNamespace()
        # Tokens of nested ``with session:`` blocks, innermost last.
        self._enter_tokens: List[Token] = []

    @classmethod
    def current(cls) -> 'CompileSession':
        """Return the session active in the current context.

        Raises:
            RuntimeError: If no session is active.  Activate one with
                ``pythoc.init()`` or ``with CompileSession():``.
        """
        session = _current_session.get()
        if session is None:
            raise RuntimeError(NO_ACTIVE_SESSION_MSG)
        return session

    @classmethod
    def active(cls) -> Optional['CompileSession']:
        """Return the active session, or None when no session is active."""
        return _current_session.get()

    def activate(self) -> Token:
        """Make this the active session; returns a token for deactivate()."""
        return _current_session.set(self)

    def deactivate(self, token: Token) -> None:
        """Restore the session that was active before the matching activate()."""
        _current_session.reset(token)

    def __enter__(self) -> 'CompileSession':
        self._enter_tokens.append(self.activate())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.deactivate(self._enter_tokens.pop())
        return False


_current_session: ContextVar[Optional[CompileSession]] = ContextVar(
    'pythoc_current_session', default=None,
)
