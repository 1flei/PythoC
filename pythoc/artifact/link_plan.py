"""Resolve compilation groups into deterministic linker inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from .model import LinkScope


@dataclass(frozen=True)
class LinkPlan:
    roots: Tuple[Tuple, ...]
    obj_files: Tuple[str, ...]
    link_libraries: Tuple[str, ...] = ()
    scope: LinkScope = LinkScope.TRANSITIVE

    @classmethod
    def from_compiled_symbols(
        cls,
        symbols: Sequence[Any],
        *,
        scope: LinkScope = LinkScope.TRANSITIVE,
        flush: bool = True,
    ) -> "LinkPlan":
        roots = []
        for symbol in symbols:
            binding = getattr(symbol, "_binding", None)
            group_key = getattr(binding, "group_key", None)
            if group_key is None:
                raise RuntimeError(
                    "compiled symbol {!r} has no compilation group".format(
                        symbol
                    )
                )
            roots.append(tuple(group_key))
        return cls.from_group_roots(roots, scope=scope, flush=flush)

    @classmethod
    def from_group_roots(
        cls,
        group_keys: Sequence[Tuple],
        *,
        scope: LinkScope = LinkScope.TRANSITIVE,
        flush: bool = True,
        output_manager=None,
        dependency_tracker=None,
    ) -> "LinkPlan":
        if flush:
            from ..build.output_manager import flush_all_pending_outputs

            flush_all_pending_outputs()

        if output_manager is None:
            from ..build.output_manager import get_output_manager

            output_manager = get_output_manager()
        if dependency_tracker is None:
            from ..build.deps import get_dependency_tracker

            dependency_tracker = get_dependency_tracker()

        roots = tuple(_normalize_group_key(key) for key in group_keys)
        groups = output_manager.get_all_groups()
        obj_files = []
        libraries = []
        seen_groups = set()
        seen_objects = set()
        seen_libraries = set()

        def add_object(path: Optional[str]) -> None:
            if (
                path
                and os.path.exists(path)
                and path not in seen_objects
            ):
                seen_objects.add(path)
                obj_files.append(path)

        def add_library(library: str) -> None:
            if library and library not in seen_libraries:
                seen_libraries.add(library)
                libraries.append(library)

        def visit(group_key: Tuple) -> None:
            if group_key in seen_groups:
                return
            seen_groups.add(group_key)

            group = groups.get(group_key)
            obj_file = group.get("obj_file") if group else None
            if obj_file is None:
                obj_file = dependency_tracker.derive_obj_file_from_group_key(
                    group_key
                )
            add_object(obj_file)

            if scope is LinkScope.OWN_GROUP:
                return

            deps = (
                dependency_tracker.get_deps(group_key, obj_file=obj_file)
                if obj_file
                else None
            )
            if deps is None:
                deps = dependency_tracker.get_deps_for_group(group_key)
            if deps is None:
                return

            for link_object in deps.link_objects:
                add_object(link_object)
            for library in deps.link_libraries:
                add_library(library)
            for dependency in deps.group_dependencies:
                target = getattr(dependency, "target_group", None)
                if target is not None:
                    visit(target.to_tuple())

        for root in roots:
            visit(root)

        if not obj_files:
            raise RuntimeError(
                "no object files found for selected compilation groups"
            )
        return cls(
            roots=roots,
            obj_files=tuple(obj_files),
            link_libraries=tuple(libraries),
            scope=scope,
        )

    @classmethod
    def from_all_groups(cls, *, flush: bool = True) -> "LinkPlan":
        if flush:
            from ..build.output_manager import flush_all_pending_outputs

            flush_all_pending_outputs()
        from ..build.output_manager import get_output_manager

        roots = tuple(
            key
            for key in get_output_manager().get_all_groups()
            if not is_internal_group(key)
        )
        if not roots:
            raise RuntimeError(
                "no @compile functions found; nothing to link"
            )
        return cls.from_group_roots(
            roots,
            scope=LinkScope.OWN_GROUP,
            flush=False,
        )


def is_internal_group(group_key) -> bool:
    """Groups compiled from pythoc's own package sources.

    The callable runtime (callable_type.py) and generated Python-adapter
    entries (python_entry_bind.py) are process implementation details and
    must never leak into a user program's link: their CPython symbols are
    only resolvable inside a Python process, and their adapter symbols do
    not belong to executable/static/dynamic-library targets.
    """
    source = group_key[0] if group_key else ""
    if not source:
        return False
    pythoc_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    try:
        return os.path.realpath(source).startswith(pythoc_dir + os.sep)
    except OSError:
        return False


def _normalize_group_key(group_key) -> Tuple:
    if hasattr(group_key, "to_tuple"):
        group_key = group_key.to_tuple()
    return tuple(group_key)
