"""Public models for planning, building, and loading native artifacts."""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from .link_plan import LinkPlan


class ArtifactKind(str, Enum):
    EXECUTABLE = "executable"
    STATIC_LIBRARY = "static_library"
    SHARED_LIBRARY = "shared_library"
    PYTHON_EXTENSION = "python_extension"


class ArtifactRole(str, Enum):
    NATIVE = "native"
    PYTHON_RUNTIME = "python_runtime"
    PYTHON_ADAPTER = "python_adapter"


class LinkScope(str, Enum):
    TRANSITIVE = "transitive"
    OWN_GROUP = "own_group"


class ArtifactPhase(str, Enum):
    PRE_LINK = "pre_link"
    POST_LINK = "post_link"


@dataclass(frozen=True)
class ExportSpec:
    export_id: str
    native_symbol: str
    adapter_symbol: Optional[str] = None
    python_name: Optional[str] = None


@dataclass(frozen=True)
class ArtifactStep:
    """One target-specific operation around the common link task."""

    id: str
    kind: str
    phase: ArtifactPhase
    run: Callable[[], Any] = field(repr=False, compare=False)
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    cache_check: Optional[Callable[[], bool]] = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ArtifactPlan:
    kind: ArtifactKind
    link: "LinkPlan"
    output_path: str
    exports: Tuple[ExportSpec, ...] = ()
    prefix_objects: Tuple[str, ...] = ()
    suffix_objects: Tuple[str, ...] = ()
    link_objects: Tuple[str, ...] = ()
    link_libraries: Tuple[str, ...] = ()
    extra_link_flags: Tuple[str, ...] = ()
    steps: Tuple[ArtifactStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_path:
            raise ValueError("artifact output_path must not be empty")

    @property
    def object_files(self) -> Tuple[str, ...]:
        return (
            self.prefix_objects
            + tuple(self.link.obj_files)
            + self.suffix_objects
        )


@dataclass(frozen=True)
class NativeArtifact:
    kind: ArtifactKind
    identity: str
    path: str
    link: "LinkPlan"
    exports: Tuple[ExportSpec, ...] = ()
    sidecars: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: ArtifactPlan) -> "NativeArtifact":
        sidecars = []
        for step in plan.steps:
            if step.phase is ArtifactPhase.POST_LINK:
                sidecars.extend(step.outputs)
        return cls(
            kind=plan.kind,
            identity=_file_digest(plan.output_path),
            path=plan.output_path,
            link=plan.link,
            exports=plan.exports,
            sidecars=tuple(sidecars),
            metadata=dict(plan.metadata),
        )

    def load(self, *, mode: Optional[int] = None) -> "LoadedArtifact":
        if self.kind in (
            ArtifactKind.EXECUTABLE,
            ArtifactKind.STATIC_LIBRARY,
        ):
            raise TypeError(
                "{} artifacts cannot be loaded into the current process".format(
                    self.kind.value
                )
            )
        load_path = os.path.abspath(self.path)
        if mode is None:
            handle = ctypes.CDLL(load_path)
        else:
            handle = ctypes.CDLL(load_path, mode=mode)
        return LoadedArtifact(artifact=self, handle=handle)


@dataclass
class LoadedArtifact:
    artifact: NativeArtifact
    handle: Any

    def resolve(self, symbol: str) -> int:
        raw = getattr(self.handle, symbol)
        address = ctypes.cast(raw, ctypes.c_void_p).value
        if address is None:
            raise RuntimeError(
                "symbol {!r} resolved to NULL in {!r}".format(
                    symbol,
                    self.artifact.path,
                )
            )
        return int(address)

    def resolve_export(self, export_id: str) -> int:
        matches = [
            export
            for export in self.artifact.exports
            if export.export_id == export_id
        ]
        if len(matches) != 1:
            raise KeyError("unknown artifact export {!r}".format(export_id))
        symbol = matches[0].adapter_symbol or matches[0].native_symbol
        return self.resolve(symbol)


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
