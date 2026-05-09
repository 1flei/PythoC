"""Normalized C import IR used by cimport backends.

The native pythoc parser and optional libclang parser expose different AST
shapes.  This module defines the small declaration/type layer that the pythoc
emitter consumes, so frontend providers do not leak into code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class CTypeIR:
    kind: str
    name: Optional[str] = None
    pointee: Optional["CTypeIR"] = None
    element: Optional["CTypeIR"] = None
    size: Optional[int] = None
    return_type: Optional["CTypeIR"] = None
    params: list["CParamIR"] = field(default_factory=list)
    is_variadic: bool = False


@dataclass(slots=True)
class CParamIR:
    name: Optional[str]
    type: CTypeIR


@dataclass(slots=True)
class CFieldIR:
    name: Optional[str]
    type: CTypeIR
    bit_width: Optional[int] = None


@dataclass(slots=True)
class CEnumValueIR:
    name: str
    value: Optional[int] = None


@dataclass(slots=True)
class CDeclIR:
    kind: str
    name: str
    type: Optional[CTypeIR] = None
    fields: list[CFieldIR] = field(default_factory=list)
    values: list[CEnumValueIR] = field(default_factory=list)
    storage: Optional[str] = None
    is_definition: bool = False


@dataclass(slots=True)
class CModuleIR:
    declarations: list[CDeclIR] = field(default_factory=list)
