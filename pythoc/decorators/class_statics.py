# -*- coding: utf-8 -*-
"""Class-level static member support for decorated classes.

@compile (struct) and @union classes may declare class-level static members
in the class body using the same annotation form as function-local statics:

    @compile
    class Counter:
        count: static[i32] = 0
        buf: static[array[i32, 4]]

Such members are not instance fields: they have static storage duration
(C++ static member semantics) and lower to one LLVM global per compilation
module, named ``ClassName[_suffix].member`` so generic instantiations stay
independent. Default linkage is weak_odr: every module that touches the
member emits a coalescable definition, so the whole program shares one
entity (C++ inline static member semantics); ``@compile(linkage='internal')``
on the class opts into module-private storage, and ``@extern`` on a class
emits a declaration (external, no initializer) so another module's
definition can provide the storage.
Access goes through the class only (``Cls.member``); instance access is
intentionally not resolved, keeping the instance/class namespace
separation used by struct attribute handling.

Initializers must be link-time constants, matching function-local statics;
an omitted initializer zero-initializes (C semantics) unless the class is
an ``@extern`` declaration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..logger import logger

_NO_INIT = object()


def split_static_members(cls, parsed_field_types: List[Any]) -> List[Any]:
    """Partition ``cls._struct_fields`` into instance fields and static members.

    ``parsed_field_types`` is the resolved parallel list of
    ``cls._struct_fields`` (string annotations already resolved where
    possible). Entries whose resolved type carries the ``static`` or
    ``thread_local`` qualifier are registered on ``cls._static_members`` as
    ``{name: (pc_type, init_value)}`` and removed from ``cls._struct_fields``
    in place, keeping the two lists index-aligned for downstream users
    (forward-ref callbacks etc.). Unresolved string annotations cannot be
    static and stay in the field list.

    Runs at most once per class, mirroring the build-once semantics of
    ``_struct_fields``. Returns the filtered ``parsed_field_types``.
    """
    from ..ir_helpers import is_static, is_thread_local

    if hasattr(cls, '_static_members'):
        return parsed_field_types

    cls._static_members: Dict[str, Tuple[Any, Any]] = {}
    keep_fields = []
    keep_types = []
    for (fname, raw_annotation), ftype in zip(cls._struct_fields, parsed_field_types):
        if not isinstance(ftype, str) and (is_static(ftype) or is_thread_local(ftype)):
            init_value = cls.__dict__.get(fname, _NO_INIT)
            cls._static_members[fname] = (ftype, init_value)
        else:
            keep_fields.append((fname, raw_annotation))
            keep_types.append(ftype)

    cls._struct_fields = keep_fields
    return keep_types


def lookup_class_static(decorated_cls, attr_name: str) -> Optional[Tuple[Any, Any, Any]]:
    """Return ``(pc_type, init_value, owner_cls)`` for a class static, or None.

    Mirrors :func:`lookup_class_method`: the class-level ``handle_attribute``
    is a classmethod bound to the unified type (e.g. a ``StructType``
    subclass) rather than to the user's Python class, so ``_python_class``
    is checked as a transparent redirect.
    """
    candidates = [decorated_cls]
    py_cls = getattr(decorated_cls, '_python_class', None)
    if py_cls is not None and py_cls is not decorated_cls:
        candidates.append(py_cls)

    for candidate in candidates:
        members = getattr(candidate, '_static_members', None)
        if members and attr_name in members:
            pc_type, init_value = members[attr_name]
            return pc_type, init_value, candidate
    return None


def get_or_create_static_global(visitor, decorated_cls, attr_name: str, node):
    """Return an address-kind ValueRef for a class-level static member.

    Lazily creates one ``ir.GlobalVariable`` per compilation module (same
    pattern as function-local statics), reusing an existing global of the
    same name when the member is accessed from several functions in the
    same module. Linkage comes from the owning class (default weak_odr, so
    all modules share one entity); ``@extern`` classes emit a declaration
    with no initializer.
    """
    from llvmlite import ir
    from ..valueref import wrap_value
    from ..ir_helpers import is_thread_local

    found = lookup_class_static(decorated_cls, attr_name)
    if found is None:
        logger.error(
            f"'{decorated_cls.get_name()}' has no class static member '{attr_name}'",
            node=node, exc_type=AttributeError,
        )
    pc_type, init_value, owner_cls = found

    name_parts = [owner_cls.__name__]
    suffix = getattr(owner_cls, '_compile_suffix', None)
    if suffix:
        name_parts.append(suffix)
    global_name = f"{'_'.join(name_parts)}.{attr_name}"

    try:
        global_var = visitor.module.get_global(global_name)
    except KeyError:
        global_var = None

    if global_var is None:
        llvm_type = pc_type.get_llvm_type(visitor.module.context)
        is_decl = getattr(owner_cls, '_static_is_decl', False)
        linkage = getattr(owner_cls, '_static_linkage', 'weak_odr')

        global_var = ir.GlobalVariable(visitor.module, llvm_type, global_name)
        if is_thread_local(pc_type):
            global_var.storage_class = 'thread_local'
        if not is_decl:
            # A defining global with an initializer already has external
            # linkage; setting linkage='external' emits invalid
            # ``external global T <init>`` IR. Only non-default kinds
            # (internal / weak_odr / ...) are written onto the definition.
            if linkage != 'external':
                global_var.linkage = linkage
            seed = _fold_static_initializer(visitor, pc_type, llvm_type,
                                            init_value, attr_name, node)
            global_var.initializer = seed
            global_var.global_constant = False

    return wrap_value(global_var, kind='address', type_hint=pc_type,
                      address=global_var)


def _fold_static_initializer(visitor, pc_type, llvm_type, init_value,
                             attr_name, node):
    """Fold a class-static initializer to a link-time constant.

    An omitted initializer zero-initializes (C semantics). Otherwise the
    Python value captured from the class body is folded through the same
    constant path used for aggregate elements, and must end up a link-time
    constant -- static storage cannot run instructions.
    """
    from ..ir_helpers import is_link_time_constant
    from ..literal_protocol import _lower_element_to_constant

    if init_value is _NO_INIT:
        return visitor.type_converter.create_zero_constant(llvm_type)

    seed = _lower_element_to_constant(visitor, init_value, pc_type)
    if seed is None or not is_link_time_constant(seed):
        logger.error(
            f"Class static member '{attr_name}' requires compile-time "
            f"constant initializer",
            node=node,
        )

    # llvmlite rejects a scalar zero for aggregate static initializers.
    from llvmlite import ir
    if isinstance(seed, ir.Constant) and isinstance(
        seed.type,
        (ir.ArrayType, ir.BaseStructType, ir.LiteralStructType,
         ir.IdentifiedStructType),
    ):
        if isinstance(seed.constant, int) and seed.constant == 0:
            from ..ast_visitor.assignments import _make_zero_aggregate
            seed = _make_zero_aggregate(seed.type)
    return seed
