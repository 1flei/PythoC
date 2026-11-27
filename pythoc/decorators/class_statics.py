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

import ast
from dataclasses import dataclass
import inspect
from typing import Any, List, Optional, Tuple

from ..logger import logger

_NO_INIT = object()


@dataclass(frozen=True)
class _StaticMember:
    """Optional native storage properties supplied by a compiler frontend."""

    initializer: Any = _NO_INIT
    symbol: Optional[str] = None
    linkage: Optional[str] = None
    visibility: Optional[str] = None
    alignment: Optional[int] = None
    section: Optional[str] = None
    used: bool = False


def _member_spec(value):
    return value if isinstance(value, _StaticMember) else _StaticMember(value)


def _static_symbol(owner, member):
    prefix = getattr(owner, '_static_member_symbol_prefix', None)
    if prefix == '':
        return member
    prefix = owner.__name__ if prefix is None else prefix
    suffix = getattr(owner, '_compile_suffix', None)
    if suffix:
        prefix += '_' + suffix
    return f'{prefix}.{member}'


def register_static_definitions(owner, node):
    """Register explicitly named storage as part of its source group."""
    from .compile import get_compiler
    from ..build.output_manager import get_output_manager
    from ..utils import get_build_paths

    if not owner._static_members and not has_deferred_storage(owner):
        return
    filename = inspect.getfile(owner)
    group_key = (filename, None, None, None)
    _, ir_file, obj_file, so_file = get_build_paths(filename)
    compiler = get_compiler(filename, {})
    manager = get_output_manager()
    group = manager.get_or_create_group(group_key, compiler, ir_file, obj_file,
                                        so_file, filename)
    owner._static_group_key = group_key
    group.setdefault('static_classes', []).append((owner, node))

    # The object also depends on class decoration inputs, which can come from
    # a factory even when the class source file is unchanged.
    import hashlib
    from ..build.cache import _stable_capture_repr
    from .visible import get_closure_variables
    fields = []
    for name, (field_type, spec) in owner._static_members.items():
        seed = spec.initializer
        if inspect.isfunction(seed) and seed.__name__ == '<lambda>':
            globals_used = {name: seed.__globals__[name] for name in seed.__code__.co_names
                            if name in seed.__globals__}
            captured = globals_used | get_closure_variables(seed)
            seed = tuple((name, item) for name, value in sorted(captured.items())
                         if (item := _stable_capture_repr(value)) is not None)
        fields.append((name, repr(field_type), spec.symbol, spec.linkage,
                       spec.visibility, spec.alignment, spec.section, spec.used,
                       _stable_capture_repr(seed)))
    identity = (owner._static_member_symbol_prefix, owner._static_linkage,
                owner._compile_suffix, fields)
    group.setdefault('_ast_content_hashes', []).append(
        hashlib.sha256(repr(identity).encode()).hexdigest())


def _native_member_address(visitor, owner, name, pc_type, spec, node):
    from ..ir_helpers import declare_native_global, is_thread_local

    symbol = spec.symbol if spec.symbol is not None else _static_symbol(owner, name)
    module = visitor.module
    llvm_type = pc_type.get_llvm_type(module.context)
    value, _ = declare_native_global(module, llvm_type, symbol,
                                    is_thread_local(pc_type), node,
                                    visibility=spec.visibility)
    return value


def _retain_static_definition(value):
    from llvmlite import ir

    module = value.parent
    byte_pointer = ir.IntType(8).as_pointer()
    name = 'llvm.compiler.used'
    retained = module.globals.get(name)
    addresses = list(retained.initializer.constant) if retained is not None else []
    addresses.append(value.bitcast(byte_pointer))
    array_type = ir.ArrayType(byte_pointer, len(addresses))
    if retained is None:
        retained = ir.GlobalVariable(module, array_type, name)
        retained.linkage = 'appending'
        retained.section = 'llvm.metadata'
    else:
        retained.type = array_type.as_pointer()
        retained.value_type = array_type
    retained.initializer = ir.Constant(array_type, addresses)


def _static_initializer_visitor(compiler, group_key, namespace=None):
    from llvmlite import ir
    from ..ast_visitor import LLVMIRVisitor
    from ..ast_visitor.control_flow_builder import ControlFlowBuilder
    from ..builder import LLVMBuilder

    visitor = LLVMIRVisitor(module=compiler.module, user_globals=namespace,
                            compiler=compiler)
    visitor.current_group_key = group_key
    visitor.builder = ControlFlowBuilder(LLVMBuilder(ir.IRBuilder()), visitor)
    return visitor


def _native_static_initializer(visitor, name, pc_type, llvm_type, seed, node):
    if not inspect.isfunction(seed) or seed.__name__ != '<lambda>':
        return _fold_static_initializer(visitor, pc_type, llvm_type, seed, name, node)

    from .visible import definition_names, get_all_accessible_symbols

    with logger.at_node(node):
        if not isinstance(node, ast.Lambda) or seed.__code__.co_argcount or (
                seed.__code__.co_kwonlyargcount
                or seed.__code__.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)):
            logger.error('A deferred static initializer must be an inline zero-argument lambda',
                         node=node, exc_type=TypeError)
        names = definition_names(seed, syntax=node.body)
        namespace = get_all_accessible_symbols(seed, include_builtins=True, names=names)
        from ..build.deps import get_dependency_tracker
        get_dependency_tracker().record_type_layout_deps_from_globals(
            visitor.current_group_key, namespace)
        expression_visitor = _static_initializer_visitor(
            visitor.compiler, visitor.current_group_key, namespace)
        value = expression_visitor._visit_expression_path(node.body, rvalue=True)
        if value is None:
            logger.error(f"Static storage '{name}' requires a compile-time constant initializer",
                         node=node.body, exc_type=TypeError)
        result = coerce_static_initializer(expression_visitor, pc_type, value,
                                           name, node.body)
        if any(block.pcir for block in expression_visitor.builder.cfg.blocks.values()):
            logger.error(f"Static storage '{name}' requires a compile-time constant initializer",
                         node=node.body, exc_type=TypeError)
        return result


def _record_storage_layout(typ, seen, records):
    from ..type_graph import _shape
    from ..type_resolver import TypeResolver

    pending = [typ]
    while pending:
        typ = TypeResolver.resolve_type(pending.pop())
        if typ in seen:
            continue
        seen.add(typ)
        if getattr(typ, '_field_types', None) is not None:
            records[id(typ)] = getattr(typ, '_python_class', None) or typ
        _, children = _shape(typ)
        pending.extend(child for child, stored in children if stored)


def emit_static_definitions(compiler, group_key, owners):
    """Lower static definitions without introducing a native function."""
    from ..ir_helpers import is_const, is_thread_local
    from ..type_descriptor import validate_static_storage
    from ..type_graph import require_prepared_type
    from ..build.deps import get_dependency_tracker

    visitor = _static_initializer_visitor(compiler, group_key)
    emitted = set()
    seen_types, records = set(), {}
    for owner, owner_node in owners:
        require_prepared_type(owner, owner_node)
        for name in owner._static_members:
            declaration = owner._static_field_nodes.get(name)
            node = declaration or owner_node
            with logger.at_node(node, source=owner._static_initializer_source):
                pc_type, spec, _ = lookup_class_static(owner, name)
                linkage = spec.linkage if spec.linkage is not None else owner._static_linkage
                zero_initialized = spec.initializer is _NO_INIT
                validate_static_storage(pc_type, node, zero_initialized=zero_initialized)
                _record_storage_layout(pc_type, seen_types, records)
                if linkage == 'common' and (
                        not zero_initialized or spec.section is not None or is_thread_local(pc_type)):
                    logger.error('Common storage requires a zero initializer, no section and no TLS',
                                 node=node, exc_type=ValueError)
                value = _native_member_address(visitor, owner, name, pc_type, spec, node)
                if value.initializer is not None:
                    logger.error(f'Duplicate definition of static member symbol {value.name!r}',
                                 node=node, exc_type=ValueError)
                value.linkage = '' if linkage == 'external' else linkage
                if spec.alignment is not None:
                    value.align = spec.alignment
                if spec.section is not None:
                    value.section = spec.section
                initializer = declaration.value if declaration is not None else None
                value.initializer = _native_static_initializer(
                    visitor, name, pc_type, value.value_type, spec.initializer, initializer or node)
                value.global_constant = linkage != 'common' and is_const(pc_type)
                if spec.used:
                    _retain_static_definition(value)
                emitted.add(value.name)
    get_dependency_tracker().record_type_layout_deps_from_globals(group_key, records)
    return emitted


def has_deferred_storage(cls):
    """Only unresolved outer types can change a field's storage class."""
    from ..type_resolver import TypeResolver
    return any(TypeResolver.is_reference(field)
               for _, _, field, _ in cls._type_declaration)


def split_static_members(cls, parsed_field_types: List[Any], namespace=None,
                         *, field_nodes=None) -> List[Any]:
    """Project declarations into static and instance storage.

    Keep the original declarations: a deferred alias can change storage kind
    before compilation fixes its binding. Repeated projections must neither
    lose fields nor retain obsolete static members.
    """
    from ..ir_helpers import is_static, is_thread_local
    from ..type_resolver import TypeResolver

    if '_type_declaration' not in cls.__dict__:
        # Source nodes preserve the original spelling. A dynamically created
        # class has no source spelling beyond the keys it was given.
        field_nodes = field_nodes or {}
        cls._type_declaration = tuple(
            (field_nodes[name].target.id if name in field_nodes else name, annotation, pc_type,
             cls.__dict__.get(name, _NO_INIT))
            for (name, annotation), pc_type in
            zip(cls._struct_fields, parsed_field_types))
    members = {}
    keep_fields = []
    keep_types = []
    for fname, raw_annotation, ftype, init_value in cls._type_declaration:
        resolved = TypeResolver.resolve_type(ftype, namespace, required=False)
        if resolved is not None and (is_static(resolved) or is_thread_local(resolved)):
            members[fname] = (ftype, _member_spec(init_value))
        else:
            keep_fields.append((fname, raw_annotation))
            keep_types.append(ftype)

    cls._static_members = members
    cls._struct_fields = keep_fields
    return keep_types


def lookup_class_static(decorated_cls, attr_name: str) -> Optional[Tuple[Any, _StaticMember, Any]]:
    """Return ``(pc_type, member_spec, owner_cls)`` for a class static, or None.

    Mirrors :func:`lookup_class_method`: the class-level ``handle_attribute``
    is a classmethod bound to the unified type (e.g. a ``StructType``
    subclass) rather than to the user's Python class, so ``_python_class``
    is checked as a transparent redirect.
    """
    from ..type_graph import prepare_type
    from ..type_resolver import TypeResolver
    prepare_type(decorated_cls)
    candidates = [decorated_cls]
    py_cls = getattr(decorated_cls, '_python_class', None)
    if py_cls is not None and py_cls is not decorated_cls:
        candidates.append(py_cls)

    for candidate in candidates:
        members = getattr(candidate, '_static_members', None)
        if members and attr_name in members:
            pc_type, spec = members[attr_name]
            return TypeResolver.resolve_type(pc_type, freeze=True), spec, candidate
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
    from ..valueref import wrap_value
    from ..ir_helpers import declare_native_global, is_thread_local

    found = lookup_class_static(decorated_cls, attr_name)
    if found is None:
        logger.error(
            f"'{decorated_cls.get_name()}' has no class static member '{attr_name}'",
            node=node, exc_type=AttributeError,
        )
    pc_type, spec, owner_cls = found
    owner_group = getattr(owner_cls, '_static_group_key', None)
    if owner_group is not None:
        if visitor.current_group_key != owner_group:
            linkage = spec.linkage if spec.linkage is not None else owner_cls._static_linkage
            if linkage == 'internal':
                logger.error(f'Internal static member {attr_name!r} belongs to another object',
                             node=node, exc_type=ValueError)
            if visitor.current_group_key is not None:
                from ..build.deps import get_dependency_tracker
                get_dependency_tracker().record_group_dependency(
                    visitor.current_group_key, owner_group, 'data_ref')
        value = _native_member_address(visitor, owner_cls, attr_name, pc_type, spec, node)
        return wrap_value(value, kind='address', type_hint=pc_type, address=value)

    from ..type_descriptor import validate_static_storage
    is_decl = getattr(owner_cls, '_static_is_decl', False)
    validate_static_storage(
        pc_type, node, zero_initialized=not is_decl and spec.initializer is _NO_INIT,
    )

    global_name = _static_symbol(owner_cls, attr_name)

    llvm_type = pc_type.get_llvm_type(visitor.module.context)
    global_var, created = declare_native_global(visitor.module, llvm_type, global_name,
                                                is_thread_local(pc_type), node)
    if created:
        linkage = getattr(owner_cls, '_static_linkage', 'weak_odr')
        if not is_decl:
            # A defining global with an initializer already has external
            # linkage; setting linkage='external' emits invalid
            # ``external global T <init>`` IR. Only non-default kinds
            # (internal / weak_odr / ...) are written onto the definition.
            if linkage != 'external':
                global_var.linkage = linkage
            else:
                global_var.linkage = ''
            seed = _fold_static_initializer(visitor, pc_type, llvm_type,
                                            spec.initializer, attr_name, node)
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
    from ..literal_protocol import wrap_literal_result

    if init_value is _NO_INIT:
        return visitor.type_converter.create_zero_constant(llvm_type)

    return coerce_static_initializer(visitor, pc_type, wrap_literal_result(init_value),
                                     attr_name, node)


def coerce_static_initializer(visitor, pc_type, value, name, node):
    """Require a constant value suitable for an LLVM storage initializer."""
    from ..ir_helpers import is_link_time_constant
    from ..valueref import ensure_ir

    seed = visitor.type_converter.try_const_aggregate(value, pc_type, node=node)
    if seed is None:
        seed = ensure_ir(visitor.implicit_coercer.coerce(value, pc_type, node))
    if seed is None or not is_link_time_constant(seed):
        logger.error(
            f"Static storage '{name}' requires compile-time constant initializer",
            node=node,
        )

    return seed
