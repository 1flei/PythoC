"""Debug information generation for pythoc.

This module wraps llvmlite's debug metadata APIs to emit DWARF line tables,
subprogram descriptors, and local variable information.  When
`config.debug_info` is enabled, every compiled function gets a `DISubprogram`,
every emitted instruction gets a `DILocation`, and every alloca'd local
variable/parameter gets a `DILocalVariable` + `llvm.dbg.declare`, which LLVM's
object emitter turns into `.debug_info` / `.debug_line` / `.debug_loc`.
"""

import os
from typing import Any, Dict, List, Optional

from llvmlite import ir


# DWARF language constants used by LLVM.
_DW_LANG_C99 = 0xC

# LLVM debug metadata version expected by the installed LLVM.
_DEBUG_INFO_VERSION = 3
_DWARF_VERSION = 4

# DWARF type encodings.
_DW_ATE_boolean = 2
_DW_ATE_signed = 5
_DW_ATE_unsigned = 7
_DW_ATE_float = 4


def _strip_qualifiers(pc_type: Any) -> Any:
    """Strip const/volatile/static qualifiers from a PC type, if applicable."""
    from .ir_helpers import strip_qualifiers
    return strip_qualifiers(pc_type)


def _resolve_type(pc_type: Any) -> Any:
    """Resolve a forward-reference string to a real type, if applicable."""
    if isinstance(pc_type, str):
        from .forward_ref import get_defined_type
        resolved = get_defined_type(pc_type)
        if resolved is not None:
            return resolved
    return pc_type


class DebugInfoBuilder:
    """Builds and caches DWARF metadata for one LLVM module.

    A single instance is tied to one ``ir.Module``.  It creates one
    ``DICompileUnit`` and ``DIFile`` and reuses them for every function
    compiled into that module.
    """

    def __init__(self, module: ir.Module, source_file: str):
        self.module = module
        self.source_file = source_file or '<unknown>'

        abs_path = os.path.abspath(self.source_file)
        self._file = module.add_debug_info(
            'DIFile',
            {
                'filename': os.path.basename(abs_path),
                'directory': os.path.dirname(abs_path),
            },
        )

        self._compile_unit = module.add_debug_info(
            'DICompileUnit',
            {
                'language': _DW_LANG_C99,
                'file': self._file,
                'producer': 'pythoc',
                'isOptimized': False,
                'emissionKind': 1,  # FullDebug
            },
            is_distinct=True,
        )

        # Generic integer type used for subroutine signatures.  We intentionally
        # keep debug signatures simple; accurate pythoc type mapping can be
        # added later without changing the line table.
        self._generic_int_type = module.add_debug_info(
            'DIBasicType',
            {'name': 'int', 'size': 32, 'encoding': _DW_ATE_signed},
        )

        module.add_named_metadata('llvm.dbg.cu', self._compile_unit)

        # Module flags required by LLVM to recognize the debug info version.
        debug_ver = module.add_metadata(
            [
                ir.Constant(ir.IntType(32), 2),
                ir.MetaDataString(module, 'Debug Info Version'),
                ir.Constant(ir.IntType(32), _DEBUG_INFO_VERSION),
            ]
        )
        module.add_named_metadata('llvm.module.flags', debug_ver)

        dwarf_ver = module.add_metadata(
            [
                ir.Constant(ir.IntType(32), 4),
                ir.MetaDataString(module, 'Dwarf Version'),
                ir.Constant(ir.IntType(32), _DWARF_VERSION),
            ]
        )
        module.add_named_metadata('llvm.module.flags', dwarf_ver)

        self._type_cache: Dict[Any, Optional[ir.DIValue]] = {}
        self._dbg_declare: Optional[ir.Function] = None

    def _ensure_dbg_declare(self) -> ir.Function:
        """Return the module-local llvm.dbg.declare intrinsic, creating it once."""
        if self._dbg_declare is None or self._dbg_declare.module is not self.module:
            fnty = ir.FunctionType(
                ir.VoidType(),
                [ir.MetaDataType(), ir.MetaDataType(), ir.MetaDataType()],
            )
            self._dbg_declare = ir.Function(self.module, fnty, 'llvm.dbg.declare')
        return self._dbg_declare

    def get_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Map a pythoc type to a DWARF debug type, or None if unsupported.

        Results are cached per type class so repeated lookups are cheap and
        self-referential types terminate.
        """
        if pc_type is None:
            return None

        pc_type = _resolve_type(pc_type)
        base_type = _strip_qualifiers(pc_type)
        if base_type is None:
            return None

        # Guard against accidental recursion and cache results.
        if base_type in self._type_cache:
            return self._type_cache[base_type]

        # Mark as in-progress with a None sentinel so recursive pointers do not
        # loop forever.  Unsupported types simply get no debug type.
        self._type_cache[base_type] = None
        debug_type = self._build_debug_type(base_type)
        self._type_cache[base_type] = debug_type
        return debug_type

    def _build_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Inner helper without caching; returns None for unsupported types."""
        if getattr(pc_type, '_is_integer', False):
            size = getattr(pc_type, '_size_bytes', 0) * 8
            name = pc_type.get_name() if hasattr(pc_type, 'get_name') else str(pc_type)
            encoding = _DW_ATE_signed if getattr(pc_type, '_is_signed', True) else _DW_ATE_unsigned
            return self.module.add_debug_info(
                'DIBasicType',
                {'name': name, 'size': size, 'encoding': encoding},
            )

        if getattr(pc_type, '_is_bool', False):
            name = pc_type.get_name() if hasattr(pc_type, 'get_name') else 'bool'
            return self.module.add_debug_info(
                'DIBasicType',
                {'name': name, 'size': 8, 'encoding': _DW_ATE_boolean},
            )

        if getattr(pc_type, '_is_float', False):
            size = getattr(pc_type, '_size_bytes', 0) * 8
            name = pc_type.get_name() if hasattr(pc_type, 'get_name') else str(pc_type)
            return self.module.add_debug_info(
                'DIBasicType',
                {'name': name, 'size': size, 'encoding': _DW_ATE_float},
            )

        if getattr(pc_type, '_is_pointer', False):
            return self._build_pointer_debug_type(pc_type)

        if callable(getattr(pc_type, 'is_array', None)) and pc_type.is_array():
            return self._build_array_debug_type(pc_type)

        if callable(getattr(pc_type, 'is_struct_type', None)) and pc_type.is_struct_type():
            return self._build_struct_debug_type(pc_type)

        if callable(getattr(pc_type, 'is_enum_type', None)) and pc_type.is_enum_type():
            return self._build_enum_debug_type(pc_type)

        if self._is_union_type(pc_type):
            return self._build_union_debug_type(pc_type)

        # Other compound types are left unmapped for now.
        return None

    def _is_union_type(self, pc_type: Any) -> bool:
        """Check whether a PC type is a union type."""
        if not isinstance(pc_type, type):
            return False
        from .builtin_entities.union import UnionType
        try:
            return issubclass(pc_type, UnionType)
        except TypeError:
            return False


    def _get_type_size_bytes(self, pc_type: Any) -> int:
        """Return the size of a PC type in bytes, using existing type APIs."""
        pc_type = _resolve_type(pc_type)
        if pc_type is None:
            return 1
        if hasattr(pc_type, 'get_size_bytes'):
            size = pc_type.get_size_bytes()
            if size is not None:
                return size
        if hasattr(pc_type, '_size_bytes'):
            return pc_type._size_bytes
        return 1

    def _get_type_size_bits(self, pc_type: Any) -> int:
        """Return the size of a PC type in bits."""
        return self._get_type_size_bytes(pc_type) * 8

    def _get_type_alignment_bytes(self, pc_type: Any) -> int:
        """Return the ABI alignment of a PC type in bytes.

        Prefer the type's own ``get_alignment`` when available; fall back to
        standard rules (pointers are pointer-sized, arrays inherit element
        alignment, scalar alignment is capped at 8 bytes).
        """
        pc_type = _resolve_type(pc_type)
        if pc_type is None:
            return 1
        if hasattr(pc_type, 'get_alignment'):
            align = pc_type.get_alignment()
            if align is not None:
                return align
        if getattr(pc_type, '_is_pointer', False):
            return 8
        if callable(getattr(pc_type, 'is_array', None)) and pc_type.is_array():
            element_type = getattr(pc_type, 'element_type', None)
            return self._get_type_alignment_bytes(element_type)
        size = self._get_type_size_bytes(pc_type)
        return min(size, 8)

    def _get_type_alignment_bits(self, pc_type: Any) -> int:
        """Return the ABI alignment of a PC type in bits."""
        return self._get_type_alignment_bytes(pc_type) * 8

    def _get_field_types(self, pc_type: Any) -> List[Any]:
        """Resolve and return a compound type's field types."""
        ensure = getattr(pc_type, '_ensure_field_types_resolved', None)
        if callable(ensure):
            ensure()
        field_types = getattr(pc_type, '_field_types', None)
        if field_types is None:
            return []
        return [_resolve_type(ft) for ft in field_types]

    def _get_field_names(self, pc_type: Any) -> List[Optional[str]]:
        """Return a compound type's field names, falling back to indices."""
        field_names = getattr(pc_type, '_field_names', None)
        field_types = self._get_field_types(pc_type)
        result = []
        for i, _ in enumerate(field_types):
            if field_names and i < len(field_names) and field_names[i] is not None:
                result.append(field_names[i])
            else:
                result.append(None)
        return result

    def _build_member_type(
        self,
        name: str,
        pc_type: Any,
        offset_bits: int,
    ) -> ir.DIValue:
        """Build a DWARF member descriptor for a struct/union field."""
        size_bits = self._get_type_size_bits(pc_type)
        field_di = self.get_debug_type(pc_type)
        fields: Dict[str, Any] = {
            'tag': ir.DIToken('DW_TAG_member'),
            'name': name,
            'size': size_bits,
            'offset': offset_bits,
        }
        if field_di is not None:
            fields['baseType'] = field_di
        return self.module.add_debug_info('DIDerivedType', fields)

    def _build_struct_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Build a DWARF structure type with member offsets."""
        field_types = self._get_field_types(pc_type)
        if not field_types:
            return None
        field_names = self._get_field_names(pc_type)

        members: List[ir.DIValue] = []
        offset_bits = 0
        for i, ft in enumerate(field_types):
            align_bits = self._get_type_alignment_bits(ft)
            offset_bits = ((offset_bits + align_bits - 1) // align_bits) * align_bits
            name = field_names[i] if field_names[i] is not None else f'field_{i}'
            members.append(self._build_member_type(name, ft, offset_bits))
            offset_bits += self._get_type_size_bits(ft)

        # Total size comes from the type itself rather than recomputing.
        total_size = self._get_type_size_bits(pc_type)

        elements_md = self.module.add_metadata(members)
        return self.module.add_debug_info(
            'DICompositeType',
            {
                'tag': ir.DIToken('DW_TAG_structure_type'),
                'name': pc_type.get_name() if hasattr(pc_type, 'get_name') else 'struct',
                'size': total_size,
                'elements': elements_md,
            },
        )

    def _build_union_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Build a DWARF union type with all members at offset 0."""
        field_types = self._get_field_types(pc_type)
        if not field_types:
            return None
        field_names = self._get_field_names(pc_type)

        members: List[ir.DIValue] = []
        for i, ft in enumerate(field_types):
            name = field_names[i] if field_names[i] is not None else f'field_{i}'
            members.append(self._build_member_type(name, ft, 0))

        # Total size comes from the type itself rather than recomputing.
        total_size = self._get_type_size_bits(pc_type)

        elements_md = self.module.add_metadata(members)
        return self.module.add_debug_info(
            'DICompositeType',
            {
                'tag': ir.DIToken('DW_TAG_union_type'),
                'name': pc_type.get_name() if hasattr(pc_type, 'get_name') else 'union',
                'size': total_size,
                'elements': elements_md,
            },
        )

    def _build_enum_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Build a DWARF structure type for pythoc enum { tag, payload }."""
        tag_type = getattr(pc_type, '_tag_type', None)
        payload_type = getattr(pc_type, '_union_payload', None)
        if tag_type is None or payload_type is None:
            return None

        tag_type = _resolve_type(tag_type)
        payload_type = _resolve_type(payload_type)
        tag_size = self._get_type_size_bytes(tag_type)
        payload_align = self._get_type_alignment_bytes(payload_type)
        payload_offset = ((tag_size + payload_align - 1) // payload_align) * payload_align * 8
        members: List[ir.DIValue] = [
            self._build_member_type('tag', tag_type, 0),
            self._build_member_type('payload', payload_type, payload_offset),
        ]

        total_size = self._get_type_size_bits(pc_type)
        elements_md = self.module.add_metadata(members)
        return self.module.add_debug_info(
            'DICompositeType',
            {
                'tag': ir.DIToken('DW_TAG_structure_type'),
                'name': pc_type.get_name() if hasattr(pc_type, 'get_name') else 'enum',
                'size': total_size,
                'elements': elements_md,
            },
        )

    def _build_pointer_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Build a DWARF pointer type around an optional pointee debug type."""
        pointee = getattr(pc_type, 'pointee_type', None)
        if isinstance(pointee, str):
            from .forward_ref import get_defined_type
            pointee = get_defined_type(pointee)

        pointee_di = self.get_debug_type(pointee) if pointee is not None else None

        fields: Dict[str, Any] = {
            'tag': ir.DIToken('DW_TAG_pointer_type'),
            'size': 64,
        }
        if pointee_di is not None:
            fields['baseType'] = pointee_di
        return self.module.add_debug_info('DIDerivedType', fields)

    def _build_array_debug_type(self, pc_type: Any) -> Optional[ir.DIValue]:
        """Build a DWARF array type for pythoc array[T, dims...]."""
        element_type = getattr(pc_type, 'element_type', None)
        dimensions = getattr(pc_type, 'dimensions', None)
        if element_type is None or not dimensions:
            return None

        element_di = self.get_debug_type(element_type)

        # Compute element size in bits as the starting size using existing APIs.
        current_size_bits = self._get_type_size_bits(element_type)

        # Build nested array types from innermost dimension outward.
        current_di = element_di
        for dim in reversed(dimensions):
            subrange = self.module.add_debug_info('DISubrange', {'count': dim})
            elements_md = self.module.add_metadata([subrange])
            current_size_bits *= dim
            fields: Dict[str, Any] = {
                'tag': ir.DIToken('DW_TAG_array_type'),
                'size': current_size_bits,
                'elements': elements_md,
            }
            if current_di is not None:
                fields['baseType'] = current_di
            current_di = self.module.add_debug_info('DICompositeType', fields)

        return current_di

    def get_subprogram(
        self,
        display_name: str,
        llvm_function: ir.Function,
        line: int,
    ) -> 'ir.DIValue':
        """Create and attach a DISubprogram for the given function.

        ``display_name`` is the human-readable name shown in debuggers;
        ``llvm_function.name`` is the actual linker symbol (which may be
        mangled).

        The ``DISubroutineType`` is created with placeholder types and should
        be finalized via ``finalize_subprogram_type()`` once the parameter
        ``VariableInfo`` objects are known.
        """
        # Placeholder subroutine type; will be replaced after parameters are
        # registered so the debugger sees accurate signatures.
        types = self.module.add_metadata([None])
        subroutine_type = self.module.add_debug_info(
            'DISubroutineType',
            {'types': types},
        )

        subprogram = self.module.add_debug_info(
            'DISubprogram',
            {
                'name': display_name,
                'linkageName': llvm_function.name,
                'scope': self._file,
                'file': self._file,
                'line': line,
                'type': subroutine_type,
                'isLocal': False,
                'isDefinition': True,
                'scopeLine': line,
                'unit': self._compile_unit,
            },
            is_distinct=True,
        )

        llvm_function.set_metadata('dbg', subprogram)
        return subprogram

    def finalize_subprogram_type(
        self,
        subprogram: 'ir.DIValue',
        return_pc_type: Any,
        param_var_infos: List[Any],
    ) -> None:
        """Replace the subroutine type with one based on actual PC types.

        ``subprogram`` is mutated in place.  This must be called before the
        function body is visited so that subsequent ``DILocation`` nodes refer
        to a subprogram whose signature is already accurate.
        """
        type_list: List[Optional[ir.DIValue]] = [
            self.get_debug_type(return_pc_type),
        ]
        for var_info in param_var_infos:
            type_list.append(self.get_debug_type(var_info.type_hint))

        types = self.module.add_metadata(type_list)
        subroutine_type = self.module.add_debug_info(
            'DISubroutineType',
            {'types': types},
        )

        new_operands = tuple(
            (key, subroutine_type if key == 'type' else value)
            for key, value in subprogram.operands
        )
        subprogram.operands = new_operands

    def get_location(
        self,
        subprogram: 'ir.DIValue',
        line: int,
        column: int = 0,
    ) -> 'ir.DIValue':
        """Create a DILocation anchored to the given subprogram."""
        return self.module.add_debug_info(
            'DILocation',
            {
                'line': line,
                'column': column,
                'scope': subprogram,
            },
        )

    def declare_local(
        self,
        subprogram: 'ir.DIValue',
        name: str,
        pc_type: Any,
        line: int,
        column: int = 0,
        is_parameter: bool = False,
        arg_index: Optional[int] = None,
    ) -> 'ir.DIValue':
        """Create a DILocalVariable for a parameter or local variable."""
        debug_type = self.get_debug_type(pc_type)

        # DILocalVariable only accepts a line number, not a column.
        fields: Dict[str, Any] = {
            'name': name,
            'scope': subprogram,
            'file': self._file,
            'line': line,
        }
        if debug_type is not None:
            fields['type'] = debug_type
        if is_parameter and arg_index is not None:
            fields['arg'] = arg_index

        return self.module.add_debug_info('DILocalVariable', fields)

    def emit_local_declare(
        self,
        builder,
        alloca: ir.Value,
        dilocal_var: 'ir.DIValue',
        location: 'ir.DIValue',
    ) -> ir.CallInstr:
        """Emit llvm.dbg.declare for an alloca'd local variable.

        The resulting call is annotated with ``!dbg`` because LLVM requires the
        intrinsic to carry a debug location.
        """
        dbg_declare = self._ensure_dbg_declare()
        expr = self.module.add_debug_info('DIExpression', {})
        call = builder.call(dbg_declare, [ir.MetaDataArgument(alloca), dilocal_var, expr])
        call.set_metadata('dbg', location)
        return call
