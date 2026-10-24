from llvmlite import ir
from .base import BuiltinFunction, BuiltinEntity, _get_unified_registry
from ..valueref import wrap_value
from ..logger import logger
from .types import u64
from .union import UnionType
import ast


class offsetof(BuiltinFunction):
    """offsetof(Type, "field") - Get byte offset of a struct field"""

    @classmethod
    def get_name(cls) -> str:
        return 'offsetof'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call) -> ir.Value:
        """Handle offsetof(type, "field_name") call."""
        if len(node.args) != 2:
            logger.error(f"offsetof() takes exactly 2 arguments ({len(node.args)} given)",
                        node=node, exc_type=TypeError)

        type_arg = node.args[0]
        field_arg = node.args[1]

        pc_type = visitor.type_resolver.parse_annotation(type_arg)
        if pc_type is None:
            logger.error(f"offsetof() first argument must be a type, got: {ast.dump(type_arg)}",
                        node=node, exc_type=TypeError)

        if not (isinstance(field_arg, ast.Constant) and isinstance(field_arg.value, str)):
            logger.error("offsetof() second argument must be a string literal field name",
                        node=node, exc_type=TypeError)
        field_name = field_arg.value

        offset = cls._get_field_offset(pc_type, field_name)
        # offsetof() is a compile-time constant; represent it as a Python value
        # (like sizeof()) so Python-level arithmetic on it remains foldable.
        # Mark the preferred PC type as u64 (size_t) so varargs calls such as
        # printf("%zu", offsetof(...)) materialize a real integer value.
        from .python_type import PythonType
        from .types import u64
        python_type = PythonType.wrap(offset, is_constant=True, preferred_pc_type=u64)
        return wrap_value(offset, kind="python", type_hint=python_type)

    @classmethod
    def _get_field_offset(cls, pc_type, field_name: str) -> int:
        """Get the byte offset of a dot-separated member path."""
        offset = 0
        for member_name in field_name.split("."):
            member_offset, pc_type = cls._get_member_layout(pc_type, member_name)
            offset += member_offset
        return offset

    @classmethod
    def _get_member_layout(cls, pc_type, field_name: str):
        """Resolve a member's local byte offset and type together."""
        pc_type = getattr(pc_type, '_union_type', pc_type)
        field_names, field_types = cls._get_struct_fields(pc_type)

        if field_name not in field_names:
            logger.error(f"offsetof(): field '{field_name}' not found in type '{cls._type_name(pc_type)}'",
                        node=None, exc_type=AttributeError)

        from ..target_layout import NATURAL_LAYOUT
        if not getattr(pc_type, '_layout_config', NATURAL_LAYOUT).is_natural:
            raise TypeError('offsetof() cannot be applied to a bit field')

        # All fields of a union share offset 0.
        if isinstance(pc_type, type) and issubclass(pc_type, UnionType):
            return 0, field_types[field_names.index(field_name)]

        if hasattr(pc_type, 'get_layout'):
            index = field_names.index(field_name)
            return pc_type.get_layout().field_offsets[index], field_types[index]

        offset = 0
        for name, field_type in zip(field_names, field_types):
            field_alignment = cls._get_type_alignment(field_type)
            offset = cls._align_to(offset, field_alignment)

            if name == field_name:
                return offset, field_type

            field_size = cls._get_type_size(field_type)
            offset += field_size

        logger.error(f"offsetof(): field '{field_name}' not found in type '{cls._type_name(pc_type)}'",
                    node=None, exc_type=AttributeError)

    @classmethod
    def _get_struct_fields(cls, pc_type):
        """Return (field_names, field_types) for a struct-like type."""
        ensure_fields = getattr(pc_type, '_ensure_field_types_resolved', None)
        if callable(ensure_fields):
            ensure_fields()
        if hasattr(pc_type, '_field_names') and hasattr(pc_type, '_field_types'):
            return pc_type._field_names, pc_type._field_types

        # Fallback to registry for types not created by pythoc's struct[...]
        struct_name = cls._type_name(pc_type)
        registry = _get_unified_registry()
        if registry.has_struct(struct_name):
            struct_info = registry.get_struct(struct_name)
            return [name for name, _ in struct_info.fields], [typ for _, typ in struct_info.fields]

        logger.error(f"offsetof(): type '{struct_name}' is not a struct",
                    node=None, exc_type=TypeError)

    @classmethod
    def _type_name(cls, pc_type) -> str:
        if hasattr(pc_type, '__name__'):
            return pc_type.__name__
        return str(pc_type)

    @classmethod
    def _get_type_size(cls, pc_type) -> int:
        """Get size of a type in bytes."""
        if hasattr(pc_type, 'get_size_bytes'):
            return pc_type.get_size_bytes()

        if isinstance(pc_type, type) and issubclass(pc_type, BuiltinEntity):
            if pc_type.can_be_type():
                return pc_type.get_size_bytes()

        registry = _get_unified_registry()
        type_name = cls._type_name(pc_type)
        entity_cls = registry.get_builtin_entity(type_name)
        if entity_cls and entity_cls.can_be_type():
            return entity_cls.get_size_bytes()

        logger.error(f"offsetof(): cannot determine size of type '{type_name}'",
                    node=None, exc_type=TypeError)

    @classmethod
    def _get_type_alignment(cls, pc_type) -> int:
        """Get alignment of a type in bytes."""
        if hasattr(pc_type, 'get_alignment'):
            return pc_type.get_alignment()

        size = cls._get_type_size(pc_type)
        return min(size, 8)

    @classmethod
    def _align_to(cls, offset: int, alignment: int) -> int:
        """Align offset to the specified alignment boundary."""
        return (offset + alignment - 1) & ~(alignment - 1)
