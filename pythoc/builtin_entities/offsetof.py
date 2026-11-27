from llvmlite import ir
from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
import ast


class offsetof(BuiltinFunction):
    """offsetof(Type, "field") - Get byte offset of a struct field"""

    @classmethod
    def get_name(cls) -> str:
        return 'offsetof'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call) -> ir.Value:
        """Handle offsetof(type, "field_name") call."""
        if len(args) != 2:
            logger.error(f"offsetof() takes exactly 2 arguments ({len(args)} given)",
                        node=node, exc_type=TypeError)

        pc_type = visitor.type_resolver.resolve_type_value(args[0], node)
        field_arg = args[1]
        if not (field_arg.is_python_value()
                and isinstance(field_arg.get_python_value(), str)):
            logger.error("offsetof() second argument must be a constant field name",
                        node=node, exc_type=TypeError)
        field_name = field_arg.get_python_value()

        offset = cls._get_field_offset(pc_type, field_name, node)
        # offsetof() is a compile-time constant; represent it as a Python value
        # (like sizeof()) so Python-level arithmetic on it remains foldable.
        # Mark the preferred PC type as u64 (size_t) so varargs calls such as
        # printf("%zu", offsetof(...)) materialize a real integer value.
        from .python_type import PythonType
        from .types import u64
        python_type = PythonType.wrap(offset, is_constant=True, preferred_pc_type=u64)
        return wrap_value(offset, kind="python", type_hint=python_type)

    @classmethod
    def _get_field_offset(cls, pc_type, field_name: str, node=None) -> int:
        """Get the byte offset of a dot-separated member path."""
        offset = 0
        for member_name in field_name.split("."):
            member_offset, pc_type = cls._get_member_layout(pc_type, member_name, node)
            offset += member_offset
        return offset

    @classmethod
    def _get_member_layout(cls, pc_type, field_name: str, node):
        """Resolve a member's local byte offset and type together."""
        from ..schema_protocol import get_field_layout_owner, iter_field_layout
        from ..target_layout import uses_bit_storage
        from ..type_converter import get_base_type
        from ..type_resolver import TypeResolver
        pc_type = TypeResolver.resolve_type(pc_type, freeze=True)
        record = get_field_layout_owner(get_base_type(pc_type))
        type_name = getattr(pc_type, '__name__', pc_type)
        if record is None:
            logger.error(f"offsetof(): type '{type_name}' is not a struct",
                         node=node, exc_type=TypeError)
        for index, name, field_type in iter_field_layout(record):
            if name != field_name:
                continue
            if uses_bit_storage(record):
                logger.error('offsetof() cannot be applied to a bit field',
                             node=node, exc_type=TypeError)
            return record.get_layout().field_offsets[index], field_type
        logger.error(f"offsetof(): field '{field_name}' not found in type '{type_name}'",
                    node=node, exc_type=AttributeError)
