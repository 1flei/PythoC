"""
Base classes for ABI handling.

Defines the interface for ABI coercion of struct types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from llvmlite import ir


class PassingKind(Enum):
    """How a type should be passed/returned."""
    DIRECT = auto()      # Pass directly (no coercion needed)
    COERCE = auto()      # Coerce to different type
    INDIRECT = auto()    # Pass via pointer (sret for returns)
    EXPAND = auto()      # Expand struct fields as separate args


@dataclass
class CoercedType:
    """Result of ABI coercion for a type.
    
    Attributes:
        kind: How the type should be passed
        coerced_type: The LLVM type to use (for COERCE kind)
        original_type: The original LLVM type
        is_return: Whether this is for a return value
    """
    kind: PassingKind
    coerced_type: Optional[ir.Type] = None
    original_type: Optional[ir.Type] = None
    is_return: bool = False
    
    @property
    def needs_coercion(self) -> bool:
        """Check if coercion is needed."""
        return self.kind in (PassingKind.COERCE, PassingKind.INDIRECT)
    
    @property
    def is_indirect(self) -> bool:
        """Check if indirect passing (sret) is needed."""
        return self.kind == PassingKind.INDIRECT

    @property
    def abi_type(self) -> ir.Type:
        if self.is_indirect:
            return ir.VoidType() if self.is_return else self.original_type.as_pointer()
        if self.needs_coercion:
            return self.coerced_type
        return self.original_type


@dataclass(frozen=True)
class FunctionABI:
    arguments: tuple[CoercedType, ...]
    result: CoercedType

    def function_type(self, var_arg=False):
        params = [argument.abi_type for argument in self.arguments]
        if self.result.is_indirect:
            params.insert(0, self.result.original_type.as_pointer())
        return ir.FunctionType(self.result.abi_type, params, var_arg=var_arg)


class ABIInfo(ABC):
    """Abstract base class for ABI information.
    
    Subclasses implement target-specific ABI rules for:
    - Struct return type coercion
    - Struct argument passing
    - Alignment requirements
    """
    
    def classify_return_type(self, pc_type, module_context) -> CoercedType:
        """Lower a PythoC return type to its C ABI representation."""
        return self._classify_type(pc_type, module_context, is_return=True)

    def classify_argument_type(self, pc_type, module_context) -> CoercedType:
        """Lower a PythoC parameter type to its C ABI representation."""
        return self._classify_type(pc_type, module_context, is_return=False)

    def _classify_type(self, pc_type, module_context, *, is_return):
        from ...logger import logger
        from ...type_resolver import TypeResolver

        pc_type = TypeResolver.resolve_type(pc_type, freeze=True)
        if not hasattr(pc_type, 'get_llvm_type'):
            logger.error(f'Expected a PythoC type in the C ABI signature, got {pc_type!r}',
                         node=logger.current_location(), exc_type=TypeError)
        llvm_type = pc_type.get_llvm_type(module_context)
        return self._classify(pc_type, llvm_type, module_context, is_return=is_return)

    @abstractmethod
    def _classify(self, pc_type, llvm_type, module_context, *, is_return) -> CoercedType:
        """Classify a PythoC type and its derived LLVM storage."""

    def classify_signature(self, param_types, return_type, module_context):
        result = self.classify_return_type(return_type, module_context)
        args = (self.classify_argument_type(pc_type, module_context)
                for pc_type in param_types)
        return FunctionABI(tuple(self._allocate_argument_registers(args, result)), result)

    def _allocate_argument_registers(self, arguments, result):
        return arguments
    
    def get_type_size(self, llvm_type: ir.Type) -> int:
        """Get the target allocation size of an LLVM ABI type."""
        if isinstance(llvm_type, ir.VoidType):
            return 0
        from ...target_layout import get_target_layout
        return get_target_layout().primitive(llvm_type).allocation_size
    
    def get_type_alignment(self, llvm_type: ir.Type) -> int:
        """Get the target ABI alignment of an LLVM ABI type."""
        if isinstance(llvm_type, ir.VoidType):
            return 1
        from ...target_layout import get_target_layout
        return get_target_layout().primitive(llvm_type).alignment
    
    def is_struct_type(self, llvm_type: ir.Type) -> bool:
        """Check if type is a struct type."""
        return isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType))
    
    def is_aggregate_type(self, llvm_type: ir.Type) -> bool:
        """Check if type is an aggregate type (struct or array).
        
        Aggregate types may need ABI coercion for passing/returning.
        """
        return isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType, ir.ArrayType))
    
    def _abi_fields(self, pc_type, llvm_type, module_context):
        """Yield nonempty members and their absolute offsets, including union alternatives."""
        # An anonymous union can contain named records even when its storage
        # has no context. A standalone type query needs a context for them.
        context = module_context if module_context is not None else ir.Context()
        return self._iter_abi_fields(pc_type, llvm_type, context)

    def _iter_abi_fields(self, pc_type, llvm_type, context, offset=0):
        from ...schema_protocol import get_field_layout_types
        from ...target_layout import uses_bit_storage
        from ...type_converter import get_base_type
        from ...type_descriptor import _array_item_type

        size = self.get_type_size(llvm_type)
        if not size:
            return
        yield offset, llvm_type, size
        if not self.is_aggregate_type(llvm_type):
            return
        pc_type = get_base_type(pc_type)
        if uses_bit_storage(pc_type):
            # Alignment carriers have no vector ABI semantics.
            for index in range(size):
                yield offset + index, ir.IntType(8), 1
        elif getattr(pc_type, 'dimensions', None) is not None:
            element = llvm_type.element
            stride = self.get_type_size(element)
            item_type = _array_item_type(pc_type)
            for index in range(llvm_type.count):
                yield from self._iter_abi_fields(item_type, element, context,
                                                 offset + stride * index)
        else:
            fields = get_field_layout_types(pc_type)
            for field, field_offset in zip(fields, pc_type.get_layout().field_offsets):
                yield from self._iter_abi_fields(field, field.get_llvm_type(context),
                                                 context, offset + field_offset)

    def uses_byval_for_indirect_args(self) -> bool:
        """Check if this ABI uses byval attribute for indirect aggregate arguments.
        
        x86-64: Uses byval attribute on pointer parameters
        ARM64: Does NOT use byval - just passes pointer directly
        
        Returns:
            True if byval attribute should be added to indirect arg pointers
        """
        return True  # Default for x86-64
