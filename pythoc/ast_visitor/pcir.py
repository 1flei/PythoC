"""
Per-block Captured IR (PCIR) - Virtual registers and deferred instructions.

During the AST visitor walk, ALL builder calls are recorded as PCIR instructions
(not emitted). After the CFG is fully built, a replay phase emits the actual
LLVM IR by walking the PCIR instructions in each block.

Key classes:
- VReg: Virtual register representing a not-yet-emitted LLVM value
- PCIRInst: A single recorded builder call
- infer_result_type: Computes LLVM type of a builder call without executing it
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, List
from llvmlite import ir


_next_vreg_id = 0


def _get_next_vreg_id() -> int:
    global _next_vreg_id
    vid = _next_vreg_id
    _next_vreg_id += 1
    return vid


def reset_vreg_counter():
    """Reset VReg ID counter (call at start of each function compilation)."""
    global _next_vreg_id
    _next_vreg_id = 0


class VReg:
    """Virtual register representing a not-yet-emitted LLVM value.

    VRegs carry type information so the visitor's type-dependent logic
    (coercion, signed vs unsigned, etc.) still works during the build phase
    without actual LLVM values.

    Duck-types as ir.Value: exposes .type, .name, .function_type so that
    outer code (valueref.py, func.py, etc.) can treat VRegs transparently.
    """
    __slots__ = ('id', 'type', '_resolved')

    def __init__(self, ir_type: ir.Type):
        self.id = _get_next_vreg_id()
        self.type = ir_type
        self._resolved = None  # Actual ir.Value (set during replay)

    def resolve(self) -> Any:
        """Get the resolved LLVM value (only valid after replay)."""
        assert self._resolved is not None, f"VReg {self.id} not yet resolved"
        return self._resolved

    def set_resolved(self, value: Any):
        """Set the resolved LLVM value during replay."""
        self._resolved = value

    @property
    def is_resolved(self) -> bool:
        return self._resolved is not None

    @property
    def name(self) -> str:
        """Duck-type as ir.Value: synthetic name based on VReg ID."""
        return f"vreg.{self.id}"

    @property
    def function_type(self):
        """Duck-type as ir.Function: derive function type from pointer type.

        When type is PointerType(FunctionType), returns the FunctionType.
        This mirrors ir.Function.function_type behavior.
        """
        if isinstance(self.type, ir.PointerType) and isinstance(self.type.pointee, ir.FunctionType):
            return self.type.pointee
        raise AttributeError(f"VReg({self.id}) with type {self.type} has no function_type")

    def get_reference(self) -> str:
        """Duck-type as ir.Value: return a reference string."""
        return f"%vreg.{self.id}"

    def __repr__(self):
        return f"VReg({self.id}, {self.type})"


class SentinelBlock:
    """Placeholder for an ir.Block during the PCIR recording phase.

    Stores the CFG block ID and will be mapped to a real ir.Block during replay.
    Also carries attributes that existing code might check (name, is_terminated).
    """
    __slots__ = ('_cfg_block_id', 'name', '_is_terminated')

    def __init__(self, cfg_block_id: int, name: str = ""):
        self._cfg_block_id = cfg_block_id
        self.name = name
        self._is_terminated = False

    @property
    def is_terminated(self) -> bool:
        return self._is_terminated

    def __repr__(self):
        return f"SentinelBlock(cfg={self._cfg_block_id}, name={self.name})"


class VRegPhi:
    """Virtual PHI node that supports add_incoming during PCIR recording.

    Unlike regular VReg, a VRegPhi records incoming edges and replays them.
    """
    __slots__ = ('id', 'type', '_resolved', '_incomings')

    def __init__(self, ir_type: ir.Type):
        self.id = _get_next_vreg_id()
        self.type = ir_type
        self._resolved = None
        self._incomings: List[Tuple[Any, Any]] = []  # (value, block) pairs

    def resolve(self) -> Any:
        assert self._resolved is not None, f"VRegPhi {self.id} not yet resolved"
        return self._resolved

    def set_resolved(self, value: Any):
        self._resolved = value

    @property
    def is_resolved(self) -> bool:
        return self._resolved is not None

    @property
    def name(self) -> str:
        return f"vreg.phi.{self.id}"

    def get_reference(self) -> str:
        return f"%vreg.phi.{self.id}"

    def add_incoming(self, value: Any, block: Any):
        """Record an incoming edge (value, block) for later replay."""
        self._incomings.append((value, block))

    def __repr__(self):
        return f"VRegPhi({self.id}, {self.type}, incomings={len(self._incomings)})"


class VRegSwitch:
    """Virtual switch instruction that supports add_case during PCIR recording."""
    __slots__ = ('id', 'type', '_resolved', '_cases', '_parent_sentinel')

    def __init__(self, parent_sentinel: SentinelBlock):
        self.id = _get_next_vreg_id()
        self.type = ir.VoidType()
        self._resolved = None  # Actual ir.SwitchInstr
        self._cases: List[Tuple[Any, Any]] = []  # (value, block) pairs
        self._parent_sentinel = parent_sentinel

    def resolve(self) -> Any:
        assert self._resolved is not None, f"VRegSwitch {self.id} not yet resolved"
        return self._resolved

    def set_resolved(self, value: Any):
        self._resolved = value

    @property
    def is_resolved(self) -> bool:
        return self._resolved is not None

    @property
    def parent(self):
        """Return the sentinel block (mimics ir.SwitchInstr.parent)."""
        return self._parent_sentinel

    def add_case(self, value: Any, block: Any):
        """Record a case for later replay."""
        self._cases.append((value, block))

    def __repr__(self):
        return f"VRegSwitch({self.id}, cases={len(self._cases)})"


@dataclass
class PCIRInst:
    """A single recorded builder call.

    Attributes:
        op: Builder method name ('add', 'store', 'branch', etc.)
        args: Positional args (may contain VRegs, ir.Type, ir.Constant, etc.)
        kwargs: Keyword args
        result: Output VReg (None for void ops: store, branch, ret, etc.)
    """
    op: str
    args: tuple
    kwargs: dict = field(default_factory=dict)
    result: Any = None  # VReg, VRegPhi, VRegSwitch, or None

    def __repr__(self):
        result_str = f" -> {self.result}" if self.result is not None else ""
        return f"PCIRInst({self.op}({len(self.args)} args){result_str})"


def _get_vreg_type(v: Any) -> Optional[ir.Type]:
    """Get the LLVM type from a VReg or an ir.Value."""
    if isinstance(v, (VReg, VRegPhi, VRegSwitch)):
        return v.type
    if hasattr(v, 'type'):
        return v.type
    return None


def infer_result_type(op: str, args: tuple, kwargs: dict) -> Optional[ir.Type]:
    """Compute the LLVM type of a builder call result without executing it.

    Returns None for void operations (store, branch, ret, etc.).
    """
    # Arithmetic: result type = type of first operand
    if op in ('add', 'sub', 'mul', 'sdiv', 'udiv', 'srem', 'urem',
              'fadd', 'fsub', 'fmul', 'fdiv', 'frem',
              'and_', 'or_', 'xor', 'shl', 'ashr', 'lshr'):
        return _get_vreg_type(args[0])

    # Comparisons: always i1
    if op in ('icmp_signed', 'icmp_unsigned', 'fcmp_ordered', 'fcmp_unordered'):
        return ir.IntType(1)

    # Memory
    if op == 'alloca':
        # alloca(typ, ...) -> ptr to typ
        return ir.PointerType(args[0])

    if op == 'load':
        # load(ptr) -> ptr.type.pointee
        ptr_type = _get_vreg_type(args[0])
        if ptr_type and isinstance(ptr_type, ir.PointerType):
            return ptr_type.pointee
        return None

    if op == 'store':
        return None

    # GEP
    if op == 'gep':
        ptr_type = _get_vreg_type(args[0])
        indices = args[1]
        if ptr_type is None:
            return None
        # Walk the type through each index
        current_type = ptr_type
        for i, idx in enumerate(indices):
            if isinstance(current_type, ir.PointerType):
                current_type = current_type.pointee
            elif isinstance(current_type, ir.ArrayType):
                current_type = current_type.element
            elif isinstance(current_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                # Index must be a constant
                if isinstance(idx, ir.Constant):
                    field_idx = int(idx.constant)
                    if hasattr(current_type, 'elements'):
                        elements = current_type.elements
                    elif hasattr(current_type, 'gep'):
                        elements = current_type.elements
                    else:
                        return ir.PointerType(ir.IntType(8))
                    if field_idx < len(elements):
                        current_type = elements[field_idx]
                    else:
                        return ir.PointerType(ir.IntType(8))
                else:
                    return ir.PointerType(ir.IntType(8))
            else:
                return ir.PointerType(ir.IntType(8))
        return ir.PointerType(current_type)

    # Casts: result type = target type (2nd arg)
    if op in ('trunc', 'zext', 'sext', 'fptrunc', 'fpext',
              'fptosi', 'fptoui', 'sitofp', 'uitofp',
              'bitcast', 'inttoptr', 'ptrtoint'):
        return args[1]  # The target type

    # C ABI varargs
    if op == 'va_start':
        return ir.PointerType(ir.IntType(8))  # returns i8*
    if op == 'va_arg':
        return args[1]  # target_type
    if op == 'va_end':
        return None

    # Call
    if op == 'call':
        # If return_type_hint is provided, use its LLVM type (ABI coercion may
        # change the actual return type from the LLVM function signature)
        return_type_hint = kwargs.get('return_type_hint')
        if return_type_hint is not None and hasattr(return_type_hint, 'get_llvm_type'):
            fn = args[0] if args else None
            module_context = None
            if fn is not None and hasattr(fn, 'module') and fn.module:
                module_context = fn.module.context

            if module_context is not None:
                try:
                    ret = return_type_hint.get_llvm_type(module_context)
                    if isinstance(ret, ir.VoidType):
                        return None
                    return ret
                except (Exception, SystemExit):
                    pass

            try:
                ret = return_type_hint.get_llvm_type(None)
                if isinstance(ret, ir.VoidType):
                    return None
                return ret
            except (Exception, SystemExit):
                pass
            # Fall through to function type inference

        fn = args[0]
        fn_type = _get_vreg_type(fn)
        if fn_type is None:
            return None
        # fn might be Function (type is PointerType(FunctionType))
        # or a function pointer (PointerType(FunctionType))
        if isinstance(fn_type, ir.PointerType) and isinstance(fn_type.pointee, ir.FunctionType):
            ret = fn_type.pointee.return_type
        elif isinstance(fn_type, ir.FunctionType):
            ret = fn_type.return_type
        else:
            # Try .function_type attribute (ir.Function has this)
            if hasattr(fn, 'function_type'):
                ret = fn.function_type.return_type
            else:
                return None
        if isinstance(ret, ir.VoidType):
            return None
        return ret

    # PHI
    if op == 'phi':
        return args[0]  # phi(typ) -> typ

    # Select
    if op == 'select':
        return _get_vreg_type(args[1])  # select(cond, a, b) -> type of a

    # Aggregate ops
    if op == 'extract_value':
        agg_type = _get_vreg_type(args[0])
        idx = args[1]
        if agg_type and hasattr(agg_type, 'elements'):
            if isinstance(idx, (list, tuple)):
                t = agg_type
                for i in idx:
                    if hasattr(t, 'elements'):
                        t = t.elements[i]
                    else:
                        return None
                return t
            else:
                return agg_type.elements[idx]
        return None

    if op == 'insert_value':
        return _get_vreg_type(args[0])  # Same type as aggregate

    # Unary
    if op in ('neg', 'fneg', 'not_'):
        return _get_vreg_type(args[0])

    # Control flow: void
    if op in ('branch', 'cbranch', 'ret', 'ret_void', 'unreachable', 'switch'):
        return None

    # Block management: void
    if op in ('position_at_end', 'position_at_start', 'position_before', 'position_after'):
        return None

    # Unknown op: try to return None (void)
    return None


def resolve_arg(arg: Any, block_map: dict) -> Any:
    """Resolve a single PCIR argument to its real LLVM value for replay.

    Handles:
    - VReg/VRegPhi/VRegSwitch -> resolved LLVM value
    - SentinelBlock -> real ir.Block from block_map
    - ir.Constant, ir.GlobalVariable, ir.Function, ir.Type -> as-is
    - Python primitives (str, int, bool, None) -> as-is
    - list/tuple of args -> recursively resolve
    """
    if isinstance(arg, (VReg, VRegPhi)):
        return arg.resolve()
    if isinstance(arg, VRegSwitch):
        return arg.resolve()
    if isinstance(arg, SentinelBlock):
        real_block = block_map.get(arg._cfg_block_id)
        if real_block is None:
            raise ValueError(f"SentinelBlock {arg._cfg_block_id} not found in block_map")
        return real_block
    if isinstance(arg, (list, tuple)):
        resolved = [resolve_arg(a, block_map) for a in arg]
        return type(arg)(resolved)
    # Everything else (ir.Constant, ir.Type, ir.Function, ir.GlobalVariable, primitives)
    return arg
