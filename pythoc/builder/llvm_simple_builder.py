"""
LLVM IR Builder - wraps llvmlite's ir.IRBuilder.

This is a simple code generation backend that produces LLVM IR.
It is a thin wrapper that delegates all operations to the underlying IRBuilder
without any ABI coercion.

For C-compatible calling conventions, use LLVMCBuilder instead.
"""

from typing import Any, List, Optional
from llvmlite import ir

from .abstract import AbstractBuilder


class LLVMBuilder(AbstractBuilder):
    """LLVM IR code generation backend.
    
    Wraps llvmlite's ir.IRBuilder to implement the AbstractBuilder interface.
    This is a thin wrapper that delegates all operations to the underlying IRBuilder.
    
    For C ABI-compatible code generation, use LLVMCBuilder instead.
    """
    
    def __init__(self, ir_builder: ir.IRBuilder = None):
        """Initialize with an optional ir.IRBuilder.
        
        Args:
            ir_builder: llvmlite IRBuilder instance. Can be set later.
        """
        self._builder = ir_builder
    
    @property
    def ir_builder(self) -> ir.IRBuilder:
        """Get the underlying ir.IRBuilder."""
        return self._builder
    
    @ir_builder.setter
    def ir_builder(self, builder: ir.IRBuilder):
        """Set the underlying ir.IRBuilder."""
        self._builder = builder
    
    # ========== Arithmetic Operations ==========
    
    def add(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.add(lhs, rhs, name=name)
    
    def sub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.sub(lhs, rhs, name=name)
    
    def mul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.mul(lhs, rhs, name=name)
    
    def sdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.sdiv(lhs, rhs, name=name)
    
    def udiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.udiv(lhs, rhs, name=name)
    
    def srem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.srem(lhs, rhs, name=name)
    
    def urem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.urem(lhs, rhs, name=name)
    
    def fadd(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fadd(lhs, rhs, name=name)
    
    def fsub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fsub(lhs, rhs, name=name)
    
    def fmul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fmul(lhs, rhs, name=name)
    
    def fdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fdiv(lhs, rhs, name=name)
    
    # ========== Bitwise Operations ==========
    
    def and_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.and_(lhs, rhs, name=name)
    
    def or_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.or_(lhs, rhs, name=name)
    
    def xor(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.xor(lhs, rhs, name=name)
    
    def shl(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.shl(lhs, rhs, name=name)
    
    def ashr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.ashr(lhs, rhs, name=name)
    
    def lshr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.lshr(lhs, rhs, name=name)
    
    # ========== Comparison Operations ==========
    
    def icmp_signed(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.icmp_signed(op, lhs, rhs, name=name)
    
    def icmp_unsigned(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.icmp_unsigned(op, lhs, rhs, name=name)
    
    def fcmp_ordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fcmp_ordered(op, lhs, rhs, name=name)
    
    def fcmp_unordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fcmp_unordered(op, lhs, rhs, name=name)
    
    # ========== Memory Operations ==========
    
    def alloca(self, typ: Any, size: Any = None, name: str = "") -> Any:
        return self._builder.alloca(typ, size=size, name=name)
    
    def load(self, ptr: Any, name: str = "", align: Any = None) -> Any:
        return self._builder.load(ptr, name=name, align=align)
    
    def store(self, value: Any, ptr: Any, align: Any = None) -> Any:
        return self._builder.store(value, ptr, align=align)
    
    def gep(self, ptr: Any, indices: List[Any], inbounds: bool = False, name: str = "") -> Any:
        return self._builder.gep(ptr, indices, inbounds=inbounds, name=name)
    
    # ========== Type Conversion ==========
    
    def trunc(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.trunc(value, typ, name=name)
    
    def zext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.zext(value, typ, name=name)
    
    def sext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.sext(value, typ, name=name)
    
    def fptrunc(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptrunc(value, typ, name=name)
    
    def fpext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fpext(value, typ, name=name)
    
    def fptosi(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptosi(value, typ, name=name)
    
    def fptoui(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptoui(value, typ, name=name)
    
    def sitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.sitofp(value, typ, name=name)
    
    def uitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.uitofp(value, typ, name=name)
    
    def ptrtoint(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.ptrtoint(value, typ, name=name)
    
    def inttoptr(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.inttoptr(value, typ, name=name)
    
    def bitcast(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.bitcast(value, typ, name=name)
    
    # ========== Control Flow ==========
    
    def branch(self, target: Any) -> Any:
        return self._builder.branch(target)
    
    def cbranch(self, cond: Any, truebr: Any, falsebr: Any) -> Any:
        return self._builder.cbranch(cond, truebr, falsebr)
    
    def switch(self, value: Any, default: Any) -> Any:
        return self._builder.switch(value, default)
    
    def ret(self, value: Any = None) -> Any:
        if value is None:
            return self._builder.ret_void()
        return self._builder.ret(value)
    
    def ret_void(self) -> Any:
        return self._builder.ret_void()
    
    def unreachable(self) -> Any:
        return self._builder.unreachable()
    
    # ========== PHI and Select ==========
    
    def phi(self, typ: Any, name: str = "") -> Any:
        return self._builder.phi(typ, name=name)
    
    def select(self, cond: Any, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.select(cond, lhs, rhs, name=name)
    
    # ========== Function Calls ==========
    
    def call(self, fn: Any, args: List[Any], name: str = "",
             return_type_hint: Any = None, arg_type_hints: List[Any] = None) -> Any:
        """Call a function.
        
        Note: return_type_hint and arg_type_hints are ignored in this simple builder.
        Use LLVMCBuilder for ABI-aware calls.
        """
        return self._builder.call(fn, args, name=name)
    
    # ========== Aggregate Operations ==========
    
    def extract_value(self, agg: Any, idx: Any, name: str = "") -> Any:
        return self._builder.extract_value(agg, idx, name=name)
    
    def insert_value(self, agg: Any, value: Any, idx: Any, name: str = "") -> Any:
        return self._builder.insert_value(agg, value, idx, name=name)
    
    # ========== Block Management ==========
    
    @property
    def block(self) -> Any:
        return self._builder.block
    
    def position_at_start(self, block: Any) -> None:
        self._builder.position_at_start(block)
    
    def position_at_end(self, block: Any) -> None:
        self._builder.position_at_end(block)
    
    def position_before(self, instr: Any) -> None:
        self._builder.position_before(instr)
    
    def position_after(self, instr: Any) -> None:
        self._builder.position_after(instr)
    
    # ========== Misc ==========
    
    def neg(self, value: Any, name: str = "") -> Any:
        return self._builder.neg(value, name=name)
    
    def fneg(self, value: Any, name: str = "") -> Any:
        return self._builder.fneg(value, name=name)
    
    def not_(self, value: Any, name: str = "") -> Any:
        return self._builder.not_(value, name=name)
