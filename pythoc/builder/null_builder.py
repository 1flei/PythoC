"""
Null Builder - for compile-time evaluation without IR generation.

This builder is used during type resolution and other compile-time operations
where no actual IR should be generated. Any attempt to generate IR will raise
a CompileTimeOnlyError.
"""

from typing import Any, List

from .abstract import AbstractBuilder


class CompileTimeOnlyError(Exception):
    """Raised when runtime IR generation is attempted in compile-time context.
    
    This error indicates that an expression in a type annotation or other
    compile-time context requires runtime evaluation, which is not allowed.
    """
    pass


class NullBuilder(AbstractBuilder):
    """No-op builder for compile-time evaluation.
    
    This builder raises CompileTimeOnlyError for any IR-generating operation.
    It's used during type resolution where we want to reuse visit_expression
    but catch any accidental IR generation.
    
    Usage:
        # Save current builder
        original_builder = visitor.builder
        try:
            visitor.builder = NullBuilder()
            value_ref = visitor.visit_expression(annotation)
            # Process value_ref...
        except CompileTimeOnlyError as e:
            raise TypeError(f"Type annotation requires runtime: {e}")
        finally:
            visitor.builder = original_builder
    """
    
    def _raise_error(self, operation: str) -> None:
        """Raise CompileTimeOnlyError for the given operation."""
        raise CompileTimeOnlyError(
            f"Cannot perform '{operation}' during compile-time evaluation. "
            f"This expression requires runtime IR generation."
        )
    
    # ========== Arithmetic Operations ==========
    
    def add(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("add")
    
    def sub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("sub")
    
    def mul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("mul")
    
    def sdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("sdiv")
    
    def udiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("udiv")
    
    def srem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("srem")
    
    def urem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("urem")
    
    def fadd(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fadd")
    
    def fsub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fsub")
    
    def fmul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fmul")
    
    def fdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fdiv")
    
    # ========== Bitwise Operations ==========
    
    def and_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("and")
    
    def or_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("or")
    
    def xor(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("xor")
    
    def shl(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("shl")
    
    def ashr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("ashr")
    
    def lshr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("lshr")
    
    # ========== Comparison Operations ==========
    
    def icmp_signed(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("icmp_signed")
    
    def icmp_unsigned(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("icmp_unsigned")
    
    def fcmp_ordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fcmp_ordered")
    
    def fcmp_unordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("fcmp_unordered")
    
    # ========== Memory Operations ==========
    
    def alloca(self, typ: Any, size: Any = None, name: str = "") -> Any:
        self._raise_error("alloca")
    
    def load(self, ptr: Any, name: str = "", align: Any = None) -> Any:
        self._raise_error("load")
    
    def store(self, value: Any, ptr: Any, align: Any = None) -> Any:
        self._raise_error("store")
    
    def gep(self, ptr: Any, indices: List[Any], inbounds: bool = False, name: str = "") -> Any:
        self._raise_error("gep")
    
    # ========== Type Conversion ==========
    
    def trunc(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("trunc")
    
    def zext(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("zext")
    
    def sext(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("sext")
    
    def fptrunc(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("fptrunc")
    
    def fpext(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("fpext")
    
    def fptosi(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("fptosi")
    
    def fptoui(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("fptoui")
    
    def sitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("sitofp")
    
    def uitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("uitofp")
    
    def ptrtoint(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("ptrtoint")
    
    def inttoptr(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("inttoptr")
    
    def bitcast(self, value: Any, typ: Any, name: str = "") -> Any:
        self._raise_error("bitcast")
    
    # ========== Control Flow ==========
    
    def branch(self, target: Any) -> Any:
        self._raise_error("branch")
    
    def cbranch(self, cond: Any, truebr: Any, falsebr: Any) -> Any:
        self._raise_error("cbranch")
    
    def switch(self, value: Any, default: Any) -> Any:
        self._raise_error("switch")
    
    def ret(self, value: Any = None) -> Any:
        self._raise_error("ret")
    
    def ret_void(self) -> Any:
        self._raise_error("ret_void")
    
    def unreachable(self) -> Any:
        self._raise_error("unreachable")
    
    # ========== PHI and Select ==========
    
    def phi(self, typ: Any, name: str = "") -> Any:
        self._raise_error("phi")
    
    def select(self, cond: Any, lhs: Any, rhs: Any, name: str = "") -> Any:
        self._raise_error("select")
    
    # ========== Function Calls ==========
    
    def call(self, fn: Any, args: List[Any], name: str = "",
             return_type_hint: Any = None) -> Any:
        self._raise_error("call")
    
    # ========== Aggregate Operations ==========
    
    def extract_value(self, agg: Any, idx: Any, name: str = "") -> Any:
        self._raise_error("extract_value")
    
    def insert_value(self, agg: Any, value: Any, idx: Any, name: str = "") -> Any:
        self._raise_error("insert_value")
    
    # ========== Block Management ==========
    # These are allowed in compile-time context (no-op)
    
    @property
    def block(self) -> Any:
        """Return None - no current block in compile-time context."""
        return None
    
    def position_at_start(self, block: Any) -> None:
        """No-op in compile-time context."""
        pass
    
    def position_at_end(self, block: Any) -> None:
        """No-op in compile-time context."""
        pass
    
    def position_before(self, instr: Any) -> None:
        """No-op in compile-time context."""
        pass
    
    def position_after(self, instr: Any) -> None:
        """No-op in compile-time context."""
        pass
    
    # ========== Misc ==========
    
    def neg(self, value: Any, name: str = "") -> Any:
        self._raise_error("neg")
    
    def fneg(self, value: Any, name: str = "") -> Any:
        self._raise_error("fneg")
    
    def not_(self, value: Any, name: str = "") -> Any:
        self._raise_error("not")
