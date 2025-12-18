"""
Abstract Builder interface for code generation backends.

This defines the interface that all code generation backends must implement.
The interface mirrors llvmlite's ir.IRBuilder API for easy migration.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class AbstractBuilder(ABC):
    """Abstract interface for code generation backends.
    
    This interface abstracts away the code generation backend, allowing:
    1. LLVM IR generation (LLVMBuilder)
    2. Compile-time evaluation without IR (NullBuilder)
    3. Future backends (C, MLIR, GPU, etc.)
    
    The API mirrors llvmlite's ir.IRBuilder for compatibility.
    """
    
    # ========== Arithmetic Operations ==========
    
    @abstractmethod
    def add(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Integer addition."""
        pass
    
    @abstractmethod
    def sub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Integer subtraction."""
        pass
    
    @abstractmethod
    def mul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Integer multiplication."""
        pass
    
    @abstractmethod
    def sdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Signed integer division."""
        pass
    
    @abstractmethod
    def udiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Unsigned integer division."""
        pass
    
    @abstractmethod
    def srem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Signed integer remainder."""
        pass
    
    @abstractmethod
    def urem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Unsigned integer remainder."""
        pass
    
    @abstractmethod
    def fadd(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Floating-point addition."""
        pass
    
    @abstractmethod
    def fsub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Floating-point subtraction."""
        pass
    
    @abstractmethod
    def fmul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Floating-point multiplication."""
        pass
    
    @abstractmethod
    def fdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Floating-point division."""
        pass
    
    # ========== Bitwise Operations ==========
    
    @abstractmethod
    def and_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Bitwise AND."""
        pass
    
    @abstractmethod
    def or_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Bitwise OR."""
        pass
    
    @abstractmethod
    def xor(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Bitwise XOR."""
        pass
    
    @abstractmethod
    def shl(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Shift left."""
        pass
    
    @abstractmethod
    def ashr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Arithmetic shift right (sign-extending)."""
        pass
    
    @abstractmethod
    def lshr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Logical shift right (zero-filling)."""
        pass
    
    # ========== Comparison Operations ==========
    
    @abstractmethod
    def icmp_signed(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Signed integer comparison."""
        pass
    
    @abstractmethod
    def icmp_unsigned(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Unsigned integer comparison."""
        pass
    
    @abstractmethod
    def fcmp_ordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Ordered floating-point comparison."""
        pass
    
    @abstractmethod
    def fcmp_unordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Unordered floating-point comparison."""
        pass
    
    # ========== Memory Operations ==========
    
    @abstractmethod
    def alloca(self, typ: Any, size: Any = None, name: str = "") -> Any:
        """Allocate stack memory."""
        pass
    
    @abstractmethod
    def load(self, ptr: Any, name: str = "", align: Any = None) -> Any:
        """Load value from memory."""
        pass
    
    @abstractmethod
    def store(self, value: Any, ptr: Any, align: Any = None) -> Any:
        """Store value to memory."""
        pass
    
    @abstractmethod
    def gep(self, ptr: Any, indices: List[Any], inbounds: bool = False, name: str = "") -> Any:
        """Get element pointer."""
        pass
    
    # ========== Type Conversion ==========
    
    @abstractmethod
    def trunc(self, value: Any, typ: Any, name: str = "") -> Any:
        """Truncate integer to smaller type."""
        pass
    
    @abstractmethod
    def zext(self, value: Any, typ: Any, name: str = "") -> Any:
        """Zero-extend integer to larger type."""
        pass
    
    @abstractmethod
    def sext(self, value: Any, typ: Any, name: str = "") -> Any:
        """Sign-extend integer to larger type."""
        pass
    
    @abstractmethod
    def fptrunc(self, value: Any, typ: Any, name: str = "") -> Any:
        """Truncate floating-point to smaller type."""
        pass
    
    @abstractmethod
    def fpext(self, value: Any, typ: Any, name: str = "") -> Any:
        """Extend floating-point to larger type."""
        pass
    
    @abstractmethod
    def fptosi(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert floating-point to signed integer."""
        pass
    
    @abstractmethod
    def fptoui(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert floating-point to unsigned integer."""
        pass
    
    @abstractmethod
    def sitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert signed integer to floating-point."""
        pass
    
    @abstractmethod
    def uitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert unsigned integer to floating-point."""
        pass
    
    @abstractmethod
    def ptrtoint(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert pointer to integer."""
        pass
    
    @abstractmethod
    def inttoptr(self, value: Any, typ: Any, name: str = "") -> Any:
        """Convert integer to pointer."""
        pass
    
    @abstractmethod
    def bitcast(self, value: Any, typ: Any, name: str = "") -> Any:
        """Bitcast between types of same size."""
        pass
    
    # ========== Control Flow ==========
    
    @abstractmethod
    def branch(self, target: Any) -> Any:
        """Unconditional branch."""
        pass
    
    @abstractmethod
    def cbranch(self, cond: Any, truebr: Any, falsebr: Any) -> Any:
        """Conditional branch."""
        pass
    
    @abstractmethod
    def switch(self, value: Any, default: Any) -> Any:
        """Switch statement."""
        pass
    
    @abstractmethod
    def ret(self, value: Any = None) -> Any:
        """Return from function."""
        pass
    
    @abstractmethod
    def ret_void(self) -> Any:
        """Return void from function."""
        pass
    
    @abstractmethod
    def unreachable(self) -> Any:
        """Mark code as unreachable."""
        pass
    
    # ========== PHI and Select ==========
    
    @abstractmethod
    def phi(self, typ: Any, name: str = "") -> Any:
        """Create PHI node."""
        pass
    
    @abstractmethod
    def select(self, cond: Any, lhs: Any, rhs: Any, name: str = "") -> Any:
        """Select between two values based on condition."""
        pass
    
    # ========== Function Calls ==========
    
    @abstractmethod
    def call(self, fn: Any, args: List[Any], name: str = "",
             return_type_hint: Any = None) -> Any:
        """Call a function.
        
        Args:
            fn: Function to call
            args: Arguments to pass
            name: Optional name for the result
            return_type_hint: Optional PC type hint for return value
                             (used by LLVM backend for struct ABI unpacking)
        """
        pass
    
    # ========== Aggregate Operations ==========
    
    @abstractmethod
    def extract_value(self, agg: Any, idx: Any, name: str = "") -> Any:
        """Extract value from aggregate."""
        pass
    
    @abstractmethod
    def insert_value(self, agg: Any, value: Any, idx: Any, name: str = "") -> Any:
        """Insert value into aggregate."""
        pass
    
    # ========== Block Management ==========
    
    @property
    @abstractmethod
    def block(self) -> Any:
        """Get current basic block."""
        pass
    
    @abstractmethod
    def position_at_start(self, block: Any) -> None:
        """Position builder at start of block."""
        pass
    
    @abstractmethod
    def position_at_end(self, block: Any) -> None:
        """Position builder at end of block."""
        pass
    
    @abstractmethod
    def position_before(self, instr: Any) -> None:
        """Position builder before instruction."""
        pass
    
    @abstractmethod
    def position_after(self, instr: Any) -> None:
        """Position builder after instruction."""
        pass
    
    # ========== Misc ==========
    
    @abstractmethod
    def neg(self, value: Any, name: str = "") -> Any:
        """Negate integer value."""
        pass
    
    @abstractmethod
    def fneg(self, value: Any, name: str = "") -> Any:
        """Negate floating-point value."""
        pass
    
    @abstractmethod
    def not_(self, value: Any, name: str = "") -> Any:
        """Bitwise NOT."""
        pass
