"""
Constexpr Backend for compile-time evaluation.

This backend is used for:
1. Type annotation resolution (TypeResolver)
2. Constant folding
3. Template/generic instantiation
4. Static assertions

It uses NullBuilder which raises CompileTimeOnlyError for any IR-generating
operation, ensuring that only compile-time evaluable expressions are allowed.
"""

from typing import Any, Optional, Dict

from .abstract import AbstractBackend
from ..builder.null_builder import NullBuilder, CompileTimeOnlyError
from ..builder.abstract import AbstractBuilder


class ConstexprBackend(AbstractBackend):
    """Backend for compile-time only evaluation.
    
    This backend does not generate any IR. It's used when we need to
    evaluate expressions at compile time, such as:
    - Type annotations: `def foo(x: ptr[i32]) -> i64`
    - Array sizes: `array[i32, N * 2]`
    - Template parameters
    
    Any attempt to perform IR-generating operations (load, store, call, etc.)
    will raise CompileTimeOnlyError.
    
    Variables in constexpr context are stored as Python values in a dict,
    not as LLVM allocas.
    
    Attributes:
        _builder: NullBuilder that raises on IR operations
        _variables: Dict storing Python values for variables
        user_globals: User-provided global namespace
    """
    
    def __init__(self, user_globals: Dict[str, Any] = None):
        """Initialize constexpr backend.
        
        Args:
            user_globals: Optional user global namespace for name resolution
        """
        self._builder = NullBuilder()
        self._variables: Dict[str, Any] = {}
        self.user_globals = user_globals or {}
    
    def get_builder(self) -> AbstractBuilder:
        """Get the NullBuilder.
        
        The NullBuilder will raise CompileTimeOnlyError for any
        IR-generating operation.
        """
        return self._builder
    
    def get_module_context(self) -> None:
        """No module context in constexpr mode.
        
        Type resolution that requires LLVM context (e.g., identified struct
        types) is not supported in pure constexpr mode.
        """
        return None
    
    def get_module(self) -> None:
        """No module in constexpr mode."""
        return None
    
    def allocate_variable(self, name: str, llvm_type: Any = None, pc_type: Any = None) -> None:
        """Variables in constexpr mode don't need allocation.
        
        Variables are stored as Python values in self._variables.
        
        Args:
            name: Variable name
            llvm_type: Ignored (no LLVM types in constexpr)
            pc_type: PC type hint (stored for type checking)
            
        Returns:
            None (no alloca needed)
        """
        # Just register that this variable exists
        # Actual value will be set via store_variable
        self._variables[name] = None
        return None
    
    def store_variable(self, name: str, value: Any) -> None:
        """Store a Python value for a variable.
        
        Args:
            name: Variable name
            value: Python value to store
        """
        self._variables[name] = value
    
    def load_variable(self, name: str) -> Any:
        """Load a Python value for a variable.
        
        Args:
            name: Variable name
            
        Returns:
            Stored Python value
            
        Raises:
            NameError: If variable not found
        """
        if name not in self._variables:
            raise NameError(f"Variable '{name}' not defined in constexpr context")
        return self._variables[name]
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists.
        
        Args:
            name: Variable name
            
        Returns:
            True if variable exists
        """
        return name in self._variables
    
    def get_global(self, name: str) -> None:
        """No globals in constexpr mode.
        
        Use user_globals for name resolution instead.
        """
        return None
    
    def declare_function(self, name: str, func_type: Any) -> Any:
        """Function declaration not supported in constexpr mode.
        
        Raises:
            CompileTimeOnlyError: Always
        """
        raise CompileTimeOnlyError(
            f"Cannot declare function '{name}' in constexpr context. "
            "Function declarations require LLVM backend."
        )
    
    def create_block(self, name: str = "") -> Any:
        """Block creation not supported in constexpr mode.
        
        Raises:
            CompileTimeOnlyError: Always
        """
        raise CompileTimeOnlyError(
            "Cannot create basic block in constexpr context. "
            "Control flow requires LLVM backend."
        )
    
    def is_constexpr(self) -> bool:
        """This is a constexpr backend."""
        return True
