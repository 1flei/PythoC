"""
llvm_asm: Direct LLVM inline assembly emission from PythoC.

A compiler intrinsic (BuiltinFunction) that emits inline assembly directly
into the LLVM IR output.  This is PythoC's "escape hatch" for platform-specific
instructions that cannot be expressed in the normal type system.

Usage:

    from pythoc import llvm_asm

    # Simple side-effect (no inputs/outputs):
    @compile
    def spin_hint() -> void:
        llvm_asm("pause", "~{memory}")

    # With input operands:
    @compile
    def memory_fence_store(addr: ptr[i32], val: i32) -> void:
        llvm_asm("mov $1, ($0)", "r,r,~{memory}", inputs=[addr, val])

    # With output (returns a value):
    @compile
    def read_cycle_counter() -> u64:
        return llvm_asm("rdtsc; shl $$32, %rdx; or %rdx, %rax",
                        "=A,~{rdx}", ret_type=u64)

Forms:
    llvm_asm(asm_str, constraints)                          -> void
    llvm_asm(asm_str, constraints, side_effect=False)       -> void
    llvm_asm(asm_str, constraints, inputs=[...])            -> void
    llvm_asm(asm_str, constraints, ret_type=T, inputs=[...]) -> T

All string/type arguments must be compile-time constants.

When to use llvm_asm vs external .S files:
    - 1-3 instructions, no complex register manipulation → llvm_asm
    - Whole functions, naked attribute, saves/restores many regs → .S file
"""
import ast
from llvmlite import ir

from .base import BuiltinFunction
from ..valueref import wrap_value, ensure_ir, get_type
from ..logger import logger


class llvm_asm(BuiltinFunction):
    """llvm_asm(asm_str, constraints, ...) -> void | T

    Emit inline assembly directly into the LLVM IR output.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'llvm_asm'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Emit inline asm IR.

        Positional args (compile-time Python values):
            args[0]: assembly string (e.g. "pause", "rdtsc")
            args[1]: constraint string (e.g. "~{memory}", "=r,r")

        Keyword args (from node.keywords, compile-time):
            side_effect: bool = True
            inputs: list of ValueRef = []
            ret_type: PythoC type = void (if set, asm returns a value)
        """
        from .types import void

        # --- Validate positional args ---
        if len(args) < 2:
            logger.error(
                "llvm_asm() requires at least 2 arguments: (asm_str, constraints)",
                node=node, exc_type=TypeError,
            )

        asm_str = cls._extract_const_str(args[0], "asm_str", node)
        constraints = cls._extract_const_str(args[1], "constraints", node)

        # --- Parse keyword args from AST node ---
        side_effect = True
        input_values = []
        ret_type = None

        if node.keywords:
            for kw in node.keywords:
                if kw.arg == 'side_effect':
                    # Evaluate at Python level
                    se_val = cls._eval_const_keyword(kw.value, visitor)
                    side_effect = bool(se_val)
                elif kw.arg == 'inputs':
                    # inputs is a list of already-evaluated ValueRefs
                    # They are passed as positional args beyond [0] and [1]
                    # Actually, handle via extra positional args
                    pass
                elif kw.arg == 'ret_type':
                    ret_type = cls._eval_const_keyword(kw.value, visitor)

        # --- Collect input operands (args beyond the first 2) ---
        if len(args) > 2:
            # Extra positional args are input operands to the asm, except for
            # compile-time controls: bool side_effect or a return type.
            third = args[2]
            third_value = cls._try_get_python_value(third)
            if isinstance(third_value, bool):
                side_effect = third_value
                input_values = list(args[3:])
            elif hasattr(third_value, 'get_llvm_type'):
                ret_type = third_value
                input_values = list(args[3:])
            else:
                input_values = list(args[2:])

        # --- Determine return type ---
        if ret_type is not None and ret_type is not void:
            llvm_ret_type = ret_type.get_llvm_type()
        else:
            llvm_ret_type = ir.VoidType()

        # --- Build input LLVM types ---
        llvm_input_types = []
        llvm_input_values = []
        for inp in input_values:
            ir_val = ensure_ir(inp)
            llvm_input_types.append(ir_val.type)
            llvm_input_values.append(ir_val)

        # --- Emit inline asm ---
        asm_func_type = ir.FunctionType(llvm_ret_type, llvm_input_types)
        inline_asm = ir.InlineAsm(
            asm_func_type, asm_str, constraints, side_effect=side_effect
        )
        result = visitor.builder.call(inline_asm, llvm_input_values)

        # --- Return ---
        if isinstance(llvm_ret_type, ir.VoidType):
            return wrap_value(None, kind='python', type_hint=void)
        else:
            return wrap_value(result, kind='value', type_hint=ret_type)

    @classmethod
    def _extract_const_str(cls, arg, param_name, node) -> str:
        """Extract a compile-time string constant from an argument ValueRef."""
        if hasattr(arg, 'get_python_value'):
            val = cls._try_get_python_value(arg)
            if isinstance(val, str):
                return val

        if hasattr(arg, 'constant') and isinstance(getattr(arg, 'constant', None), str):
            return arg.constant

        if hasattr(arg, 'value') and isinstance(arg.value, str):
            return arg.value

        logger.error(
            f"llvm_asm() argument '{param_name}' must be a compile-time string constant, "
            f"got {type(arg).__name__}",
            node=node, exc_type=TypeError,
        )

    @classmethod
    def _try_get_python_value(cls, arg):
        if hasattr(arg, 'get_python_value') and arg.is_python_value():
            return arg.get_python_value()
        return arg

    @classmethod
    def _eval_const_keyword(cls, value_node, visitor):
        """Evaluate a keyword argument node as a compile-time Python value."""
        # Simple constant nodes
        if isinstance(value_node, ast.Constant):
            return value_node.value
        if isinstance(value_node, ast.NameConstant):  # Python 3.7 compat
            return value_node.value
        # Name reference (e.g., a type like u64)
        if isinstance(value_node, ast.Name):
            name = value_node.id
            if name in visitor.user_globals:
                return visitor.user_globals[name]
        return None
