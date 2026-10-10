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
        llvm_asm("movl $1, ($0)", "r,r,~{memory}", inputs=[addr, val])

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
    llvm_asm(asm_str, constraints, T, *operands)            -> T

All string/type arguments must be compile-time constants.
"~{memory}" is an LLVM memory clobber: the asm may read and write any
memory.  It is not a prefetch hint.
"""
import ast
from llvmlite import ir

from .base import BuiltinFunction
from ..valueref import wrap_value, ensure_ir
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

        Positional args:
            args[0]: assembly string
            args[1]: constraint string
            args[2:]: optional side_effect bool, optional ret type, then operands

        Keyword args:
            side_effect: bool = True
            inputs: list of operands
            ret_type: PythoC type (asm returns a value)
        """
        from .types import void

        if len(args) < 2:
            logger.error(
                "llvm_asm() requires at least 2 arguments: (asm_str, constraints)",
                node=node, exc_type=TypeError,
            )

        asm_str = cls._extract_const_str(args[0], "asm_str", node)
        constraints = cls._extract_const_str(args[1], "constraints", node)

        side_effect = True
        ret_type = None
        positional_operands = []
        for item in args[2:]:
            py = cls._try_get_python_value(item)
            if type(py) is bool:
                side_effect = py
            elif cls._is_pc_type(py):
                ret_type = py
            else:
                positional_operands.append(item)

        kw_inputs = None
        if node.keywords:
            for kw in node.keywords:
                if kw.arg == 'side_effect':
                    se_val = cls._eval_const_keyword(kw.value, visitor)
                    if type(se_val) is not bool:
                        logger.error(
                            "llvm_asm() side_effect= must be a bool constant",
                            node=node, exc_type=TypeError,
                        )
                    side_effect = se_val
                elif kw.arg == 'ret_type':
                    parsed = visitor.type_resolver.parse_annotation(kw.value)
                    if parsed is None:
                        logger.error(
                            "llvm_asm() ret_type= must be a PythoC type",
                            node=node, exc_type=TypeError,
                        )
                    ret_type = parsed
                elif kw.arg == 'inputs':
                    kw_inputs = cls._eval_inputs(kw.value, visitor, node)
                else:
                    logger.error(
                        "llvm_asm() unexpected keyword '{}'".format(kw.arg),
                        node=node, exc_type=TypeError,
                    )

        if kw_inputs is not None and positional_operands:
            logger.error(
                "llvm_asm() operands given both positionally and as inputs=",
                node=node, exc_type=TypeError,
            )
        input_values = kw_inputs if kw_inputs is not None else positional_operands

        if ret_type is not None and ret_type is not void:
            llvm_ret_type = ret_type.get_llvm_type()
        else:
            llvm_ret_type = ir.VoidType()

        llvm_input_types = []
        llvm_input_values = []
        for inp in input_values:
            ir_val = ensure_ir(inp)
            llvm_input_types.append(ir_val.type)
            llvm_input_values.append(ir_val)

        logger.debug(
            "llvm_asm asm={!r} constraints={!r} n_inputs={}".format(
                asm_str, constraints, len(llvm_input_values),
            )
        )

        asm_func_type = ir.FunctionType(llvm_ret_type, llvm_input_types)
        inline_asm = ir.InlineAsm(
            asm_func_type, asm_str, constraints, side_effect=side_effect
        )
        result = visitor.builder.call(inline_asm, llvm_input_values)

        if isinstance(llvm_ret_type, ir.VoidType):
            return wrap_value(None, kind='python', type_hint=void)
        return wrap_value(result, kind='value', type_hint=ret_type)

    @classmethod
    def _is_pc_type(cls, value):
        return isinstance(value, type) and hasattr(value, 'get_llvm_type')

    @classmethod
    def _eval_inputs(cls, value_node, visitor, node):
        if not isinstance(value_node, ast.List):
            logger.error(
                "llvm_asm() inputs= must be a list of operands",
                node=node, exc_type=TypeError,
            )
        return [visitor.visit_rvalue_expression(elt) for elt in value_node.elts]

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
        if isinstance(value_node, ast.Constant):
            return value_node.value
        if isinstance(value_node, ast.NameConstant):
            return value_node.value
        if isinstance(value_node, ast.Name):
            name = value_node.id
            binding = visitor._lookup_python_global_binding(name)
            if binding is not None:
                return binding.value_ref.get_python_value()
        return None
