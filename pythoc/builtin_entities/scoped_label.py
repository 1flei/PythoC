"""
Scoped goto/label intrinsics for structured control flow

with label("name"):    - Define a scoped label
    goto("name")       - Jump to beginning of label scope
    goto_end("name")   - Jump to end of label scope

Key properties:
1. Labels define scopes (via with statement)
2. Visibility rules:
   - goto: Can target self, ancestors, siblings, uncles
   - goto_end: Can ONLY target self and ancestors (must be inside target)
3. Defer execution follows parent_scope_depth model:
   - Both goto and goto_end exit to target's parent depth
   - Execute defers for all scopes being exited

This is a thin facade: validation and argument extraction only.
All business logic lives in scope_manager (label/goto orchestration)
and control_flow_builder (IR lowering, PendingGoto).
"""
import ast
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value
from ..logger import logger


class label(BuiltinFunction):
    """label("name") - Scoped label context manager

    Used with 'with' statement to define a scoped label:

        with label("loop"):
            # code here can use goto("loop") or goto_end("loop")
            pass

    The label creates two IR blocks:
    - begin_block: target for goto (at 'with' level)
    - end_block: target for goto_end (inside body)
    """

    @classmethod
    def get_name(cls) -> str:
        return 'label'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle label("name") call - returns context manager marker."""
        if len(args) != 1:
            logger.error("label() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)

        label_arg = args[0]

        if not label_arg.is_python_value():
            logger.error("label() argument must be a string literal",
                        node=node, exc_type=TypeError)

        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"label() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)

        return wrap_value(('__scoped_label__', label_name), kind='python', type_hint=void)

    @classmethod
    def enter_label_scope(cls, visitor, label_name: str, node: ast.With):
        """Called by visit_With to set up the label scope."""
        return visitor.scope_manager.create_label_scope(
            label_name, visitor._get_cf_builder(), node)

    @classmethod
    def exit_label_scope(cls, visitor, ctx):
        """Called by visit_With to clean up the label scope."""
        visitor.scope_manager.exit_label_scope(ctx, visitor._get_cf_builder())


class goto(BuiltinFunction):
    """goto("name") - Jump to beginning of label scope

    Can target self, ancestors, siblings, and uncles.
    Executes defers for all scopes being exited.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'goto'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle goto("name") call."""
        if len(args) != 1:
            logger.error("goto() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)

        label_arg = args[0]

        if not label_arg.is_python_value():
            logger.error("goto() argument must be a string literal",
                        node=node, exc_type=TypeError)

        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"goto() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)

        cf = visitor._get_cf_builder()

        if cf.is_terminated():
            return wrap_value(None, kind='python', type_hint=void)

        sm = visitor.scope_manager
        ctx = sm.find_label(label_name)

        if ctx is not None:
            sm.resolve_backward_goto(ctx, cf)
        else:
            sm.register_forward_goto(label_name, cf, node, is_goto_end=False)

        return wrap_value(None, kind='python', type_hint=void)


# Backward compatibility alias
goto_begin = goto


class goto_end(BuiltinFunction):
    """goto_end("name") - Jump to end of label scope

    Can ONLY target self and ancestors (must be inside the label).
    Executes defers for all scopes being exited.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'goto_end'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle goto_end("name") call."""
        if len(args) != 1:
            logger.error("goto_end() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)

        label_arg = args[0]

        if not label_arg.is_python_value():
            logger.error("goto_end() argument must be a string literal",
                        node=node, exc_type=TypeError)

        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"goto_end() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)

        cf = visitor._get_cf_builder()

        if cf.is_terminated():
            return wrap_value(None, kind='python', type_hint=void)

        sm = visitor.scope_manager
        ctx = sm.find_label_for_end(label_name)
        if ctx is None:
            logger.error(f"goto_end: label '{label_name}' not visible. "
                        f"goto_end can only target self or ancestors (must be inside the label).",
                        node=node, exc_type=SyntaxError)

        sm.resolve_backward_goto_end(ctx, cf)

        return wrap_value(None, kind='python', type_hint=void)
