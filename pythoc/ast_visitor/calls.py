"""
Calls mixin for LLVMIRVisitor
"""

import ast

from ..logger import logger


class CallsMixin:
    """Mixin containing calls-related visitor methods"""
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls with unified call protocol.

        The visitor evaluates the callee, positional and keyword arguments,
        then binds parameters and packs typed collectors into literal carriers
        before delegating to ``ValueRefDispatcher.handle_call()``.
        """
        return self._visit_call_with_target(node, self.visit_expression(node.func))

    def _visit_call_with_target(self, node, target):
        """Evaluate arguments once, reusing an already resolved callee."""
        from ..call_normalization import (
            CallArguments, normalize_call_args, prepare_call_target,
        )
        func_ref, positional_args = prepare_call_target(target)

        evaluate_argument = getattr(
            func_ref.type_hint, "evaluate_call_argument", None
        )
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                expanded_args = self._expand_starred_struct(arg.value)
                positional_args.extend(expanded_args)
            else:
                value = (evaluate_argument(self, arg)
                         if evaluate_argument is not None
                         else self.visit_rvalue_expression(arg))
                positional_args.append(value)

        keywords = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                logger.error('**kwargs splat in calls is not supported',
                             node=node, exc_type=TypeError)
            if keyword.arg in keywords:
                logger.error(f"duplicate keyword argument: '{keyword.arg}'",
                             node=node, exc_type=TypeError)
            keywords[keyword.arg] = self.visit_rvalue_expression(keyword.value)
        args = normalize_call_args(
            func_ref, CallArguments(positional_args, keywords), node, visitor=self)

        return self.value_dispatcher.handle_call(
            func_ref,
            args,
            node,
            name=getattr(node.func, 'id', None),
        )

    def _expand_starred_struct(self, struct_expr):
        """Expand a starred carrier through the shared unpack protocol."""
        from ..literal_protocol import get_unpack_values

        struct_val = self.visit_rvalue_expression(struct_expr)
        return get_unpack_values(self, struct_val, struct_expr)
