"""
Calls mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..logger import logger


class CallsMixin:
    """Mixin containing calls-related visitor methods"""
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls with unified call protocol.

        The visitor evaluates the callee expression and arguments, then hands the
        resulting `ValueRef` objects to `ValueRefDispatcher.handle_call()` for
        protocol resolution, rvalue materialization, and linear-transfer policy.

        Keyword arguments (``f(key=value)``) are resolved here by looking up
        parameter names from the callee's type metadata and reordering them
        into the correct positional slots before entering the dispatch layer.
        This keeps all downstream ``handle_call`` implementations unchanged.

        Excess positional arguments for ``*args: T`` callees are packed into
        a ``pc_tuple`` carrier, symmetric with ``**kwargs: T`` / ``pc_dict``.
        """
        func_ref = self.visit_expression(node.func)

        # Pre-evaluate positional arguments (handle *struct unpacking)
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                expanded_args = self._expand_starred_struct(arg.value)
                args.extend(expanded_args)
            else:
                arg_value = self.visit_rvalue_expression(arg)
                args.append(arg_value)

        # Pack excess positional args into pc_tuple for *args: T callees
        args = self._try_varargs_tuple_packing(func_ref, args, node)

        # Merge keyword arguments into positional args list
        if node.keywords:
            args = self._merge_keyword_args(func_ref, args, node)

        return self.value_dispatcher.handle_call(
            func_ref,
            args,
            node,
            name=getattr(node.func, 'id', None),
        )

    def _merge_keyword_args(self, func_ref, positional_args, node):
        """Merge keyword arguments into the positional args list.

        Two modes:
        1. Simple reorder: keyword names match function parameter names
           -> place each keyword arg at the correct positional index.
        2. **kwargs: T packing: keyword names don't match any param name
           -> pack into a pc_dict carrier and pass as the kwargs parameter.
           The callee's parameter init does ``kwargs: T = pc_dict``.
        """
        param_names = self._get_callee_param_names(func_ref, node)
        if param_names is None:
            logger.error(
                "keyword arguments require a callable with named parameters",
                node=node, exc_type=TypeError,
            )

        # Evaluate keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                logger.error(
                    "**kwargs splat in calls is not supported",
                    node=node, exc_type=TypeError,
                )
            if kw.arg in kwargs:
                logger.error(
                    f"duplicate keyword argument: '{kw.arg}'",
                    node=node, exc_type=TypeError,
                )
            kwargs[kw.arg] = self.visit_rvalue_expression(kw.value)

        # Check if any keyword name doesn't match a param name
        unmatched = [name for name in kwargs if name not in param_names]
        if unmatched:
            packed = self._try_kwargs_dict_packing(
                func_ref, param_names, positional_args, kwargs, node,
            )
            if packed is not None:
                return packed
            logger.error(
                f"unexpected keyword argument: '{unmatched[0]}' "
                f"(valid: {', '.join(param_names)})",
                node=node, exc_type=TypeError,
            )

        # Simple reorder: all keywords match param names
        merged = list(positional_args)
        n_positional = len(positional_args)

        for name, val in kwargs.items():
            idx = param_names.index(name)
            if idx < n_positional:
                logger.error(
                    f"argument '{name}' given both as positional (index {idx}) "
                    f"and keyword",
                    node=node, exc_type=TypeError,
                )
            while len(merged) <= idx:
                merged.append(None)
            merged[idx] = val

        for i, v in enumerate(merged):
            if v is None:
                name = param_names[i] if i < len(param_names) else f"<index {i}>"
                logger.error(
                    f"missing required argument: '{name}'",
                    node=node, exc_type=TypeError,
                )

        return merged

    def _try_kwargs_dict_packing(
        self, func_ref, param_names, positional_args, kwargs, node,
    ):
        """Pack all keyword args into a pc_dict for a **kwargs: T parameter.

        The pc_dict carrier preserves key/value pairs at compile time.
        The callee's parameter initialization does ``kwargs: T = pc_dict``,
        which triggers the normal type conversion / assignment path.

        Returns a merged args list if the callee has a kwargs parameter,
        or None if not applicable.
        """
        # Check if callee has a **kwargs: T parameter (last param with a
        # type that could accept a dict-like initializer).
        if not param_names:
            return None

        # Look for metadata indicating a **kwargs parameter
        has_kwargs_param = self._callee_has_kwargs_param(func_ref)
        if not has_kwargs_param:
            return None

        # Build pc_dict carrier from keyword args
        from ..builtin_entities.pc_dict import create_pc_dict_type
        from ..builtin_entities.python_type import PythonType

        entries = []
        for name, val in kwargs.items():
            key_hint = PythonType.wrap(name, is_constant=True)
            key_ref = wrap_value(name, kind='python', type_hint=key_hint)
            entries.append((key_ref, val))

        dict_type = create_pc_dict_type(entries)
        dict_hint = PythonType.wrap(dict_type, is_constant=True)
        dict_ref = wrap_value(dict_type, kind='python', type_hint=dict_hint)

        # Place at the kwargs parameter position (last param)
        kwargs_idx = len(param_names) - 1
        merged = list(positional_args)
        while len(merged) <= kwargs_idx:
            merged.append(None)
        merged[kwargs_idx] = dict_ref

        # Verify no gaps in non-kwargs positions
        for i, v in enumerate(merged):
            if v is None:
                name = param_names[i] if i < len(param_names) else f"<index {i}>"
                logger.error(
                    f"missing required argument: '{name}'",
                    node=node, exc_type=TypeError,
                )

        return merged

    def _try_varargs_tuple_packing(self, func_ref, positional_args, node):
        """Pack excess positional args into a pc_tuple for a *args: T parameter.

        When the callee has ``*args: T``, the func type contains a single
        parameter named ``args`` of type ``T``.  The caller may pass N
        individual positional arguments that correspond to the fields of T.
        This method detects the excess and packs them into a ``pc_tuple``
        carrier so the normal type conversion path (``pc_tuple -> T``)
        handles the rest.

        Returns the (possibly modified) args list.
        """
        from ..builtin_entities.func import func as func_cls

        hint = getattr(func_ref, 'type_hint', None)
        if hint is None:
            return positional_args

        # Determine the number of fixed params from the func type
        param_types = None
        has_varargs = False
        if isinstance(hint, type) and issubclass(hint, func_cls):
            param_types = getattr(hint, 'param_types', None)
            has_varargs = getattr(hint, 'has_varargs', False)

        if not has_varargs or param_types is None:
            # Also check _func_info for PythonType-wrapped callees
            func_info = self._get_func_info(func_ref)
            if func_info:
                has_varargs = getattr(func_info, 'has_varargs', False)
                if has_varargs and param_types is None:
                    # Reconstruct param count from func_info
                    param_types = tuple(
                        func_info.param_type_hints[n]
                        for n in func_info.param_names
                    )

        if not has_varargs or param_types is None:
            return positional_args

        # The varargs param is the one right after normal params.
        # In param_types, the varargs param is at a known position:
        # For func with *args and optionally **kwargs, the layout is:
        #   [normal_params..., args_param, (kwargs_param)]
        has_kwargs = False
        if isinstance(hint, type) and issubclass(hint, func_cls):
            has_kwargs = getattr(hint, 'has_kwargs', False)
        else:
            func_info = self._get_func_info(func_ref)
            if func_info:
                has_kwargs = getattr(func_info, 'has_kwargs', False)

        kwargs_count = 1 if has_kwargs else 0
        varargs_idx = len(param_types) - 1 - kwargs_count
        normal_count = varargs_idx  # number of params before the varargs param

        n_positional = len(positional_args)
        if n_positional <= normal_count:
            # Not enough args to trigger varargs packing
            return positional_args

        # Split: normal args + excess args to pack
        normal_args = positional_args[:normal_count]
        excess_args = positional_args[normal_count:]

        # Build pc_tuple carrier from excess args
        from ..builtin_entities.pc_tuple import create_pc_tuple_type
        from ..builtin_entities.python_type import PythonType

        tuple_type = create_pc_tuple_type(excess_args)
        tuple_hint = PythonType.wrap(tuple_type, is_constant=True)
        tuple_ref = wrap_value(tuple_type, kind='python', type_hint=tuple_hint)

        return list(normal_args) + [tuple_ref]

    @staticmethod
    def _get_func_info(func_ref):
        """Extract FunctionInfo from a func_ref (any source)."""
        from ..builtin_entities.python_type import PythonType

        hint = getattr(func_ref, 'type_hint', None)
        func_info = None
        if isinstance(hint, type) and issubclass(hint, PythonType):
            obj = PythonType.unwrap(hint)
            func_info = getattr(obj, '_func_info', None)
        elif isinstance(hint, PythonType):
            obj = hint._python_object
            func_info = getattr(obj, '_func_info', None)
        if func_info is None:
            val = getattr(func_ref, 'value', None) if func_ref else None
            if val is not None:
                func_info = getattr(val, '_func_info', None)
        return func_info

    @staticmethod
    def _callee_has_kwargs_param(func_ref):
        """Check if callee was declared with **kwargs: T."""
        from ..builtin_entities.func import func as func_cls

        hint = getattr(func_ref, 'type_hint', None)
        if hint is not None and isinstance(hint, type) and issubclass(hint, func_cls):
            if getattr(hint, 'has_kwargs', False):
                return True

        func_info = CallsMixin._get_func_info(func_ref)
        return func_info is not None and getattr(func_info, 'has_kwargs', False)

    @staticmethod
    def _get_callee_param_names(func_ref, node):
        """Extract parameter names from a callee ValueRef.

        Tries multiple sources in priority order:
        1. func type's param_names (for lowered @compile/@extern pointers)
        2. PythonType wrapping a @compile wrapper with _func_info
        3. ExternFunctionWrapper.param_types (list of (name, type) pairs)
        """
        from ..builtin_entities.func import func as func_cls
        from ..builtin_entities.python_type import PythonType

        hint = getattr(func_ref, 'type_hint', None)

        # 1. func[...] type with param_names
        if hint is not None and isinstance(hint, type) and issubclass(hint, func_cls):
            names = getattr(hint, 'param_names', None)
            if names:
                return list(names)

        # 2. PythonType wrapping a compiled wrapper
        if isinstance(hint, type) and issubclass(hint, PythonType):
            obj = PythonType.unwrap(hint)
            func_info = getattr(obj, '_func_info', None)
            if func_info and func_info.param_names:
                return list(func_info.param_names)

        # 3. Try the raw python value
        val = getattr(func_ref, 'value', None) if func_ref else None
        if val is not None:
            func_info = getattr(val, '_func_info', None)
            if func_info and func_info.param_names:
                return list(func_info.param_names)
            # ExternFunctionWrapper stores param_types as [(name, type), ...]
            pt = getattr(val, 'param_types', None)
            if pt and isinstance(pt, list) and pt and isinstance(pt[0], tuple):
                return [name for name, _ in pt]

        return None
    
    def _expand_starred_struct(self, struct_expr):
        """Expand a starred carrier through the shared unpack protocol."""
        from ..literal_protocol import get_unpack_values

        struct_val = self.visit_rvalue_expression(struct_expr)
        return get_unpack_values(self, struct_val, struct_expr)
    

    def _perform_call(self, node: ast.Call, func_callable, param_types, return_type_hint=None, evaluated_args=None, param_pc_types=None):
        """Unified function call handler
        
        Args:
            node: ast.Call node
            func_callable: ir.Function or loaded function pointer
            param_types: List of expected parameter types (LLVM types)
            return_type_hint: Optional PC type hint for return value
            evaluated_args: Optional pre-evaluated arguments (for overloading)
            param_pc_types: Optional list of PC types for parameters (for type conversion)
        
        Returns:
            ValueRef with call result
            
        Note: ABI coercion for struct returns is handled by LLVMBuilder.call().
        """
        # Evaluate arguments (unless already evaluated for overloading)
        if evaluated_args is not None:
            args = evaluated_args
        else:
            args = [self.visit_expression(arg) for arg in node.args]
        
        # Type conversion for arguments using PC type hints
        converted_args = []
        for idx, (arg, expected_type) in enumerate(zip(args, param_types)):
            # Get PC type hint from param_pc_types if provided
            target_pc_type = None
            if param_pc_types and idx < len(param_pc_types):
                target_pc_type = param_pc_types[idx]
            
            if target_pc_type is not None:
                converted = self.implicit_coercer.coerce(arg, target_pc_type, node)
                converted_args.append(ensure_ir(converted))
            else:
                # No PC hint: pass-through only if already matching
                if ensure_ir(arg).type == expected_type:
                    converted_args.append(ensure_ir(arg))
                else:
                    func_name_dbg = getattr(func_callable, 'name', '<unknown>')
                    logger.error(f"Function '{func_name_dbg}' parameter {idx} missing PC type hint; cannot convert",
                                node=node, exc_type=TypeError)
        
        # Debug: check argument count
        func_name = getattr(func_callable, 'name', '<unknown>')
        expected_param_count = len(func_callable.function_type.args)
        actual_arg_count = len(converted_args)
        if expected_param_count != actual_arg_count:
            logger.error(f"Function '{func_name}' expects {expected_param_count} arguments, got {actual_arg_count}",
                        node=node, exc_type=TypeError)
        
        # Return type must be available
        if return_type_hint is None:
            func_name = getattr(func_callable, 'name', '<unknown>')
            logger.error(f"Cannot infer return type for function '{func_name}' - missing type hint",
                        node=node, exc_type=TypeError)
        
        # Make the call - LLVMBuilder.call() handles ABI coercion for struct returns
        logger.debug(f"_perform_call: calling {getattr(func_callable, 'name', func_callable)}, args={len(converted_args)}, return_type_hint={return_type_hint}")
        call_result = self.builder.call(func_callable, converted_args,
                                        return_type_hint=return_type_hint)
        
        # Return with type hint (tracking happens in visit_expression if needed)
        return wrap_value(call_result, kind="value", type_hint=return_type_hint)
