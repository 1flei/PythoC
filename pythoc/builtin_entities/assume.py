from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
import ast


class assume(BuiltinFunction):
    """assume(value, pred1, pred2, "tag1", "tag2", ...) -> refined type
    
    Create a refined type instance without checking the predicates.
    Supports multiple predicates and tags.
    
    Forms:
    1. Single value: assume(value, pred1, pred2, "tag1", ...)
       - Multiple predicates/tags supported
       
    2. Multi-value (auto-detect): assume(val1, val2, predicate)
       - Last arg is N-param predicate, first N args are values
       - Only ONE predicate supported (auto-detected)
       
    3. Multi-value (explicit): assume((val1, val2), pred1, pred2, "tag1", ...)
       - First arg is tuple of values
       - Multiple predicates/tags supported
    
    Example:
        # Single value
        x = assume(5, is_positive)  # refined[i32, is_positive]
        y = assume(10, is_positive, "validated")  # refined[i32, is_positive, "validated"]
        
        # Multi-value auto-detect
        r = assume(10, 20, is_valid_range)  # refined[is_valid_range]
        
        # Multi-value explicit
        r = assume((10, 20), pred1, pred2, "tag")  # refined[pred1, pred2, "tag"]
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'assume'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle assume(value, pred1, pred2, "tag1", ...) call
        
        Args:
            visitor: AST visitor instance
            func_ref: ValueRef of the callable (assume function)
            args: Pre-evaluated argument ValueRefs
            node: ast.Call node
            
        Returns:
            ValueRef containing refined type value
        """
        from .refined import refined
        from ..valueref import ValueRef
        
        if len(args) < 2:
            logger.error("assume() requires at least 2 arguments", node=node, exc_type=TypeError)
        if len(node.args) < 2:
            logger.error("assume() requires at least 2 arguments", node=node, exc_type=TypeError)
        
        last_arg = args[-1]
        is_multi_param_form = False
        
        if isinstance(last_arg, ValueRef):
            if last_arg.kind == 'python' and last_arg.value is not None and callable(last_arg.value):
                predicate = last_arg.value
                import inspect
                try:
                    sig = inspect.signature(predicate)
                    params = list(sig.parameters.values())
                    if len(params) > 1 and len(args) == len(params) + 1:
                        is_multi_param_form = True
                except:
                    pass
            elif last_arg.kind == 'pointer':
                arg_idx = len(args) - 1
                if arg_idx < len(node.args):
                    arg_node = node.args[arg_idx]
                    if isinstance(arg_node, ast.Name):
                        user_globals = {}
                        if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
                            user_globals = visitor.ctx.user_globals
                        elif hasattr(visitor, 'user_globals'):
                            user_globals = visitor.user_globals
                        
                        if arg_node.id in user_globals:
                            obj = user_globals[arg_node.id]
                            if hasattr(obj, '_original_func'):
                                predicate = obj._original_func
                            elif callable(obj):
                                predicate = obj
                            else:
                                predicate = None
                            
                            if predicate:
                                import inspect
                                try:
                                    sig = inspect.signature(predicate)
                                    params = list(sig.parameters.values())
                                    if len(params) > 1 and len(args) == len(params) + 1:
                                        is_multi_param_form = True
                                except:
                                    pass
        
        if is_multi_param_form:
            value_args = args[:-1]
            predicate_arg = args[-1]
            
            if isinstance(predicate_arg, ValueRef):
                if predicate_arg.kind == 'python' and predicate_arg.value is not None:
                    pred_callable = predicate_arg.value
                elif predicate_arg.kind == 'pointer':
                    pred_callable = predicate
                else:
                    logger.error("assume() last argument must be a predicate", node=node, exc_type=TypeError)
            else:
                pred_callable = predicate_arg
            
            refined_type = refined[pred_callable]
            return refined_type.handle_call(visitor, refined_type, value_args, node)
        
        value_arg = args[0]
        
        constraint_args = []
        for arg in args[1:]:
            if isinstance(arg, ValueRef):
                if arg.kind == 'python' and arg.value is not None:
                    constraint_args.append(arg.value)
                elif arg.kind == 'pointer':
                    arg_idx = len(constraint_args) + 1
                    if arg_idx < len(node.args):
                        arg_node = node.args[arg_idx]
                        if isinstance(arg_node, ast.Name):
                            user_globals = {}
                            if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
                                user_globals = visitor.ctx.user_globals
                            elif hasattr(visitor, 'user_globals'):
                                user_globals = visitor.user_globals
                            
                            if arg_node.id in user_globals:
                                obj = user_globals[arg_node.id]
                                if hasattr(obj, '_original_func'):
                                    constraint_args.append(obj._original_func)
                                elif callable(obj):
                                    constraint_args.append(obj)
                                else:
                                    logger.error(f"assume() constraint must be callable or string, got {type(obj)}",
                                                node=node, exc_type=TypeError)
                            else:
                                logger.error(f"assume() constraint '{arg_node.id}' not found",
                                            node=node, exc_type=TypeError)
                        else:
                            logger.error(f"assume() constraint must be a name reference",
                                        node=node, exc_type=TypeError)
                    else:
                        logger.error(f"Cannot resolve constraint argument", node=node, exc_type=TypeError)
                else:
                    logger.error(f"assume() constraint must be Python value or function reference",
                                node=node, exc_type=TypeError)
            else:
                constraint_args.append(arg)
        
        if not isinstance(value_arg, ValueRef):
            logger.error("assume() first argument must be a value", node=node, exc_type=TypeError)
        
        base_type = value_arg.type_hint
        
        from .python_type import PythonType
        needs_type_inference = (base_type is None or isinstance(base_type, PythonType))
        
        if needs_type_inference:
            if len(constraint_args) > 0 and callable(constraint_args[0]):
                import inspect
                try:
                    sig = inspect.signature(constraint_args[0])
                    params = list(sig.parameters.values())
                    if len(params) > 0 and params[0].annotation != inspect.Parameter.empty:
                        base_type = params[0].annotation
                        needs_type_inference = False
                except:
                    pass
            
            if needs_type_inference and value_arg.is_python_value():
                py_value = value_arg.get_python_value()
                if isinstance(py_value, bool):
                    from .types import i32
                    base_type = i32
                elif isinstance(py_value, int):
                    from .types import i32
                    base_type = i32
                elif isinstance(py_value, float):
                    from .types import f64
                    base_type = f64
                else:
                    logger.error(f"Cannot infer PC type for Python value of type {type(py_value)}",
                                node=node, exc_type=TypeError)
        
        if base_type is None or isinstance(base_type, PythonType):
            logger.error("assume() value must have PC type information or provide a predicate for type inference",
                        node=node, exc_type=TypeError)
        
        refined_args = [base_type] + constraint_args
        
        if len(refined_args) == 1:
            refined_type = refined[refined_args[0]]
        else:
            refined_type = refined[tuple(refined_args)]
        
        return refined_type.handle_call(visitor, refined_type, [value_arg], node)
    
    @classmethod
    def _create_refined_type_from_predicate(cls, predicate, param_types, visitor):
        """Create refined type from multi-param predicate"""
        from .refined import RefinedType
        from .struct import create_struct_type
        import inspect
        
        sig = inspect.signature(predicate)
        params = list(sig.parameters.values())
        param_names = [p.name for p in params]
        
        struct_type = create_struct_type(param_types, param_names)
        
        class_name = f"RefinedType_{predicate.__name__}"
        new_refined_type = type(class_name, (RefinedType,), {
            '_base_type': None,
            '_predicates': [predicate],
            '_tags': [],
            '_struct_type': struct_type,
            '_field_types': param_types,
            '_field_names': param_names,
            '_param_types': param_types,
            '_param_names': param_names,
            '_is_refined': True,
            '_is_single_param': False,
        })
        
        return new_refined_type
    
    @classmethod
    def _create_refined_type(cls, base_type, predicates, tags, visitor):
        """Create refined type from base + predicates + tags"""
        from .refined import RefinedType
        
        base_name = base_type.get_name() if hasattr(base_type, 'get_name') else str(base_type)
        pred_names = [p.__name__ for p in predicates]
        tag_names = ['tag_' + t for t in tags]
        all_names = [base_name] + pred_names + tag_names
        # Build class name by sanitizing all special characters
        sanitized_names = []
        for n in all_names:
            s = str(n).replace('[', '_').replace(']', '_').replace(',', '_')
            s = s.replace(' ', '').replace('"', '').replace("'", '')
            sanitized_names.append(s)
        class_name = "RefinedType_" + '_'.join(sanitized_names)
        
        new_refined_type = type(class_name, (RefinedType,), {
            '_base_type': base_type,
            '_predicates': predicates,
            '_tags': tags,
            '_struct_type': None,
            '_field_types': [base_type],
            '_field_names': ['value'],
            '_param_types': [base_type],
            '_param_names': ['value'],
            '_is_refined': True,
            '_is_single_param': True,
        })
        
        return new_refined_type
