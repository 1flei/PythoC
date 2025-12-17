from .base import BuiltinFunction
from ..valueref import wrap_value, ValueRef
from ..logger import logger
import ast


class refine(BuiltinFunction):
    """refine(value, pred1, pred2, "tag1", "tag2", ...) -> yield refined[T, pred1, pred2, "tag1", "tag2"]
    
    Check all predicates and yield refined type if all satisfied.
    Must be used in for-else loop.
    
    Example:
        for x in refine(5, is_positive):
            # x is refined[i32, is_positive] type
            use(x)
        else:
            # Predicate failed
            handle_error()
        
        for p in refine(ptr, "owned"):
            # p is refined[ptr[T], "owned"] type
            use(p)
        else:
            # never happens (tags are always true)
            pass
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'refine'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle refine(value, pred1, "tag1", ...) call
        
        This function is special - it's a yield function that should be
        handled by the yield inlining system.
        
        Args:
            visitor: AST visitor instance
            func_ref: ValueRef of the callable (refine function)
            args: Pre-evaluated argument ValueRefs (NOT used - we need AST)
            node: ast.Call node
            
        Returns:
            ValueRef with yield inline info for for-loop processing
        """
        func_ast = cls._create_refine_inline_ast(node, visitor)
        
        from .python_type import PythonType
        
        def refine_placeholder(*args, **kwargs):
            raise RuntimeError("refine() must be used in a for loop and will be inlined")
        
        refine_placeholder._is_yield_generated = True
        refine_placeholder.__name__ = 'refine'
        refine_placeholder._original_ast = func_ast
        
        result = wrap_value(refine_placeholder, kind='python', type_hint=PythonType(refine_placeholder))
        result._yield_inline_info = {
            'func_obj': refine_placeholder,
            'original_ast': func_ast,
            'call_node': node,
            'call_args': node.args
        }
        
        return result
    
    @classmethod
    def _create_refine_inline_ast(cls, call_node: ast.Call, visitor) -> ast.FunctionDef:
        """Create AST for inline refine function
        
        Supports two forms:
        1. Single-param: refine(value, pred1, pred2, "tag1", "tag2")
           -> if pred1(value) and pred2(value): yield assume(value, pred1, pred2, "tag1", "tag2")
        
        2. Multi-param: refine(val1, val2, pred)
           -> if pred(val1, val2): yield assume(val1, val2, pred)
        
        Args:
            call_node: The refine() call node
            visitor: AST visitor for context
            
        Returns:
            ast.FunctionDef for the inline function
        """
        import copy
        
        if len(call_node.args) < 2:
            logger.error("refine() requires at least 2 arguments", node=call_node, exc_type=TypeError)
        
        user_globals = {}
        if hasattr(visitor, 'ctx') and hasattr(visitor.ctx, 'user_globals'):
            user_globals = visitor.ctx.user_globals
        elif hasattr(visitor, 'user_globals'):
            user_globals = visitor.user_globals
        
        last_arg_node = call_node.args[-1]
        is_multi_param_form = False
        
        has_string_tag = any(
            isinstance(call_node.args[i], ast.Constant) and isinstance(call_node.args[i].value, str)
            for i in range(1, len(call_node.args))
        )
        
        if not has_string_tag and isinstance(last_arg_node, ast.Name):
            func_name = last_arg_node.id
            if func_name in user_globals:
                maybe_pred = user_globals[func_name]
                if hasattr(maybe_pred, '_original_func'):
                    maybe_pred = maybe_pred._original_func
                if callable(maybe_pred):
                    import inspect
                    sig = inspect.signature(maybe_pred)
                    params = list(sig.parameters.values())
                    if len(params) > 1 and len(call_node.args) == len(params) + 1:
                        is_multi_param_form = True
        
        if is_multi_param_form:
            value_args = call_node.args[:-1]
            pred_node = call_node.args[-1]
            
            pred_call = ast.Call(
                func=copy.deepcopy(pred_node),
                args=[copy.deepcopy(arg) for arg in value_args],
                keywords=[]
            )
            
            assume_call = ast.Call(
                func=ast.Name(id='assume', ctx=ast.Load()),
                args=[copy.deepcopy(arg) for arg in call_node.args],
                keywords=[]
            )
            
            yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
            
            if_stmt = ast.If(
                test=pred_call,
                body=[yield_stmt],
                orelse=[]
            )
            
            func_def = ast.FunctionDef(
                name='__refine_inline__',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None
                ),
                body=[if_stmt],
                decorator_list=[],
                returns=None,
                lineno=call_node.lineno,
                col_offset=call_node.col_offset
            )
        
        else:
            value_arg = call_node.args[0]
            
            predicate_nodes = []
            all_constraint_nodes = []
            
            for i in range(1, len(call_node.args)):
                arg_node = call_node.args[i]
                all_constraint_nodes.append(arg_node)
                
                if not (isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str)):
                    predicate_nodes.append(arg_node)
            
            if len(predicate_nodes) == 0:
                assume_call = ast.Call(
                    func=ast.Name(id='assume', ctx=ast.Load()),
                    args=[copy.deepcopy(arg) for arg in call_node.args],
                    keywords=[]
                )
                
                yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
                
                func_def = ast.FunctionDef(
                    name='__refine_inline__',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None
                    ),
                    body=[yield_stmt],
                    decorator_list=[],
                    returns=None,
                    lineno=call_node.lineno,
                    col_offset=call_node.col_offset
                )
            else:
                predicate_calls = []
                for pred_node in predicate_nodes:
                    pred_call = ast.Call(
                        func=copy.deepcopy(pred_node),
                        args=[copy.deepcopy(value_arg)],
                        keywords=[]
                    )
                    predicate_calls.append(pred_call)
                
                if len(predicate_calls) == 1:
                    combined_test = predicate_calls[0]
                else:
                    combined_test = ast.BoolOp(
                        op=ast.And(),
                        values=predicate_calls
                    )
                
                assume_call = ast.Call(
                    func=ast.Name(id='assume', ctx=ast.Load()),
                    args=[copy.deepcopy(arg) for arg in call_node.args],
                    keywords=[]
                )
                
                yield_stmt = ast.Expr(value=ast.Yield(value=assume_call))
                
                if_stmt = ast.If(
                    test=combined_test,
                    body=[yield_stmt],
                    orelse=[]
                )
                
                func_def = ast.FunctionDef(
                    name='__refine_inline__',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None
                    ),
                    body=[if_stmt],
                    decorator_list=[],
                    returns=None,
                    lineno=call_node.lineno,
                    col_offset=call_node.col_offset
                )
        
        ast.fix_missing_locations(func_def)
        return func_def
