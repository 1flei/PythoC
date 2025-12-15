"""
Inline Visitor for executing PC functions inline at call sites

This visitor is specialized for inline execution - it shares the parent
visitor's context but handles control flow differently (e.g., capturing
return values instead of generating ret instructions).
"""

import ast
from typing import Dict, Optional, Any, TYPE_CHECKING
from llvmlite import ir
from .base import LLVMIRVisitor as BaseVisitor
from .expressions import ExpressionsMixin
from .calls import CallsMixin
from .subscripts import SubscriptsMixin
from .statements import StatementsMixin
from .assignments import AssignmentsMixin
from .functions import FunctionsMixin
from .helpers import HelpersMixin
from ..valueref import ValueRef, ensure_ir, wrap_value
from ..context import VariableInfo
from ..logger import logger

if TYPE_CHECKING:
    from .visitor_impl import LLVMIRVisitor


class InlineVisitor(
    ExpressionsMixin,
    CallsMixin,
    SubscriptsMixin,
    StatementsMixin,
    AssignmentsMixin,
    FunctionsMixin,
    HelpersMixin,
    BaseVisitor
):
    """
    Specialized visitor for inline function execution.
    
    Key differences from normal visitor:
    - visit_Return captures value instead of generating ret instruction
    - Executes in caller's scope (shares builder, module, context)
    - Parameters are bound from caller's arguments
    - No function prologue/epilogue
    
    This allows PC functions decorated with @inline to be executed
    inline at call sites, generating IR instructions directly in the
    caller's basic block.
    """
    
    def __init__(self, parent_visitor: 'LLVMIRVisitor', param_bindings: Dict[str, ValueRef]):
        """
        Initialize inline visitor from parent visitor context.
        
        Args:
            parent_visitor: The visitor that's calling the inline function
            param_bindings: Dict mapping parameter names to ValueRefs or Python objects
        """
        # Share all context with parent visitor
        self.module = parent_visitor.module
        self.builder = parent_visitor.builder
        self.ctx = parent_visitor.ctx
        self.current_function = parent_visitor.current_function
        self.type_converter = parent_visitor.type_converter
        self.type_resolver = parent_visitor.type_resolver
        self.loop_stack = parent_visitor.loop_stack
        self.label_counter = parent_visitor.label_counter
        self.func_type_hints = parent_visitor.func_type_hints
        self.struct_types = parent_visitor.struct_types
        self.source_globals = parent_visitor.source_globals
        self.compiler = parent_visitor.compiler
        
        # Inline-specific state
        self.param_bindings = param_bindings
        self.return_value = None
        self.has_returned = False
        
        # For handling early returns in inline functions
        self.inline_merge_block = None
        self.inline_return_blocks = []  # List of (block, value) tuples
    
    def visit_Attribute(self, node: ast.Attribute):
        """
        Handle attribute access with support for Python objects.
        
        For inline methods with Python object parameters (method=True),
        we need to access attributes of Python objects like self.limit.
        """
        # Evaluate the base expression
        result = self.visit_expression(node.value)
        
        # Check if this is a Python object with direct attribute access
        if result.kind == "python" and hasattr(result.value, node.attr):
            # Access Python object attribute directly
            attr_value = getattr(result.value, node.attr)
            logger.debug("Accessing Python object attribute", obj=result.value, attr=node.attr, value=attr_value)
            
            # Determine type hint from attribute annotation if available
            type_hint = None
            if hasattr(result.value.__class__, '__annotations__'):
                annotations = result.value.__class__.__annotations__
                if node.attr in annotations:
                    # Annotation may be a string or a type object
                    annotation = annotations[node.attr]
                    from ..builtin_entities import get_builtin_entity
                    if isinstance(annotation, str):
                        type_hint = get_builtin_entity(annotation)
                    elif isinstance(annotation, type):
                        # If it's already a type, try to get it as builtin entity
                        # or use it directly if it's a BuiltinEntity subclass
                        from ..builtin_entities.base import BuiltinEntity
                        if issubclass(annotation, BuiltinEntity):
                            type_hint = annotation
                        else:
                            # Try to get by name
                            type_hint = get_builtin_entity(annotation.__name__)
            
            # If attr_value is already a ValueRef, return it
            if isinstance(attr_value, ValueRef):
                return attr_value
            
            # Convert Python value to PC value
            if type_hint is None:
                # Infer type from Python value
                from ..builtin_entities import get_builtin_entity
                if isinstance(attr_value, int):
                    type_hint = get_builtin_entity('i32')
                elif isinstance(attr_value, bool):
                    type_hint = get_builtin_entity('bool')
                elif isinstance(attr_value, float):
                    type_hint = get_builtin_entity('f64')
            
            if type_hint:
                # Create LLVM constant
                llvm_type = self.get_llvm_type(type_hint)
                llvm_const = ir.Constant(llvm_type, attr_value)
                return wrap_value(llvm_const, kind="value", type_hint=type_hint)
            else:
                # Keep as Python value
                from ..builtin_entities.python_type import PythonType
                return wrap_value(attr_value, kind="python",
                              type_hint=PythonType.wrap(attr_value, is_constant=True))
        
        # Fall back to parent implementation for normal attribute access
        return super().visit_Attribute(node)
    
    def visit_Return(self, node: ast.Return):
        """
        Inline version: capture return value and branch to merge block.
        
        This is the key difference from normal visitor - we don't generate
        a ret instruction, we branch to a merge block and collect the return
        value using a phi node.
        """
        if not self.builder.block.is_terminated:
            if node.value:
                self.return_value = self.visit_expression(node.value)
            else:
                self.return_value = None
            
            # Record this return point
            current_block = self.builder.block
            self.inline_return_blocks.append((current_block, self.return_value))
            
            # Branch to merge block
            if self.inline_merge_block:
                self.builder.branch(self.inline_merge_block)
            
            self.has_returned = True
    
    def execute_inline(self, func_ast: ast.FunctionDef) -> Optional[ValueRef]:
        """
        Execute function body inline.
        
        Args:
            func_ast: The function AST to execute
        
        Returns:
            ValueRef of the return value, or None if no return
        """
        # Create merge block for collecting return values
        self.inline_merge_block = self.current_function.append_basic_block(
            self.get_next_label("inline_merge")
        )
        
        # Enter new scope for local variables
        with self.ctx.var_registry.scope():
            # Declare parameters in the new scope
            # This allows local variables to shadow parameters if needed
            for param_name, param_value in self.param_bindings.items():
                # Check if param_value is a raw Python object (not ValueRef)
                if not isinstance(param_value, ValueRef):
                    # Raw Python object - store it directly
                    logger.debug("Inline param binding (Python object)", name=param_name, value=param_value)
                    from ..builtin_entities.python_type import PythonType
                    param_info = VariableInfo(
                        name=param_name,
                        value_ref=wrap_value(param_value, kind="python",
                                         type_hint=PythonType.wrap(param_value, is_constant=True)),
                        alloca=None,
                        source="inline_param_python_obj",
                        is_parameter=True
                    )
                    self.ctx.var_registry.declare(param_info, allow_shadow=True)
                    continue
                
                # Create alloca for parameter (so it can be reassigned)
                param_type = param_value.type_hint if hasattr(param_value, 'type_hint') else None
                if param_type is None:
                    logger.error(f"Cannot determine type for parameter {param_name}",
                                node=func_ast, exc_type=ValueError)
                logger.debug("Inline param binding", name=param_name, value=param_value)
                
                if param_value.is_python_value():
                    param_info = VariableInfo(
                        name=param_name,
                        value_ref=param_value,
                        alloca=None,
                        source="inline_param_const",
                        is_parameter=True
                    )
                    self.ctx.var_registry.declare(param_info, allow_shadow=True)
                else:
                    llvm_type = self.get_llvm_type(param_type)
                    param_alloca = self._create_alloca_in_entry(llvm_type, f"{param_name}_addr")
                    
                    # Store parameter value
                    self.builder.store(ensure_ir(param_value), param_alloca)
                    
                    # Create ValueRef with proper wrapper for function pointers
                    from ..builtin_entities.func import func
                    
                    # Check if this is a function pointer parameter
                    if param_type and isinstance(param_type, type) and issubclass(param_type, func):
                        # Store alloca directly, func.handle_call will load it when needed
                        value_ref = wrap_value(
                            param_alloca,
                            kind='address',
                            type_hint=param_type,
                            address=param_alloca
                        )
                    else:
                        value_ref = wrap_value(
                            param_alloca,
                            kind='address',
                            type_hint=param_type,
                            address=param_alloca
                        )
                    
                    # Register parameter variable
                    param_info = VariableInfo(
                        name=param_name,
                        value_ref=value_ref,
                        alloca=param_alloca,
                        source="inline_param",
                        is_parameter=True
                    )
                    self.ctx.var_registry.declare(param_info, allow_shadow=True)
            
            # Execute function body
            for stmt in func_ast.body:
                if self.builder.block.is_terminated:
                    break
                self.visit(stmt)
            
            # If we reached the end without returning, branch to merge
            if not self.builder.block.is_terminated:
                # Implicit return None
                self.inline_return_blocks.append((self.builder.block, None))
                self.builder.branch(self.inline_merge_block)
            
            # Position at merge block
            self.builder.position_at_end(self.inline_merge_block)
            
            # Create phi node if there are multiple return points or return values
            if self.inline_return_blocks:
                # Check if all returns have values
                return_values = [(b, v) for b, v in self.inline_return_blocks]
                if not return_values:
                    return None
                
                target_llvm_type = None
                target_pc_type = None
                
                # Try to get type from IR return values first
                for _, v in return_values:
                    if v and not v.is_python_value():
                        target_llvm_type = ensure_ir(v).type
                        target_pc_type = v.type_hint
                        logger.debug("Got type from IR return", pc_type=target_pc_type, llvm_type=target_llvm_type)
                        break
                
                # If all returns are Python constants, infer type from function signature or first return value
                if target_llvm_type is None:
                    # Try to get type from function return annotation
                    inferred_type = self._infer_return_type_from_ast(func_ast)
                    if inferred_type:
                        target_pc_type = inferred_type
                        target_llvm_type = self.get_llvm_type(inferred_type)
                        logger.debug("Inferred type from function annotation", pc_type=target_pc_type, llvm_type=target_llvm_type)
                    else:
                        # Use type from first return value
                        for _, v in return_values:
                            if v and v.type_hint:
                                target_pc_type = v.type_hint
                                target_llvm_type = self.get_llvm_type(target_pc_type)
                                logger.debug("Got type from first return value", pc_type=target_pc_type, llvm_type=target_llvm_type)
                                break
                
                if target_llvm_type is None:
                    logger.error("Cannot determine return type for inline function",
                                node=func_ast, exc_type=ValueError)
                
                # Create phi node
                phi = self.builder.phi(target_llvm_type)
                logger.debug("Created phi node", type=target_llvm_type)
                
                for block, val in return_values:
                    if val.is_python_value():
                        # Convert to LLVM constant
                        py_value = val.type_hint.get_python_object()
                        llvm_const = ir.Constant(target_llvm_type, py_value)
                        phi.add_incoming(llvm_const, block)
                        logger.debug("Added Python constant to phi", value=py_value, block=block.name)
                    else:
                        # LLVM value
                        val_ir = ensure_ir(val)
                        if val_ir.type != target_llvm_type:
                            # Type conversion
                            saved_block = self.builder.block
                            self.builder.position_before(block.terminator)
                            converted = self.type_converter.convert(val, target_pc_type)
                            converted_ir = ensure_ir(converted)
                            self.builder.position_at_end(saved_block)
                            phi.add_incoming(converted_ir, block)
                            logger.debug("Added converted IR value to phi", block=block.name)
                        else:
                            phi.add_incoming(val_ir, block)
                            logger.debug("Added IR value to phi", block=block.name)
                
                return wrap_value(phi, kind="value", type_hint=target_pc_type)
            
            return None
    
    def _infer_return_type_from_ast(self, func_ast: ast.FunctionDef):
        """Infer function return value type"""
        if func_ast.returns:
            return self.type_resolver.parse_annotation(func_ast.returns)
        return None
    
    def _create_alloca_in_entry(self, llvm_type: ir.Type, name: str = "") -> ir.AllocaInstr:
        """
        Create alloca instruction in function entry block.
        
        This is a helper method to ensure allocas are created in the entry block
        for better optimization (LLVM prefers all allocas at function start).
        """
        # Save current position
        current_block = self.builder.block
        
        # Find entry block
        entry_block = self.current_function.entry_basic_block
        
        # Position at start of entry block
        if entry_block.instructions:
            self.builder.position_before(entry_block.instructions[0])
        else:
            self.builder.position_at_end(entry_block)
        
        # Create alloca
        alloca = self.builder.alloca(llvm_type, name=name)
        
        # Restore position
        self.builder.position_at_end(current_block)
        
        return alloca
