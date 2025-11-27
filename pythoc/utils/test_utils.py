"""
Utility functions for the PC compiler
"""

import ast
import os
from typing import List, Dict, Any, Optional, TypeVar, Generic, Type
from llvmlite import binding
from ..logger import logger


def analyze_function(source_code: str, func_name: str) -> Dict[str, Any]:
    """Analyze a Python function and extract metadata"""
    tree = ast.parse(source_code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Extract function information
            info = {
                'name': node.name,
                'parameters': [],
                'return_type': None,
                'has_control_flow': False,
                'operations': set(),
                'line_count': len(node.body)
            }
            
            # Extract parameter information
            for arg in node.args.args:
                param_info = {
                    'name': arg.arg,
                    'type': None
                }
                if arg.annotation:
                    # Convert annotation to string representation
                    param_info['type'] = ast.unparse(arg.annotation)
                info['parameters'].append(param_info)
            
            # Extract return type
            if node.returns:
                # Convert annotation to string representation
                info['return_type'] = ast.unparse(node.returns)
            
            # Analyze function body
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.If):
                    info['has_control_flow'] = True
                elif isinstance(stmt, ast.BinOp):
                    if isinstance(stmt.op, ast.Add):
                        info['operations'].add('add')
                    elif isinstance(stmt.op, ast.Sub):
                        info['operations'].add('sub')
                    elif isinstance(stmt.op, ast.Mult):
                        info['operations'].add('mul')
                    elif isinstance(stmt.op, ast.Div):
                        info['operations'].add('div')
                elif isinstance(stmt, ast.Compare):
                    info['operations'].add('compare')
                elif isinstance(stmt, ast.UnaryOp):
                    if isinstance(stmt.op, ast.USub):
                        info['operations'].add('neg')
            
            info['operations'] = list(info['operations'])
            return info
    
    return {}


def get_llvm_version() -> str:
    """Get the LLVM version"""
    try:
        return binding.llvm_version_info
    except:
        return "Unknown"


def print_module_info(ir_code: str):
    """Print information about an LLVM module"""
    try:
        module = binding.parse_assembly(ir_code)
        print("Module Information:")
        print(f"   Target Triple: {module.triple}")
        print(f"   Data Layout: {module.data_layout}")
        
        functions = list(module.functions)
        print(f"   Functions: {len(functions)}")
        for func in functions:
            print(f"     - {func.name}: {func.type}")
            print(f"       Basic Blocks: {len(list(func.blocks))}")
            instruction_count = sum(len(list(block.instructions)) for block in func.blocks)
            print(f"       Instructions: {instruction_count}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to analyze module: {e}")


def validate_ir(ir_code: str) -> bool:
    """Validate LLVM IR code"""
    module = binding.parse_assembly(ir_code)
    module.verify()
    return True


def compare_performance(original_ir: str, optimized_ir: str):
    """Compare performance between original and optimized IR"""
    try:
        orig_module = binding.parse_assembly(original_ir)
        opt_module = binding.parse_assembly(optimized_ir)
        
        orig_instructions = 0
        opt_instructions = 0
        
        for func in orig_module.functions:
            for block in func.blocks:
                orig_instructions += len(list(block.instructions))
        
        for func in opt_module.functions:
            for block in func.blocks:
                opt_instructions += len(list(block.instructions))
        
        reduction = orig_instructions - opt_instructions
        percentage = (reduction / orig_instructions) * 100 if orig_instructions > 0 else 0
        
        print("Performance Comparison:")
        print(f"   Original Instructions: {orig_instructions}")
        print(f"   Optimized Instructions: {opt_instructions}")
        print(f"   Reduction: {reduction} ({percentage:.1f}%)")
    
    except Exception as e:
        raise RuntimeError(f"Performance comparison failed: {e}")


def disassemble_to_native(ir_code: str, target_triple: Optional[str] = None) -> str:
    """Disassemble LLVM IR to native assembly"""
    try:
        module = binding.parse_assembly(ir_code)
        
        if target_triple:
            module.triple = target_triple
        
        # Create target machine
        target = binding.Target.from_triple(module.triple)
        target_machine = target.create_target_machine()
        
        # Generate assembly
        asm_code = target_machine.emit_assembly(module)
        return asm_code
    
    except Exception as e:
        raise RuntimeError(f"Assembly generation failed: {e}")


def benchmark_function(ir_code: str, func_name: str, iterations: int = 1000000):
    """Benchmark a compiled function (placeholder for future implementation)"""
    print(f"Benchmarking {func_name} with {iterations} iterations...")
    logger.warning("Benchmarking not yet implemented - requires proper JIT integration")


def create_build_info() -> Dict[str, str]:
    """Create build information"""
    return {
        'llvm_version': get_llvm_version(),
        'target_triple': binding.get_default_triple(),
        'host_cpu': binding.get_host_cpu_name(),
        'host_cpu_features': binding.get_host_cpu_features(),
    }
