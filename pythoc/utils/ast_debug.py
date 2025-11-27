"""
AST Debugging Utilities

Provides tools to inspect and debug AST transformations, especially useful
for debugging inline function expansions.

Environment Variables:
    PC_DEBUG_AST=1          Enable AST debugging (saves to build/debug_ast/)
    PC_DEBUG_AST_FORMAT=all Export format: 'unparse', 'tree', 'all' (default: 'all')
    PC_DEBUG_AST_DIFF=1     Show diff between before/after AST

Usage:
    from pythoc.utils.ast_debug import ast_debugger, dump_ast, compare_ast
    
    # Method 1: Context manager (automatic before/after capture)
    with ast_debugger.trace("inline_expansion", node):
        # ... perform AST transformation ...
        new_node = transform(node)
    
    # Method 2: Manual capture
    ast_debugger.capture("before_inline", node, func_name="square")
    new_node = transform(node)
    ast_debugger.capture("after_inline", new_node, func_name="square")
    
    # Method 3: One-off dump
    dump_ast(node, title="My AST", format='unparse')
    
    # Method 4: Compare two ASTs
    compare_ast(old_node, new_node, "Before", "After")
"""

import ast
import os
import sys
from pathlib import Path
from typing import Union, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ASTSnapshot:
    """Represents a captured AST snapshot"""
    name: str
    node: ast.AST
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def get_source(self) -> str:
        """Get Python source code from AST"""
        try:
            return ast.unparse(self.node)
        except Exception as e:
            return f"<unparseable: {e}>"
    
    def get_tree(self, indent: int = 2) -> str:
        """Get tree representation of AST"""
        return ast.dump(self.node, indent=indent)


class ASTDebugger:
    """
    AST debugging tool for tracking transformations
    
    Features:
    - Capture AST snapshots at different stages
    - Export to multiple formats (source code, tree view)
    - Compare before/after transformations
    - Automatic file saving to build/debug_ast/
    """
    
    def __init__(self):
        self.enabled = self._check_enabled()
        self.export_format = os.environ.get('PC_DEBUG_AST_FORMAT', 'all')
        self.show_diff = os.environ.get('PC_DEBUG_AST_DIFF', '0') == '1'
        self.snapshots: List[ASTSnapshot] = []
        self.output_dir = None
        
        if self.enabled:
            self._setup_output_dir()
    
    def _check_enabled(self) -> bool:
        """Check if AST debugging is enabled via environment variable"""
        return os.environ.get('PC_DEBUG_AST', '0') == '1'
    
    def _setup_output_dir(self):
        """Setup output directory for debug files"""
        from pythoc.utils.path_utils import get_build_paths
        try:
            build_dir, _ = get_build_paths()
            self.output_dir = Path(build_dir) / "debug_ast"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except:
            # Fallback to current directory
            self.output_dir = Path("build/debug_ast")
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def capture(self, name: str, node: Union[ast.AST, List[ast.stmt]], **metadata):
        """
        Capture an AST snapshot
        
        Args:
            name: Snapshot name (e.g., "before_inline", "after_transform")
            node: AST node or list of statements
            **metadata: Additional metadata (func_name, inline_id, etc.)
        """
        if not self.enabled:
            return
        
        # Wrap list of statements in Module node
        if isinstance(node, list):
            node = ast.Module(body=node, type_ignores=[])
        
        snapshot = ASTSnapshot(name=name, node=node, metadata=metadata)
        self.snapshots.append(snapshot)
        
        # Auto-export if enabled
        self._export_snapshot(snapshot)
    
    def _export_snapshot(self, snapshot: ASTSnapshot):
        """Export a snapshot to file"""
        if not self.output_dir:
            return
        
        # Build filename
        func_name = snapshot.metadata.get('func_name', 'unknown')
        inline_id = snapshot.metadata.get('inline_id', '')
        timestamp = snapshot.timestamp.strftime('%H%M%S')
        
        base_name = f"{func_name}_{snapshot.name}"
        if inline_id:
            base_name += f"_{inline_id}"
        base_name += f"_{timestamp}"
        
        # Export source code (unparse)
        if self.export_format in ('unparse', 'all'):
            source_file = self.output_dir / f"{base_name}.py"
            with open(source_file, 'w') as f:
                f.write(f"# AST Snapshot: {snapshot.name}\n")
                f.write(f"# Function: {func_name}\n")
                f.write(f"# Time: {snapshot.timestamp}\n")
                if snapshot.metadata:
                    f.write(f"# Metadata: {snapshot.metadata}\n")
                f.write("\n")
                f.write(snapshot.get_source())
            print(f"[AST DEBUG] Exported source: {source_file}", file=sys.stderr)
        
        # Export tree view
        if self.export_format in ('tree', 'all'):
            tree_file = self.output_dir / f"{base_name}.ast"
            with open(tree_file, 'w') as f:
                f.write(f"AST Snapshot: {snapshot.name}\n")
                f.write(f"Function: {func_name}\n")
                f.write(f"Time: {snapshot.timestamp}\n")
                if snapshot.metadata:
                    f.write(f"Metadata: {snapshot.metadata}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(snapshot.get_tree())
            print(f"[AST DEBUG] Exported tree: {tree_file}", file=sys.stderr)
    
    @contextmanager
    def trace(self, operation_name: str, before_node: ast.AST, **metadata):
        """
        Context manager to trace AST transformation
        
        Usage:
            with ast_debugger.trace("inline_expand", call_node, func_name="square"):
                result = inline_transform(call_node)
            # Automatically captures before/after
        
        Args:
            operation_name: Name of operation (e.g., "inline_expand")
            before_node: AST node before transformation
            **metadata: Additional metadata
        """
        if self.enabled:
            self.capture(f"before_{operation_name}", before_node, **metadata)
        
        # Yield control back to caller
        class Result:
            """Container for storing result node"""
            def __init__(self):
                self.after_node = None
        
        result_container = Result()
        
        try:
            yield result_container
        finally:
            # Capture after transformation if result was set
            if self.enabled and result_container.after_node is not None:
                self.capture(f"after_{operation_name}", result_container.after_node, **metadata)
    
    def compare(self, name1: str, name2: str):
        """
        Compare two captured snapshots
        
        Args:
            name1: First snapshot name
            name2: Second snapshot name
        """
        if not self.enabled:
            return
        
        snap1 = next((s for s in self.snapshots if s.name == name1), None)
        snap2 = next((s for s in self.snapshots if s.name == name2), None)
        
        if not snap1 or not snap2:
            print(f"[AST DEBUG] Cannot compare: snapshot not found", file=sys.stderr)
            return
        
        print(f"\n{'='*80}")
        print(f"AST COMPARISON: {name1} vs {name2}")
        print(f"{'='*80}\n")
        
        source1 = snap1.get_source()
        source2 = snap2.get_source()
        
        print(f"=== {name1} ===")
        print(source1)
        print(f"\n=== {name2} ===")
        print(source2)
        print(f"\n{'='*80}\n")
    
    def clear(self):
        """Clear all captured snapshots"""
        self.snapshots.clear()
    
    def summary(self):
        """Print summary of captured snapshots"""
        if not self.snapshots:
            print("[AST DEBUG] No snapshots captured")
            return
        
        print(f"\n[AST DEBUG] Captured {len(self.snapshots)} snapshots:")
        for i, snap in enumerate(self.snapshots):
            func_name = snap.metadata.get('func_name', '?')
            print(f"  {i+1}. {snap.name} (func={func_name}, time={snap.timestamp.strftime('%H:%M:%S')})")


# Global instance
ast_debugger = ASTDebugger()


def dump_ast(node: Union[ast.AST, List[ast.stmt]], 
             title: str = "AST Dump",
             format: str = 'unparse',
             show_tree: bool = False):
    """
    One-off AST dump for quick inspection
    
    Args:
        node: AST node or list of statements
        title: Title for the dump
        format: 'unparse' (source code) or 'tree' (AST structure)
        show_tree: If True, also show tree view
    
    Example:
        dump_ast(func_node, "My Function AST")
    """
    # Wrap list in Module
    if isinstance(node, list):
        node = ast.Module(body=node, type_ignores=[])
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")
    
    if format == 'unparse' or not show_tree:
        try:
            source = ast.unparse(node)
            print(source)
        except Exception as e:
            print(f"<unparseable: {e}>")
    
    if format == 'tree' or show_tree:
        print(f"\n--- AST Tree ---\n")
        print(ast.dump(node, indent=2))
    
    print(f"\n{'='*80}\n")


def compare_ast(node1: Union[ast.AST, List[ast.stmt]], 
                node2: Union[ast.AST, List[ast.stmt]],
                label1: str = "Before",
                label2: str = "After"):
    """
    Compare two AST nodes side-by-side
    
    Args:
        node1: First AST node
        node2: Second AST node
        label1: Label for first node
        label2: Label for second node
    
    Example:
        compare_ast(original_func, inlined_func, "Original", "After Inline")
    """
    # Wrap lists in Module
    if isinstance(node1, list):
        node1 = ast.Module(body=node1, type_ignores=[])
    if isinstance(node2, list):
        node2 = ast.Module(body=node2, type_ignores=[])
    
    print(f"\n{'='*80}")
    print(f"  AST COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"=== {label1} ===\n")
    try:
        print(ast.unparse(node1))
    except Exception as e:
        print(f"<unparseable: {e}>")
    
    print(f"\n=== {label2} ===\n")
    try:
        print(ast.unparse(node2))
    except Exception as e:
        print(f"<unparseable: {e}>")
    
    print(f"\n{'='*80}\n")


def enable_ast_debug():
    """Programmatically enable AST debugging"""
    os.environ['PC_DEBUG_AST'] = '1'
    global ast_debugger
    ast_debugger = ASTDebugger()


def disable_ast_debug():
    """Programmatically disable AST debugging"""
    os.environ['PC_DEBUG_AST'] = '0'
    global ast_debugger
    ast_debugger.enabled = False
