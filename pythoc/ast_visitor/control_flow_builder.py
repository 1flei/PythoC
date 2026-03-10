"""
Control Flow Builder - Transparent wrapper around LLVMBuilder with CFG tracking

This module provides a ControlFlowBuilder that wraps the real LLVMBuilder and
delegates all non-overridden methods via __getattr__. Control-flow methods are
explicitly overridden to maintain CFG state for linear type checking.

Design:
- ControlFlowBuilder IS the visitor's builder (visitor.builder = cf)
- All IR emission goes through it transparently
- Control-flow methods (branch, cbranch, etc.) also update CFG
- External consumers (type_converter, builtin_entities) need zero changes
"""

from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from llvmlite import ir
import ast

from ..cfg import CFG, CFGBlock, CFGEdge
from ..cfg.linear_checker import (
    LinearSnapshot,
    LinearRegister,
    LinearTransition,
    LinearEvent,
    copy_snapshot,
)
from ..logger import logger

if TYPE_CHECKING:
    from .base import LLVMIRVisitor
    from ..builder import LLVMBuilder


class ControlFlowBuilder:
    """Transparent wrapper around LLVMBuilder with CFG tracking

    Uses __getattr__ to delegate all non-overridden methods to the real builder.
    Control-flow methods are explicitly overridden to maintain CFG state.

    Usage:
        real_builder = LLVMBuilder(ir_builder)
        cf = ControlFlowBuilder(real_builder, visitor, func_name)
        visitor.builder = cf  # All IR goes through cf now

        # Control flow methods update CFG:
        then_block = cf.create_block("then")
        cf.cbranch(condition, then_block, else_block)

        # All other methods delegate transparently:
        cf.store(value, ptr)   # -> real_builder.store(value, ptr)
        cf.load(ptr)           # -> real_builder.load(ptr)
    """

    def __init__(self, real_builder: "LLVMBuilder", visitor: "LLVMIRVisitor", func_name: str = ""):
        """Initialize with a real builder and visitor

        Args:
            real_builder: The LLVMBuilder instance to wrap
            visitor: LLVMIRVisitor instance (for get_next_label, current_function)
            func_name: Name of the function being compiled
        """
        self._real_builder = real_builder
        self._visitor = visitor
        self._func_name = func_name or "unknown"
        self._finalized = False

        # CFG data structure
        self._cfg = CFG(func_name=self._func_name)

        # Create entry block in CFG
        self._entry_block = self._cfg.add_block()
        self._cfg.entry_id = self._entry_block.id

        # Create virtual exit block in CFG
        # All return statements and fall-through paths connect to this block.
        # This makes the exit block a merge point for linear state checking.
        self._exit_block = self._cfg.add_block()
        self._cfg.exit_id = self._exit_block.id

        # Current block in CFG
        self._current_block_id = self._entry_block.id

        # Map CFGBlock ID -> LLVM ir.Block
        # Entry block maps to the function's entry block
        self._block_map: Dict[int, ir.Block] = {}
        if real_builder and real_builder.block:
            self._block_map[self._entry_block.id] = real_builder.block

        # Track if current CFG block is terminated (has outgoing edge)
        self._terminated: Dict[int, bool] = {self._entry_block.id: False}

    def __getattr__(self, name):
        """Delegate all non-overridden attribute access to the real builder"""
        return getattr(self._real_builder, name)

    @property
    def cfg(self) -> CFG:
        """Get the CFG being built"""
        return self._cfg

    @property
    def block(self):
        """Get the current LLVM basic block (properties bypass __getattr__)"""
        return self._real_builder.block

    @property
    def ir_builder(self):
        """Get the underlying llvmlite IRBuilder (properties bypass __getattr__)"""
        return self._real_builder.ir_builder

    @property
    def current_function(self) -> ir.Function:
        """Get the current LLVM function"""
        return self._visitor.current_function

    @property
    def current_block(self) -> ir.Block:
        """Get the current LLVM basic block"""
        return self._real_builder.block

    @property
    def current_block_id(self) -> int:
        """Get current CFG block ID"""
        return self._current_block_id

    def is_terminated(self) -> bool:
        """Check if current block is already terminated"""
        # Check both CFG and LLVM
        cfg_terminated = self._terminated.get(self._current_block_id, False)
        ir_terminated = self._real_builder.block.is_terminated
        return cfg_terminated or ir_terminated

    def create_block(self, name: str) -> ir.Block:
        """Create a new basic block

        Creates both a CFGBlock and an LLVM ir.Block.

        Args:
            name: Label prefix for the block

        Returns:
            The LLVM ir.Block (for compatibility with existing code)
        """
        # Create CFG block
        cfg_block = self._cfg.add_block()
        self._terminated[cfg_block.id] = False

        # Create LLVM block
        label = self._visitor.get_next_label(name)
        ir_block = self.current_function.append_basic_block(label)

        # Map CFG block to IR block
        self._block_map[cfg_block.id] = ir_block

        # Store CFG block ID in IR block for reverse lookup
        ir_block._cfg_block_id = cfg_block.id

        logger.debug(f"CFG: create_block '{name}' -> CFGBlock({cfg_block.id}), IR={label}")

        return ir_block

    def _get_cfg_block_id(self, ir_block: ir.Block) -> int:
        """Get CFG block ID for an IR block"""
        if hasattr(ir_block, '_cfg_block_id'):
            return ir_block._cfg_block_id
        # Fallback: search in map
        for cfg_id, mapped_block in self._block_map.items():
            if mapped_block is ir_block:
                return cfg_id
        raise ValueError(f"IR block {ir_block.name} not found in CFG")

    def position_at_end(self, block: ir.Block):
        """Position at the end of the given block

        NOTE: We do NOT record exit_snapshot here. Exit snapshots should only
        be recorded when the block actually terminates (via branch/cbranch/return).

        Args:
            block: The LLVM block to position at
        """
        # Update CFG current block
        cfg_block_id = self._get_cfg_block_id(block)

        self._current_block_id = cfg_block_id

        # Update IR builder
        self._real_builder.position_at_end(block)

        logger.debug(f"CFG: position_at_end CFGBlock({cfg_block_id})")

    def branch(self, target: ir.Block):
        """Emit an unconditional branch

        Also records exit snapshot for CFG-based linear checking.

        Args:
            target: The target block to branch to
        """
        src_id = self._current_block_id
        dst_id = self._get_cfg_block_id(target)

        # Add edge to CFG
        self._cfg.add_edge(src_id, dst_id, kind='sequential')
        self._terminated[src_id] = True

        # Generate IR
        self._real_builder.branch(target)

        logger.debug(f"CFG: branch {src_id} -> {dst_id}")

    def cbranch(self, condition, true_block: ir.Block, false_block: ir.Block):
        """Emit a conditional branch

        Args:
            condition: LLVM i1 value for the condition
            true_block: Block to branch to if condition is true
            false_block: Block to branch to if condition is false
        """
        src_id = self._current_block_id
        true_id = self._get_cfg_block_id(true_block)
        false_id = self._get_cfg_block_id(false_block)

        # Add edges to CFG
        self._cfg.add_edge(src_id, true_id, kind='branch_true')
        self._cfg.add_edge(src_id, false_id, kind='branch_false')
        self._terminated[src_id] = True

        # Generate IR
        self._real_builder.cbranch(condition, true_block, false_block)

        logger.debug(f"CFG: cbranch {src_id} -> T:{true_id}, F:{false_id}")

    def unreachable(self):
        """Emit an unreachable instruction"""
        self._terminated[self._current_block_id] = True
        self._real_builder.unreachable()

        logger.debug(f"CFG: unreachable at CFGBlock({self._current_block_id})")

    def switch(self, value, default_block: ir.Block) -> ir.SwitchInstr:
        """Emit a switch instruction

        Args:
            value: The value to switch on
            default_block: The default case block

        Returns:
            The switch instruction (for adding cases)
        """
        src_id = self._current_block_id
        default_id = self._get_cfg_block_id(default_block)

        # Add default edge to CFG
        self._cfg.add_edge(src_id, default_id, kind='sequential')
        self._terminated[src_id] = True

        # Generate IR and return switch for adding cases
        switch_instr = self._real_builder.switch(value, default_block)

        logger.debug(f"CFG: switch at {src_id}, default -> {default_id}")

        return switch_instr

    def ret(self, value):
        """Emit a return instruction with CFG tracking

        Args:
            value: The return value
        """
        if not self._finalized:
            self.mark_return()
        self._real_builder.ret(value)

    def ret_void(self):
        """Emit a void return instruction with CFG tracking"""
        if not self._finalized:
            self.mark_return()
        self._real_builder.ret_void()

    def add_switch_case(self, switch_instr: ir.SwitchInstr, value, block: ir.Block):
        """Add a case to a switch instruction

        Args:
            switch_instr: The switch instruction
            value: The case value
            block: The case block
        """
        # Find source block (the one with the switch)
        src_id = None
        for cfg_id, ir_block in self._block_map.items():
            if ir_block is switch_instr.parent:
                src_id = cfg_id
                break

        if src_id is not None:
            dst_id = self._get_cfg_block_id(block)
            # Add case edge
            self._cfg.add_edge(src_id, dst_id, kind='branch_true')
            logger.debug(f"CFG: switch case {src_id} -> {dst_id}")

        # Add case to IR switch
        switch_instr.add_case(value, block)

    def add_stmt(self, stmt: ast.stmt):
        """Record an AST statement in the current CFG block

        Args:
            stmt: AST statement to record
        """
        cfg_block = self._cfg.blocks[self._current_block_id]
        cfg_block.stmts.append(stmt)

    # ========== Linear Event Recording Methods ==========

    def record_linear_register(self, var_id: int, var_name: str, path: Tuple[int, ...],
                               initial_state: str, line_number: int = None,
                               node: ast.AST = None):
        """Record a LinearRegister event - variable with linear type enters scope"""
        event = LinearRegister(var_id, var_name, path, initial_state, line_number, node)
        self._add_linear_event(event)
        logger.debug(f"CFG: LinearRegister(id={var_id}, {var_name}, path={path}, initial={initial_state})")

    def record_linear_transition(self, var_id: int, var_name: str, path: Tuple[int, ...],
                                  old_state: str, new_state: str, line_number: int = None,
                                  node: ast.AST = None):
        """Record a LinearTransition event - linear state change"""
        event = LinearTransition(var_id, var_name, path, old_state, new_state, line_number, node)
        self._add_linear_event(event)
        logger.debug(f"CFG: LinearTransition(id={var_id}, {var_name}, path={path}, {old_state}->{new_state})")

    def _add_linear_event(self, event: LinearEvent):
        """Add a linear event to the current block"""
        cfg_block = self._cfg.blocks[self._current_block_id]
        cfg_block.linear_events.append(event)

    def mark_return(self):
        """Mark current block as containing a return statement

        Records exit snapshot and connects to the virtual exit block.
        This makes the exit block a merge point for linear state checking.
        """
        # Connect to virtual exit block (makes exit block a merge point)
        self._cfg.add_edge(self._current_block_id, self._exit_block.id, kind='return')

        self._cfg.return_blocks.append(self._current_block_id)
        self._terminated[self._current_block_id] = True
        logger.debug(f"CFG: return at CFGBlock({self._current_block_id}) -> exit({self._exit_block.id})")

    def mark_loop_back(self, target: ir.Block):
        """Mark a loop back edge (for loop analysis)

        Args:
            target: The loop header block
        """
        src_id = self._current_block_id
        dst_id = self._get_cfg_block_id(target)

        # Update the last edge to be a loop_back edge
        for edge in reversed(self._cfg.edges):
            if edge.source_id == src_id and edge.target_id == dst_id:
                edge.kind = 'loop_back'
                break

        self._cfg.loop_headers.add(dst_id)
        logger.debug(f"CFG: loop_back {src_id} -> {dst_id}")

    def finalize(self):
        """Finalize CFG construction

        Call this at the end of function compilation to:
        1. Connect fall-through blocks to exit block
        2. Record exit snapshot for blocks that need it
        3. Compute loop headers
        4. Log CFG structure

        Sets _finalized flag so subsequent ret/ret_void calls (from structural
        fixup) skip CFG tracking.
        """
        # For blocks without explicit terminator, connect to exit block
        # This handles void functions that fall through without return
        reachable = self._cfg.get_reachable_blocks()
        for block_id in reachable:
            # Skip the exit block itself
            if block_id == self._exit_block.id:
                continue

            successors = self._cfg.get_successors(block_id)
            # If block has no successors, it's a fall-through to function end
            if not successors:
                # Connect to exit block (makes exit block a merge point)
                self._cfg.add_edge(block_id, self._exit_block.id, kind='fallthrough')
                logger.debug(f"CFG: fallthrough at CFGBlock({block_id}) -> exit({self._exit_block.id})")

        # Compute loop headers
        self._cfg.compute_loop_headers()

        # Mark as finalized - subsequent ret/ret_void skip CFG tracking
        self._finalized = True

        # Log CFG summary
        logger.debug(f"CFG finalized: {self._cfg}")

    def run_cfg_linear_check(self, initial_snapshot: LinearSnapshot = None) -> bool:
        """Run CFG-based linear type checking

        Args:
            initial_snapshot: Initial linear state (defaults to empty)

        Returns:
            True if no errors, False if errors found
        """
        from ..cfg.linear_checker import LinearChecker

        if initial_snapshot is None:
            initial_snapshot = {}

        checker = LinearChecker()
        errors = checker.check(self._cfg, initial_snapshot)

        if errors:
            for err in errors:
                logger.error(f"CFG linear check: {err.format()}", node=err.source_node)
            return False

        logger.debug(f"CFG linear check passed for {self._func_name}")
        return True

    def dump_cfg(self, level: int = 0, use_print: bool = False):
        """Dump CFG structure for debugging

        Args:
            level: Log level (0=debug, 1=info)
            use_print: If True, use print() instead of logger
        """
        lines = [f"\n=== CFG for {self._func_name} ==="]
        lines.append(f"Entry: B{self._cfg.entry_id}, Exit: B{self._cfg.exit_id}")
        lines.append(f"Blocks: {len(self._cfg.blocks)}, Edges: {len(self._cfg.edges)}")

        # Dump blocks
        lines.append("\nBlocks:")
        for block_id in sorted(self._cfg.blocks.keys()):
            block = self._cfg.blocks[block_id]
            markers = []
            if block_id == self._cfg.entry_id:
                markers.append("entry")
            if block_id == self._cfg.exit_id:
                markers.append("exit")
            if block_id in self._cfg.loop_headers:
                markers.append("loop_header")
            if block_id in self._cfg.return_blocks:
                markers.append("return")

            marker_str = f" ({', '.join(markers)})" if markers else ""
            ir_block = self._block_map.get(block_id)
            ir_name = ir_block.name if ir_block else "?"

            lines.append(f"  B{block_id}{marker_str} [IR: {ir_name}]:")

            # Show statements
            if block.stmts:
                for stmt in block.stmts:
                    stmt_str = ast.unparse(stmt)[:60]
                    lines.append(f"    {stmt_str}")
            else:
                lines.append("    (no statements recorded)")

            # Show linear events
            if block.linear_events:
                lines.append(f"    Linear events: {len(block.linear_events)}")
                for event in block.linear_events:
                    lines.append(f"      {event}")

        # Dump edges
        lines.append("\nEdges:")
        for edge in self._cfg.edges:
            lines.append(f"  B{edge.source_id} -> B{edge.target_id} [{edge.kind}]")

        lines.append("=== End CFG ===\n")

        msg = "\n".join(lines)
        if use_print:
            print(msg)
        elif level == 0:
            logger.debug(msg)
        else:
            logger.info(msg)
