"""
Control Flow Builder - Wrapper around LLVMBuilder with CFG tracking and PCIR recording.

Design (PCIR = Per-block Captured IR):
- During the AST visitor walk (build phase), ALL builder calls are recorded
  as PCIR instructions. No LLVM IR is emitted.
- Control-flow methods (branch, cbranch, ret, etc.) update CFG state AND
  record PCIR instructions.
- create_block() returns SentinelBlock placeholders (not real ir.Block).
- After the CFG is fully built, emit_ir() replays PCIR to produce real LLVM IR.

Key benefits:
- Full separation of CFG construction from IR emission
- Forward goto becomes trivial (append PCIR, no save/restore)
- Dead block elimination can happen before IR emission
"""

from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
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
from .pcir import (
    VReg, VRegPhi, VRegSwitch, SentinelBlock, PCIRInst,
    infer_result_type, resolve_arg, reset_vreg_counter,
)

if TYPE_CHECKING:
    from .base import LLVMIRVisitor
    from ..builder import LLVMBuilder


class ControlFlowBuilder:
    """Wrapper around LLVMBuilder with CFG tracking and PCIR recording.

    Build phase (before finalize()):
    - __getattr__ returns a recorder function that creates PCIRInst + VReg
    - Control-flow methods update CFG state AND record PCIR
    - create_block() returns SentinelBlock placeholders
    - No real LLVM IR is emitted

    Replay phase (emit_ir()):
    - Creates real ir.Blocks for each CFG block
    - Replays PCIR instructions block by block
    - Resolves VRegs to actual ir.Values
    - Patches unterminated blocks

    Usage:
        cf = ControlFlowBuilder(real_builder, visitor, func_name)
        visitor.builder = cf
        # ... visit AST body (records PCIR, builds CFG) ...
        cf.finalize()   # Finalize CFG structure
        cf.emit_ir()    # Replay PCIR -> actual LLVM IR
    """

    def __init__(self, real_builder: "LLVMBuilder", visitor: "LLVMIRVisitor", func_name: str = ""):
        self._real_builder = real_builder
        self._visitor = visitor
        self._func_name = func_name or "unknown"
        self._finalized = False
        self._emitted = False

        # Reset VReg counter for each function
        reset_vreg_counter()

        # CFG data structure
        self._cfg = CFG(func_name=self._func_name)

        # Create entry block in CFG
        self._entry_block = self._cfg.add_block()
        self._cfg.entry_id = self._entry_block.id

        # Create virtual exit block in CFG
        self._exit_block = self._cfg.add_block()
        self._cfg.exit_id = self._exit_block.id

        # Current block in CFG
        self._current_block_id = self._entry_block.id

        # Map CFGBlock ID -> SentinelBlock (build phase) or ir.Block (replay phase)
        self._block_map: Dict[int, Any] = {}

        # Create sentinel for entry block
        entry_sentinel = SentinelBlock(self._entry_block.id, "entry")
        self._block_map[self._entry_block.id] = entry_sentinel

        # Tag the real entry ir.Block with the CFG block ID so that
        # code accessing current_function.entry_basic_block can map back to CFG
        if real_builder and real_builder.block:
            real_builder.block._cfg_block_id = self._entry_block.id

        # Store the block label names for creating real blocks during replay
        self._block_labels: Dict[int, str] = {self._entry_block.id: "entry"}

        # Track if current CFG block is terminated
        self._terminated: Dict[int, bool] = {self._entry_block.id: False}

        # Pending forward gotos (opaque objects with .block, .label_name, etc.)
        self._pending_gotos: List[Any] = []

        # Track insertion mode for position_at_start vs position_at_end
        # When _insert_at_start is True, PCIR instructions are inserted at
        # _insert_idx instead of appended (for _create_alloca_in_entry pattern)
        self._insert_at_start: bool = False
        self._insert_idx: int = 0
        # Per-block watermark: tracks how many instructions have been inserted
        # at the start of each block. When position_at_start is called again,
        # new instructions go after the watermark (not at index 0).
        self._start_watermark: Dict[int, int] = {}

    def __getattr__(self, name):
        """Record PCIR instruction instead of delegating to real builder.

        During build phase: returns a recorder function.
        After emit_ir: delegates to real builder for post-emission operations.
        """
        if self._emitted:
            # After emission, delegate to real builder
            return getattr(self._real_builder, name)

        # During build phase: return a PCIR recorder
        def _record(*args, **kwargs):
            result_type = infer_result_type(name, args, kwargs)
            if result_type is not None:
                vreg = VReg(result_type)
                inst = PCIRInst(op=name, args=args, kwargs=kwargs, result=vreg)
            else:
                vreg = None
                inst = PCIRInst(op=name, args=args, kwargs=kwargs, result=None)

            self._append_pcir(inst)
            return vreg

        return _record

    def _append_pcir(self, inst: PCIRInst):
        """Append a PCIR instruction to the current block.

        Respects _insert_at_start mode for position_at_start semantics
        (used by _create_alloca_in_entry to insert allocas at the beginning).
        Tracks per-block watermark so subsequent position_at_start calls
        insert after previously-inserted-at-start instructions.
        """
        cfg_block = self._cfg.blocks[self._current_block_id]
        if self._insert_at_start:
            cfg_block.pcir.insert(self._insert_idx, inst)
            self._insert_idx += 1
            # Update watermark for this block
            self._start_watermark[self._current_block_id] = self._insert_idx
        else:
            cfg_block.pcir.append(inst)

    @property
    def cfg(self) -> CFG:
        return self._cfg

    @property
    def block(self):
        """Get the current block (SentinelBlock during build, ir.Block after emit)."""
        if self._emitted:
            return self._real_builder.block
        return self._block_map.get(self._current_block_id)

    @property
    def ir_builder(self):
        """Get the underlying llvmlite IRBuilder.

        During build phase, this is still accessible for operations that need it
        (e.g., parameter initialization in compiler.py which happens before PCIR).
        """
        return self._real_builder.ir_builder

    @property
    def current_function(self) -> ir.Function:
        return self._visitor.current_function

    @property
    def current_block(self):
        """Get the current block."""
        if self._emitted:
            return self._real_builder.block
        return self._block_map.get(self._current_block_id)

    @property
    def current_block_id(self) -> int:
        return self._current_block_id

    def is_terminated(self) -> bool:
        """Check if current block is already terminated."""
        if self._emitted:
            cfg_terminated = self._terminated.get(self._current_block_id, False)
            ir_terminated = self._real_builder.block.is_terminated
            return cfg_terminated or ir_terminated
        return self._terminated.get(self._current_block_id, False)

    def create_block(self, name: str):
        """Create a new basic block.

        During build phase: creates CFGBlock + SentinelBlock.
        Returns SentinelBlock (has _cfg_block_id and name attributes).
        """
        # Create CFG block
        cfg_block = self._cfg.add_block()
        self._terminated[cfg_block.id] = False

        # Create sentinel block (placeholder for real ir.Block)
        label = self._visitor.get_next_label(name)
        sentinel = SentinelBlock(cfg_block.id, label)
        self._block_map[cfg_block.id] = sentinel
        self._block_labels[cfg_block.id] = label

        logger.debug(f"CFG: create_block '{name}' -> CFGBlock({cfg_block.id}), label={label}")

        return sentinel

    def _get_cfg_block_id(self, block) -> int:
        """Get CFG block ID for a block (SentinelBlock or ir.Block)."""
        if isinstance(block, SentinelBlock):
            return block._cfg_block_id
        if hasattr(block, '_cfg_block_id'):
            return block._cfg_block_id
        # Fallback: search in map
        for cfg_id, mapped_block in self._block_map.items():
            if mapped_block is block:
                return cfg_id
        raise ValueError(f"Block {getattr(block, 'name', block)} not found in CFG")

    def position_at_end(self, block):
        """Position at the end of the given block.

        During build phase: only updates CFG current block tracking.
        After emit: delegates to real builder.
        """
        if self._emitted:
            cfg_block_id = self._get_cfg_block_id(block)
            self._current_block_id = cfg_block_id
            self._real_builder.position_at_end(block)
            return

        if not self._finalized:
            cfg_block_id = self._get_cfg_block_id(block)
            self._current_block_id = cfg_block_id
            # Switch back to append mode
            self._insert_at_start = False
            logger.debug(f"CFG: position_at_end CFGBlock({cfg_block_id})")

    def position_at_start(self, block):
        """Position at the start of the given block.

        During build phase: updates CFG current block tracking and enables
        insert-at-start mode for subsequent PCIR instructions.
        Uses per-block watermark so repeated calls insert after previously
        inserted-at-start instructions (e.g., allocas before va_start).
        After emit: delegates to real builder.
        """
        if self._emitted:
            self._real_builder.position_at_start(block)
            return

        cfg_block_id = self._get_cfg_block_id(block)
        self._current_block_id = cfg_block_id
        # Enable insert-at-start mode: subsequent PCIR goes after the watermark
        self._insert_at_start = True
        self._insert_idx = self._start_watermark.get(cfg_block_id, 0)

    def branch(self, target, kind='sequential'):
        """Record an unconditional branch (CFG + PCIR).

        Args:
            target: Target block (SentinelBlock or ir.Block)
            kind: Edge kind for CFG ('sequential', 'break', 'continue', 'goto', 'goto_end')
        """
        src_id = self._current_block_id
        dst_id = self._get_cfg_block_id(target)

        # Add edge to CFG
        self._cfg.add_edge(src_id, dst_id, kind=kind)
        self._terminated[src_id] = True

        # Record PCIR
        self._append_pcir(PCIRInst(op='branch', args=(target,), kwargs={}))

        logger.debug(f"CFG: branch {src_id} -> {dst_id} [{kind}]")

    def cbranch(self, condition, true_block, false_block):
        """Record a conditional branch (CFG + PCIR)."""
        src_id = self._current_block_id
        true_id = self._get_cfg_block_id(true_block)
        false_id = self._get_cfg_block_id(false_block)

        # Add edges to CFG
        self._cfg.add_edge(src_id, true_id, kind='branch_true')
        self._cfg.add_edge(src_id, false_id, kind='branch_false')
        self._terminated[src_id] = True

        # Record PCIR
        self._append_pcir(PCIRInst(op='cbranch', args=(condition, true_block, false_block), kwargs={}))

        logger.debug(f"CFG: cbranch {src_id} -> T:{true_id}, F:{false_id}")

    def unreachable(self):
        """Record an unreachable instruction."""
        if not self._finalized:
            self._terminated[self._current_block_id] = True
            logger.debug(f"CFG: unreachable at CFGBlock({self._current_block_id})")

        self._append_pcir(PCIRInst(op='unreachable', args=(), kwargs={}))

    def switch(self, value, default_block):
        """Record a switch instruction (CFG + PCIR).

        Returns VRegSwitch for adding cases.
        """
        src_id = self._current_block_id
        default_id = self._get_cfg_block_id(default_block)

        # Add default edge to CFG
        self._cfg.add_edge(src_id, default_id, kind='sequential')
        self._terminated[src_id] = True

        # Create VRegSwitch for case tracking
        current_sentinel = self._block_map.get(src_id)
        vswitch = VRegSwitch(current_sentinel)

        # Record PCIR
        self._append_pcir(PCIRInst(op='switch', args=(value, default_block), kwargs={}, result=vswitch))

        logger.debug(f"CFG: switch at {src_id}, default -> {default_id}")

        return vswitch

    def ret(self, value):
        """Record a return instruction with CFG tracking."""
        if not self._finalized:
            self.mark_return()

        self._append_pcir(PCIRInst(op='ret', args=(value,), kwargs={}))

    def ret_void(self):
        """Record a void return instruction with CFG tracking."""
        if not self._finalized:
            self.mark_return()

        self._append_pcir(PCIRInst(op='ret_void', args=(), kwargs={}))

    def phi(self, typ, name: str = ""):
        """Record a PHI node. Returns VRegPhi for add_incoming calls."""
        vphi = VRegPhi(typ)

        self._append_pcir(PCIRInst(op='phi', args=(typ,), kwargs={'name': name}, result=vphi))

        return vphi

    def select(self, cond, lhs, rhs, name: str = ""):
        """Record a select instruction."""
        result_type = infer_result_type('select', (cond, lhs, rhs), {'name': name})
        vreg = VReg(result_type) if result_type else None

        self._append_pcir(PCIRInst(op='select', args=(cond, lhs, rhs), kwargs={'name': name}, result=vreg))

        return vreg

    def add_switch_case(self, switch_instr, value, block):
        """Add a case to a switch instruction.

        Works with both VRegSwitch (build phase) and ir.SwitchInstr (post-emit).
        """
        if isinstance(switch_instr, VRegSwitch):
            # Build phase: find source block via VRegSwitch's parent sentinel
            src_sentinel = switch_instr._parent_sentinel
            if src_sentinel is not None:
                src_id = src_sentinel._cfg_block_id
                dst_id = self._get_cfg_block_id(block)
                self._cfg.add_edge(src_id, dst_id, kind='branch_true')
                logger.debug(f"CFG: switch case {src_id} -> {dst_id}")

            # Record case in VRegSwitch for replay
            switch_instr.add_case(value, block)
        else:
            # Post-emit: use real switch instruction
            src_id = None
            for cfg_id, ir_block in self._block_map.items():
                if ir_block is switch_instr.parent:
                    src_id = cfg_id
                    break

            if src_id is not None:
                dst_id = self._get_cfg_block_id(block)
                self._cfg.add_edge(src_id, dst_id, kind='branch_true')
                logger.debug(f"CFG: switch case {src_id} -> {dst_id}")

            switch_instr.add_case(value, block)

    def add_stmt(self, stmt: ast.stmt):
        """Record an AST statement in the current CFG block."""
        cfg_block = self._cfg.blocks[self._current_block_id]
        cfg_block.stmts.append(stmt)

    # ========== Linear Event Recording Methods ==========

    def record_linear_register(self, var_id: int, var_name: str, path: Tuple[int, ...],
                               initial_state: str, line_number: int = None,
                               node: ast.AST = None):
        event = LinearRegister(var_id, var_name, path, initial_state, line_number, node)
        self._add_linear_event(event)
        logger.debug(f"CFG: LinearRegister(id={var_id}, {var_name}, path={path}, initial={initial_state})")

    def record_linear_transition(self, var_id: int, var_name: str, path: Tuple[int, ...],
                                  old_state: str, new_state: str, line_number: int = None,
                                  node: ast.AST = None):
        event = LinearTransition(var_id, var_name, path, old_state, new_state, line_number, node)
        self._add_linear_event(event)
        logger.debug(f"CFG: LinearTransition(id={var_id}, {var_name}, path={path}, {old_state}->{new_state})")

    def _add_linear_event(self, event: LinearEvent):
        cfg_block = self._cfg.blocks[self._current_block_id]
        cfg_block.linear_events.append(event)

    def mark_return(self):
        """Mark current block as containing a return statement."""
        self._cfg.add_edge(self._current_block_id, self._exit_block.id, kind='return')
        self._cfg.return_blocks.append(self._current_block_id)
        self._terminated[self._current_block_id] = True
        logger.debug(f"CFG: return at CFGBlock({self._current_block_id}) -> exit({self._exit_block.id})")

    def mark_loop_back(self, target):
        """Mark a loop back edge."""
        src_id = self._current_block_id
        dst_id = self._get_cfg_block_id(target)

        for edge in reversed(self._cfg.edges):
            if edge.source_id == src_id and edge.target_id == dst_id:
                edge.kind = 'loop_back'
                break

        self._cfg.loop_headers.add(dst_id)
        logger.debug(f"CFG: loop_back {src_id} -> {dst_id}")

    def resume_recording_at(self, block):
        """Switch PCIR recording target to the given block, clearing its terminated flag.

        Returns a saved state tuple to pass to restore_recording().
        Used by scope_manager for forward-goto defer emission.
        """
        cfg_block_id = self._get_cfg_block_id(block)
        saved_block_id = self._current_block_id
        was_terminated = self._terminated.get(cfg_block_id, False)
        self._current_block_id = cfg_block_id
        self._terminated[cfg_block_id] = False
        return (saved_block_id, cfg_block_id, was_terminated)

    def restore_recording(self, saved):
        """Restore recording position from a resume_recording_at() return value."""
        saved_block_id, pending_block_id, was_terminated = saved
        self._terminated[pending_block_id] = was_terminated
        self._current_block_id = saved_block_id

    def mark_terminated(self, block=None):
        """Mark a block as terminated (e.g. for forward goto pending resolution).

        Args:
            block: Block to mark. If None, marks the current block.
        """
        if block is None:
            block_id = self._current_block_id
        else:
            block_id = self._get_cfg_block_id(block)
        self._terminated[block_id] = True

    def has_predecessors(self, block) -> bool:
        """Check if a block has any predecessor edges in the CFG."""
        block_id = self._get_cfg_block_id(block)
        return bool(self._cfg.get_predecessors(block_id))

    @property
    def entry_block(self):
        """Get the entry block (SentinelBlock during build, ir.Block after emit)."""
        return self._block_map.get(self._entry_block.id)

    # ========== Pending Forward Goto API ==========

    def register_pending_goto(self, pending):
        """Register a forward goto from current block. Marks block terminated.

        Args:
            pending: Opaque PendingGoto object with .block attribute set to
                     current block by caller.
        """
        self._pending_gotos.append(pending)
        self.mark_terminated()

    def emit_defers_for_pending_gotos(self, scope_depth, defer_executor):
        """Called during scope exit. Emit defers at scope_depth into each
        relevant pending goto block.

        For each pending goto whose goto_scope_depth >= scope_depth and
        emitted_to_depth > scope_depth, switch recording to the pending block,
        call defer_executor for matching defers, then restore recording.

        Args:
            scope_depth: The depth of the scope being exited.
            defer_executor: callable(callable_obj, func_ref, args, node) that
                            emits a single defer call (appends PCIR to current block).
        """
        if not self._pending_gotos:
            return

        for pending in self._pending_gotos:
            if pending.goto_scope_depth < scope_depth:
                continue
            if pending.emitted_to_depth <= scope_depth:
                continue

            defers_for_this_scope = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in pending.defer_snapshot
                if depth == scope_depth
            ]

            if not defers_for_this_scope:
                pending.emitted_to_depth = scope_depth
                continue

            logger.debug(f"CFG: emitting {len(defers_for_this_scope)} defers for pending goto "
                        f"'{pending.label_name}' at scope depth {scope_depth}")

            # Mark as emitted BEFORE executing (prevents re-entrant emission)
            pending.emitted_to_depth = scope_depth

            # Switch recording to pending block
            saved = self.resume_recording_at(pending.block)

            for callable_obj, func_ref, args, defer_node in defers_for_this_scope:
                defer_executor(callable_obj, func_ref, args, defer_node)

            # Update pending.block (defers may create new blocks via inlining)
            pending.block = self._block_map.get(self._current_block_id)

            # Restore recording position
            self.restore_recording(saved)

    def resolve_pending_gotos(self, label_name, target_block, target_parent_depth, defer_executor):
        """Resolve all pending gotos for label_name.

        Emits remaining defers + branch into each pending goto's block.

        Args:
            label_name: The label being defined.
            target_block: The begin_block of the label (branch target).
            target_parent_depth: The label's parent_scope_depth.
            defer_executor: callable(callable_obj, func_ref, args, node).
        """
        resolved = []

        for pending in self._pending_gotos:
            if pending.label_name != label_name:
                continue

            if pending.is_goto_end:
                logger.error(f"Internal error: forward goto_end to '{label_name}'",
                            node=pending.node)
                continue

            # Switch recording to pending block
            saved = self.resume_recording_at(pending.block)

            # Emit defers that haven't been emitted yet
            defers_to_execute = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in pending.defer_snapshot
                if target_parent_depth < depth < pending.emitted_to_depth
            ]
            logger.debug(f"CFG: resolving forward goto '{pending.label_name}': "
                        f"executing {len(defers_to_execute)} defers "
                        f"(goto_scope={pending.goto_scope_depth}, "
                        f"target_parent={target_parent_depth}, "
                        f"emitted_to_depth={pending.emitted_to_depth})")

            for callable_obj, func_ref, args, defer_node in defers_to_execute:
                defer_executor(callable_obj, func_ref, args, defer_node)

            # Emit branch to target
            self.branch(target_block, kind='goto')

            # Restore recording position
            self.restore_recording(saved)

            resolved.append(pending)

        for item in resolved:
            self._pending_gotos.remove(item)

    def get_unresolved_pending_gotos(self):
        """Return list of unresolved pending gotos (for consistency check)."""
        return list(self._pending_gotos)

    def reset_pending_gotos(self):
        """Clear pending gotos."""
        self._pending_gotos = []

    def finalize(self):
        """Finalize CFG construction.

        1. Connect fall-through CFG blocks to exit block
        2. Compute loop headers
        3. Set _finalized flag

        Does NOT emit any IR (that's emit_ir()'s job).
        """
        # For blocks without explicit terminator, connect to exit block
        reachable = self._cfg.get_reachable_blocks()
        for block_id in reachable:
            if block_id == self._exit_block.id:
                continue
            successors = self._cfg.get_successors(block_id)
            if not successors:
                self._cfg.add_edge(block_id, self._exit_block.id, kind='fallthrough')
                logger.debug(f"CFG: fallthrough at CFGBlock({block_id}) -> exit({self._exit_block.id})")

        # Compute loop headers
        self._cfg.compute_loop_headers()

        self._finalized = True
        logger.debug(f"CFG finalized: {self._cfg}")

    def emit_ir(self):
        """Replay PCIR instructions to produce actual LLVM IR.

        Steps:
        1. Create real ir.Blocks for each CFG block (except entry which exists)
        2. Build a mapping from CFG block ID -> real ir.Block
        3. Replay PCIR instructions block by block using block creation order
        4. Resolve VReg arguments and map results
        5. Replay PHI incoming edges and switch cases
        6. Patch unterminated blocks
        """
        assert self._finalized, "Must call finalize() before emit_ir()"
        assert not self._emitted, "emit_ir() already called"

        # Step 1: Create real ir.Blocks for all CFG blocks (except entry and exit)
        real_block_map: Dict[int, ir.Block] = {}

        # Entry block already exists (it's the function's entry block)
        entry_ir_block = self._real_builder.block
        if entry_ir_block is None:
            # Fallback: get entry from function
            entry_ir_block = self.current_function.entry_basic_block
        real_block_map[self._entry_block.id] = entry_ir_block
        entry_ir_block._cfg_block_id = self._entry_block.id

        # Create ir.Blocks for all other CFG blocks (in creation order, skip exit block)
        for block_id in sorted(self._cfg.blocks.keys()):
            if block_id == self._entry_block.id:
                continue
            if block_id == self._exit_block.id:
                continue
            label = self._block_labels.get(block_id, f"block_{block_id}")
            ir_block = self.current_function.append_basic_block(label)
            ir_block._cfg_block_id = block_id
            real_block_map[block_id] = ir_block

        # Step 2: Update block_map to point to real blocks
        self._block_map = real_block_map

        # Step 3: Replay PCIR instructions block by block
        # Use block creation order (sorted by ID)
        phi_nodes: List[Tuple[VRegPhi, Any]] = []  # (vphi, real_phi) pairs
        switch_nodes: List[Tuple[VRegSwitch, Any]] = []  # (vswitch, real_switch) pairs

        for block_id in sorted(self._cfg.blocks.keys()):
            if block_id == self._exit_block.id:
                continue
            if block_id not in real_block_map:
                continue

            ir_block = real_block_map[block_id]
            cfg_block = self._cfg.blocks[block_id]

            # Position at the real block
            self._real_builder.position_at_end(ir_block)

            for inst in cfg_block.pcir:
                self._replay_inst(inst, real_block_map, phi_nodes, switch_nodes)

        # Step 4: Replay PHI incoming edges
        for vphi, real_phi in phi_nodes:
            for value, block in vphi._incomings:
                resolved_value = resolve_arg(value, real_block_map)
                resolved_block = resolve_arg(block, real_block_map)
                real_phi.add_incoming(resolved_value, resolved_block)

        # Step 5: Replay switch cases
        for vswitch, real_switch in switch_nodes:
            for value, block in vswitch._cases:
                resolved_value = resolve_arg(value, real_block_map)
                resolved_block = resolve_arg(block, real_block_map)
                real_switch.add_case(resolved_value, resolved_block)

        # Step 6: Patch unterminated IR blocks
        raw = self._real_builder.ir_builder
        ret_type = self.current_function.function_type.return_type
        blocks_cleaned = 0
        for block in self.current_function.blocks:
            if not block.is_terminated:
                raw.position_at_end(block)
                if isinstance(ret_type, ir.VoidType):
                    raw.ret_void()
                else:
                    raw.ret(ir.Constant(ret_type, ir.Undefined))
                blocks_cleaned += 1
        if blocks_cleaned > 0:
            logger.debug(
                f"Patched {blocks_cleaned} unterminated blocks in "
                f"{self.current_function.name}"
            )

        self._emitted = True
        logger.debug(f"PCIR emit_ir completed for {self._func_name}")

    def _replay_inst(self, inst: PCIRInst, block_map: Dict[int, ir.Block],
                     phi_nodes: list, switch_nodes: list):
        """Replay a single PCIR instruction on the real builder."""
        op = inst.op

        # Resolve all args
        resolved_args = tuple(resolve_arg(a, block_map) for a in inst.args)
        resolved_kwargs = {k: resolve_arg(v, block_map) for k, v in inst.kwargs.items()}

        # Call the real builder method
        method = getattr(self._real_builder, op)
        result = method(*resolved_args, **resolved_kwargs)

        # Map result to VReg
        if inst.result is not None:
            if isinstance(inst.result, VRegPhi):
                inst.result.set_resolved(result)
                phi_nodes.append((inst.result, result))
            elif isinstance(inst.result, VRegSwitch):
                inst.result.set_resolved(result)
                switch_nodes.append((inst.result, result))
            elif isinstance(inst.result, VReg):
                inst.result.set_resolved(result)

    def run_cfg_linear_check(self, initial_snapshot: LinearSnapshot = None) -> bool:
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
        lines = [f"\n=== CFG for {self._func_name} ==="]
        lines.append(f"Entry: B{self._cfg.entry_id}, Exit: B{self._cfg.exit_id}")
        lines.append(f"Blocks: {len(self._cfg.blocks)}, Edges: {len(self._cfg.edges)}")

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
            ir_name = getattr(ir_block, 'name', '?')

            lines.append(f"  B{block_id}{marker_str} [IR: {ir_name}]:")

            if block.stmts:
                for stmt in block.stmts:
                    stmt_str = ast.unparse(stmt)[:60]
                    lines.append(f"    {stmt_str}")
            else:
                lines.append("    (no statements recorded)")

            if block.linear_events:
                lines.append(f"    Linear events: {len(block.linear_events)}")
                for event in block.linear_events:
                    lines.append(f"      {event}")

            if block.pcir:
                lines.append(f"    PCIR: {len(block.pcir)} instructions")

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
