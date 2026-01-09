"""
CFG Data Structures

Core data structures for Control Flow Graphs:
- CFGBlock: Basic block containing a sequence of statements
- CFGEdge: Directed edge representing control flow
- CFG: Complete control flow graph for a function

Design: PCIR blocks contain AST nodes (not a fully flattened IR).
Control flow is explicit in CFG (blocks + edges), statements/expressions
are AST nodes stored in blocks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Iterator, Tuple, Any
import ast
import copy


@dataclass
class CFGBlock:
    """Basic block - maximal sequence of linearly-executed statements
    
    A basic block has:
    - Single entry point (first statement)
    - Single exit point (last statement)
    - All statements execute sequentially
    
    Attributes:
        id: Unique identifier for this block
        stmts: List of AST statements in this block
        source_range: Optional (start_line, end_line) for error reporting
        linear_events: List of linear type events for CFG-based checking
    """
    id: int
    stmts: List[ast.stmt] = field(default_factory=list)
    source_range: Optional[Tuple[int, int]] = None
    linear_events: List[Any] = field(default_factory=list)  # List[LinearEvent]
    
    def __repr__(self) -> str:
        stmt_count = len(self.stmts)
        return f"CFGBlock(id={self.id}, stmts={stmt_count})"
    
    def is_empty(self) -> bool:
        """Check if block has no statements"""
        return len(self.stmts) == 0
    
    def get_first_line(self) -> Optional[int]:
        """Get line number of first statement"""
        if self.stmts and hasattr(self.stmts[0], 'lineno'):
            return self.stmts[0].lineno
        return None
    
    def get_last_line(self) -> Optional[int]:
        """Get line number of last statement"""
        if self.stmts and hasattr(self.stmts[-1], 'lineno'):
            return self.stmts[-1].lineno
        return None
    
    def append_stmt(self, stmt: ast.stmt):
        """Append a statement to this block"""
        self.stmts.append(stmt)
    
    def extend_stmts(self, stmts: List[ast.stmt]):
        """Extend this block with multiple statements"""
        self.stmts.extend(stmts)


@dataclass
class CFGEdge:
    """Directed edge representing control flow between blocks
    
    Attributes:
        source_id: ID of source block
        target_id: ID of target block
        kind: Type of edge:
            - 'sequential': Normal fall-through
            - 'branch_true': Conditional branch (condition true)
            - 'branch_false': Conditional branch (condition false)
            - 'loop_back': Back edge to loop header
            - 'loop_exit': Exit from loop
            - 'break': Break statement
            - 'continue': Continue statement
            - 'return': Return statement
        condition: AST expression for conditional edges
        condition_value: Compile-time constant value (if known)
    """
    source_id: int
    target_id: int
    kind: str = 'sequential'
    condition: Optional[ast.expr] = None
    condition_value: Optional[bool] = None
    
    def __repr__(self) -> str:
        cond_str = ""
        if self.condition_value is not None:
            cond_str = f", const={self.condition_value}"
        elif self.condition is not None:
            cond_str = ", has_cond"
        return f"CFGEdge({self.source_id}->{self.target_id}, {self.kind}{cond_str})"
    
    def is_conditional(self) -> bool:
        """Check if this is a conditional edge"""
        return self.kind in ('branch_true', 'branch_false')
    
    def is_back_edge(self) -> bool:
        """Check if this is a loop back edge"""
        return self.kind == 'loop_back'


@dataclass
class CFG:
    """Control Flow Graph for a function
    
    A CFG represents the control flow structure of a function as a directed graph
    where nodes are basic blocks and edges represent possible control flow paths.
    
    Attributes:
        func_name: Name of the function
        entry_id: ID of entry block
        exit_id: ID of exit block
        blocks: Mapping from block ID to CFGBlock
        edges: List of all edges
        return_blocks: IDs of blocks containing return statements
        loop_headers: IDs of loop header blocks (computed after construction)
    """
    func_name: str
    entry_id: int = 0
    exit_id: int = 0
    blocks: Dict[int, CFGBlock] = field(default_factory=dict)
    edges: List[CFGEdge] = field(default_factory=list)
    return_blocks: List[int] = field(default_factory=list)
    loop_headers: Set[int] = field(default_factory=set)
    
    # Internal counter for block IDs
    _next_block_id: int = field(default=0, repr=False)
    
    def _get_next_id(self) -> int:
        """Get next unique block ID"""
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id
    
    def add_block(self, stmts: Optional[List[ast.stmt]] = None) -> CFGBlock:
        """Add a new block to the CFG
        
        Args:
            stmts: Optional list of statements for the block
            
        Returns:
            The newly created CFGBlock
        """
        block = CFGBlock(id=self._get_next_id(), stmts=stmts or [])
        self.blocks[block.id] = block
        return block
    
    def add_edge(self, source_id: int, target_id: int,
                 kind: str = 'sequential', **kwargs) -> CFGEdge:
        """Add an edge between blocks
        
        Args:
            source_id: Source block ID
            target_id: Target block ID
            kind: Edge type
            **kwargs: Additional edge attributes (condition, condition_value)
            
        Returns:
            The newly created CFGEdge
        """
        edge = CFGEdge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            **kwargs
        )
        self.edges.append(edge)
        return edge
    
    def remove_edge(self, edge: CFGEdge):
        """Remove an edge from the CFG"""
        if edge in self.edges:
            self.edges.remove(edge)
    
    def get_block(self, block_id: int) -> Optional[CFGBlock]:
        """Get block by ID"""
        return self.blocks.get(block_id)
    
    def get_successors(self, block_id: int) -> List[CFGEdge]:
        """Get all outgoing edges from a block"""
        return [e for e in self.edges if e.source_id == block_id]
    
    def get_predecessors(self, block_id: int) -> List[CFGEdge]:
        """Get all incoming edges to a block"""
        return [e for e in self.edges if e.target_id == block_id]
    
    def get_successor_blocks(self, block_id: int) -> List[CFGBlock]:
        """Get all successor blocks"""
        return [self.blocks[e.target_id] for e in self.get_successors(block_id)
                if e.target_id in self.blocks]
    
    def get_predecessor_blocks(self, block_id: int) -> List[CFGBlock]:
        """Get all predecessor blocks"""
        return [self.blocks[e.source_id] for e in self.get_predecessors(block_id)
                if e.source_id in self.blocks]
    
    def is_merge_point(self, block_id: int) -> bool:
        """Check if block is a merge point (multiple predecessors excluding back edges)"""
        preds = [e for e in self.get_predecessors(block_id) if e.kind != 'loop_back']
        return len(preds) > 1
    
    def compute_loop_headers(self):
        """Identify loop header blocks (targets of back edges)"""
        self.loop_headers.clear()
        for edge in self.edges:
            if edge.kind == 'loop_back':
                self.loop_headers.add(edge.target_id)
    
    def topological_order(self) -> List[CFGBlock]:
        """Return blocks in topological order (reverse post-order DFS)
        
        This ordering ensures that all predecessors of a block (except back edges)
        are visited before the block itself, which is required for dataflow analysis.
        
        Returns:
            List of CFGBlocks in topological order
        """
        visited: Set[int] = set()
        post_order: List[int] = []
        
        def dfs(block_id: int):
            if block_id in visited:
                return
            visited.add(block_id)
            
            for edge in self.get_successors(block_id):
                # Skip back edges to avoid infinite recursion
                if edge.kind != 'loop_back':
                    dfs(edge.target_id)
            
            post_order.append(block_id)
        
        # Start DFS from entry
        dfs(self.entry_id)
        
        # Reverse post-order gives topological order
        result = []
        for block_id in reversed(post_order):
            if block_id in self.blocks:
                result.append(self.blocks[block_id])
        
        return result
    
    def iter_blocks(self) -> Iterator[CFGBlock]:
        """Iterate over all blocks"""
        return iter(self.blocks.values())
    
    def iter_edges(self) -> Iterator[CFGEdge]:
        """Iterate over all edges"""
        return iter(self.edges)
    
    def get_reachable_blocks(self) -> Set[int]:
        """Get IDs of all blocks reachable from entry"""
        reachable: Set[int] = set()
        worklist = [self.entry_id]
        
        while worklist:
            block_id = worklist.pop()
            if block_id in reachable:
                continue
            reachable.add(block_id)
            
            for edge in self.get_successors(block_id):
                if edge.target_id not in reachable:
                    worklist.append(edge.target_id)
        
        return reachable
    
    def validate(self) -> List[str]:
        """Validate CFG structure, return list of errors
        
        Checks:
        - Entry and exit blocks exist
        - All edges reference valid blocks
        - Entry has no predecessors
        """
        errors = []
        
        if self.entry_id not in self.blocks:
            errors.append(f"Entry block {self.entry_id} not found")
        
        if self.exit_id not in self.blocks:
            errors.append(f"Exit block {self.exit_id} not found")
        
        for edge in self.edges:
            if edge.source_id not in self.blocks:
                errors.append(f"Edge source {edge.source_id} not found")
            if edge.target_id not in self.blocks:
                errors.append(f"Edge target {edge.target_id} not found")
        
        # Entry should have no predecessors
        entry_preds = self.get_predecessors(self.entry_id)
        if entry_preds:
            errors.append(f"Entry block has {len(entry_preds)} predecessors")
        
        return errors
    
    def splice(self, at_block_id: int, callee_cfg: 'CFG', 
               result_var: Optional[str] = None) -> Dict[int, int]:
        """Splice callee's CFG into this CFG at the given block
        
        This is the core operation for inline/yield/closure expansion.
        
        Args:
            at_block_id: Block ID where to splice (call site block)
            callee_cfg: CFG to splice in
            result_var: Variable name for return value (if any)
            
        Returns:
            Mapping of callee block IDs to new block IDs in this CFG
        """
        # 1. Copy callee's blocks with fresh IDs
        id_map: Dict[int, int] = {}
        for old_id, block in callee_cfg.blocks.items():
            if old_id == callee_cfg.exit_id:
                continue  # Don't copy exit block
            new_block = self.add_block(copy.deepcopy(block.stmts))
            id_map[old_id] = new_block.id
        
        # 2. Create continuation block
        cont_block = self.add_block()
        
        # 3. Transform return statements to assignments
        for ret_block_id in callee_cfg.return_blocks:
            if ret_block_id not in id_map:
                continue
            new_id = id_map[ret_block_id]
            block = self.blocks[new_id]
            # Last stmt should be return
            if block.stmts and isinstance(block.stmts[-1], ast.Return):
                ret_stmt = block.stmts[-1]
                if ret_stmt.value and result_var:
                    block.stmts[-1] = ast.Assign(
                        targets=[ast.Name(id=result_var, ctx=ast.Store())],
                        value=ret_stmt.value
                    )
                else:
                    block.stmts.pop()  # Remove return with no value
        
        # 4. Copy callee's edges with remapped IDs
        for edge in callee_cfg.edges:
            if edge.source_id == callee_cfg.exit_id or edge.target_id == callee_cfg.exit_id:
                continue
            if edge.source_id in id_map and edge.target_id in id_map:
                self.add_edge(
                    id_map[edge.source_id],
                    id_map[edge.target_id],
                    edge.kind,
                    condition=copy.deepcopy(edge.condition) if edge.condition else None,
                    condition_value=edge.condition_value
                )
        
        # 5. Connect: current block -> callee entry
        self.add_edge(at_block_id, id_map[callee_cfg.entry_id])
        
        # 6. Connect: callee returns -> continuation
        for ret_block_id in callee_cfg.return_blocks:
            if ret_block_id in id_map:
                self.add_edge(id_map[ret_block_id], cont_block.id)
        
        # 7. Store continuation block ID for caller to use
        id_map['_continuation'] = cont_block.id
        
        return id_map
    
    def to_dot(self) -> str:
        """Generate DOT format for visualization
        
        Returns:
            DOT format string for graphviz
        """
        lines = [f'digraph "{self.func_name}" {{']
        lines.append('  node [shape=box];')
        
        # Add nodes
        for block_id, block in self.blocks.items():
            label_parts = [f"B{block_id}"]
            if block_id == self.entry_id:
                label_parts.append("(entry)")
            if block_id == self.exit_id:
                label_parts.append("(exit)")
            if block_id in self.loop_headers:
                label_parts.append("(loop)")
            
            # Add statement summary
            if block.stmts:
                for stmt in block.stmts[:3]:  # Show first 3 statements
                    try:
                        stmt_str = ast.unparse(stmt)[:40]
                    except Exception:
                        stmt_str = type(stmt).__name__
                    label_parts.append(stmt_str.replace('"', '\\"'))
                if len(block.stmts) > 3:
                    label_parts.append(f"... ({len(block.stmts) - 3} more)")
            
            label = "\\n".join(label_parts)
            lines.append(f'  B{block_id} [label="{label}"];')
        
        # Add edges
        for edge in self.edges:
            attrs = []
            if edge.kind == 'branch_true':
                attrs.append('label="T"')
                attrs.append('color=green')
            elif edge.kind == 'branch_false':
                attrs.append('label="F"')
                attrs.append('color=red')
            elif edge.kind == 'loop_back':
                attrs.append('style=dashed')
                attrs.append('color=blue')
            elif edge.kind == 'return':
                attrs.append('label="ret"')
            elif edge.kind == 'break':
                attrs.append('label="break"')
            
            attr_str = f' [{", ".join(attrs)}]' if attrs else ''
            lines.append(f'  B{edge.source_id} -> B{edge.target_id}{attr_str};')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return (f"CFG(func={self.func_name}, blocks={len(self.blocks)}, "
                f"edges={len(self.edges)}, entry={self.entry_id}, exit={self.exit_id})")
