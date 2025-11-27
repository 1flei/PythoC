import os


class OutputManager:
    """
    Manages compilation groups and output file generation.
    
    A compilation group represents a set of functions compiled into the same
    .ll/.o/.so files. Functions are grouped by:
    - (source_file, None, None) for normal functions
    - (definition_file, scope, suffix) for suffix-specialized functions
    """
    
    def __init__(self):
        """Initialize output manager with empty group registry."""
        # Key: (source_file, scope_name, suffix) tuple
        # Value: dict with compiler, wrappers, file paths, etc.
        self._pending_groups = {}
    
    def get_or_create_group(self, group_key, compiler, ir_file, obj_file, so_file, 
                           source_file, skip_codegen=False):
        """
        Get existing group or create a new one.
        
        Args:
            group_key: (source_file, scope, suffix) tuple identifying the group
            compiler: LLVMCompiler instance for this group
            ir_file: Path to output .ll file
            obj_file: Path to output .o file
            so_file: Path to output .so file
            source_file: Original source file path
            skip_codegen: If True, skip code generation (files are up-to-date)
        
        Returns:
            dict: Group info with keys: compiler, wrappers, ir_file, obj_file, so_file, 
                  skip_codegen, source_file
        """
        if group_key not in self._pending_groups:
            # If skipping codegen, try to load existing IR
            if skip_codegen and os.path.exists(ir_file):
                try:
                    compiler.load_ir_from_file(ir_file)
                except Exception:
                    # If loading fails, force recompilation
                    skip_codegen = False
            
            self._pending_groups[group_key] = {
                'compiler': compiler,
                'wrappers': [],
                'source_file': source_file,
                'ir_file': ir_file,
                'obj_file': obj_file,
                'so_file': so_file,
                'skip_codegen': skip_codegen
            }
        
        return self._pending_groups[group_key]
    
    def add_wrapper_to_group(self, group_key, wrapper):
        """
        Add a compiled function wrapper to its group.
        
        Args:
            group_key: Group identifier
            wrapper: Function wrapper to add
        """
        if group_key in self._pending_groups:
            group = self._pending_groups[group_key]
            if not group['skip_codegen']:
                group['wrappers'].append(wrapper)
    
    def flush_all(self):
        """
        Flush all pending output files to disk.
        
        This should be called before native execution to ensure
        all .ll and .o files have been written.
        """
        for group_key, group in self._pending_groups.items():
            if group.get('skip_codegen', False):
                # Already up-to-date
                continue
            
            if not group.get('wrappers'):
                # No functions compiled
                continue
            
            compiler = group['compiler']
            
            # Verify module
            if not compiler.verify_module():
                source_file, scope, suffix = group_key
                raise RuntimeError(f"Module verification failed for group {group_key}")
            
            # Optimize
            opt_level = int(os.environ.get('PC_OPT_LEVEL', '2'))
            compiler.optimize_module(optimization_level=opt_level)
            
            # Write files
            compiler.save_ir_to_file(group['ir_file'])
            compiler.compile_to_object(group['obj_file'])
        
        # Don't clear pending groups - they serve as metadata cache for subsequent runs
    
    def get_group(self, group_key):
        """
        Get group info by key.
        
        Args:
            group_key: Group identifier
        
        Returns:
            dict or None: Group info if exists
        """
        return self._pending_groups.get(group_key)
    
    def clear_all(self):
        """Clear all pending groups (for testing/reset)."""
        self._pending_groups.clear()


# Global singleton instance
_output_manager = OutputManager()


def get_output_manager():
    """Get the global OutputManager singleton."""
    return _output_manager


def flush_all_pending_outputs():
    """
    Convenience function to flush all pending outputs.
    
    This is the main entry point used by the runtime.
    """
    _output_manager.flush_all()
