import os


class BuildCache:
    """Manages build cache and timestamp checking for incremental compilation."""
    
    @staticmethod
    def check_timestamp_skip(ir_file, obj_file, source_file):
        """
        Check if output files are up-to-date based on timestamps.
        
        If files are outdated, delete them to force recompilation.
        
        Args:
            ir_file: Path to .ll file
            obj_file: Path to .o file
            source_file: Path to source .py file
        
        Returns:
            bool: True if can skip compilation, False if must compile
        """
        if not os.path.exists(source_file):
            return False
        
        # If output files don't exist, must compile
        if not (os.path.exists(ir_file) and os.path.exists(obj_file)):
            return False
        
        source_mtime = os.path.getmtime(source_file)
        ir_mtime = os.path.getmtime(ir_file)
        obj_mtime = os.path.getmtime(obj_file)
        
        # If both outputs newer than source, can skip
        if ir_mtime > source_mtime and obj_mtime > source_mtime:
            return True
        
        # Otherwise, delete outdated files to force recompilation
        try:
            if os.path.exists(ir_file):
                os.remove(ir_file)
            if os.path.exists(obj_file):
                os.remove(obj_file)
        except OSError:
            pass  # Ignore deletion errors, will try to overwrite anyway
        
        return False
    
    @staticmethod
    def invalidate(source_file):
        """
        Invalidate all cached files for a source file.
        
        Args:
            source_file: Path to source file
        """
        # This could be extended to maintain an index of all cached files
        # For now, individual files handle their own cache invalidation
        pass
