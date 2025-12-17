"""
Simple logging system for PC compiler

Usage:
    from pythoc.logger import logger, set_log_level, LogLevel
    
    # Set log level (default: WARNING)
    set_log_level(LogLevel.DEBUG)
    
    # Or use environment variable PC_LOG_LEVEL:
    # export PC_LOG_LEVEL=0
    # export PC_LOG_LEVEL=1
    # export PC_LOG_LEVEL=2
    
    # Log messages
    logger.debug("Variable lookup", var_name="x", type="i32")
    logger.warning("Type mismatch detected", expected="i32", got="i64")
    logger.error("Compilation failed", reason="undefined variable")
    
Log levels (in order of severity):
    DEBUG: Detailed diagnostic information (includes file:line, disabled by default)
    WARNING: Warning messages that don't stop compilation (includes file:line)
    ERROR: Error messages for compilation failures (includes file:line)

Environment variables:
    PC_LOG_LEVEL: 0=DEBUG, 1=WARNING, 2=ERROR (default: 1)
    PC_LOG_MODULES: Comma-separated module path prefixes to enable debug for.
                    Automatically detects caller module from stack frame.
                    Examples:
                      PC_LOG_MODULES=decorators          # pythoc/decorators/*
                      PC_LOG_MODULES=effect              # pythoc/effect.py
                      PC_LOG_MODULES=ast_visitor.calls   # pythoc/ast_visitor/calls.py
                      PC_LOG_MODULES=decorators,effect   # Multiple modules
                      PC_LOG_MODULES=-build,-registry    # Exclude specific modules
                    Default: all modules enabled
    PC_RAISE_ON_ERROR: Set to "1" to raise exceptions on errors (for tests)

Error handling:
    By default, logger.error() exits with code 1 (production mode).
    For tests that need to catch exceptions, use:
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)
"""

import sys
import os
import inspect
import ast
from enum import IntEnum
from typing import Any, Optional, Union


class LogLevel(IntEnum):
    """Log levels in order of severity"""
    DEBUG = 0
    WARNING = 1
    ERROR = 2


class Logger:
    """Simple logger for PC compiler"""
    
    def __init__(self, level: LogLevel = LogLevel.WARNING):
        self.level = self._get_level_from_env(level)
        self.enabled = True
        self.current_source_file = None  # Track current source file being compiled
        self.current_line_offset = 0  # Line offset for AST nodes (function start line - 1)
        # Default: exit on error (production mode). Set to True for tests that expect exceptions.
        self.raise_on_error = os.environ.get('PC_RAISE_ON_ERROR') == '1'
        # Module filtering for debug messages
        self._enabled_modules, self._excluded_modules = self._parse_modules_from_env()
    
    def _get_level_from_env(self, default_level: LogLevel) -> LogLevel:
        """Get log level from environment variable PC_LOG_LEVEL or use default"""
        env_level = int(os.environ.get('PC_LOG_LEVEL', '1'))
        return env_level
    
    def _parse_modules_from_env(self) -> tuple:
        """Parse PC_LOG_MODULES environment variable.
        
        Returns:
            (enabled_modules, excluded_modules): Lists of module prefixes.
            If enabled_modules is None, all modules are enabled (unless excluded).
        """
        env_modules = os.environ.get('PC_LOG_MODULES', '').strip()
        if not env_modules or env_modules.lower() == 'all':
            return None, []  # All enabled, none excluded
        
        enabled = []
        excluded = []
        
        for mod in env_modules.split(','):
            mod = mod.strip()
            if not mod:
                continue
            if mod.startswith('-'):
                # Exclusion: -build means exclude 'build'
                excluded.append(mod[1:])
            else:
                enabled.append(mod)
        
        # If only exclusions specified, enable all except excluded
        if not enabled and excluded:
            return None, excluded
        
        return enabled if enabled else None, excluded
    
    def _get_caller_module(self, skip_frames: int = 4) -> Optional[str]:
        """Get the module path of the caller (relative to pythoc/).
        
        Args:
            skip_frames: Number of frames to skip (default 4 for debug->_log->_get_caller_module)
        
        Returns module path like 'decorators.compile' or 'ast_visitor.calls'.
        """
        try:
            frame = inspect.currentframe()
            # Skip internal logger frames to get to actual caller
            for _ in range(skip_frames):
                if frame is not None:
                    frame = frame.f_back
            
            if frame is None:
                return None
            
            filename = frame.f_code.co_filename
            
            # Extract module path relative to pythoc/
            if 'pythoc/' in filename:
                # Get path after pythoc/
                rel_path = filename.split('pythoc/')[-1]
                # Remove .py extension and convert / to .
                if rel_path.endswith('.py'):
                    rel_path = rel_path[:-3]
                module_path = rel_path.replace('/', '.')
                return module_path
            
            return None
        except Exception:
            return None
    
    def _is_module_enabled(self, caller_module: Optional[str]) -> bool:
        """Check if a module is enabled for debug logging.
        
        Args:
            caller_module: Module path like 'decorators.compile' or None.
        
        Returns:
            True if the module is enabled, False otherwise.
        """
        if caller_module is None:
            # Unknown caller: enabled unless we have specific module filters
            return self._enabled_modules is None
        
        # Check exclusions first
        for excluded in self._excluded_modules:
            if caller_module.startswith(excluded) or caller_module == excluded:
                return False
        
        # If enabled_modules is None, all modules are enabled (except excluded)
        if self._enabled_modules is None:
            return True
        
        # Otherwise, check if caller matches any enabled prefix
        for enabled in self._enabled_modules:
            if caller_module.startswith(enabled) or caller_module == enabled:
                return True
        
        return False
    
    def set_modules(self, enabled: Optional[list] = None, excluded: Optional[list] = None):
        """Programmatically set enabled/excluded modules.
        
        Args:
            enabled: List of module prefixes to enable (None = all modules)
            excluded: List of module prefixes to exclude
        """
        self._enabled_modules = enabled
        self._excluded_modules = excluded or []
    
    def set_level(self, level: LogLevel):
        """Set minimum log level to display"""
        self.level = level
    
    def enable(self):
        """Enable logging output"""
        self.enabled = True
    
    def disable(self):
        """Disable all logging output"""
        self.enabled = False
    
    def set_source_file(self, filename: Optional[str]):
        """Set the current source file being compiled"""
        self.current_source_file = filename
    
    def set_line_offset(self, offset: int):
        """Set line offset for AST nodes.
        
        When parsing function source with ast.parse(), line numbers start from 1.
        To get the actual line number in the source file, we need to add the
        function's starting line number minus 1.
        
        Args:
            offset: The starting line number of the function minus 1
                   (e.g., if function starts at line 10, offset should be 9)
        """
        self.current_line_offset = offset
    
    def set_source_context(self, filename: Optional[str], line_offset: int = 0):
        """Set both source file and line offset at once.
        
        Args:
            filename: Source file path
            line_offset: Starting line number of the function minus 1
        """
        self.current_source_file = filename
        self.current_line_offset = line_offset
    
    def set_raise_on_error(self, raise_on_error: bool):
        """Set whether to raise exception on error or exit.
        
        Args:
            raise_on_error: If True, raise exception on error (for tests).
                           If False, exit with code 1 (default, for production).
        """
        self.raise_on_error = raise_on_error
    
    def _get_source_location(self, node: Optional[Union[ast.AST, Any]] = None) -> str:
        """Get source code location from AST node or caller info"""
        # Priority 1: If node is provided and has location info, use it
        if node is not None and isinstance(node, ast.AST):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                filename = self.current_source_file or "source"
                # Shorten path for readability
                if '/pc/' in filename:
                    filename = filename.split('/pc/')[-1]
                elif '/' in filename:
                    filename = filename.split('/')[-1]
                # Add line offset to get actual line number in source file
                actual_lineno = node.lineno + self.current_line_offset
                return f"{filename}:{actual_lineno}:{node.col_offset}"
        
        # Priority 2: Fall back to Python caller location
        try:
            frame = inspect.currentframe()
            for _ in range(4):  # Skip _log, debug/warning/error, and this method
                if frame is not None:
                    frame = frame.f_back
            
            if frame is not None:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                if '/pc/' in filename:
                    filename = filename.split('/pc/')[-1]
                    filename = 'pc/' + filename
                return f"{filename}:{lineno}"
        except:
            pass
        return "unknown"
    
    def _format_message(self, level_name: str, msg: str, node: Optional[ast.AST] = None, 
                       include_location: bool = True, caller_module: Optional[str] = None, 
                       **kwargs) -> str:
        """Format log message with optional key-value pairs and location"""
        location = ""
        if include_location:
            location = f" [{self._get_source_location(node)}]"
        
        # Include module in output for DEBUG if available
        module_prefix = f"[{caller_module}] " if caller_module else ""
        
        if not kwargs:
            return f"[{level_name}]{location} {module_prefix}{msg}"
        
        # Format key-value pairs (exclude 'node' from kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'node'}
        if not filtered_kwargs:
            return f"[{level_name}]{location} {module_prefix}{msg}"
        
        details = " ".join(f"{k}={v}" for k, v in filtered_kwargs.items())
        return f"[{level_name}]{location} {module_prefix}{msg}: {details}"
    
    def _log(self, level: LogLevel, level_name: str, msg: str, node: Optional[ast.AST] = None, 
             **kwargs):
        """Internal logging method"""
        if not self.enabled or level < self.level:
            return
        
        caller_module = None
        # For DEBUG level, check module filtering
        if level == LogLevel.DEBUG:
            # skip_frames=3: debug -> _log -> _get_caller_module
            caller_module = self._get_caller_module(skip_frames=3)
            if not self._is_module_enabled(caller_module):
                return
        
        # Include location for all log levels
        include_location = True
        formatted = self._format_message(level_name, msg, node=node, 
                                         include_location=include_location, 
                                         caller_module=caller_module, **kwargs)
        
        # Print to stderr for WARNING and ERROR, stdout for DEBUG
        output = sys.stderr if level >= LogLevel.WARNING else sys.stdout
        print(formatted, file=output)
    
    def debug(self, msg: str, node: Optional[ast.AST] = None, **kwargs):
        """Log debug message (detailed diagnostic info)
        
        Filtering is automatic based on caller module.
        Use PC_LOG_MODULES env var to filter by module path.
        
        Args:
            msg: Message to log
            node: Optional AST node for source location
            **kwargs: Additional key-value pairs to log
        """
        self._log(LogLevel.DEBUG, "DEBUG", msg, node=node, **kwargs)
    
    def warning(self, msg: str, node: Optional[ast.AST] = None, **kwargs):
        """Log warning message (non-fatal issues)
        
        Args:
            msg: Message to log
            node: Optional AST node for source location (recommended for PC code warnings)
            **kwargs: Additional key-value pairs to log
        """
        self._log(LogLevel.WARNING, "WARNING", msg, node=node, **kwargs)
    
    def error(self, msg: str, node: Optional[ast.AST] = None, 
              exc_type: type = RuntimeError, **kwargs):
        """Log error message and raise exception or exit (compilation failures)
        
        Args:
            msg: Message to log
            node: Optional AST node for source location (recommended for PC code errors)
            exc_type: Exception type to raise (default: RuntimeError)
            **kwargs: Additional key-value pairs to log
        
        Raises:
            exc_type: The specified exception type with the message (if raise_on_error=True)
        
        Note:
            By default, exits with code 1 (production mode).
            Set raise_on_error=True or PC_RAISE_ON_ERROR=1 to raise exceptions (for tests).
        """
        self._log(LogLevel.ERROR, "ERROR", msg, node=node, **kwargs)
        
        if self.raise_on_error:
            # Test mode: raise exception with full stack trace
            raise exc_type(msg)
        else:
            # Production mode (default): exit cleanly without stack trace
            sys.exit(1)


# Global logger instance
logger = Logger()


def set_log_level(level: LogLevel):
    """Set global log level"""
    logger.set_level(level)


def set_source_file(filename: Optional[str]):
    """Set the current source file being compiled (for better error messages)"""
    logger.set_source_file(filename)


def set_line_offset(offset: int):
    """Set line offset for AST nodes (function start line - 1)"""
    logger.set_line_offset(offset)


def set_source_context(filename: Optional[str], line_offset: int = 0):
    """Set both source file and line offset at once"""
    logger.set_source_context(filename, line_offset)


def set_raise_on_error(raise_on_error: bool):
    """Set whether to raise exception on error or exit.
    
    Args:
        raise_on_error: If True, raise exception on error (for tests).
                       If False, exit with code 1 (default, for production).
    """
    logger.set_raise_on_error(raise_on_error)


def enable_logging():
    """Enable logging globally"""
    logger.enable()


def disable_logging():
    """Disable all logging globally"""
    logger.disable()


# Convenience function for enabling debug mode
def enable_debug():
    """Enable debug logging"""
    set_log_level(LogLevel.DEBUG)
