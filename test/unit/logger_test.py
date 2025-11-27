#!/usr/bin/env python3
"""
Test logger system with different levels and line numbers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.logger import logger, set_log_level, LogLevel

print("=" * 60)
print("Testing Logger System")
print("=" * 60)
print()

# Test DEBUG level (disabled by default)
print("1. DEBUG messages (should not appear by default):")
logger.debug("This is a debug message", var="x", value=42)
logger.debug("Another debug message")
print()

# Test WARNING level (enabled by default, with line numbers)
print("2. WARNING messages (with file:line):")
logger.warning("This is a warning message", code="W001")
logger.warning("Type mismatch detected", expected="i32", got="i64")
logger.warning("Deprecated feature used")
print()

# Test ERROR level (with line numbers)
print("3. ERROR messages (with file:line):")
logger.error("Compilation failed", reason="undefined variable")
logger.error("Type error", message="Cannot convert float to int")
print()

# Enable DEBUG and test again
print("4. Enabling DEBUG level:")
set_log_level(LogLevel.DEBUG)
logger.debug("Now debug is enabled", status="active")
logger.debug("Variable lookup", name="foo", type="i32")
logger.warning("Warning still shows with line number")
logger.error("Error still shows with line number")
print()

print("=" * 60)
print("Logger test completed!")
print("=" * 60)
