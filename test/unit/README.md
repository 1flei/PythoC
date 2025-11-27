# PC Compiler Unit Tests

This directory contains comprehensive unit tests for the PC compiler project.

## Test Coverage

The unit tests cover the following modules:

### Core Modules
- **test_ir_value.py** - Tests for ValueRef wrapper and IR value utilities
- **test_type_converter.py** - Tests for the unified type conversion system
- **test_registry.py** - Tests for the unified compilation registry
- **test_type_resolver.py** - Tests for type annotation parsing and resolution
- **test_utils.py** - Tests for utility functions

### Type System
- **test_builtin_types.py** - Tests for builtin type entities (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, ptr, array)
- **test_builtin_functions.py** - Tests for builtin function entities (sizeof, etc.)

### Context and Compilation
- **test_context.py** - Tests for compilation context management
- **test_decorators.py** - Tests for the @compile decorator
- **test_meta.py** - Tests for meta programming utilities

### Library Bindings
- **test_libc_bindings.py** - Tests for libc function bindings (printf, malloc, free, memcpy, strlen)

## Running Tests

### Run all unit tests:
```bash
cd /data/workspace/pc
PYTHONPATH=/data/workspace/pc conda run -n py python test/run_unit_tests.py
```

### Run with pytest (more detailed output):
```bash
cd /data/workspace/pc
PYTHONPATH=/data/workspace/pc conda run -n py python -m pytest test/unit/ -v
```

### Run specific test file:
```bash
PYTHONPATH=/data/workspace/pc conda run -n py python -m pytest test/unit/test_type_converter.py -v
```

### Run specific test:
```bash
PYTHONPATH=/data/workspace/pc conda run -n py python -m pytest test/unit/test_type_converter.py::TestTypeConverter::test_int_to_int_sext -v
```

## Test Statistics

- **Total Tests**: 105
- **Test Files**: 11
- **All tests passing**: ✓

## Test Organization

Tests are organized by module and functionality:

1. **Value and Type Tests** - Core IR value handling and type conversion
2. **Registry Tests** - Variable, function, struct, and builtin entity registration
3. **Type System Tests** - Type parsing, resolution, and builtin types
4. **Context Tests** - Compilation context and scope management
5. **Utility Tests** - Helper functions and LLVM utilities
6. **Integration Tests** - Decorator and compilation workflow tests

## Adding New Tests

When adding new tests:

1. Create a new test file in `test/unit/` with prefix `test_`
2. Import `unittest` and the module to test
3. Create test classes inheriting from `unittest.TestCase`
4. Name test methods with prefix `test_`
5. Use descriptive docstrings for each test
6. Run tests to ensure they pass

Example:
```python
import unittest
from pc.your_module import YourClass

class TestYourClass(unittest.TestCase):
    """Test YourClass functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
```

## Continuous Integration

These tests should be run:
- Before committing changes
- As part of CI/CD pipeline
- After refactoring
- When adding new features

## Notes

- Tests use the `py` conda environment
- PYTHONPATH must be set to project root
- Some tests create temporary LLVM IR for validation
- All tests are isolated and don't depend on external state
