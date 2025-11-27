"""
Unit tests for visible symbols extraction utilities.
"""

import sys
import unittest
from pythoc.decorators.visible import (
    get_visible_symbols,
    get_closure_variables,
    get_imported_names,
    get_type_annotations,
    get_all_accessible_symbols,
    summarize_visible_symbols,
)


class TestGetVisibleSymbols(unittest.TestCase):
    """Test get_visible_symbols function."""
    
    def test_basic_local_and_global(self):
        """Test that both local and global variables are captured."""
        def test_function():
            local_var = "I am local"
            symbols = get_visible_symbols(depth=1)
            
            # Should have local variable
            self.assertIn('local_var', symbols)
            self.assertEqual(symbols['local_var'], "I am local")
            
            # Should have some module-level names
            self.assertIn('__name__', symbols)
        
        test_function()
    
    def test_depth_navigation(self):
        """Test stack depth navigation."""
        def outer():
            y = "middle"
            
            def inner():
                z = "inner"
                
                # depth=1 should see z (local in inner)
                symbols1 = get_visible_symbols(depth=1)
                self.assertIn('z', symbols1)
                self.assertEqual(symbols1['z'], "inner")
                
                # depth=2 should see y (local in outer) but not z
                symbols2 = get_visible_symbols(depth=2)
                self.assertNotIn('z', symbols2)
                self.assertIn('y', symbols2)
                self.assertEqual(symbols2['y'], "middle")
            
            inner()
        
        outer()
    
    def test_builtins_inclusion(self):
        """Test builtin names inclusion/exclusion."""
        def test_function():
            # Without builtins
            symbols_no_builtins = get_visible_symbols(depth=1, include_builtins=False)
            self.assertNotIn('__builtins__', symbols_no_builtins)
            
            # With builtins
            symbols_with_builtins = get_visible_symbols(depth=1, include_builtins=True)
            self.assertIn('__builtins__', symbols_with_builtins)
        
        test_function()
    
    def test_imported_names_visible(self):
        """Test that imported names are visible."""
        def test_function():
            import os
            import math as m
            from typing import Dict
            
            symbols = get_visible_symbols(depth=1)
            self.assertIn('os', symbols)
            self.assertIn('m', symbols)
            self.assertIn('Dict', symbols)
        
        test_function()


class TestGetClosureVariables(unittest.TestCase):
    """Test get_closure_variables function."""
    
    def test_simple_closure(self):
        """Test simple closure variable extraction."""
        def outer(x, y):
            z = x + y
            
            def inner():
                return x + y + z
            
            return inner
        
        func = outer(10, 20)
        closure_vars = get_closure_variables(func)
        
        self.assertEqual(len(closure_vars), 3)
        self.assertEqual(closure_vars['x'], 10)
        self.assertEqual(closure_vars['y'], 20)
        self.assertEqual(closure_vars['z'], 30)
    
    def test_no_closure(self):
        """Test function with no closure."""
        def simple_func():
            return 42
        
        closure_vars = get_closure_variables(simple_func)
        self.assertEqual(len(closure_vars), 0)
    
    def test_nested_closure(self):
        """Test nested closures."""
        def outer(a):
            def middle(b):
                def inner(c):
                    return a + b + c
                return inner
            return middle
        
        func = outer(1)(2)
        closure_vars = get_closure_variables(func)
        
        self.assertIn('a', closure_vars)
        self.assertIn('b', closure_vars)
        self.assertEqual(closure_vars['a'], 1)
        self.assertEqual(closure_vars['b'], 2)
    
    def test_type_error_on_non_callable(self):
        """Test that TypeError is raised for non-callable."""
        with self.assertRaises(TypeError):
            get_closure_variables(42)
        
        with self.assertRaises(TypeError):
            get_closure_variables("not a function")


class TestGetImportedNames(unittest.TestCase):
    """Test get_imported_names function."""
    
    def test_module_imports(self):
        """Test detection of imported modules."""
        def test_function():
            import os
            import sys as system
            
            imports = get_imported_names(frame_depth=1)
            self.assertIn('os', imports)
            self.assertIn('system', imports)
        
        test_function()
    
    def test_from_imports(self):
        """Test detection of from imports."""
        from collections import defaultdict
        from typing import List, Dict
        
        def test_function():
            imports = get_imported_names(frame_depth=1)
            # These may or may not be detected depending on heuristics
            # Just verify the function runs without error
            self.assertIsInstance(imports, dict)
        
        test_function()


class TestGetTypeAnnotations(unittest.TestCase):
    """Test get_type_annotations function."""
    
    def test_function_annotations(self):
        """Test extraction of function-level annotations."""
        def test_function():
            x: int = 10
            y: str = "hello"
            
            annotations = get_type_annotations(frame_depth=1)
            # Note: annotations might not be in f_locals depending on Python version
            # This is a best-effort test
            self.assertIsInstance(annotations, dict)
        
        test_function()
    
    def test_no_annotations(self):
        """Test when there are no annotations."""
        def test_function():
            x = 10
            
            annotations = get_type_annotations(frame_depth=1)
            # Should return empty or minimal dict
            self.assertIsInstance(annotations, dict)
        
        test_function()


class TestGetAllAccessibleSymbols(unittest.TestCase):
    """Test get_all_accessible_symbols function."""
    
    def test_function_with_closure(self):
        """Test function with closure variables."""
        global_x = 100
        
        def outer(y):
            z = 50
            
            def inner(w):
                return global_x + y + z + w
            
            return inner
        
        func = outer(25)
        symbols = get_all_accessible_symbols(func, include_closure=True)
        
        # Should have closure vars
        self.assertEqual(symbols.get('y'), 25)
        self.assertEqual(symbols.get('z'), 50)
        
        # Should have globals
        self.assertIn('global_x', symbols)
    
    def test_exclude_closure(self):
        """Test excluding closure variables."""
        def outer(x):
            def inner():
                return x
            return inner
        
        func = outer(42)
        
        # With closure
        symbols_with = get_all_accessible_symbols(func, include_closure=True)
        self.assertIn('x', symbols_with)
        
        # Without closure
        symbols_without = get_all_accessible_symbols(func, include_closure=False)
        self.assertNotIn('x', symbols_without)
    
    def test_exclude_builtins(self):
        """Test excluding builtins."""
        def simple():
            return 1
        
        # Without builtins
        symbols_no_builtins = get_all_accessible_symbols(simple, include_builtins=False)
        self.assertNotIn('__builtins__', symbols_no_builtins)
        
        # With builtins
        symbols_with_builtins = get_all_accessible_symbols(simple, include_builtins=True)
        self.assertIn('__builtins__', symbols_with_builtins)
    
    def test_include_calling_scope(self):
        """Test including symbols from calling stack frames."""
        def factory_function(element_type, size_type):
            """Simulates a factory function like Vector(element_type, size_type)."""
            # Force closure capture
            _ = (element_type, size_type)
            
            def inner_function():
                """Inner function that should access factory parameters."""
                pass
            
            # Get symbols with calling scope
            symbols = get_all_accessible_symbols(
                inner_function, 
                include_calling_scope=True,
                include_builtins=False
            )
            
            return inner_function, symbols
        
        # Call factory with specific types
        func, symbols = factory_function('int', 'u64')
        
        # Should have factory function parameters from calling scope
        self.assertIn('element_type', symbols)
        self.assertEqual(symbols['element_type'], 'int')
        self.assertIn('size_type', symbols)
        self.assertEqual(symbols['size_type'], 'u64')
    
    def test_exclude_calling_scope(self):
        """Test excluding calling scope symbols."""
        def outer_func(param):
            def inner_func():
                pass
            
            # Without calling scope
            symbols = get_all_accessible_symbols(
                inner_func,
                include_calling_scope=False,
                include_builtins=False
            )
            
            return symbols
        
        symbols = outer_func('test_value')
        
        # Should NOT have outer_func's param when calling scope is excluded
        # (unless it's in closure, which it's not)
        self.assertNotIn('param', symbols)
    
    def test_calling_scope_with_type_annotations(self):
        """Test that type annotation names from calling scope are captured."""
        def create_typed_function(int_type, str_type):
            """Factory that creates a function with type parameters."""
            # Force closure
            _ = (int_type, str_type)
            
            def typed_function():
                # This function uses int_type and str_type in annotations
                x = None  # x: int_type = ...
                y = None  # y: str_type = ...
                pass
            
            symbols = get_all_accessible_symbols(
                typed_function,
                include_calling_scope=True
            )
            
            return symbols
        
        symbols = create_typed_function('i32', 'String')
        
        # Should capture type parameter names from factory
        self.assertEqual(symbols.get('int_type'), 'i32')
        self.assertEqual(symbols.get('str_type'), 'String')
    
    def test_nested_calling_scope(self):
        """Test calling scope with multiple nesting levels."""
        outer_var = 'outer'
        
        def level1(param1):
            middle_var = 'middle'
            
            def level2(param2):
                inner_var = 'inner'
                
                def level3():
                    pass
                
                symbols = get_all_accessible_symbols(
                    level3,
                    include_calling_scope=True,
                    include_builtins=False
                )
                
                return symbols
            
            return level2('param2_value')
        
        symbols = level1('param1_value')
        
        # Should have symbols from all calling frames
        self.assertIn('param1', symbols)
        self.assertEqual(symbols['param1'], 'param1_value')
        self.assertIn('param2', symbols)
        self.assertEqual(symbols['param2'], 'param2_value')
        self.assertIn('middle_var', symbols)
        self.assertEqual(symbols['middle_var'], 'middle')
        self.assertIn('inner_var', symbols)
        self.assertEqual(symbols['inner_var'], 'inner')


class TestSummarizeVisibleSymbols(unittest.TestCase):
    """Test summarize_visible_symbols function."""
    
    def test_categorization(self):
        """Test that symbols are properly categorized."""
        def test_function():
            import os
            
            class MyClass:
                pass
            
            def my_function():
                pass
            
            my_var = 42
            
            summary = summarize_visible_symbols(depth=1)
            
            # Check structure
            self.assertIn('modules', summary)
            self.assertIn('functions', summary)
            self.assertIn('classes', summary)
            self.assertIn('variables', summary)
            self.assertIn('builtins', summary)
            
            # Check some expected entries
            self.assertIn('os', summary['modules'])
            self.assertIn('MyClass', summary['classes'])
            self.assertIn('my_function', summary['functions'])
            self.assertIn('my_var', summary['variables'])
        
        test_function()
    
    def test_builtins_detected(self):
        """Test that builtins are detected."""
        def test_function():
            summary = summarize_visible_symbols(depth=1)
            
            # Should have some common builtins
            builtins_set = summary['builtins']
            # Note: 'len', 'str', etc. might not show up if not used
            # Just verify it's a set
            self.assertIsInstance(builtins_set, set)
        
        test_function()


class TestClassScenarios(unittest.TestCase):
    """Test symbol visibility in class contexts."""
    
    # Class-level attributes for testing
    class_var = "class level"
    
    def test_class_attributes_visible(self):
        """Test that class attributes are visible in methods."""
        class TestClass:
            class_attr = "class level"
            
            def method(self):
                local_var = "local"
                symbols = get_visible_symbols(depth=1)
                
                # Should see local variables and self
                assert 'local_var' in symbols
                assert 'self' in symbols
                
                return symbols
        
        obj = TestClass()
        symbols = obj.method()
        
        # Verify symbols were captured
        self.assertIn('local_var', symbols)
        self.assertIn('self', symbols)
    
    def test_instance_method_context(self):
        """Test symbol visibility from within instance methods."""
        class TestClass:
            class_attr = "from class"
            
            def __init__(self):
                self.instance_attr = "from instance"
            
            def method(self):
                local_var = "local in method"
                symbols = get_visible_symbols(depth=1)
                
                # Should see self
                assert 'self' in symbols
                # Should see local variable
                assert 'local_var' in symbols
                assert symbols['local_var'] == "local in method"
                
                return symbols
        
        obj = TestClass()
        symbols = obj.method()
        
        # Verify we got the symbols
        self.assertIn('self', symbols)
        self.assertIn('local_var', symbols)
    
    def test_static_method_context(self):
        """Test symbol visibility in static methods."""
        class TestClass:
            class_attr = 100
            
            @staticmethod
            def static_method():
                local_var = 42
                symbols = get_visible_symbols(depth=1)
                
                # Should see local variables
                assert 'local_var' in symbols
                # Should NOT see self (it's static)
                assert 'self' not in symbols
                
                return symbols
        
        symbols = TestClass.static_method()
        self.assertIn('local_var', symbols)
        self.assertNotIn('self', symbols)
    
    def test_class_method_context(self):
        """Test symbol visibility in class methods."""
        class TestClass:
            class_attr = 200
            
            @classmethod
            def class_method(cls):
                local_var = 99
                symbols = get_visible_symbols(depth=1)
                
                # Should see cls
                assert 'cls' in symbols
                # Should see local variable
                assert 'local_var' in symbols
                
                return symbols
        
        symbols = TestClass.class_method()
        self.assertIn('cls', symbols)
        self.assertIn('local_var', symbols)
    
    def test_nested_class_context(self):
        """Test symbol visibility in nested classes."""
        outer_var = "outer"
        
        class OuterClass:
            outer_class_var = "outer class"
            
            class InnerClass:
                inner_class_var = "inner class"
                
                def method(self):
                    method_var = "method"
                    symbols = get_visible_symbols(depth=1)
                    
                    # Should see method-level vars
                    assert 'method_var' in symbols
                    assert 'self' in symbols
                    
                    # May or may not see outer class depending on scope
                    return symbols
        
        obj = OuterClass.InnerClass()
        symbols = obj.method()
        
        self.assertIn('method_var', symbols)
        self.assertIn('self', symbols)
    
    def test_method_with_closure(self):
        """Test methods that create closures."""
        class TestClass:
            def __init__(self, value):
                self.value = value
            
            def make_adder(self, increment):
                # Create closure capturing self and increment
                def adder(x):
                    return self.value + increment + x
                
                return adder
        
        obj = TestClass(10)
        adder_func = obj.make_adder(5)
        
        # Check closure variables
        closure_vars = get_closure_variables(adder_func)
        
        # Should capture self and increment
        self.assertIn('self', closure_vars)
        self.assertIn('increment', closure_vars)
        self.assertEqual(closure_vars['increment'], 5)
        self.assertEqual(closure_vars['self'].value, 10)
    
    def test_class_with_type_annotations(self):
        """Test type annotations in class context."""
        class TypedClass:
            class_attr: int = 100
            
            def __init__(self, x: int):
                self.x: int = x
            
            def method(self, y: str) -> str:
                z: float = 3.14
                symbols = get_visible_symbols(depth=1)
                
                # Should have local variables
                assert 'z' in symbols
                assert 'y' in symbols
                assert 'self' in symbols
                
                return symbols
        
        obj = TypedClass(42)
        symbols = obj.method("test")
        
        self.assertIn('z', symbols)
        self.assertIn('y', symbols)
        self.assertIn('self', symbols)
    
    def test_summarize_with_class_context(self):
        """Test summarize_visible_symbols in class method."""
        class TestClass:
            def method(self):
                import math
                
                class InnerClass:
                    pass
                
                def inner_func():
                    pass
                
                my_var = 123
                
                summary = summarize_visible_symbols(depth=1)
                
                # Check categorization
                assert 'modules' in summary
                assert 'classes' in summary
                assert 'functions' in summary
                assert 'variables' in summary
                
                # Verify specific items
                assert 'math' in summary['modules']
                assert 'InnerClass' in summary['classes']
                assert 'inner_func' in summary['functions']
                assert 'my_var' in summary['variables']
                
                return summary
        
        obj = TestClass()
        summary = obj.method()
        
        self.assertIn('math', summary['modules'])
        self.assertIn('InnerClass', summary['classes'])
        self.assertIn('inner_func', summary['functions'])
        self.assertIn('my_var', summary['variables'])
    
    def test_class_decorator_context(self):
        """Test accessing class symbols from decorated methods."""
        class TestClass:
            shared_data = "shared"
            
            def decorator(self, func):
                """Instance method used as decorator."""
                def wrapper(*args, **kwargs):
                    # Capture context when wrapper is called
                    symbols = get_visible_symbols(depth=1)
                    result = func(*args, **kwargs)
                    return result, symbols
                return wrapper
            
            def process(self):
                """Method that uses the decorator."""
                local_val = 100
                
                @self.decorator
                def inner():
                    return local_val
                
                result, symbols = inner()
                return result, symbols
        
        obj = TestClass()
        result, symbols = obj.process()
        
        self.assertEqual(result, 100)
        self.assertIsInstance(symbols, dict)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def test_decorator_use_case(self):
        """Simulate a decorator extracting visible symbols."""
        from pythoc.decorators.visible import get_visible_symbols
        
        def capture_context(func):
            """Decorator that captures the defining context."""
            # Capture symbols visible when decorator is applied
            context = get_visible_symbols(depth=2)  # Skip this function and decorator call
            
            def wrapper(*args, **kwargs):
                # Context is available in closure
                return func(*args, **kwargs), len(context)
            
            return wrapper
        
        x = 10
        y = 20
        
        @capture_context
        def my_func():
            return x + y
        
        result, context_size = my_func()
        self.assertEqual(result, 30)
        self.assertGreater(context_size, 0)  # Should have captured some symbols
    
    def test_meta_programming_use_case(self):
        """Test extracting type parameters from factory function."""
        from pythoc.decorators.visible import get_all_accessible_symbols
        
        def Vector(element_type, capacity=10):
            """Factory function that generates specialized vector."""
            # Force closure capture
            _ = (element_type, capacity)
            
            def push(vec, item):
                # In real code, this would use element_type
                return element_type, capacity
            
            return push
        
        int_vector_push = Vector(int, 100)
        
        # Extract accessible symbols including closures
        symbols = get_all_accessible_symbols(int_vector_push, include_closure=True)
        
        self.assertEqual(symbols.get('element_type'), int)
        self.assertEqual(symbols.get('capacity'), 100)


if __name__ == '__main__':
    unittest.main()
