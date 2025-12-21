"""
Unit tests for visible symbols extraction utilities.
"""

import unittest
from pythoc.decorators.visible import (
    get_closure_variables,
    get_all_accessible_symbols,
    capture_caller_symbols,
)


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


class TestCaptureCallerSymbols(unittest.TestCase):
    """Test capture_caller_symbols function."""
    
    def test_capture_locals_and_globals(self):
        """Test that both locals and globals are captured."""
        global_var = 'global'
        
        def outer():
            local_var = 'local'
            captured = capture_caller_symbols(depth=0)
            return captured
        
        symbols = outer()
        self.assertIn('local_var', symbols)
        self.assertEqual(symbols['local_var'], 'local')
    
    def test_locals_override_globals(self):
        """Test that locals take precedence over globals."""
        def test_func():
            # Shadow a global name
            unittest = 'shadowed'
            captured = capture_caller_symbols(depth=0)
            return captured
        
        symbols = test_func()
        self.assertEqual(symbols['unittest'], 'shadowed')


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
    
    def test_captured_symbols_parameter(self):
        """Test using captured_symbols parameter."""
        def factory_function(element_type, size_type):
            """Simulates a factory function like Vector(element_type, size_type)."""
            _ = (element_type, size_type)
            
            def inner_function():
                pass
            
            captured = capture_caller_symbols(depth=0)
            
            symbols = get_all_accessible_symbols(
                inner_function, 
                captured_symbols=captured,
                include_builtins=False
            )
            
            return inner_function, symbols
        
        func, symbols = factory_function('int', 'u64')
        
        self.assertIn('element_type', symbols)
        self.assertEqual(symbols['element_type'], 'int')
        self.assertIn('size_type', symbols)
        self.assertEqual(symbols['size_type'], 'u64')
    
    def test_captured_symbols_override(self):
        """Test that captured_symbols override other sources."""
        def inner_func():
            pass
        
        captured = {'override_var': 'captured_value'}
        symbols = get_all_accessible_symbols(
            inner_func,
            captured_symbols=captured,
            include_builtins=False
        )
        
        self.assertEqual(symbols.get('override_var'), 'captured_value')
    
    def test_closure_captures_type_annotations(self):
        """Test that closure variables capture type parameters."""
        def create_typed_function(int_type, str_type):
            def typed_function():
                return (int_type, str_type)
            
            symbols = get_all_accessible_symbols(
                typed_function,
                include_closure=True
            )
            
            return symbols
        
        symbols = create_typed_function('i32', 'String')
        
        self.assertEqual(symbols.get('int_type'), 'i32')
        self.assertEqual(symbols.get('str_type'), 'String')
    
    def test_captured_symbols_with_closure(self):
        """Test captured_symbols combined with closure variables."""
        def level1(param1):
            middle_var = 'middle'
            _ = (param1, middle_var)
            
            def level2():
                pass
            
            captured = capture_caller_symbols(depth=0)
            
            symbols = get_all_accessible_symbols(
                level2,
                captured_symbols=captured,
                include_builtins=False
            )
            
            return symbols
        
        symbols = level1('param1_value')
        
        self.assertIn('param1', symbols)
        self.assertEqual(symbols['param1'], 'param1_value')
        self.assertIn('middle_var', symbols)
        self.assertEqual(symbols['middle_var'], 'middle')


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def test_meta_programming_use_case(self):
        """Test extracting type parameters from factory function."""
        def Vector(element_type, capacity=10):
            _ = (element_type, capacity)
            
            def push(vec, item):
                return element_type, capacity
            
            return push
        
        int_vector_push = Vector(int, 100)
        
        symbols = get_all_accessible_symbols(int_vector_push, include_closure=True)
        
        self.assertEqual(symbols.get('element_type'), int)
        self.assertEqual(symbols.get('capacity'), 100)


if __name__ == '__main__':
    unittest.main()
