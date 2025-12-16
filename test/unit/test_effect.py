# -*- coding: utf-8 -*-
"""
Unit tests for the effect system.

Tests the Effect class, EffectNamespace, and EffectContext
without involving actual compilation.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc.effect import Effect, EffectNamespace, EffectContext, effect


class MockRNG:
    """Mock RNG implementation for testing"""
    def __init__(self, value=42):
        self.value = value
        self.seed_value = None
    
    def seed(self, s):
        self.seed_value = s
    
    def next(self):
        return self.value


class MockAllocator:
    """Mock allocator implementation for testing"""
    def __init__(self):
        self.allocated = []
    
    def alloc(self, size):
        self.allocated.append(size)
        return size  # Return size as mock pointer
    
    def free(self, ptr):
        if ptr in self.allocated:
            self.allocated.remove(ptr)


class TestEffectNamespace(unittest.TestCase):
    """Test EffectNamespace class"""
    
    def test_create_unbound_namespace(self):
        """Test creating an unbound namespace"""
        ns = EffectNamespace('test')
        self.assertEqual(repr(ns), "<EffectNamespace 'test' (unbound)>")
    
    def test_create_bound_namespace(self):
        """Test creating a bound namespace"""
        mock = MockRNG()
        ns = EffectNamespace('rng', mock)
        self.assertIn('MockRNG', repr(ns))
    
    def test_attribute_access_bound(self):
        """Test accessing attributes on bound namespace"""
        mock = MockRNG(100)
        ns = EffectNamespace('rng', mock)
        
        # Access method
        self.assertEqual(ns.next(), 100)
        
        # Access attribute
        self.assertEqual(ns.value, 100)
    
    def test_attribute_access_unbound_raises(self):
        """Test that accessing attributes on unbound namespace raises"""
        ns = EffectNamespace('rng')
        
        with self.assertRaises(AttributeError) as ctx:
            _ = ns.next
        
        self.assertIn("has no implementation bound", str(ctx.exception))
    
    def test_missing_attribute_raises(self):
        """Test that accessing missing attribute raises"""
        mock = MockRNG()
        ns = EffectNamespace('rng', mock)
        
        with self.assertRaises(AttributeError) as ctx:
            _ = ns.nonexistent
        
        self.assertIn("has no attribute 'nonexistent'", str(ctx.exception))
    
    def test_set_impl(self):
        """Test _set_impl method"""
        ns = EffectNamespace('test')
        mock = MockRNG()
        ns._set_impl(mock)
        
        self.assertEqual(ns._get_impl(), mock)
        self.assertEqual(ns.next(), 42)


class TestEffect(unittest.TestCase):
    """Test Effect class"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_direct_assignment(self):
        """Test direct assignment (effect.xxx = impl)"""
        mock = MockRNG()
        self.effect.rng = mock
        
        self.assertTrue(self.effect.has_effect('rng'))
        self.assertEqual(self.effect.get_effect_impl('rng'), mock)
        self.assertTrue(self.effect.is_direct_assignment('rng'))
    
    def test_default_setting(self):
        """Test effect.default() method"""
        mock = MockRNG()
        self.effect.default(rng=mock)
        
        self.assertTrue(self.effect.has_effect('rng'))
        self.assertEqual(self.effect.get_effect_impl('rng'), mock)
        # default() does NOT create direct assignment
        self.assertFalse(self.effect.is_direct_assignment('rng'))
    
    def test_attribute_access_creates_unbound(self):
        """Test that accessing undefined effect creates unbound namespace"""
        ns = self.effect.undefined_effect
        
        self.assertIsInstance(ns, EffectNamespace)
        self.assertFalse(self.effect.has_effect('undefined_effect'))
    
    def test_delete_effect(self):
        """Test deleting an effect"""
        mock = MockRNG()
        self.effect.rng = mock
        
        self.assertTrue(self.effect.has_effect('rng'))
        
        del self.effect.rng
        
        self.assertFalse(self.effect.has_effect('rng'))
        self.assertFalse(self.effect.is_direct_assignment('rng'))
    
    def test_list_effects(self):
        """Test list_effects method"""
        mock_rng = MockRNG()
        mock_alloc = MockAllocator()
        
        self.effect.rng = mock_rng
        self.effect.allocator = mock_alloc
        
        effects = self.effect.list_effects()
        
        self.assertIn('rng', effects)
        self.assertIn('allocator', effects)
        self.assertTrue(effects['rng'])
        self.assertTrue(effects['allocator'])
    
    def test_method_forwarding(self):
        """Test that method calls are forwarded to implementation"""
        mock = MockRNG(999)
        self.effect.rng = mock
        
        result = self.effect.rng.next()
        self.assertEqual(result, 999)
        
        self.effect.rng.seed(123)
        self.assertEqual(mock.seed_value, 123)


class TestEffectContext(unittest.TestCase):
    """Test EffectContext (with effect(...) block)"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_context_override(self):
        """Test that context manager overrides effects"""
        default_rng = MockRNG(1)
        override_rng = MockRNG(2)
        
        self.effect.default(rng=default_rng)
        
        self.assertEqual(self.effect.rng.next(), 1)
        
        with self.effect(rng=override_rng, suffix="test"):
            self.assertEqual(self.effect.rng.next(), 2)
        
        # Should restore after context
        self.assertEqual(self.effect.rng.next(), 1)
    
    def test_context_requires_suffix(self):
        """Test that context with overrides requires suffix"""
        mock = MockRNG()
        
        with self.assertRaises(ValueError) as ctx:
            with self.effect(rng=mock):
                pass
        
        self.assertIn("suffix", str(ctx.exception))
    
    def test_context_suffix_only(self):
        """Test that context with only suffix is allowed"""
        # Should not raise
        with self.effect(suffix="namespace"):
            pass
    
    def test_context_suffix_stack(self):
        """Test suffix stack management"""
        self.assertIsNone(self.effect._get_current_suffix())
        
        with self.effect(suffix="outer"):
            self.assertEqual(self.effect._get_current_suffix(), "outer")
            
            with self.effect(suffix="inner"):
                self.assertEqual(self.effect._get_current_suffix(), "inner")
            
            self.assertEqual(self.effect._get_current_suffix(), "outer")
        
        self.assertIsNone(self.effect._get_current_suffix())
    
    def test_direct_assignment_immune_to_override(self):
        """Test that direct assignments cannot be overridden by context"""
        secure_rng = MockRNG(999)
        weak_rng = MockRNG(1)
        
        # Direct assignment - should be immune to override
        self.effect.rng = secure_rng
        
        self.assertEqual(self.effect.rng.next(), 999)
        
        # Try to override with context
        with self.effect(rng=weak_rng, suffix="weak"):
            # Should still use secure_rng because direct assignment is immune
            self.assertEqual(self.effect.rng.next(), 999)
        
        self.assertEqual(self.effect.rng.next(), 999)
    
    def test_default_can_be_overridden(self):
        """Test that effect.default() can be overridden by context"""
        default_rng = MockRNG(1)
        override_rng = MockRNG(2)
        
        # Set via default - should be overridable
        self.effect.default(rng=default_rng)
        
        self.assertEqual(self.effect.rng.next(), 1)
        
        # Override with context
        with self.effect(rng=override_rng, suffix="override"):
            self.assertEqual(self.effect.rng.next(), 2)
        
        # Restored after context
        self.assertEqual(self.effect.rng.next(), 1)
    
    def test_multiple_overrides(self):
        """Test overriding multiple effects at once"""
        rng1 = MockRNG(1)
        rng2 = MockRNG(2)
        alloc1 = MockAllocator()
        alloc2 = MockAllocator()
        
        self.effect.default(rng=rng1, allocator=alloc1)
        
        with self.effect(rng=rng2, allocator=alloc2, suffix="multi"):
            self.assertEqual(self.effect.rng.next(), 2)
            self.effect.allocator.alloc(100)
            self.assertIn(100, alloc2.allocated)
            self.assertNotIn(100, alloc1.allocated)
        
        # Both restored
        self.assertEqual(self.effect.rng.next(), 1)


class TestGlobalEffectSingleton(unittest.TestCase):
    """Test the global effect singleton"""
    
    def tearDown(self):
        """Clean up global effect state"""
        # Remove any effects we added
        for name in list(effect._effects.keys()):
            try:
                delattr(effect, name)
            except:
                pass
    
    def test_singleton_exists(self):
        """Test that global effect singleton exists"""
        from pythoc.effect import effect as global_effect
        self.assertIsInstance(global_effect, Effect)
    
    def test_singleton_usage(self):
        """Test basic usage of global singleton"""
        mock = MockRNG(42)
        effect.test_rng = mock
        
        self.assertEqual(effect.test_rng.next(), 42)
        
        del effect.test_rng


class TestEffectResolutionHelpers(unittest.TestCase):
    """Test compile-time resolution helper functions"""
    
    def setUp(self):
        """Set up test effects"""
        self.mock_rng = MockRNG(42)
        effect.test_rng = self.mock_rng
    
    def tearDown(self):
        """Clean up"""
        try:
            del effect.test_rng
        except:
            pass
    
    def test_get_current_effect_suffix(self):
        """Test get_current_effect_suffix helper"""
        from pythoc.effect import get_current_effect_suffix
        
        self.assertIsNone(get_current_effect_suffix())
        
        with effect(suffix="test"):
            self.assertEqual(get_current_effect_suffix(), "test")
        
        self.assertIsNone(get_current_effect_suffix())


class TestEffectResolutionOrder(unittest.TestCase):
    """Test effect resolution order (priority)"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_direct_assignment_highest_priority(self):
        """Test that direct assignment has highest priority"""
        direct_rng = MockRNG(999)
        default_rng = MockRNG(1)
        override_rng = MockRNG(2)
        
        # Set default first
        self.effect.default(rng=default_rng)
        
        # Direct assignment should override default
        self.effect.rng = direct_rng
        self.assertEqual(self.effect.rng.next(), 999)
        
        # Context override should NOT override direct assignment
        with self.effect(rng=override_rng, suffix="test"):
            self.assertEqual(self.effect.rng.next(), 999)
    
    def test_caller_override_over_default(self):
        """Test that caller override beats default"""
        default_rng = MockRNG(1)
        override_rng = MockRNG(2)
        
        self.effect.default(rng=default_rng)
        self.assertEqual(self.effect.rng.next(), 1)
        
        with self.effect(rng=override_rng, suffix="test"):
            self.assertEqual(self.effect.rng.next(), 2)
        
        # Restored after context
        self.assertEqual(self.effect.rng.next(), 1)
    
    def test_default_lowest_priority(self):
        """Test that default has lowest priority among set effects"""
        default_rng = MockRNG(1)
        
        self.effect.default(rng=default_rng)
        self.assertEqual(self.effect.rng.next(), 1)
        
        # Direct assignment should override
        direct_rng = MockRNG(999)
        self.effect.rng = direct_rng
        self.assertEqual(self.effect.rng.next(), 999)
    
    def test_resolution_order_complete(self):
        """Test complete resolution order: direct > caller > default"""
        direct_impl = MockRNG(1)
        caller_impl = MockRNG(2)
        default_impl = MockRNG(3)
        
        # Start with default
        self.effect.default(rng=default_impl)
        self.assertEqual(self.effect.rng.next(), 3)
        
        # Caller override beats default
        with self.effect(rng=caller_impl, suffix="caller"):
            self.assertEqual(self.effect.rng.next(), 2)
        
        # Direct assignment beats everything
        self.effect.rng = direct_impl
        self.assertEqual(self.effect.rng.next(), 1)
        
        # Even caller override cannot beat direct
        with self.effect(rng=caller_impl, suffix="caller"):
            self.assertEqual(self.effect.rng.next(), 1)


class TestEffectNestedContexts(unittest.TestCase):
    """Test nested effect contexts"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_nested_suffix_stack(self):
        """Test that suffix stacks correctly in nested contexts"""
        self.assertIsNone(self.effect._get_current_suffix())
        
        with self.effect(suffix="outer"):
            self.assertEqual(self.effect._get_current_suffix(), "outer")
            
            with self.effect(suffix="middle"):
                self.assertEqual(self.effect._get_current_suffix(), "middle")
                
                with self.effect(suffix="inner"):
                    self.assertEqual(self.effect._get_current_suffix(), "inner")
                
                self.assertEqual(self.effect._get_current_suffix(), "middle")
            
            self.assertEqual(self.effect._get_current_suffix(), "outer")
        
        self.assertIsNone(self.effect._get_current_suffix())
    
    def test_nested_effect_overrides(self):
        """Test nested effect overrides restore correctly"""
        rng1 = MockRNG(1)
        rng2 = MockRNG(2)
        rng3 = MockRNG(3)
        
        self.effect.default(rng=rng1)
        self.assertEqual(self.effect.rng.next(), 1)
        
        with self.effect(rng=rng2, suffix="outer"):
            self.assertEqual(self.effect.rng.next(), 2)
            
            with self.effect(rng=rng3, suffix="inner"):
                self.assertEqual(self.effect.rng.next(), 3)
            
            self.assertEqual(self.effect.rng.next(), 2)
        
        self.assertEqual(self.effect.rng.next(), 1)
    
    def test_nested_mixed_effects(self):
        """Test nested contexts with different effects"""
        rng1 = MockRNG(1)
        rng2 = MockRNG(2)
        alloc1 = MockAllocator()
        alloc2 = MockAllocator()
        
        self.effect.default(rng=rng1, allocator=alloc1)
        
        # Outer: override rng only
        with self.effect(rng=rng2, suffix="outer"):
            self.assertEqual(self.effect.rng.next(), 2)
            self.assertEqual(self.effect.get_effect_impl('allocator'), alloc1)
            
            # Inner: override allocator only
            with self.effect(allocator=alloc2, suffix="inner"):
                # rng should still be rng2 from outer context
                self.assertEqual(self.effect.rng.next(), 2)
                self.assertEqual(self.effect.get_effect_impl('allocator'), alloc2)
            
            # allocator restored to alloc1
            self.assertEqual(self.effect.get_effect_impl('allocator'), alloc1)
        
        # Both restored
        self.assertEqual(self.effect.rng.next(), 1)
        self.assertEqual(self.effect.get_effect_impl('allocator'), alloc1)


class TestEffectTransitiveDependencies(unittest.TestCase):
    """Test effect propagation through transitive dependencies"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_effect_propagates_to_inner_access(self):
        """Test that effects propagate when accessed from inner scope"""
        outer_rng = MockRNG(100)
        
        with self.effect(rng=outer_rng, suffix="outer"):
            # Simulating inner module accessing effect
            # In real usage, this would be from an imported module
            impl = self.effect.get_effect_impl('rng')
            self.assertEqual(impl, outer_rng)
            self.assertEqual(impl.next(), 100)
    
    def test_direct_assignment_blocks_propagation(self):
        """Test that direct assignment blocks effect propagation"""
        outer_rng = MockRNG(100)
        inner_rng = MockRNG(999)
        
        # Simulate inner module doing direct assignment
        self.effect.rng = inner_rng
        
        # Outer context tries to override
        with self.effect(rng=outer_rng, suffix="outer"):
            # Should still be inner_rng due to direct assignment
            impl = self.effect.get_effect_impl('rng')
            self.assertEqual(impl, inner_rng)
    
    def test_default_allows_propagation(self):
        """Test that default() allows effect propagation from caller"""
        outer_rng = MockRNG(100)
        default_rng = MockRNG(1)
        
        # Simulate inner module setting default
        self.effect.default(rng=default_rng)
        
        # Outer context overrides
        with self.effect(rng=outer_rng, suffix="outer"):
            impl = self.effect.get_effect_impl('rng')
            self.assertEqual(impl, outer_rng)


class TestEffectModuleCaching(unittest.TestCase):
    """Test that different effect configurations produce different results"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_different_effects_different_results(self):
        """Test that different effect configs give different implementations"""
        fast_rng = MockRNG(1)
        crypto_rng = MockRNG(999)
        
        results = []
        
        with self.effect(rng=fast_rng, suffix="fast"):
            results.append(('fast', self.effect.get_effect_impl('rng')))
        
        with self.effect(rng=crypto_rng, suffix="crypto"):
            results.append(('crypto', self.effect.get_effect_impl('rng')))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], fast_rng)
        self.assertEqual(results[1][1], crypto_rng)
        self.assertNotEqual(results[0][1], results[1][1])
    
    def test_same_suffix_same_effect(self):
        """Test that same suffix returns same effect within context"""
        rng = MockRNG(42)
        
        with self.effect(rng=rng, suffix="test"):
            impl1 = self.effect.get_effect_impl('rng')
            impl2 = self.effect.get_effect_impl('rng')
            self.assertIs(impl1, impl2)
    
    def test_effect_hash_for_caching(self):
        """Test that effect configurations can be hashed for caching"""
        rng1 = MockRNG(1)
        rng2 = MockRNG(2)
        
        # Simulate what compiler would do for caching
        config1 = ('test_module', 'test_func', id(rng1), 'suffix1')
        config2 = ('test_module', 'test_func', id(rng2), 'suffix2')
        config3 = ('test_module', 'test_func', id(rng1), 'suffix1')  # Same as config1
        
        cache = {}
        cache[config1] = 'compiled_v1'
        cache[config2] = 'compiled_v2'
        
        # Same config should hit cache
        self.assertEqual(cache.get(config3), 'compiled_v1')
        # Different config should be different
        self.assertNotEqual(cache.get(config1), cache.get(config2))


class TestEffectFlags(unittest.TestCase):
    """Test using effects as compile-time flags"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_boolean_flag_default(self):
        """Test boolean flag with default value"""
        self.effect.default(DEBUG=False)
        
        self.assertFalse(self.effect.get_effect_impl('DEBUG'))
    
    def test_boolean_flag_override(self):
        """Test boolean flag override in context"""
        self.effect.default(DEBUG=False)
        
        with self.effect(DEBUG=True, suffix="debug"):
            self.assertTrue(self.effect.get_effect_impl('DEBUG'))
        
        self.assertFalse(self.effect.get_effect_impl('DEBUG'))
    
    def test_numeric_flag(self):
        """Test numeric flag"""
        self.effect.default(BUFFER_SIZE=4096, MAX_CONNECTIONS=100)
        
        self.assertEqual(self.effect.get_effect_impl('BUFFER_SIZE'), 4096)
        self.assertEqual(self.effect.get_effect_impl('MAX_CONNECTIONS'), 100)
        
        with self.effect(BUFFER_SIZE=8192, suffix="large"):
            self.assertEqual(self.effect.get_effect_impl('BUFFER_SIZE'), 8192)
            # MAX_CONNECTIONS unchanged
            self.assertEqual(self.effect.get_effect_impl('MAX_CONNECTIONS'), 100)
    
    def test_string_flag(self):
        """Test string flag"""
        self.effect.default(LOG_LEVEL="INFO")
        
        self.assertEqual(self.effect.get_effect_impl('LOG_LEVEL'), "INFO")
        
        with self.effect(LOG_LEVEL="DEBUG", suffix="verbose"):
            self.assertEqual(self.effect.get_effect_impl('LOG_LEVEL'), "DEBUG")


class TestEffectExceptionSafety(unittest.TestCase):
    """Test that effects are properly restored on exceptions"""
    
    def setUp(self):
        """Create a fresh Effect instance for each test"""
        self.effect = Effect()
    
    def test_restore_on_exception(self):
        """Test that effects are restored when exception occurs in context"""
        default_rng = MockRNG(1)
        override_rng = MockRNG(2)
        
        self.effect.default(rng=default_rng)
        
        try:
            with self.effect(rng=override_rng, suffix="test"):
                self.assertEqual(self.effect.rng.next(), 2)
                raise ValueError("test exception")
        except ValueError:
            pass
        
        # Effect should be restored despite exception
        self.assertEqual(self.effect.rng.next(), 1)
    
    def test_restore_suffix_on_exception(self):
        """Test that suffix is restored when exception occurs"""
        self.assertIsNone(self.effect._get_current_suffix())
        
        try:
            with self.effect(suffix="test"):
                self.assertEqual(self.effect._get_current_suffix(), "test")
                raise RuntimeError("test")
        except RuntimeError:
            pass
        
        self.assertIsNone(self.effect._get_current_suffix())


if __name__ == '__main__':
    unittest.main()
