import pytest
from torchcp.utils.registry import Registry

def test_registry_init():
    """Test registry initialization"""
    registry = Registry("test")
    assert registry._name == "test"
    assert registry._obj_map == {}

def test_registry_basic_registration():
    """Test basic object registration"""
    registry = Registry("test")
    
    class DummyClass:
        pass
    
    # Test function call registration
    registry.register(DummyClass)
    assert "DummyClass" in registry.registered_names()
    assert registry.get("DummyClass") == DummyClass

def test_registry_decorator():
    """Test decorator registration"""
    registry = Registry("test")
    
    @registry.register()
    class TestClass:
        pass
    
    assert "TestClass" in registry.registered_names()
    assert registry.get("TestClass") == TestClass

def test_duplicate_registration():
    """Test handling of duplicate registration"""
    registry = Registry("test")
    
    class DummyClass:
        pass
    
    registry.register(DummyClass)
    
    # Try to register the same class again
    with pytest.raises(KeyError, match="was already registered"):
        registry.register(DummyClass)
    
    # Test force registration
    registry.register(DummyClass, force=True)
    assert registry.get("DummyClass") == DummyClass

def test_get_nonexistent():
    """Test getting non-existent object"""
    registry = Registry("test")
    
    with pytest.raises(KeyError, match="does not exist"):
        registry.get("NonexistentClass")

def test_registered_names():
    """Test listing registered names"""
    registry = Registry("test")
    
    # Register multiple classes
    class Class1:
        pass
    
    class Class2:
        pass
    
    registry.register(Class1)
    registry.register(Class2)
    
    names = registry.registered_names()
    assert isinstance(names, list)
    assert "Class1" in names
    assert "Class2" in names
    assert len(names) == 2