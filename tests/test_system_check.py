"""Basic system check test module.

Provides simple test to verify test infrastructure is working.
"""
from typing import Any


def test_system_responsive() -> None:
    """Verify that the test system is responsive and working.

    This is a basic test that always passes to confirm the testing
    infrastructure is properly configured.
    """
    assert True, "System check test passed"


def test_python_basics() -> None:
    """Verify basic Python functionality is working as expected.

    Tests fundamental Python operations to ensure environment is correct.
    """
    # Test basic arithmetic
    assert 1 + 1 == 2, "Basic addition failed"
    
    # Test string operations
    test_str = "hello"
    assert test_str.upper() == "HELLO", "String methods failed"
    
    # Test list operations
    test_list = [1, 2, 3]
    assert len(test_list) == 3, "List operations failed"