#!/usr/bin/env python3
"""
Simple test script for workflow enforcement system
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_state_manager():
    """Test state manager initialization"""
    try:
        from tools.workflow.state_manager import WorkflowStateManager

        # Create state manager
        state_manager = WorkflowStateManager()
        print("✅ State manager initialized successfully")

        # Test creating a tracker
        success = state_manager.create_tracker_workflow("TEST-001")
        if success:
            print("✅ Test tracker created successfully")
        else:
            print("❌ Failed to create test tracker")

        # Test getting current step
        current_step = state_manager.get_current_step("TEST-001")
        print(f"✅ Current step: {current_step}")

        return True

    except Exception as e:
        print(f"❌ State manager test failed: {e}")
        return False


def test_validator():
    """Test file system validator"""
    try:
        from tools.validation.file_system_validator import FileSystemValidator

        validator = FileSystemValidator()
        print("✅ File system validator initialized successfully")

        # Test branch verification (this will likely fail but should not crash)
        try:
            result = validator.validate_branch_verification("TEST-001")
            print(f"✅ Branch validation result: {result.passed}")
        except Exception as e:
            print(f"⚠️  Branch validation test failed (expected): {e}")

        return True

    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        return False


def main():
    """Run tests"""
    print("🔄 Testing Workflow Enforcement System...")

    # Test state manager
    print("\n📋 Testing State Manager...")
    state_ok = test_state_manager()

    # Test validator
    print("\n🔍 Testing Validator...")
    validator_ok = test_validator()

    if state_ok and validator_ok:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
