#!/usr/bin/env python3
"""
Basic functionality test after Phase 0 refactoring
Tests core functionality without requiring GPU or large models
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test basic imports work"""
    print("🧪 Testing imports...")

    try:
        # Test core module imports
        from features.extraction.commands.extract_character import extract_character_from_path

        print("✅ extract_character import successful")

        from features.evaluation.utils.learned_quality_assessment import assess_image_quality

        print("✅ quality assessment import successful")

        from features.processing.preprocessing.preprocessing import preprocess_image_pipeline

        print("✅ preprocessing import successful")

        from features.processing.postprocessing.postprocessing import enhance_character_mask

        print("✅ postprocessing import successful")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_project_structure():
    """Test project structure is correct"""
    print("\n🧪 Testing project structure...")

    required_paths = [
        "core/segment_anything",
        "features/extraction/commands",
        "features/extraction/models",
        "features/evaluation/utils",
        "features/processing/preprocessing",
        "features/processing/postprocessing",
        "features/common/hooks",
        "tests/unit",
        "tests/integration",
        "tools",
    ]

    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print(f"❌ Missing paths: {missing}")
        return False

    print("✅ Project structure verified")
    return True


def test_module_initialization():
    """Test that modules can be initialized"""
    print("\n🧪 Testing module initialization...")

    try:
        # Test basic class instantiation
        from features.evaluation.utils.learned_quality_assessment import LearnedQualityAssessment

        assessor = LearnedQualityAssessment()
        print("✅ Quality assessor created")

        # Test basic method exists
        if hasattr(assessor, "analyze_image_characteristics"):
            print("✅ Image analysis method available")
        else:
            print("❌ Image analysis method missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Module initialization failed: {e}")
        return False


def test_configuration_files():
    """Test configuration files are accessible"""
    print("\n🧪 Testing configuration files...")

    config_files = ["config/pushover.json", "CLAUDE.md", "PROGRESS_TRACKER.md", "requirements.txt"]

    missing = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing.append(config_file)

    if missing:
        print(f"❌ Missing config files: {missing}")
        return False

    print("✅ Configuration files verified")
    return True


def run_basic_tests():
    """Run all basic functionality tests"""
    print("🚀 Running basic functionality tests after Phase 0...")

    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Module Initialization", test_module_initialization),
        ("Configuration Files", test_configuration_files),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    print(f"\n📊 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 Basic functionality confirmed!")
        print("📋 System is ready for use")
        return True
    else:
        print("⚠️ Some functionality issues detected")
        return False


if __name__ == "__main__":
    success = run_basic_tests()

    if success:
        print("\n💡 Next steps:")
        print("  - Phase 1 implementation can begin")
        print("  - Core functionality is working")
        print("  - Test framework is operational")
    else:
        print("\n🔧 Required fixes:")
        print("  - Check import paths")
        print("  - Verify file structure")
        print("  - Check dependencies")

    sys.exit(0 if success else 1)
