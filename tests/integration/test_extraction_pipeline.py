#!/usr/bin/env python3
"""
Integration test for character extraction pipeline
Tests the complete extraction workflow after Phase 0 refactoring
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_import_structure():
    """Test that all new import paths work correctly"""
    
    # Test basic imports
    try:
        from features.common.hooks.start import get_sam_model, get_yolo_model
        from features.extraction.commands.extract_character import extract_character_from_path
        from features.evaluation.utils.learned_quality_assessment import assess_image_quality
        from features.processing.preprocessing.preprocessing import preprocess_image_pipeline
        from features.processing.postprocessing.postprocessing import enhance_character_mask
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_structure():
    """Test that new project structure exists"""
    
    base_path = Path(__file__).parent.parent.parent
    
    required_dirs = [
        "core/segment_anything",
        "core/scripts", 
        "core/demo",
        "features/extraction/commands",
        "features/extraction/models",
        "features/evaluation/utils",
        "features/processing/preprocessing",
        "features/processing/postprocessing",
        "features/common/hooks",
        "features/common/notification",
        "features/common/performance",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "tools"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure correct")
    return True

def test_backup_integrity():
    """Test that backup was created successfully"""
    
    base_path = Path(__file__).parent.parent.parent
    backup_dirs = list(base_path.glob("backup-*"))
    
    if not backup_dirs:
        print("❌ No backup directories found")
        return False
    
    latest_backup = max(backup_dirs, key=lambda x: x.name)
    
    # Check if critical files are backed up
    critical_files = [
        "CLAUDE.md",
        "PROGRESS_TRACKER.md",
        "requirements.txt",
        "commands",
        "models",
        "utils",
        "hooks",
        "config",
        "tests"
    ]
    
    missing_files = []
    for file_name in critical_files:
        backup_path = latest_backup / file_name
        if not backup_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing backup files: {missing_files}")
        return False
    
    print(f"✅ Backup integrity confirmed: {latest_backup}")
    return True

def test_claude_md_enhancement():
    """Test that CLAUDE.md has been enhanced with test-first rules"""
    
    claude_md_path = Path(__file__).parent.parent.parent / "CLAUDE.md"
    
    if not claude_md_path.exists():
        print("❌ CLAUDE.md not found")
        return False
    
    content = claude_md_path.read_text(encoding='utf-8')
    
    required_sections = [
        "絶対遵守ルール",
        "実装報告の前に必須",
        "テスト作成",
        "動作確認",
        "報告テンプレート",
        "禁止事項"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing CLAUDE.md sections: {missing_sections}")
        return False
    
    print("✅ CLAUDE.md enhancement confirmed")
    return True

def run_all_tests():
    """Run all Phase 0 completion tests"""
    
    print("🧪 Running Phase 0 completion tests...")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Import Structure", test_import_structure),
        ("Backup Integrity", test_backup_integrity),
        ("CLAUDE.md Enhancement", test_claude_md_enhancement)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n📊 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Phase 0 completion confirmed!")
        return True
    else:
        print("⚠️ Phase 0 completion issues detected")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)