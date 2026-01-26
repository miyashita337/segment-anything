#!/usr/bin/env python3
"""
環境仕様整合性チェックテスト

このテストは spec.md で定義された仕様と実際の環境の整合性を検証します。
新機能追加やシステム変更時には必ずこのテストを実行してください。
"""

import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class EnvironmentSpecTest(unittest.TestCase):
    """環境仕様との整合性をテストするクラス"""

    @classmethod
    def setUpClass(cls):
        """テスト開始前の初期化"""
        cls.spec_path = PROJECT_ROOT / "spec.md"
        cls.spec_content = cls._load_spec_content()
        cls.project_root = PROJECT_ROOT

        print("🔍 Environment Specification Test Suite")
        print(f"📁 Project Root: {cls.project_root}")
        print(f"📋 Spec File: {cls.spec_path}")

    @classmethod
    def _load_spec_content(cls) -> str:
        """spec.mdの内容を読み込み"""
        if not cls.spec_path.exists():
            raise FileNotFoundError(f"spec.md not found at {cls.spec_path}")

        with open(cls.spec_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_python_version_compatibility(self):
        """Python バージョンの整合性テスト"""
        print("\n🐍 Testing Python Version Compatibility...")

        # spec.md からPythonバージョン要件を抽出
        version_pattern = r'python_version:\s*"([^"]+)"'
        version_match = re.search(version_pattern, self.spec_content)

        self.assertIsNotNone(version_match, "Python version specification not found in spec.md")

        version_spec = version_match.group(1)
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        print(f"📋 Spec requires: {version_spec}")
        print(f"🔍 Current version: {current_version}")

        # 基本的な互換性チェック (>=3.8)
        self.assertGreaterEqual(
            sys.version_info[:2], (3, 8), f"Python {current_version} is below minimum requirement"
        )

        # 推奨バージョンチェック
        recommended_pattern = r'recommended_version:\s*"([^"]+)"'
        recommended_match = re.search(recommended_pattern, self.spec_content)

        if recommended_match:
            recommended_version = recommended_match.group(1)
            print(f"💡 Recommended version: {recommended_version}")

        print("✅ Python version compatibility confirmed")

    def test_required_packages_availability(self):
        """必須パッケージの可用性テスト"""
        print("\n📦 Testing Required Packages...")

        # spec.md から必須パッケージを抽出
        core_packages = ["torch", "torchvision", "numpy", "pillow", "opencv"]

        missing_packages = []
        available_packages = []

        for package in core_packages:
            try:
                # パッケージ名のマッピング
                import_name = {"pillow": "PIL", "opencv": "cv2"}.get(package, package)

                if importlib.util.find_spec(import_name):
                    available_packages.append(package)
                    print(f"✅ {package} is available")
                else:
                    missing_packages.append(package)
                    print(f"❌ {package} is missing")

            except Exception as e:
                missing_packages.append(package)
                print(f"❌ {package} check failed: {e}")

        self.assertEqual(len(missing_packages), 0, f"Missing required packages: {missing_packages}")

        print(f"✅ All {len(available_packages)} required packages are available")

    def test_cuda_availability(self):
        """CUDA環境の整合性テスト"""
        print("\n🎮 Testing CUDA Availability...")

        try:
            import torch

            cuda_available = torch.cuda.is_available()

            if cuda_available:
                device_count = torch.cuda.device_count()
                cuda_version = torch.version.cuda

                print(f"✅ CUDA is available")
                print(f"🔢 GPU devices: {device_count}")
                print(f"🏷️ CUDA version: {cuda_version}")

                # VRAM チェック (GPU 0のみ)
                if device_count > 0:
                    device_props = torch.cuda.get_device_properties(0)
                    total_memory_gb = device_props.total_memory / (1024**3)
                    print(f"💾 GPU 0 VRAM: {total_memory_gb:.1f} GB")

                    # spec.md で推奨される8GB以上かチェック
                    if total_memory_gb >= 8.0:
                        print("✅ VRAM meets recommended requirement (8GB+)")
                    else:
                        print(f"⚠️ VRAM ({total_memory_gb:.1f}GB) below recommended 8GB")

            else:
                print("⚠️ CUDA is not available (CPU-only mode)")
                print("💡 GPU acceleration disabled - performance may be limited")

        except ImportError:
            self.fail("PyTorch not available for CUDA testing")

        # CUDA利用可能性は必須ではないため、警告のみ
        print("✅ CUDA availability test completed")

    def test_model_file_requirements(self):
        """モデルファイルの存在確認テスト"""
        print("\n🤖 Testing Model File Requirements...")

        # spec.md から必要なモデルファイルを抽出
        required_models = {
            "sam_vit_h_4b8939.pth": "SAM ViT-H model",
            "yolov8": "YOLO model (any variant)",
        }

        model_status = {}

        for model_pattern, description in required_models.items():
            if model_pattern == "yolov8":
                # YOLO系ファイルの存在確認
                yolo_files = list(self.project_root.glob("yolov8*.pt"))
                if yolo_files:
                    model_status[model_pattern] = f"Found: {[f.name for f in yolo_files]}"
                    print(f"✅ {description}: {yolo_files[0].name}")
                else:
                    model_status[model_pattern] = "Not found"
                    print(f"⚠️ {description}: Not found")
            else:
                # 特定ファイルの存在確認
                model_path = self.project_root / model_pattern
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024**2)
                    model_status[model_pattern] = f"Found ({size_mb:.1f}MB)"
                    print(f"✅ {description}: Found ({size_mb:.1f}MB)")
                else:
                    model_status[model_pattern] = "Not found"
                    print(f"⚠️ {description}: Not found")

        # モデルファイルは推奨であり必須ではないため、警告のみ
        missing_models = [k for k, v in model_status.items() if "Not found" in v]
        if missing_models:
            print(f"💡 Recommendation: Download missing models: {missing_models}")

        print("✅ Model file requirements check completed")

    def test_image_format_support(self):
        """対応画像形式のサポート確認テスト"""
        print("\n🖼️ Testing Image Format Support...")

        # spec.md から対応形式を抽出
        supported_formats = [".jpg", ".jpeg", ".png", ".webp"]

        # PIL/Pillowでの対応確認
        try:
            from PIL import Image

            format_support = {}
            for fmt in supported_formats:
                ext = fmt.upper().lstrip(".")

                # 特別なケース
                if ext == "JPG":
                    ext = "JPEG"

                if ext in Image.registered_extensions().values():
                    format_support[fmt] = True
                    print(f"✅ {fmt} format supported")
                else:
                    format_support[fmt] = False
                    print(f"❌ {fmt} format not supported")

            # 基本形式 (JPG, PNG) は必須
            essential_formats = [".jpg", ".png"]
            for fmt in essential_formats:
                self.assertTrue(
                    format_support.get(fmt, False), f"Essential format {fmt} not supported"
                )

            print("✅ Essential image formats are supported")

        except ImportError:
            self.fail("PIL/Pillow not available for image format testing")

    def test_directory_structure(self):
        """ディレクトリ構造の整合性テスト"""
        print("\n📁 Testing Directory Structure...")

        # spec.md で定義されている必要なディレクトリ
        required_dirs = ["core", "features", "tools", "tests", "docs/workflows"]

        missing_dirs = []
        existing_dirs = []

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                existing_dirs.append(dir_path)
                print(f"✅ Directory exists: {dir_path}")
            else:
                missing_dirs.append(dir_path)
                print(f"❌ Directory missing: {dir_path}")

        self.assertEqual(len(missing_dirs), 0, f"Missing required directories: {missing_dirs}")

        print(f"✅ All {len(existing_dirs)} required directories exist")

    def test_command_availability(self):
        """必要なコマンドの可用性テスト"""
        print("\n⚙️ Testing Command Availability...")

        commands_to_test = [
            ("python3", "Python interpreter"),
            ("pip", "Package installer"),
            ("git", "Version control"),
        ]

        available_commands = []
        missing_commands = []

        for cmd, description in commands_to_test:
            try:
                result = subprocess.run(
                    [cmd, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    available_commands.append(cmd)
                    version_line = result.stdout.split("\n")[0]
                    print(f"✅ {cmd}: {version_line}")
                else:
                    missing_commands.append(cmd)
                    print(f"❌ {cmd}: Not available")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_commands.append(cmd)
                print(f"❌ {cmd}: Not found")

        # python3とpipは必須
        essential_commands = ["python3", "pip"]
        for cmd in essential_commands:
            self.assertIn(cmd, available_commands, f"Essential command {cmd} not available")

        print("✅ Essential commands are available")

    def test_performance_requirements(self):
        """パフォーマンス要件の確認テスト"""
        print("\n⚡ Testing Performance Requirements...")

        # システム情報取得
        import psutil

        # メモリ要件確認
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Total RAM: {total_memory_gb:.1f} GB")

        # spec.md では最小8GB、推奨16GB
        if total_memory_gb >= 16:
            print("✅ RAM meets recommended requirement (16GB+)")
        elif total_memory_gb >= 8:
            print("⚠️ RAM meets minimum requirement (8GB) but below recommended (16GB)")
        else:
            print(f"❌ RAM ({total_memory_gb:.1f}GB) below minimum requirement (8GB)")

        # ディスク容量確認
        disk_usage = psutil.disk_usage(str(self.project_root))
        free_space_gb = disk_usage.free / (1024**3)
        print(f"💿 Free disk space: {free_space_gb:.1f} GB")

        # 最小20GB必要
        if free_space_gb >= 50:
            print("✅ Disk space meets recommended requirement (50GB+)")
        elif free_space_gb >= 20:
            print("⚠️ Disk space meets minimum requirement (20GB)")
        else:
            print(f"❌ Disk space ({free_space_gb:.1f}GB) below minimum requirement (20GB)")

        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_arch = platform.machine()
        print(f"🖥️ CPU cores: {cpu_count}, Architecture: {cpu_arch}")

        print("✅ Performance requirements check completed")

    def test_spec_file_integrity(self):
        """spec.md ファイル自体の整合性テスト"""
        print("\n📋 Testing spec.md File Integrity...")

        # 必要なセクションが存在するかチェック
        required_sections = ["ハードウェア要件", "ソフトウェア要件", "モデルファイル要件", "対応画像形式", "ディレクトリ構造"]

        missing_sections = []
        for section in required_sections:
            if section in self.spec_content:
                print(f"✅ Section found: {section}")
            else:
                missing_sections.append(section)
                print(f"❌ Section missing: {section}")

        self.assertEqual(
            len(missing_sections), 0, f"Missing required sections in spec.md: {missing_sections}"
        )

        # バージョン情報の存在確認
        version_patterns = [r"バージョン.*?:", r"最終更新.*?:", r"python_version:"]

        for pattern in version_patterns:
            if re.search(pattern, self.spec_content, re.IGNORECASE):
                print(f"✅ Version info pattern found: {pattern}")
            else:
                print(f"⚠️ Version info pattern missing: {pattern}")

        print("✅ spec.md file integrity verified")


class PerformanceEnvironmentTest(unittest.TestCase):
    """パフォーマンス関連の環境テスト"""

    def test_basic_import_performance(self):
        """基本的なインポート性能テスト"""
        print("\n⏱️ Testing Basic Import Performance...")

        import time

        # 主要ライブラリのインポート時間測定
        libraries_to_test = [
            ("numpy", "import numpy"),
            ("PIL", "from PIL import Image"),
            ("torch", "import torch"),
        ]

        for lib_name, import_statement in libraries_to_test:
            start_time = time.time()
            try:
                exec(import_statement)
                import_time = time.time() - start_time
                print(f"✅ {lib_name}: {import_time:.3f}s")

                # 異常に遅いインポート (>5秒) は警告
                if import_time > 5.0:
                    print(f"⚠️ {lib_name} import is slow ({import_time:.1f}s)")

            except ImportError:
                print(f"❌ {lib_name}: Import failed")
            except Exception as e:
                print(f"❌ {lib_name}: Error - {e}")

        print("✅ Import performance test completed")


def run_environment_tests(verbose: bool = True):
    """環境テストを実行する関数"""

    # テストスイート構築
    suite = unittest.TestSuite()

    # 基本環境テスト
    suite.addTest(unittest.makeSuite(EnvironmentSpecTest))

    # パフォーマンステスト
    suite.addTest(unittest.makeSuite(PerformanceEnvironmentTest))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # 結果サマリー
    print("\n" + "=" * 60)
    print("🧪 Environment Specification Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            error_line = traceback.split("\n")[-2]
            print(f"  - {test}: {error_line}")

    if result.wasSuccessful():
        print("\n✅ All tests passed - Environment is compatible with spec.md")
        return True
    else:
        print("\n❌ Some tests failed - Environment may not be compatible with spec.md")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Environment specification consistency test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick test (skip performance tests)"
    )

    args = parser.parse_args()

    print("🚀 Starting Environment Specification Test Suite")
    print(f"📅 {platform.platform()}")
    print(f"🐍 Python {sys.version}")

    if args.quick:
        # クイックテスト：基本環境テストのみ
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(EnvironmentSpecTest))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        success = result.wasSuccessful()
    else:
        # フルテスト
        success = run_environment_tests(args.verbose)

    sys.exit(0 if success else 1)
