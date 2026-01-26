#!/usr/bin/env python3
"""
統合ダッシュボードシステムのテストスクリプト

統合システムの動作確認とデバッグ用
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.common.dashboard_config import DashboardConfigManager
from features.common.unified_dashboard_generator import UnifiedDashboardGenerator


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_test_images(test_dir: Path, count: int = 5) -> List[str]:
    """テスト用画像作成"""
    import numpy as np
    import cv2

    image_paths = []
    extraction_dir = test_dir / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        # ランダムな品質の画像を作成
        if i == 0:
            # 高品質画像
            image = np.random.randint(100, 255, (800, 600, 3), dtype=np.uint8)
        elif i == 1:
            # 中品質画像
            image = np.random.randint(50, 200, (640, 480, 3), dtype=np.uint8)
        elif i == 2:
            # 低品質画像
            image = np.random.randint(0, 100, (320, 240, 3), dtype=np.uint8)
        elif i == 3:
            # 黒画面（品質不良）
            image = np.full((400, 300, 3), 10, dtype=np.uint8)
        else:
            # 標準画像
            image = np.random.randint(80, 180, (512, 384, 3), dtype=np.uint8)

        # 中央に図形を描画（キャラクター風）
        center = (image.shape[1] // 2, image.shape[0] // 2)
        if i != 3:  # 黒画面以外
            cv2.circle(image, center, min(center) // 3, (255, 255, 255), -1)
            cv2.rectangle(
                image,
                (center[0] - 20, center[1] - 40),
                (center[0] + 20, center[1] - 10),
                (0, 0, 255),
                -1,
            )

        image_path = extraction_dir / f"test_image_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))

    return image_paths


def test_unified_dashboard_basic():
    """基本機能テスト"""
    print("\n🔍 統合ダッシュボード基本機能テスト")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # 1. テスト用画像作成
            print("📸 テスト画像作成中...")
            image_paths = create_test_images(temp_path, 5)
            print(f"✅ テスト画像作成完了: {len(image_paths)}枚")

            # 2. 統合ダッシュボードジェネレーター初期化
            print("🏗️ 統合ダッシュボードジェネレーター初期化中...")
            generator = UnifiedDashboardGenerator()
            print("✅ 初期化完了")

            # 3. ダッシュボード生成
            print("📊 ダッシュボード生成中...")
            output_dir = temp_path / "output"

            dashboard_path = generator.generate_dashboard(
                tracker_id="TEST-016",
                extraction_dir=str(temp_path / "extraction"),
                output_dir=str(output_dir),
            )

            # 4. 結果確認
            if dashboard_path.exists():
                file_size = dashboard_path.stat().st_size
                print(f"✅ ダッシュボード生成成功: {dashboard_path}")
                print(f"📄 ファイルサイズ: {file_size / 1024:.1f}KB")

                # HTML内容の基本確認
                content = dashboard_path.read_text(encoding="utf-8")
                if "TEST-016" in content and "品質評価ダッシュボード" in content:
                    print("✅ HTML内容確認OK")
                else:
                    print("❌ HTML内容に問題があります")

                return True
            else:
                print("❌ ダッシュボード生成失敗: ファイルが存在しません")
                return False

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_config_system():
    """設定システムテスト"""
    print("\n🔧 設定システムテスト")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. 設定マネージャー初期化
            config_manager = DashboardConfigManager(temp_dir)
            print("✅ 設定マネージャー初期化完了")

            # 2. デフォルト設定ロード
            default_config = config_manager.load_config("DEFAULT-TEST")
            print(f"✅ デフォルト設定ロード: {default_config.title}")

            # 3. カスタム設定作成
            template_path = config_manager.create_config_template("CUSTOM-TEST")
            print(f"✅ カスタム設定テンプレート作成: {template_path}")

            # 4. カスタム設定ロード
            custom_config = config_manager.load_config("CUSTOM-TEST")
            print(f"✅ カスタム設定ロード: {custom_config.title}")

            return True

        except Exception as e:
            print(f"❌ 設定システムテストエラー: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_plugin_system():
    """プラグインシステムテスト"""
    print("\n🔌 プラグインシステムテスト")

    try:
        # 統合ダッシュボードジェネレーター初期化
        generator = UnifiedDashboardGenerator()

        # プラグイン確認
        plugins = list(generator.plugins.keys())
        print(f"✅ 検出されたプラグイン: {plugins}")

        if "image_quality" in plugins:
            print("✅ image_qualityプラグイン: 利用可能")
        else:
            print("⚠️ image_qualityプラグイン: 未検出")

        if "statistics" in plugins:
            print("✅ statisticsプラグイン: 利用可能")
        else:
            print("⚠️ statisticsプラグイン: 未検出")

        return len(plugins) > 0

    except Exception as e:
        print(f"❌ プラグインシステムテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_compatibility():
    """互換性テスト"""
    print("\n🔄 既存システム互換性テスト")

    try:
        # 既存システムのインポートテスト
        from features.common.dashboard_generator import StandardDashboardGenerator

        print("✅ StandardDashboardGenerator: インポート成功")

        # QI-004システム統合テスト（オプショナル）
        try:
            from features.evaluation.qi004_dashboard_optimization_system import ImageQualityAnalyzer

            print("✅ QI-004システム: 利用可能")
            qi004_available = True
        except ImportError:
            print("⚠️ QI-004システム: 利用不可（インポートエラー）")
            qi004_available = False

        # quality_dashboardシステム統合テスト（オプショナル）
        try:
            import sys

            sys.path.append(str(Path(__file__).parent / "tools" / "core"))
            from quality_dashboard import QualityDashboard

            print("✅ QualityDashboardシステム: 利用可能")
            quality_dashboard_available = True
        except ImportError:
            print("⚠️ QualityDashboardシステム: 利用不可（インポートエラー）")
            quality_dashboard_available = False

        return True

    except Exception as e:
        print(f"❌ 互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """全テスト実行"""
    print("🚀 統合ダッシュボードシステム テスト開始")
    print("=" * 60)

    results = {}

    # 各テスト実行
    results["basic"] = test_unified_dashboard_basic()
    results["config"] = test_config_system()
    results["plugins"] = test_plugin_system()
    results["compatibility"] = test_compatibility()

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:15s}: {status}")

    print(f"\n🎯 総合結果: {passed_tests}/{total_tests} テスト合格")

    if passed_tests == total_tests:
        print("🎉 全テスト合格！統合システムは正常に動作します")
        return True
    else:
        print("⚠️ 一部テストで問題が発生しました")
        return False


if __name__ == "__main__":
    setup_logging()

    # 個別テスト実行オプション
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "basic":
            success = test_unified_dashboard_basic()
        elif test_type == "config":
            success = test_config_system()
        elif test_type == "plugins":
            success = test_plugin_system()
        elif test_type == "compatibility":
            success = test_compatibility()
        else:
            print("使用方法: python test_unified_dashboard.py [basic|config|plugins|compatibility]")
            success = False
    else:
        # 全テスト実行
        success = run_all_tests()

    sys.exit(0 if success else 1)
