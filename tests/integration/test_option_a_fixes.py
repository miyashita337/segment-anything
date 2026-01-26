#!/usr/bin/env python3
"""
Option A修正内容の統合テスト
Phase 0リファクタリング後のデグレード修正確認
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_error_message_fix():
    """エラーメッセージ修正のテスト"""
    print("🧪 エラーメッセージ修正テスト...")

    try:
        # extract_character.py の内容確認
        extract_file = (
            Path(__file__).parent.parent.parent
            / "features/extraction/commands/extract_character.py"
        )

        if not extract_file.exists():
            print("❌ extract_character.py が見つかりません")
            return False

        content = extract_file.read_text(encoding="utf-8")

        # 新しいパスのエラーメッセージが含まれているか確認
        if "python3 features/common/hooks/start.py" in content:
            print("✅ エラーメッセージが新構造パスに更新済み")
            return True
        else:
            print("❌ エラーメッセージが更新されていません")
            return False

    except Exception as e:
        print(f"❌ エラーメッセージテスト失敗: {e}")
        return False


def test_auto_init_system():
    """自動初期化システムのテスト"""
    print("🧪 自動初期化システムテスト...")

    try:
        # start.py の initialize_models 関数存在確認
        start_file = Path(__file__).parent.parent.parent / "features/common/hooks/start.py"

        if not start_file.exists():
            print("❌ start.py が見つかりません")
            return False

        content = start_file.read_text(encoding="utf-8")

        # initialize_models 関数が追加されているか確認
        if "def initialize_models():" in content:
            print("✅ initialize_models 関数が追加済み")

            # Phase 0対応メッセージが含まれているか確認
            if "Phase 0対応モデル初期化開始" in content:
                print("✅ Phase 0対応の初期化メッセージ確認")
                return True
            else:
                print("❌ Phase 0対応メッセージが不足")
                return False
        else:
            print("❌ initialize_models 関数が見つかりません")
            return False

    except Exception as e:
        print(f"❌ 自動初期化システムテスト失敗: {e}")
        return False


def test_unified_init_script():
    """統合初期化スクリプトのテスト"""
    print("🧪 統合初期化スクリプトテスト...")

    try:
        # init_models.py の存在確認
        init_script = Path(__file__).parent.parent.parent / "init_models.py"

        if not init_script.exists():
            print("❌ init_models.py が見つかりません")
            return False

        # 実行権限確認
        if not os.access(init_script, os.X_OK):
            # 実行権限を付与
            os.chmod(init_script, 0o755)
            print("✅ init_models.py に実行権限を付与")

        # スクリプトの基本構造確認
        content = init_script.read_text(encoding="utf-8")

        required_elements = [
            "統合モデル初期化スクリプト",
            "Phase 0リファクタリング後の新構造対応版",
            "def main():",
            "def run_test_mode(",
            "initialize_models",
        ]

        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)

        if missing_elements:
            print(f"❌ 必要な要素が不足: {missing_elements}")
            return False

        print("✅ 統合初期化スクリプトの構造確認完了")
        return True

    except Exception as e:
        print(f"❌ 統合初期化スクリプトテスト失敗: {e}")
        return False


def test_import_compatibility():
    """インポート互換性のテスト"""
    print("🧪 インポート互換性テスト...")

    try:
        # 新構造でのインポートテスト
        from features.common.hooks.start import initialize_models

        print("✅ initialize_models のインポート成功")

        from features.extraction.commands.extract_character import extract_character_from_path

        print("✅ extract_character_from_path のインポート成功")

        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ インポートテスト失敗: {e}")
        return False


def test_init_script_execution():
    """初期化スクリプト実行テスト"""
    print("🧪 初期化スクリプト実行テスト...")

    try:
        init_script = Path(__file__).parent.parent.parent / "init_models.py"

        if not init_script.exists():
            print("❌ init_models.py が見つかりません")
            return False

        # ヘルプメッセージ確認（軽量テスト）
        result = subprocess.run(
            [sys.executable, str(init_script), "--help"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print("✅ 初期化スクリプトのヘルプ実行成功")
            if "統合モデル初期化スクリプト" in result.stdout:
                print("✅ ヘルプメッセージ内容確認")
                return True
            else:
                print("❌ ヘルプメッセージ内容不正")
                return False
        else:
            print(f"❌ 初期化スクリプト実行失敗: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 初期化スクリプト実行タイムアウト")
        return False
    except Exception as e:
        print(f"❌ 初期化スクリプト実行テスト失敗: {e}")
        return False


def run_all_tests():
    """全テスト実行"""
    print("🚀 Option A修正内容の統合テスト開始")
    print("=" * 50)

    tests = [
        ("エラーメッセージ修正", test_error_message_fix),
        ("自動初期化システム", test_auto_init_system),
        ("統合初期化スクリプト", test_unified_init_script),
        ("インポート互換性", test_import_compatibility),
        ("初期化スクリプト実行", test_init_script_execution),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}テスト実行中...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ テスト例外: {e}")
            results.append((test_name, False))

    print(f"\n{'='*50}")
    print("📊 テスト結果:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 総合結果: {passed}/{len(tests)} テスト成功")

    if passed == len(tests):
        print("🎉 Option A修正完了！")
        print("💡 次のステップ:")
        print("   1. python3 init_models.py --test")
        print("   2. python3 run_batch_extraction.py")
        print("   3. バッチ抽出の動作確認")
        return True
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("🔧 修正が必要な項目を確認してください")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
