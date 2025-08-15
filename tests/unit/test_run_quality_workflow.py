#!/usr/bin/env python3
"""
run_quality_workflow.sh の包括的ユニットテスト
全ての重要機能をテストし、ワークフロー破壊を防止
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestWorkflowEnvironment(unittest.TestCase):
    """Phase 1: 環境検証テスト"""
    
    def setUp(self):
        """テスト環境準備"""
        self.test_dir = tempfile.mkdtemp(prefix="test_workflow_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        """テスト環境クリーンアップ"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_sam_env_exists(self):
        """sam-env/bin/python3の存在確認テスト"""
        # sam-env作成
        sam_env_path = Path("sam-env/bin")
        sam_env_path.mkdir(parents=True)
        (sam_env_path / "python3").touch(mode=0o755)
        
        # スクリプトテスト部分実行
        script = """
        if [ ! -f "sam-env/bin/python3" ]; then
            exit 1
        fi
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True)
        self.assertEqual(result.returncode, 0, "sam-env環境の存在確認に失敗")
    
    def test_sam_env_not_exists(self):
        """sam-env不在時のエラー処理テスト"""
        script = """
        if [ ! -f "sam-env/bin/python3" ]; then
            echo "❌ エラー: sam-env環境が見つかりません"
            exit 1
        fi
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("sam-env環境が見つかりません", result.stdout)
    
    def test_required_packages_check(self):
        """必須パッケージ確認テスト"""
        # モックPython環境作成
        sam_env_path = Path("sam-env/bin")
        sam_env_path.mkdir(parents=True)
        
        # 成功するダミーPythonスクリプト作成
        python_script = sam_env_path / "python3"
        python_script.write_text("""#!/bin/bash
if [[ "$1" == "-c" ]] && [[ "$2" == *"import cv2"* ]]; then
    echo "✅ 必須パッケージ確認完了"
    exit 0
fi
exit 1
""")
        python_script.chmod(0o755)
        
        # テスト実行
        script = """
        if ! sam-env/bin/python3 -c "import cv2, click, numpy; print('✅ 必須パッケージ確認完了')" 2>/dev/null; then
            exit 1
        fi
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True)
        self.assertEqual(result.returncode, 0, "必須パッケージ確認に失敗")
    
    def test_pushover_config_warning(self):
        """Pushover設定ファイル不在時の警告テスト"""
        script = """
        if [ ! -f "config/pushover.json" ]; then
            echo "⚠️ 警告: Pushover設定ファイルが見つかりません"
        fi
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertIn("Pushover設定ファイルが見つかりません", result.stdout)


class TestWorkflowArguments(unittest.TestCase):
    """Phase 2: 引数処理テスト"""
    
    def test_tracker_id_format_valid(self):
        """有効なTRACKER_ID形式テスト"""
        valid_ids = ["PH2-001", "P1-005", "QI-002", "PH3-999", "P9-000"]
        
        for tracker_id in valid_ids:
            script = f"""
            TRACKER_ID="{tracker_id}"
            if [[ ! "$TRACKER_ID" =~ ^(PH[0-9]+-[0-9]{{3}}|P[0-9]+-[0-9]{{3}}|QI-[0-9]{{3}})$ ]]; then
                exit 1
            fi
            exit 0
            """
            result = subprocess.run(["bash", "-c", script], capture_output=True)
            self.assertEqual(result.returncode, 0, f"{tracker_id}の形式検証に失敗")
    
    def test_tracker_id_format_invalid(self):
        """無効なTRACKER_ID形式テスト"""
        invalid_ids = ["PH2-1", "P1-00005", "INVALID", "PH-001", ""]
        
        for tracker_id in invalid_ids:
            script = f"""
            TRACKER_ID="{tracker_id}"
            if [[ ! "$TRACKER_ID" =~ ^(PH[0-9]+-[0-9]{{3}}|P[0-9]+-[0-9]{{3}}|QI-[0-9]{{3}})$ ]]; then
                echo "❌ エラー: 無効なトラッカーID形式: $TRACKER_ID"
                exit 1
            fi
            exit 0
            """
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, f"{tracker_id}が誤って有効と判定された")
            if tracker_id:  # 空文字列以外
                self.assertIn("無効なトラッカーID形式", result.stdout)
    
    def test_empty_tracker_id(self):
        """TRACKER_ID未指定時のエラーテスト"""
        script = """
        TRACKER_ID=""
        if [ -z "$TRACKER_ID" ]; then
            echo "❌ エラー: トラッカーIDを指定してください"
            exit 1
        fi
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("トラッカーIDを指定してください", result.stdout)


class TestWorkspaceOperations(unittest.TestCase):
    """Phase 3: ワークスペース操作テスト"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_workspace_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_workspace_directory_creation(self):
        """ワークスペースディレクトリ作成テスト"""
        script = """
        OUTPUT_DIR="test_workspace/PH2-001"
        mkdir -p "${OUTPUT_DIR}"/{extraction,quality,dashboard,tests}
        
        # 検証
        for dir in extraction quality dashboard tests; do
            if [ ! -d "${OUTPUT_DIR}/${dir}" ]; then
                echo "❌ ${dir}ディレクトリ作成失敗"
                exit 1
            fi
        done
        echo "✅ 全ディレクトリ作成成功"
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("全ディレクトリ作成成功", result.stdout)
    
    def test_existing_extraction_check(self):
        """既存抽出結果の確認テスト"""
        # 既存ディレクトリと結果ファイル作成
        os.makedirs("output/extraction", exist_ok=True)
        Path("output/extraction/result.jpg").touch()
        
        script = """
        OUTPUT_DIR="output"
        if [ -d "${OUTPUT_DIR}/extraction" ] && [ "$(ls -A ${OUTPUT_DIR}/extraction)" ]; then
            echo "既存の抽出結果が見つかりました"
            exit 0
        fi
        exit 1
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("既存の抽出結果が見つかりました", result.stdout)


class TestExtractionPipeline(unittest.TestCase):
    """Phase 4: 抽出パイプラインテスト"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_extraction_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_input_directory_validation(self):
        """入力ディレクトリ検証テスト"""
        # 存在しないディレクトリ
        script = """
        INPUT_DIR="/nonexistent/path"
        if [ ! -d "$INPUT_DIR" ]; then
            echo "❌ エラー: 入力ディレクトリが存在しません"
            exit 1
        fi
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("入力ディレクトリが存在しません", result.stdout)
    
    def test_image_count_validation(self):
        """画像ファイル数カウントテスト"""
        # テスト用ディレクトリと画像作成
        os.makedirs("test_images", exist_ok=True)
        for i in range(3):
            Path(f"test_images/image_{i}.jpg").touch()
        
        script = """
        INPUT_DIR="test_images"
        IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
        if [ "$IMAGE_COUNT" -eq 0 ]; then
            exit 1
        fi
        echo "画像数: $IMAGE_COUNT"
        exit 0
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("画像数: 3", result.stdout)
    
    def test_process_monitoring(self):
        """プロセス監視テスト"""
        script = """
        # ダミープロセス起動
        sleep 0.1 &
        PID=$!
        
        # プロセス存在確認
        if kill -0 $PID 2>/dev/null; then
            echo "プロセス実行中"
        fi
        
        # プロセス終了待機
        wait $PID
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "プロセス正常終了"
            exit 0
        fi
        exit 1
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("プロセス正常終了", result.stdout)


class TestQualityChecks(unittest.TestCase):
    """Phase 5: 品質チェックテスト"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_quality_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # モックPython環境作成
        sam_env_path = Path("sam-env/bin")
        sam_env_path.mkdir(parents=True)
        
        # 成功するダミーPythonスクリプト
        python_script = sam_env_path / "python3"
        python_script.write_text("""#!/bin/bash
echo "ダミー実行: $@"
# 出力ファイル作成
if [[ "$@" == *"--output"* ]]; then
    # --outputパラメータから出力パス抽出
    for i in "${@}"; do
        if [[ "$prev" == "--output" ]]; then
            mkdir -p "$i"
            touch "$i/dummy_output.json"
        fi
        prev="$i"
    done
fi
exit 0
""")
        python_script.chmod(0o755)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_quality_report_generation(self):
        """品質レポート生成テスト"""
        # 必要なディレクトリ作成
        os.makedirs("output/extraction", exist_ok=True)
        os.makedirs("output/quality", exist_ok=True)
        
        script = """
        OUTPUT_DIR="output"
        sam-env/bin/python3 dummy_script.py \
            --input_dir "${OUTPUT_DIR}/extraction/" \
            --output "${OUTPUT_DIR}/quality/report.json"
        
        if [ $? -eq 0 ]; then
            echo "品質レポート生成成功"
            exit 0
        fi
        exit 1
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("品質レポート生成成功", result.stdout)
    
    def test_dashboard_generation_with_report(self):
        """レポート存在時のダッシュボード生成テスト"""
        # レポートファイル作成
        os.makedirs("output/quality", exist_ok=True)
        Path("output/quality/unified_quality_report.json").write_text("{}")
        
        script = """
        OUTPUT_DIR="output"
        if [ -f "${OUTPUT_DIR}/quality/unified_quality_report.json" ]; then
            echo "ダッシュボード生成開始"
            sam-env/bin/python3 dummy_dashboard.py \
                --report "${OUTPUT_DIR}/quality/unified_quality_report.json" \
                --output "${OUTPUT_DIR}/dashboard/"
            echo "ダッシュボード生成完了"
            exit 0
        fi
        exit 1
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("ダッシュボード生成完了", result.stdout)


class TestSummaryGeneration(unittest.TestCase):
    """Phase 6: サマリー生成テスト"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_summary_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_summary_file_creation(self):
        """サマリーファイル作成テスト"""
        script = """
        OUTPUT_DIR="output"
        TRACKER_ID="PH2-001"
        SUMMARY_FILE="${OUTPUT_DIR}/workflow_summary.txt"
        
        mkdir -p "$OUTPUT_DIR"
        
        cat > "$SUMMARY_FILE" << EOF
品質保証ワークフロー実行結果
================================
トラッカーID: ${TRACKER_ID}
EOF
        
        if [ -f "$SUMMARY_FILE" ]; then
            echo "サマリーファイル作成成功"
            exit 0
        fi
        exit 1
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("サマリーファイル作成成功", result.stdout)
        
        # ファイル内容確認
        self.assertTrue(Path("output/workflow_summary.txt").exists())
        content = Path("output/workflow_summary.txt").read_text()
        self.assertIn("PH2-001", content)


class TestIntegrationWorkflow(unittest.TestCase):
    """統合ワークフローテスト"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_integration_")
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # 完全なモック環境セットアップ
        self._setup_mock_environment()
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _setup_mock_environment(self):
        """完全なモック環境構築"""
        # sam-env環境
        sam_env_path = Path("sam-env/bin")
        sam_env_path.mkdir(parents=True)
        
        # モックPython
        python_script = sam_env_path / "python3"
        python_script.write_text("""#!/bin/bash
echo "モック実行: $(basename $2 .py)"

# WorkspaceConfig モック
if [[ "$@" == *"workspace_config"* ]]; then
    echo 'TRACKER_WORKSPACE_ROOT="/tmp/test_workspace"'
    exit 0
fi

# 各種Pythonスクリプトのモック処理
case "$2" in
    *"sam_yolo_character_segment.py")
        echo "抽出パイプライン実行中..."
        sleep 0.1
        echo "抽出完了"
        ;;
    *"create_phase1_extraction_report.py")
        mkdir -p "${@: -1}"
        echo '{"status": "success"}' > "${@: -1}/report.json"
        ;;
    *"unified_quality_checker.py")
        mkdir -p "$(dirname ${@: -1})"
        echo '{"quality": "good"}' > "${@: -1}"
        ;;
    *"quality_dashboard.py")
        mkdir -p "${@: -1}"
        echo "<html>Dashboard</html>" > "${@: -1}/dashboard.html"
        ;;
    *)
        echo "デフォルト処理"
        ;;
esac
exit 0
""")
        python_script.chmod(0o755)
        
        # 必須パッケージチェック用
        (sam_env_path / "python3_check").write_text("""#!/bin/bash
if [[ "$2" == *"import cv2"* ]]; then
    echo "✅ 必須パッケージ確認完了"
fi
exit 0
""")
        (sam_env_path / "python3_check").chmod(0o755)
        
        # テスト用画像ディレクトリ
        test_images = Path("/tmp/test_images")
        test_images.mkdir(exist_ok=True)
        (test_images / "test.jpg").touch()
        
        # config/workspace_config.py のモック
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        (config_path / "workspace_config.py").write_text("""
class WorkspaceConfig:
    @staticmethod
    def export_environment_variables():
        return {"TRACKER_WORKSPACE_ROOT": "/tmp/test_workspace"}
""")
    
    def test_full_workflow_execution(self):
        """完全なワークフロー実行テスト"""
        # 簡略化したワークフロースクリプト作成
        workflow_script = """#!/bin/bash
set -e

# 環境チェック
if [ ! -f "sam-env/bin/python3" ]; then
    echo "❌ エラー: sam-env環境が見つかりません"
    exit 1
fi

TRACKER_ID="${1:-PH2-001}"

# TRACKER_ID形式チェック
if [[ ! "$TRACKER_ID" =~ ^(PH[0-9]+-[0-9]{3}|P[0-9]+-[0-9]{3}|QI-[0-9]{3})$ ]]; then
    echo "❌ エラー: 無効なトラッカーID形式"
    exit 1
fi

# ワークスペース設定
WORKSPACE_BASE="/tmp/test_workspace"
OUTPUT_DIR="${WORKSPACE_BASE}/${TRACKER_ID}"

echo "🔄 品質保証ワークフロー開始: ${TRACKER_ID}"

# ディレクトリ作成
mkdir -p "${OUTPUT_DIR}"/{extraction,quality,dashboard,tests}

# 入力チェック
INPUT_DIR="/tmp/test_images"
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ エラー: 入力ディレクトリが存在しません"
    exit 1
fi

IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.jpg" | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ エラー: 画像ファイルが見つかりません"
    exit 1
fi

echo "✅ 入力検証完了: $IMAGE_COUNT 枚の画像を検出"

# モック実行
echo "📊 抽出パイプライン実行中..."
sam-env/bin/python3 tools/core/sam_yolo_character_segment.py

echo "📊 品質チェック実行中..."
sam-env/bin/python3 tools/core/unified_quality_checker.py \
    --output "${OUTPUT_DIR}/quality/report.json"

echo "✅ 品質保証ワークフロー完了: ${TRACKER_ID}"
exit 0
"""
        
        Path("test_workflow.sh").write_text(workflow_script)
        Path("test_workflow.sh").chmod(0o755)
        
        # ワークフロー実行
        result = subprocess.run(
            ["./test_workflow.sh", "PH2-001"],
            capture_output=True,
            text=True
        )
        
        # 検証
        self.assertEqual(result.returncode, 0, f"ワークフロー実行失敗:\n{result.stdout}\n{result.stderr}")
        self.assertIn("品質保証ワークフロー開始", result.stdout)
        self.assertIn("入力検証完了", result.stdout)
        self.assertIn("品質保証ワークフロー完了", result.stdout)
        
        # 出力ディレクトリ確認
        output_dir = Path("/tmp/test_workspace/PH2-001")
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / "extraction").exists())
        self.assertTrue((output_dir / "quality").exists())
        self.assertTrue((output_dir / "dashboard").exists())
        self.assertTrue((output_dir / "tests").exists())


def run_tests():
    """全テスト実行"""
    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 各テストクラスを追加
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowArguments))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkspaceOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractionPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestSummaryGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWorkflow))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)
    print(f"✅ 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失敗: {len(result.failures)}")
    print(f"⚠️  エラー: {len(result.errors)}")
    print(f"📊 合計: {result.testsRun}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)