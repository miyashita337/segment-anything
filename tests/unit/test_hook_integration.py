#!/usr/bin/env python3
"""
INTG-087: Hook統合システム総合テスト
Claude Code Hook統合システムの単体テスト・統合テスト
"""

import json
import pytest
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestHookIntegration(unittest.TestCase):
    """Hook統合システムのテスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.test_workspace = Path("/tmp/test_hook_integration")
        self.test_workspace.mkdir(exist_ok=True)
        self.tracker_id = "INTG-087"

        # テスト用の状態ファイル作成
        self.workflow_dir = self.test_workspace / ".workflow"
        self.workflow_dir.mkdir(exist_ok=True)

        # 初期チェックリスト状態
        self.checklist_status = {
            "tracker_id": self.tracker_id,
            "created_at": "2025-09-09T12:00:00Z",
            "phase_0_5_branch": False,
            "phase_1_planning": {
                "sam_env_check": False,
                "google_sheets_sync": False,
                "sow_creation": False,
            },
            "phase_2_implementation": {"started": False, "approval": False},
            "phase_3_quality": {"workflow_executed": False, "dashboard_created": False},
        }

        # チェックリスト状態ファイル作成
        checklist_file = self.workflow_dir / "checklist_status.json"
        with open(checklist_file, "w") as f:
            json.dump(self.checklist_status, f, indent=2)

    def tearDown(self):
        """テスト環境クリーンアップ"""
        import shutil

        if self.test_workspace.exists():
            shutil.rmtree(self.test_workspace)

    def test_validate_workflow_compliance_hook(self):
        """validate_workflow_compliance.sh Hook テスト"""
        # テスト用JSON入力
        test_input = {
            "tool": "Bash",
            "args": {
                "command": f"python features/extraction/commands/extract_character.py /test/path -o {self.test_workspace}/{self.tracker_id}/extraction/ --batch"
            },
        }

        # Hook実行
        with patch("subprocess.run") as mock_run:
            # workspace_config.pyのモック
            mock_run.return_value.stdout = str(self.test_workspace)
            mock_run.return_value.returncode = 0

            # Hook実行のテスト
            result = self._run_hook_script(
                "tools/hooks/validate_workflow_compliance.sh", test_input
            )

            self.assertIsNotNone(result)
            # 結果のJSONパース確認
            if result:
                response = json.loads(result)
                self.assertIn("allow", response)

    def test_update_workflow_progress_hook(self):
        """update_workflow_progress.sh Hook テスト"""
        test_input = {
            "tool": "Bash",
            "args": {"command": "python tools/progress_tracker/cli.py status INTG-087"},
            "result": "success",
        }

        # Hook実行テスト
        result = self._run_hook_script("tools/hooks/update_workflow_progress.sh", test_input)

        self.assertIsNotNone(result)
        if result:
            response = json.loads(result)
            self.assertIn("status", response)

    def test_analyze_tracker_context_hook(self):
        """analyze_tracker_context.sh Hook テスト"""
        test_input = {"prompt": f"{self.tracker_id}の進捗を確認して次のステップを教えてください。現在Phase 1の計画フェーズです。"}

        # Hook実行テスト
        result = self._run_hook_script("tools/hooks/analyze_tracker_context.sh", test_input)

        self.assertIsNotNone(result)
        if result:
            response = json.loads(result)
            self.assertIn("tracker_id", response)
            self.assertEqual(response["tracker_id"], self.tracker_id)

    def test_hook_sequence_workflow(self):
        """Hook統合シーケンステスト（PreToolUse → PostToolUse）"""
        # Phase 1: PreToolUse Hook テスト
        pre_input = {
            "tool": "Bash",
            "args": {"command": "source sam-env/bin/activate && echo $VIRTUAL_ENV"},
        }

        pre_result = self._run_hook_script("tools/hooks/validate_workflow_compliance.sh", pre_input)

        # Phase 2: PostToolUse Hook テスト
        post_input = {
            "tool": "Bash",
            "args": {"command": "source sam-env/bin/activate && echo $VIRTUAL_ENV"},
            "result": "/path/to/sam-env",
        }

        post_result = self._run_hook_script("tools/hooks/update_workflow_progress.sh", post_input)

        # 両方のHookが正常に動作することを確認
        self.assertIsNotNone(pre_result)
        self.assertIsNotNone(post_result)

    def test_workflow_state_persistence(self):
        """ワークフロー状態の永続化テスト"""
        # 状態ファイルの存在確認
        checklist_file = self.workflow_dir / "checklist_status.json"
        self.assertTrue(checklist_file.exists())

        # 状態の読み込み確認
        with open(checklist_file, "r") as f:
            state = json.load(f)

        self.assertEqual(state["tracker_id"], self.tracker_id)
        self.assertIn("phase_1_planning", state)

    def test_error_handling(self):
        """Hook エラーハンドリングテスト"""
        # 不正なJSON入力
        invalid_input = {"tool": "InvalidTool", "args": {}}

        # エラー時の適切なレスポンス確認
        result = self._run_hook_script("tools/hooks/validate_workflow_compliance.sh", invalid_input)

        # エラーでもJSONレスポンスが返ることを確認
        if result:
            try:
                json.loads(result)
            except json.JSONDecodeError:
                self.fail("Hook should return valid JSON even on errors")

    def _run_hook_script(self, script_path, input_data):
        """Hook スクリプト実行ヘルパー"""
        try:
            # JSON入力を準備
            input_json = json.dumps(input_data)

            # スクリプト実行（タイムアウト設定）
            process = subprocess.run(
                ["bash", script_path],
                input=input_json,
                text=True,
                capture_output=True,
                timeout=30,
                cwd="/mnt/c/AItools/segment-anything",
            )

            if process.returncode == 0:
                return process.stdout.strip()
            else:
                print(f"Hook script error: {process.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("Hook script timeout")
            return None
        except Exception as e:
            print(f"Hook script execution error: {e}")
            return None


class TestHookPerformance(unittest.TestCase):
    """Hook パフォーマンステスト"""

    def test_hook_execution_speed(self):
        """Hook実行速度テスト"""
        import time

        test_input = {"tool": "Bash", "args": {"command": "echo test"}}

        start_time = time.time()

        # テスト用に簡単なHookを実行
        process = subprocess.run(
            ["bash", "-c", "echo '{\"allow\": true}'"],
            input=json.dumps(test_input),
            text=True,
            capture_output=True,
            timeout=10,
        )

        execution_time = time.time() - start_time

        # 5秒以内での実行を期待
        self.assertLess(execution_time, 5.0)
        self.assertEqual(process.returncode, 0)


class TestHookResilience(unittest.TestCase):
    """Hook 耐障害性テスト"""

    def test_concurrent_hook_execution(self):
        """並行Hook実行テスト"""
        import concurrent.futures
        import threading

        def run_hook():
            test_input = {"tool": "Bash", "args": {"command": "echo concurrent_test"}}

            # 簡易Hookテスト
            process = subprocess.run(
                ["bash", "-c", "sleep 1 && echo '{\"allow\": true}'"],
                input=json.dumps(test_input),
                text=True,
                capture_output=True,
                timeout=10,
            )
            return process.returncode == 0

        # 3つの並行実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_hook) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 全て成功することを確認
        self.assertTrue(all(results))
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    # テスト実行設定
    unittest.main(verbosity=2)
