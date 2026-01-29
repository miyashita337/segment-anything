#!/usr/bin/env python3
"""
Workflow Hooks Unit Tests
KIRO-016: Layer 1 Hook Implementation

Test cases:
    H1: ワークフロー正常でBash実行 → exit 0でツール実行許可
    H2: 承認待ち状態でBash実行 → exit 2でツール実行中断
    H3: 不正フェーズ状態でEdit実行 → exit 2 + エラーメッセージ
    H4: Bash完了後 → PostToolUseで完了条件チェック
    H5: エラー検出時 → stdout にSkill利用提案
    H6: settings.json削除時 → フックなしで通常動作
    H7: workflow_cli.pyでError発生 → Skill更新提案メッセージ表示
    H8: workflow_cli.py正常終了 → 提案メッセージなし
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestWorkflowHooks(unittest.TestCase):
    """Workflow Hook tests"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent.parent.parent
        cls.hooks_dir = cls.project_root / ".claude" / "hooks"
        cls.settings_file = cls.project_root / ".claude" / "settings.json"
        cls.precheck_script = cls.hooks_dir / "workflow_precheck.sh"
        cls.postcheck_script = cls.hooks_dir / "workflow_postcheck.sh"

    def test_hook_files_exist(self):
        """Verify all hook files exist"""
        self.assertTrue(self.settings_file.exists(), "settings.json should exist")
        self.assertTrue(self.precheck_script.exists(), "workflow_precheck.sh should exist")
        self.assertTrue(self.postcheck_script.exists(), "workflow_postcheck.sh should exist")

    def test_settings_json_valid(self):
        """Verify settings.json is valid JSON with correct structure"""
        with open(self.settings_file) as f:
            settings = json.load(f)

        self.assertIn("hooks", settings)
        self.assertIn("PreToolUse", settings["hooks"])
        self.assertIn("PostToolUse", settings["hooks"])

        # Verify PreToolUse structure
        pre_hooks = settings["hooks"]["PreToolUse"]
        self.assertIsInstance(pre_hooks, list)
        self.assertGreater(len(pre_hooks), 0)
        self.assertIn("matcher", pre_hooks[0])
        self.assertIn("hooks", pre_hooks[0])

        # Verify PostToolUse structure
        post_hooks = settings["hooks"]["PostToolUse"]
        self.assertIsInstance(post_hooks, list)
        self.assertGreater(len(post_hooks), 0)

    def test_scripts_executable(self):
        """Verify hook scripts are executable"""
        self.assertTrue(os.access(self.precheck_script, os.X_OK),
                        "workflow_precheck.sh should be executable")
        self.assertTrue(os.access(self.postcheck_script, os.X_OK),
                        "workflow_postcheck.sh should be executable")

    def test_h1_normal_workflow_allows_execution(self):
        """H1: ワークフロー正常でBash実行 → exit 0でツール実行許可"""
        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env={**os.environ, "WORKFLOW_DB": "nonexistent.db"}
        )
        # Without DB, should allow (exit 0)
        self.assertEqual(result.returncode, 0,
                         f"Should exit 0 when no workflow DB exists. stderr: {result.stderr}")

    def test_h4_postcheck_normal_execution(self):
        """H4: Bash完了後 → PostToolUseで完了条件チェック（正常終了）"""
        result = subprocess.run(
            [str(self.postcheck_script)],
            input="Normal command output without errors",
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        # PostToolUse always returns 0
        self.assertEqual(result.returncode, 0,
                         "PostToolUse should always return 0")
        # Output should be passed through
        self.assertIn("Normal command output", result.stdout)

    def test_h5_error_detection_skill_suggestion(self):
        """H5: エラー検出時 → stdout にSkill利用提案"""
        # Use echo to pipe input to the script
        error_output = (
            "Traceback (most recent call last):\n"
            "  File \"test.py\", line 1, in <module>\n"
            "    raise Exception(\"Test error\")\n"
            "Exception: Test error"
        )
        result = subprocess.run(
            ["bash", "-c", f"echo '{error_output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        # Should suggest checking virtual environment
        self.assertIn("sam-env", result.stderr,
                      "Should suggest virtual environment check on Python error")

    def test_h7_workflow_cli_error_skill_suggestion(self):
        """H7: workflow_cli.pyでError発生 → Skill更新提案メッセージ表示"""
        error_output = """
        workflow_cli.py execution
        Error: Something went wrong
        """
        result = subprocess.run(
            [str(self.postcheck_script)],
            input=error_output,
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        # Should suggest workflow troubleshoot skill
        self.assertIn("workflow-troubleshoot", result.stderr,
                      "Should suggest workflow-troubleshoot skill on workflow_cli error")

    def test_h8_workflow_cli_normal_no_suggestion(self):
        """H8: workflow_cli.py正常終了 → 提案メッセージなし"""
        normal_output = """
        workflow_cli.py execution
        Status: OK
        Completed successfully
        """
        result = subprocess.run(
            [str(self.postcheck_script)],
            input=normal_output,
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        # Should NOT suggest workflow-troubleshoot on success
        self.assertNotIn("workflow-troubleshoot", result.stderr,
                         "Should not suggest skill on successful workflow execution")

    def test_precheck_syntax_valid(self):
        """Verify workflow_precheck.sh has valid bash syntax"""
        result = subprocess.run(
            ["bash", "-n", str(self.precheck_script)],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                         f"Syntax error in precheck script: {result.stderr}")

    def test_postcheck_syntax_valid(self):
        """Verify workflow_postcheck.sh has valid bash syntax"""
        result = subprocess.run(
            ["bash", "-n", str(self.postcheck_script)],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                         f"Syntax error in postcheck script: {result.stderr}")

    def test_precheck_failsafe_on_missing_jq(self):
        """Verify failsafe: precheck allows execution when jq is missing"""
        # Skip this test if jq is not installed (can't test missing jq behavior)
        # Instead, verify the script contains failsafe logic
        with open(self.precheck_script) as f:
            script_content = f.read()

        # Verify failsafe logic exists in the script
        self.assertIn("jq", script_content,
                      "Script should check for jq command")
        self.assertIn("exit 0", script_content,
                      "Script should have failsafe exit 0")
        # Verify the failsafe message pattern
        self.assertIn("WARNING", script_content,
                      "Script should log WARNING when jq is not found")

    def test_postcheck_git_error_detection(self):
        """Verify postcheck detects git errors"""
        git_error = "fatal: not a git repository"
        result = subprocess.run(
            [str(self.postcheck_script)],
            input=git_error,
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Git issue", result.stderr,
                      "Should detect git errors")

    def test_postcheck_permission_error_detection(self):
        """Verify postcheck detects permission errors"""
        perm_error = "Permission denied: /some/path"
        result = subprocess.run(
            [str(self.postcheck_script)],
            input=perm_error,
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Permission issue", result.stderr,
                      "Should detect permission errors")


class TestWorkflowHooksWithMockDB(unittest.TestCase):
    """Workflow Hook tests with mock database"""

    def setUp(self):
        """Create temporary directory and mock database"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.precheck_script = self.project_root / ".claude" / "hooks" / "workflow_precheck.sh"

        # Create mock workflow database
        self.mock_db = Path(self.temp_dir) / "workflow_state.db"
        self._create_mock_db()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_db(self):
        """Create mock SQLite database for testing"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_states (
                tracker_id TEXT PRIMARY KEY,
                current_phase TEXT,
                current_step TEXT,
                status TEXT DEFAULT 'active',
                requires_approval INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY,
                tracker_id TEXT,
                step_id TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _insert_workflow(self, tracker_id, phase, step, requires_approval=0):
        """Insert workflow state into mock database"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO workflow_states "
            "(tracker_id, current_phase, current_step, status, requires_approval) "
            "VALUES (?, ?, ?, 'active', ?)",
            (tracker_id, phase, step, requires_approval)
        )
        conn.commit()
        conn.close()

    def _insert_approval(self, tracker_id, step_id, status="approved"):
        """Insert approval into mock database"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO approvals (tracker_id, step_id, status) VALUES (?, ?, ?)",
            (tracker_id, step_id, status)
        )
        conn.commit()
        conn.close()

    def test_h1_active_workflow_without_approval_required(self):
        """H1: Active workflow without approval required → exit 0"""
        self._insert_workflow("TEST-001", "phase_1", "implementation", requires_approval=0)

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 0,
                         f"Should allow execution when no approval required. stderr: {result.stderr}")

    def test_h2_approval_required_without_approval(self):
        """H2: 承認待ち状態でBash実行 → exit 2でツール実行中断"""
        self._insert_workflow("TEST-002", "phase_1", "review", requires_approval=1)

        env = os.environ.copy()
        # Use absolute path for WORKFLOW_DB
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 2,
                         f"Should block (exit 2) when approval required but not approved. "
                         f"stderr: {result.stderr}")
        self.assertIn("requires approval", result.stderr.lower(),
                      "Should indicate approval is required")

    def test_h2_approval_required_with_approval(self):
        """H2 variation: Approval required AND approved → exit 0"""
        self._insert_workflow("TEST-003", "phase_1", "review", requires_approval=1)
        self._insert_approval("TEST-003", "review", "approved")

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 0,
                         f"Should allow execution when approval exists. stderr: {result.stderr}")


class TestWorkflowPostcheckSkillSuggestions(unittest.TestCase):
    """Test Skill suggestions based on workflow step"""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent.parent.parent
        cls.postcheck_script = cls.project_root / ".claude" / "hooks" / "workflow_postcheck.sh"

    def test_implementation_step_suggests_skills(self):
        """Implementation step should suggest pre-implementation skills"""
        # Must include workflow_cli.py to trigger the skill suggestion logic
        output = "workflow_cli.py step KIRO-016\\n次のステップ: implementation"
        result = subprocess.run(
            ["bash", "-c", f"echo -e '{output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("pre-implementation-check", result.stderr,
                      "Should suggest /pre-implementation-check")
        self.assertIn("impact-analysis", result.stderr,
                      "Should suggest /impact-analysis")
        self.assertIn("test-first", result.stderr,
                      "Should suggest /test-first")

    def test_testing_step_suggests_test_skill(self):
        """Testing step should suggest test-first skill"""
        output = "workflow_cli.py step KIRO-016\\n次のステップ: testing"
        result = subprocess.run(
            ["bash", "-c", f"echo -e '{output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("test-first", result.stderr,
                      "Should suggest /test-first for testing phase")

    def test_review_step_suggests_architecture_review(self):
        """Review step should suggest architecture-review skill"""
        output = "workflow_cli.py step KIRO-016\\n次のステップ: review"
        result = subprocess.run(
            ["bash", "-c", f"echo -e '{output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("architecture-review", result.stderr,
                      "Should suggest /architecture-review for review phase")

    def test_other_step_no_skill_suggestion(self):
        """Other steps should not suggest implementation skills"""
        output = "workflow_cli.py step KIRO-016\\n次のステップ: cleanup"
        result = subprocess.run(
            ["bash", "-c", f"echo -e '{output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("pre-implementation-check", result.stderr,
                         "Should not suggest skills for non-implementation steps")


class TestSkillFilesExist(unittest.TestCase):
    """Test that Skill placeholder files exist"""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent.parent.parent
        cls.commands_dir = cls.project_root / ".claude" / "commands"

    def test_commands_directory_exists(self):
        """Commands directory should exist"""
        self.assertTrue(self.commands_dir.exists(),
                        ".claude/commands directory should exist")

    def test_pre_implementation_check_skill_exists(self):
        """pre-implementation-check.md should exist"""
        skill_file = self.commands_dir / "pre-implementation-check.md"
        self.assertTrue(skill_file.exists(),
                        "pre-implementation-check.md should exist")

    def test_impact_analysis_skill_exists(self):
        """impact-analysis.md should exist"""
        skill_file = self.commands_dir / "impact-analysis.md"
        self.assertTrue(skill_file.exists(),
                        "impact-analysis.md should exist")

    def test_test_first_skill_exists(self):
        """test-first.md should exist"""
        skill_file = self.commands_dir / "test-first.md"
        self.assertTrue(skill_file.exists(),
                        "test-first.md should exist")

    def test_architecture_review_skill_exists(self):
        """architecture-review.md should exist"""
        skill_file = self.commands_dir / "architecture-review.md"
        self.assertTrue(skill_file.exists(),
                        "architecture-review.md should exist")


class TestKIRO016ExtractionApprovalRequired(unittest.TestCase):
    """KIRO-016: 画像抽出判断の強制確認機能テスト"""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent.parent.parent
        cls.precheck_script = cls.project_root / ".claude" / "hooks" / "workflow_precheck.sh"
        cls.postcheck_script = cls.project_root / ".claude" / "hooks" / "workflow_postcheck.sh"

    def test_subagent_extraction_approval_required_in_controller(self):
        """subagent_extraction は approval_required=True であること"""
        import sys
        sys.path.insert(0, str(self.project_root))
        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()
        step = controller.workflow_config.get("subagent_extraction")
        self.assertIsNotNone(step, "subagent_extraction step not found")
        self.assertTrue(
            step.approval_required,
            "subagent_extraction must have approval_required=True (KIRO-016)",
        )

    def test_subagent_extraction_not_auto_executable_in_controller(self):
        """subagent_extraction は auto_executable=False であること"""
        import sys
        sys.path.insert(0, str(self.project_root))
        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()
        step = controller.workflow_config.get("subagent_extraction")
        self.assertIsNotNone(step, "subagent_extraction step not found")
        self.assertFalse(
            step.auto_executable,
            "subagent_extraction must have auto_executable=False (KIRO-016)",
        )

    def test_subagent_validation_approval_required_in_controller(self):
        """subagent_validation は approval_required=True であること"""
        import sys
        sys.path.insert(0, str(self.project_root))
        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()
        step = controller.workflow_config.get("subagent_validation")
        self.assertIsNotNone(step, "subagent_validation step not found")
        self.assertTrue(
            step.approval_required,
            "subagent_validation must have approval_required=True (KIRO-016)",
        )

    def test_subagent_validation_not_auto_executable_in_controller(self):
        """subagent_validation は auto_executable=False であること"""
        import sys
        sys.path.insert(0, str(self.project_root))
        from tools.interface.workflow_controller import WorkflowController

        controller = WorkflowController()
        step = controller.workflow_config.get("subagent_validation")
        self.assertIsNotNone(step, "subagent_validation step not found")
        self.assertFalse(
            step.auto_executable,
            "subagent_validation must have auto_executable=False (KIRO-016)",
        )

    def test_precheck_contains_extraction_step_block_logic(self):
        """workflow_precheck.sh にシングルトン抽出制御処理が含まれること"""
        with open(self.precheck_script, "r", encoding="utf-8") as f:
            content = f.read()

        # KIRO-016で追加されたシングルトン抽出制御処理の確認
        self.assertIn(
            "check_extraction_in_progress",
            content,
            "check_extraction_in_progress function should be defined in precheck hook",
        )
        self.assertIn(
            "subagent_extraction",
            content,
            "subagent_extraction step should be in extraction check",
        )
        self.assertIn(
            "subagent_validation",
            content,
            "subagent_validation step should be in extraction check",
        )
        self.assertIn(
            "シングルトン制御",
            content,
            "Japanese message for singleton control should be present",
        )

    def test_postcheck_contains_extraction_warning_message(self):
        """workflow_postcheck.sh に画像抽出警告メッセージが含まれること"""
        with open(self.postcheck_script, "r", encoding="utf-8") as f:
            content = f.read()

        # KIRO-016で追加された警告メッセージの確認
        self.assertIn(
            "画像抽出フェーズに入ります",
            content,
            "Extraction phase warning should be present in postcheck hook",
        )
        self.assertIn(
            "次のステップ: subagent_extraction",
            content,
            "subagent_extraction step detection should be present",
        )
        self.assertIn(
            "Claudeが勝手にスキップすることは禁止されています",
            content,
            "Skip prevention message should be present",
        )

    def test_postcheck_extraction_warning_output(self):
        """postcheck が subagent_extraction ステップで警告を出力すること"""
        output = "workflow_cli.py step KIRO-016\\n次のステップ: subagent_extraction"
        result = subprocess.run(
            ["bash", "-c", f"echo -e '{output}' | {self.postcheck_script}"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("画像抽出フェーズに入ります", result.stderr,
                      "Should show extraction phase warning")
        self.assertIn("Claudeが勝手にスキップすることは禁止されています", result.stderr,
                      "Should show skip prevention message")


class TestKIRO016ExtractionStepBlocking(unittest.TestCase):
    """KIRO-016: 抽出ステップブロック機能の統合テスト"""

    def setUp(self):
        """Create temporary directory and mock database"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.precheck_script = self.project_root / ".claude" / "hooks" / "workflow_precheck.sh"

        # Create mock workflow database
        self.mock_db = Path(self.temp_dir) / "workflow_state.db"
        self._create_mock_db()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_db(self):
        """Create mock SQLite database for testing"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_states (
                tracker_id TEXT PRIMARY KEY,
                current_phase TEXT,
                current_step TEXT,
                status TEXT DEFAULT 'active',
                requires_approval INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY,
                tracker_id TEXT,
                step_id TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _insert_workflow(self, tracker_id, phase, step, requires_approval=0):
        """Insert workflow state into mock database"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO workflow_states "
            "(tracker_id, current_phase, current_step, status, requires_approval) "
            "VALUES (?, ?, ?, 'active', ?)",
            (tracker_id, phase, step, requires_approval)
        )
        conn.commit()
        conn.close()

    def _insert_approval(self, tracker_id, step_id, status="approved"):
        """Insert approval into mock database"""
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO approvals (tracker_id, step_id, status) VALUES (?, ?, ?)",
            (tracker_id, step_id, status)
        )
        conn.commit()
        conn.close()

    def test_subagent_extraction_blocked_singleton_control(self):
        """subagent_extraction はシングルトン制御でブロックされること"""
        # KIRO-016: シングルトン制御 - 抽出ステップにあるワークフローが存在すれば無条件ブロック
        self._insert_workflow("TEST-EXT-001", "phase_2", "subagent_extraction", requires_approval=0)

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 2,
                         f"Should block (exit 2) subagent_extraction due to singleton control. "
                         f"stderr: {result.stderr}")
        self.assertIn("シングルトン制御", result.stderr,
                      "Should show singleton control message")

    def test_subagent_validation_blocked_singleton_control(self):
        """subagent_validation はシングルトン制御でブロックされること"""
        # KIRO-016: シングルトン制御 - 抽出ステップにあるワークフローが存在すれば無条件ブロック
        self._insert_workflow("TEST-VAL-001", "phase_2", "subagent_validation", requires_approval=0)

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 2,
                         f"Should block (exit 2) subagent_validation due to singleton control. "
                         f"stderr: {result.stderr}")
        self.assertIn("シングルトン制御", result.stderr,
                      "Should show singleton control message")

    def test_multiple_extraction_records_all_blocked(self):
        """複数ワークフローで抽出ステップが1つでもあればブロック"""
        # KIRO-016: シングルトン制御 - いずれかのレコードが抽出ステップならブロック
        # 古いレコード: 非抽出ステップ
        self._insert_workflow("TEST-IMPL-001", "phase_2", "implementation", requires_approval=0)
        # 新しいレコード: 抽出ステップ
        self._insert_workflow("TEST-EXT-002", "phase_2", "subagent_extraction", requires_approval=0)

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 2,
                         f"Should block (exit 2) when any workflow is in extraction step. "
                         f"stderr: {result.stderr}")

    def test_old_extraction_new_non_extraction_blocked(self):
        """古い抽出ワークフロー + 新しい非抽出でもブロック"""
        # KIRO-016: 最新が非抽出でも、古いレコードが抽出ステップならブロック
        import sqlite3
        conn = sqlite3.connect(str(self.mock_db))
        cursor = conn.cursor()
        # 古いレコード: 抽出ステップ
        cursor.execute(
            "INSERT INTO workflow_states "
            "(tracker_id, current_phase, current_step, status, requires_approval, updated_at) "
            "VALUES (?, ?, ?, 'active', 0, datetime('now', '-1 hour'))",
            ("TEST-EXT-OLD", "phase_2", "subagent_extraction")
        )
        # 新しいレコード: 非抽出ステップ（最新）
        cursor.execute(
            "INSERT INTO workflow_states "
            "(tracker_id, current_phase, current_step, status, requires_approval, updated_at) "
            "VALUES (?, ?, ?, 'active', 0, datetime('now'))",
            ("TEST-IMPL-NEW", "phase_2", "implementation")
        )
        conn.commit()
        conn.close()

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 2,
                         f"Should block when older workflow is in extraction. "
                         f"stderr: {result.stderr}")

    def test_non_extraction_step_not_blocked_by_extraction_logic(self):
        """非抽出ステップは抽出ロジックでブロックされないこと"""
        self._insert_workflow("TEST-IMPL-001", "phase_2", "implementation", requires_approval=0)

        env = os.environ.copy()
        env["WORKFLOW_DB"] = str(self.mock_db)

        result = subprocess.run(
            [str(self.precheck_script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=env
        )
        self.assertEqual(result.returncode, 0,
                         f"Should allow (exit 0) non-extraction steps. "
                         f"stderr: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
