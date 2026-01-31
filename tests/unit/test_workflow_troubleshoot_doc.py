#!/usr/bin/env python3
"""
workflow-troubleshoot.md ドキュメント検証テスト

統合されたワークフロートラブルシューティングガイドの
構造と内容を検証するユニットテスト
"""

import re
from pathlib import Path

import pytest


@pytest.fixture
def doc_path():
    """テスト対象ドキュメントのパス"""
    return Path(__file__).parents[2] / ".claude/commands/workflow-troubleshoot.md"


@pytest.fixture
def doc_content(doc_path):
    """ドキュメント内容を読み込む"""
    if not doc_path.exists():
        pytest.skip(f"Document not found: {doc_path}")
    return doc_path.read_text(encoding="utf-8")


class TestDocumentExists:
    """ドキュメント存在確認テスト"""

    def test_file_exists(self, doc_path):
        """ファイルが存在すること"""
        assert doc_path.exists(), f"File not found: {doc_path}"

    def test_file_not_empty(self, doc_content):
        """ファイルが空でないこと"""
        assert len(doc_content) > 0


class TestFrontmatter:
    """フロントマター検証テスト"""

    def test_has_frontmatter(self, doc_content):
        """フロントマターが存在すること"""
        assert doc_content.startswith("---"), "Missing frontmatter"
        assert doc_content.count("---") >= 2, "Incomplete frontmatter"

    def test_has_name_field(self, doc_content):
        """name フィールドが存在すること"""
        assert re.search(r'^name:\s*"workflow-troubleshoot"', doc_content, re.MULTILINE)

    def test_has_description_field(self, doc_content):
        """description フィールドが存在すること"""
        assert re.search(r'^description:\s*".*"', doc_content, re.MULTILINE)


class TestRequiredSections:
    """必須セクション検証テスト"""

    @pytest.mark.parametrize(
        "section_title",
        [
            "⚠️ 重要な前提条件",
            "Quick Reference",
            "環境セットアップ問題",
            "Step-Specific Error Patterns",
            "Common Issues",
            "Google Sheets接続エラー",
        ],
    )
    def test_required_section_exists(self, doc_content, section_title):
        """必須セクションが存在すること"""
        assert section_title in doc_content, f"Missing section: {section_title}"


class TestEnvironmentSection:
    """環境セットアップ問題セクション検証テスト（fix-environment.md統合）"""

    @pytest.mark.parametrize(
        "error_type",
        [
            "sympy循環インポートエラー",
            "torchvision インポートエラー",
            "CUDA関連エラー",
            "診断コマンド",
        ],
    )
    def test_environment_error_types(self, doc_content, error_type):
        """環境エラータイプが記載されていること"""
        assert error_type in doc_content, f"Missing error type: {error_type}"

    def test_sympy_solution_exists(self, doc_content):
        """sympyエラーの解決方法が記載されていること"""
        assert "pip install --force-reinstall sympy" in doc_content

    def test_torchvision_solution_exists(self, doc_content):
        """torchvisionエラーの解決方法が記載されていること"""
        assert "pip install torchvision --upgrade" in doc_content


class TestBaselineSection:
    """BaseLine参照エラーセクション検証テスト（baseline-reference.md統合）"""

    def test_baseline_section_exists(self, doc_content):
        """BaseLine参照エラーセクションが存在すること"""
        assert "BaseLine参照エラー" in doc_content

    def test_baseline_error_messages(self, doc_content):
        """BaseLineエラーメッセージ例が記載されていること"""
        assert "BaseLineが有効な値ではありません" in doc_content
        assert "p値が有効な値ではありません" in doc_content
        assert "効果サイズが有効な値ではありません" in doc_content

    def test_baseline_solution_script(self, doc_content):
        """BaseLine設定スクリプトが含まれていること"""
        assert "statistical_analysis" in doc_content
        assert "baseline_score" in doc_content

    def test_recommended_tracker_table(self, doc_content):
        """推奨参照トラッカー一覧が存在すること"""
        assert "推奨参照トラッカー" in doc_content
        assert "KIRO-016" in doc_content


class TestWorkflowCommands:
    """ワークフローコマンド検証テスト"""

    @pytest.mark.parametrize(
        "command",
        [
            "workflow_cli.py plan",
            "workflow_cli.py create",
            "workflow_cli.py instructions",
            "workflow_cli.py step",
            "workflow_cli.py status",
            "workflow_cli.py approvals",
        ],
    )
    def test_command_documented(self, doc_content, command):
        """ワークフローコマンドが記載されていること"""
        assert command in doc_content, f"Missing command: {command}"


class TestStepSpecificErrors:
    """ステップ別エラーパターン検証テスト"""

    @pytest.mark.parametrize(
        "step_name",
        [
            "quality_workflow ステップ",
            "dashboard_generation ステップ",
            "final_approval ステップ",
            "extraction ステップ",
        ],
    )
    def test_step_section_exists(self, doc_content, step_name):
        """ステップセクションが存在すること"""
        assert step_name in doc_content, f"Missing step section: {step_name}"


class TestCodeBlocks:
    """コードブロック検証テスト"""

    def test_has_bash_code_blocks(self, doc_content):
        """bashコードブロックが存在すること"""
        assert "```bash" in doc_content

    def test_has_python_code_blocks(self, doc_content):
        """Pythonコードブロックが存在すること"""
        assert "```python" in doc_content

    def test_code_blocks_closed(self, doc_content):
        """コードブロックが閉じられていること"""
        # ``` の総数は偶数であるべき（開始+終了）
        triple_backtick_count = len(re.findall(r"^```", doc_content, re.MULTILINE))
        assert triple_backtick_count % 2 == 0, (
            f"Unclosed code blocks: found {triple_backtick_count} ``` markers (should be even)"
        )


class TestRelatedFiles:
    """関連ファイルセクション検証テスト"""

    def test_related_files_section_exists(self, doc_content):
        """関連ファイルセクションが存在すること"""
        assert "関連ファイル" in doc_content

    @pytest.mark.parametrize(
        "related_file",
        [
            "dashboard_generator.py",
            "workflow_controller.py",
        ],
    )
    def test_related_file_listed(self, doc_content, related_file):
        """関連ファイルがリストされていること"""
        assert related_file in doc_content, f"Missing related file: {related_file}"


class TestNoDeletedContent:
    """削除済みファイルの内容が統合されていることを確認"""

    def test_fix_environment_content_integrated(self, doc_content):
        """fix-environment.mdの内容が統合されていること"""
        # fix-environment.mdの主要コンテンツ
        assert "sam-env/bin/activate" in doc_content
        assert "pip cache purge" in doc_content

    def test_baseline_reference_content_integrated(self, doc_content):
        """baseline-reference.mdの内容が統合されていること"""
        # baseline-reference.mdの主要コンテンツ
        assert "extraction_result.json" in doc_content
        assert "average_quality_score" in doc_content
