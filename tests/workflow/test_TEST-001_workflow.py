#!/usr/bin/env python3
"""TEST-001ワークフロー統合テスト

Phase 2ステップ7: 統合テスト実装
- 抽出結果検証
- ワークスペース構造確認
- 品質評価システム動作確認
"""

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestTEST001Workflow(unittest.TestCase):
    """TEST-001ワークフロー統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.tracker_id = "TEST-001"

        # テスト用一時ディレクトリを使用
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir) / self.tracker_id
        self.extraction_dir = self.workspace / "extraction"

        # テスト用ディレクトリ構造作成
        for dir_name in ["extraction", "dashboard", "quality", "tests"]:
            (self.workspace / dir_name).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """テスト後クリーンアップ"""
        import shutil

        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_workspace_structure(self):
        """ワークスペース構造検証"""
        # 必須ディレクトリ存在確認
        required_dirs = ["extraction", "dashboard", "quality", "tests"]
        for dir_name in required_dirs:
            dir_path = self.workspace / dir_name
            self.assertTrue(dir_path.exists(), f"必須ディレクトリ {dir_name} が存在しません")

        # 計画書存在確認
        plan_file = self.workspace / f"{self.tracker_id}_implementation_plan.md"
        self.assertTrue(plan_file.exists(), "実装計画書が存在しません")

    def test_extraction_results(self):
        """抽出結果検証"""
        # 抽出ファイル存在確認
        extracted_files = list(self.extraction_dir.glob("extracted_*.jpg"))
        self.assertGreater(len(extracted_files), 0, "抽出ファイルが存在しません")

        # 26枚の成功を確認（ログから）
        self.assertEqual(len(extracted_files), 26, f"期待される抽出数26枚に対し、実際は{len(extracted_files)}枚")

        # 0バイトファイル不在確認
        for file in extracted_files:
            self.assertGreater(file.stat().st_size, 0, f"{file.name}が0バイトファイルです")

    def test_extraction_quality(self):
        """抽出品質検証"""
        # 各ファイルサイズ確認（50KB以上を基準）
        quality_threshold = 50 * 1024  # 50KB
        high_quality_count = 0

        extracted_files = list(self.extraction_dir.glob("extracted_*.jpg"))
        for file in extracted_files:
            if file.stat().st_size > quality_threshold:
                high_quality_count += 1

        # 80%以上が高品質であることを確認
        quality_ratio = high_quality_count / len(extracted_files) if extracted_files else 0
        self.assertGreaterEqual(quality_ratio, 0.8, f"高品質ファイル比率 {quality_ratio:.1%} が基準80%未満")

    def test_missing_files(self):
        """欠損ファイル確認"""
        # 期待されるファイル番号
        expected_numbers = set(range(1, 29))  # 1-28

        # 実際に存在するファイル番号
        actual_numbers = set()
        for file in self.extraction_dir.glob("extracted_*.jpg"):
            num = int(file.stem.split("_")[1].lstrip("0"))
            actual_numbers.add(num)

        # 欠損ファイル特定
        missing = expected_numbers - actual_numbers
        self.assertEqual(missing, {2, 27}, f"予期しない欠損ファイル: {missing}")  # ログから2と27が失敗

    def test_checkpoint_system(self):
        """チェックポイントシステム検証"""
        checkpoint_dir = self.extraction_dir / ".checkpoint"
        self.assertTrue(checkpoint_dir.exists(), "チェックポイントディレクトリが存在しません")

        # チェックポイントファイル確認
        checkpoint_file = checkpoint_dir / "batch_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                self.assertIn("processed", checkpoint_data)
                self.assertIn("failed", checkpoint_data)

                # 処理済み26件確認
                self.assertEqual(len(checkpoint_data.get("processed", [])), 26, "チェックポイントの処理済み数が不正")


if __name__ == "__main__":
    unittest.main()
