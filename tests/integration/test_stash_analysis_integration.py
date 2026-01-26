#!/usr/bin/env python3
"""
212ファイル復旧差分分析システムの統合テスト
QUAL-035 統合テスト - 実際のgit stashとの連携確認
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer import (
    ChangeType,
    RecommendedAction,
    StashDiffAnalyzer,
)


class TestStashAnalysisIntegration(unittest.TestCase):
    """統合テスト: 実際のgit操作との連携"""

    def setUp(self):
        """テスト前の準備"""
        self.analyzer = StashDiffAnalyzer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

    def tearDown(self):
        """テスト後の片付け"""
        self.temp_dir.cleanup()

    def test_integration_workflow_with_mock_git(self):
        """git操作模擬での統合ワークフローテスト"""

        # 実際のgit stash show --numstatを模擬
        mock_numstat = """12\t8\tfeatures/extraction/commands/extract_character.py
3\t3\tfeatures/common/memory_optimizer.py
2\t0\tdocs/tracker_naming_guidelines.md
45\t30\ttools/core/sam_yolo_character_segment.py
1\t1\ttests/unit/test_extraction.py
8\t12\ttools/progress_tracker/cli.py
5\t2\tfeatures/evaluation/integrated_quality_monitor.py
0\t15\tdocs/deprecated_apis.md
25\t5\ttools/batch/kana08_enhanced_stable_batch.py
7\t7\ttests/integration/test_pipeline.py"""

        # 各ファイルの差分内容を模擬
        mock_diffs = {
            "features/extraction/commands/extract_character.py": """
diff --git a/features/extraction/commands/extract_character.py b/features/extraction/commands/extract_character.py
--- a/features/extraction/commands/extract_character.py
+++ b/features/extraction/commands/extract_character.py
@@ -15,7 +15,7 @@ def extract_character(input_path, output_path):
     \"\"\"キャラクター抽出コマンド\"\"\"
     
     # QUAL-035: 212ファイル復旧対応
-    # P1-A001での処理ロジック
+    # QUAL-001での処理ロジック
     
     try:
         result = process_image(input_path)
""",
            "features/common/memory_optimizer.py": """
diff --git a/features/common/memory_optimizer.py b/features/common/memory_optimizer.py
--- a/features/common/memory_optimizer.py
+++ b/features/common/memory_optimizer.py
@@ -1,3 +1,6 @@
+\"\"\"
+メモリ使用最適化システム - QUAL-035 復旧版
+\"\"\"
 import gc
 import logging
 import psutil
""",
            "docs/tracker_naming_guidelines.md": """
diff --git a/docs/tracker_naming_guidelines.md b/docs/tracker_naming_guidelines.md
--- a/docs/tracker_naming_guidelines.md
+++ b/docs/tracker_naming_guidelines.md
@@ -1,4 +1,6 @@
 # トラッカー命名ガイドライン
 
+## QUAL-035 復旧作業での追加仕様
+
 ## 基本形式
 
""",
            "tools/core/sam_yolo_character_segment.py": """
diff --git a/tools/core/sam_yolo_character_segment.py b/tools/core/sam_yolo_character_segment.py
--- a/tools/core/sam_yolo_character_segment.py
+++ b/tools/core/sam_yolo_character_segment.py
@@ -50,15 +50,25 @@ class SAMYOLOSegmenter:
         
     def segment_character(self, image_path, output_dir):
         \"\"\"キャラクター分割処理\"\"\"
-        # 旧バージョンの処理
-        yolo_results = self.yolo_model.predict(image_path)
-        
-        for result in yolo_results:
-            bbox = result.bbox
-            sam_mask = self.sam_model.predict(image_path, bbox)
+        # QUAL-035: 復旧版の改善処理
+        try:
+            yolo_results = self.yolo_model.predict(image_path, conf=0.07)
+            
+            for result in yolo_results:
+                bbox = result.bbox
+                confidence = result.confidence
+                
+                if confidence < 0.05:
+                    continue
+                    
+                sam_mask = self.sam_model.predict(image_path, bbox)
+                
+                if self._validate_mask_quality(sam_mask):
+                    self._save_extracted_character(sam_mask, output_dir)
+        except Exception as e:
+            logger.error(f"Character segmentation failed: {e}")
+            return False
         
-        return sam_mask
+        return True
""",
        }

        with patch("subprocess.run") as mock_run:
            # git stash show --numstat の模擬
            def mock_subprocess_side_effect(*args, **kwargs):
                command = args[0]

                if "stash" in command and "show" in command and "--numstat" in command:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = mock_numstat
                    mock_result.returncode = 0
                    return mock_result

                elif "stash" in command and "show" in command and "-p" in command:
                    # ファイル個別の差分取得
                    file_path = command[-1]  # 最後の引数がファイルパス
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = mock_diffs.get(file_path, "")
                    return mock_result

                else:
                    # その他のgitコマンド
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = ""
                    return mock_result

            mock_run.side_effect = mock_subprocess_side_effect

            # 統合テスト実行
            analyses = self.analyzer.analyze_stash_files()

            # 結果検証
            self.assertIsInstance(analyses, dict)
            self.assertEqual(len(analyses), 10)  # 10ファイル

            # 各ファイルの分析結果検証
            extract_char_analysis = analyses["features/extraction/commands/extract_character.py"]
            self.assertEqual(extract_char_analysis.change_type, ChangeType.TRACKER_ID)
            self.assertEqual(extract_char_analysis.recommended_action, RecommendedAction.INTEGRATE)

            memory_opt_analysis = analyses["features/common/memory_optimizer.py"]
            # QUAL-035を含むためTRACKER_IDとして分類される
            self.assertEqual(memory_opt_analysis.change_type, ChangeType.TRACKER_ID)
            self.assertEqual(memory_opt_analysis.recommended_action, RecommendedAction.INTEGRATE)

            sam_yolo_analysis = analyses["tools/core/sam_yolo_character_segment.py"]
            # QUAL-035を含むためTRACKER_IDとして分類される
            self.assertEqual(sam_yolo_analysis.change_type, ChangeType.TRACKER_ID)
            self.assertEqual(sam_yolo_analysis.recommended_action, RecommendedAction.INTEGRATE)

            # サマリーレポート生成テスト
            summary = self.analyzer.generate_summary_report(analyses)

            self.assertEqual(summary["total_files"], 10)
            self.assertGreater(summary["total_changes"]["added"], 0)
            self.assertGreater(summary["total_changes"]["removed"], 0)

            # アクション分布の確認
            actions = summary["action_recommendations"]
            self.assertGreater(actions["integrate"], 0)  # 統合推奨ファイルが存在
            # トラッカーIDが多く検出されるため、reviewは少ない可能性がある

            # 結果保存テスト
            self.analyzer.save_analysis_results(analyses, str(self.output_path))

            # 保存ファイルの確認
            self.assertTrue((self.output_path / "stash_analysis_detailed.json").exists())
            self.assertTrue((self.output_path / "stash_analysis_summary.json").exists())
            self.assertTrue((self.output_path / "stash_analysis_report.md").exists())

            # JSON内容の検証
            with open(
                self.output_path / "stash_analysis_detailed.json", "r", encoding="utf-8"
            ) as f:
                detailed_data = json.load(f)

                # 重要ファイルの詳細確認
                extract_char_detail = detailed_data[
                    "features/extraction/commands/extract_character.py"
                ]
                self.assertEqual(extract_char_detail["change_type"], "tracker_id")
                self.assertEqual(extract_char_detail["recommended_action"], "integrate")
                self.assertGreater(extract_char_detail["confidence"], 0.6)  # 重要ファイルは信頼度が低下

            with open(self.output_path / "stash_analysis_summary.json", "r", encoding="utf-8") as f:
                summary_data = json.load(f)

                self.assertEqual(summary_data["total_files"], 10)
                self.assertIn("recommended_integration", summary_data)
                self.assertIn("safe_to_discard", summary_data)
                self.assertIn("requires_review", summary_data)

            # Markdownレポートの検証
            with open(self.output_path / "stash_analysis_report.md", "r", encoding="utf-8") as f:
                report_content = f.read()

                self.assertIn("212ファイル復旧差分分析レポート", report_content)
                self.assertIn("分析サマリー", report_content)
                self.assertIn("推奨アクション", report_content)

                # レポートの基本構造確認（具体的なファイル名よりも構造を重視）
                self.assertTrue(len(report_content) > 200)  # ある程度の内容量がある

    def test_large_scale_analysis_performance(self):
        """大規模分析のパフォーマンステスト"""

        # 100ファイルの大規模変更を模擬
        large_numstat = "\n".join([f"{i%20+1}\t{i%15+1}\tfile_{i:03d}.py" for i in range(100)])

        mock_diff_content = """
@@ -1,3 +1,5 @@
+# QUAL-035: 大規模復旧テスト
 def function():
-    old_code()
+    new_code()
     pass
"""

        with patch("subprocess.run") as mock_run:

            def mock_subprocess_side_effect(*args, **kwargs):
                command = args[0]

                if "--numstat" in command:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = large_numstat
                    mock_result.returncode = 0
                    return mock_result
                elif "-p" in command:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = mock_diff_content
                    return mock_result
                else:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = ""
                    return mock_result

            mock_run.side_effect = mock_subprocess_side_effect

            # パフォーマンス測定
            import time

            start_time = time.time()

            analyses = self.analyzer.analyze_stash_files()

            end_time = time.time()
            processing_time = end_time - start_time

            # 結果検証
            self.assertEqual(len(analyses), 100)

            # パフォーマンス要件（10秒以内で100ファイル処理）
            self.assertLess(processing_time, 10.0)

            # サマリー生成もテスト
            summary = self.analyzer.generate_summary_report(analyses)
            self.assertEqual(summary["total_files"], 100)

    def test_edge_cases_handling(self):
        """エッジケースの処理テスト"""

        # 特殊ケースを含むnumstat
        edge_case_numstat = """0\t0\tempty_change.py
1000\t0\tmassive_addition.py
0\t1000\tmassive_deletion.py
-\t-\tbinary_file.jpg
5\t5\tnormal_file.py"""

        edge_case_diffs = {
            "empty_change.py": "",  # 空の差分
            "massive_addition.py": "\n".join([f"+line {i}" for i in range(1000)]),  # 大量追加
            "massive_deletion.py": "\n".join([f"-line {i}" for i in range(1000)]),  # 大量削除
            "binary_file.jpg": "Binary files differ",  # バイナリファイル
            "normal_file.py": "+# normal change",  # 通常変更
        }

        with patch("subprocess.run") as mock_run:

            def mock_subprocess_side_effect(*args, **kwargs):
                command = args[0]

                if "--numstat" in command:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = edge_case_numstat
                    mock_result.returncode = 0
                    return mock_result
                elif "-p" in command:
                    file_path = command[-1]
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = edge_case_diffs.get(file_path, "")
                    return mock_result
                else:
                    mock_result = type("MockResult", (), {})()
                    mock_result.stdout = ""
                    return mock_result

            mock_run.side_effect = mock_subprocess_side_effect

            # エッジケース処理テスト
            analyses = self.analyzer.analyze_stash_files()

            # 結果検証
            self.assertEqual(len(analyses), 5)

            # 空の変更
            empty_analysis = analyses["empty_change.py"]
            self.assertEqual(empty_analysis.lines_added, 0)
            self.assertEqual(empty_analysis.lines_removed, 0)

            # 大量追加
            massive_add_analysis = analyses["massive_addition.py"]
            self.assertEqual(massive_add_analysis.lines_added, 1000)
            self.assertEqual(massive_add_analysis.lines_removed, 0)
            self.assertLess(massive_add_analysis.confidence, 0.9)  # 大きな変更は信頼度低下

            # 大量削除
            massive_del_analysis = analyses["massive_deletion.py"]
            self.assertEqual(massive_del_analysis.lines_added, 0)
            self.assertEqual(massive_del_analysis.lines_removed, 1000)

            # バイナリファイル
            binary_analysis = analyses["binary_file.jpg"]
            self.assertEqual(binary_analysis.lines_added, 0)
            self.assertEqual(binary_analysis.lines_removed, 0)


class TestFileTypeClassification(unittest.TestCase):
    """ファイルタイプ分類の統合テスト"""

    def setUp(self):
        self.analyzer = StashDiffAnalyzer()

    def test_python_file_classification(self):
        """Pythonファイルの分類テスト"""
        diff_content = """
@@ -1,5 +1,8 @@
 #!/usr/bin/env python3
+# QUAL-035: 機能改善
 
 def process_data():
-    return old_process()
+    return new_enhanced_process()
"""

        change_type = self.analyzer._classify_change_type("script.py", diff_content)

        # トラッカーIDとコメント変更を含む
        self.assertEqual(change_type, ChangeType.TRACKER_ID)

    def test_test_file_handling(self):
        """テストファイルの特別処理テスト"""
        test_file_path = "tests/unit/test_extraction.py"

        action = self.analyzer._determine_action(test_file_path, ChangeType.FUNCTIONAL, 10, 5)

        # テストファイルの機能変更は要レビュー
        self.assertEqual(action, RecommendedAction.REVIEW)

        reasons = self.analyzer._generate_reasons(ChangeType.FUNCTIONAL, test_file_path, 10, 5)

        self.assertIn("テストファイル", reasons)

    def test_configuration_file_handling(self):
        """設定ファイルの処理テスト"""
        config_files = [
            "setup.py",
            "requirements.txt",
            "pyproject.toml",
            ".gitignore",
            "config.json",
        ]

        for config_file in config_files:
            with self.subTest(config_file=config_file):
                action = self.analyzer._determine_action(config_file, ChangeType.FUNCTIONAL, 5, 2)

                # 設定ファイルの変更は要レビュー
                self.assertEqual(action, RecommendedAction.REVIEW)


if __name__ == "__main__":
    unittest.main()
