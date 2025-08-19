#!/usr/bin/env python3
"""
212ファイル復旧差分分析システムのユニットテスト
QUAL-035 実装テスト
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer import (
    StashDiffAnalyzer, ChangeType, RecommendedAction, FileAnalysis
)


class TestStashDiffAnalyzer(unittest.TestCase):
    """StashDiffAnalyzer のユニットテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.analyzer = StashDiffAnalyzer()
        
        # テスト用の模擬差分データ
        self.mock_numstat_output = """5\t2\tfeatures/extraction/commands/extract_character.py
0\t1\tdocs/README.md
15\t8\ttools/core/sam_yolo_character_segment.py
3\t3\ttests/unit/test_extraction.py"""
        
        self.mock_diff_content = """diff --git a/features/extraction/commands/extract_character.py b/features/extraction/commands/extract_character.py
index 1234567..abcdefg 100644
--- a/features/extraction/commands/extract_character.py
+++ b/features/extraction/commands/extract_character.py
@@ -1,5 +1,8 @@
 #!/usr/bin/env python3
+# QUAL-035対応: 改善された抽出機能
 
 def extract_character(image_path, output_path):
+    # P1-A001 → QUAL-001 への参照更新
-    # 旧処理
+    # 新処理
     pass"""
    
    def test_init(self):
        """初期化テスト"""
        analyzer = StashDiffAnalyzer()
        
        self.assertEqual(analyzer.stash_ref, "stash@{0}")
        self.assertEqual(analyzer.base_ref, "HEAD")
        self.assertIsInstance(analyzer.format_patterns, list)
        self.assertIsInstance(analyzer.tracker_id_patterns, list)
        self.assertIsInstance(analyzer.critical_files, list)
    
    def test_parse_numstat(self):
        """numstat出力パースのテスト"""
        result = self.analyzer._parse_numstat(self.mock_numstat_output)
        
        expected = {
            'features/extraction/commands/extract_character.py': (5, 2),
            'docs/README.md': (0, 1),
            'tools/core/sam_yolo_character_segment.py': (15, 8),
            'tests/unit/test_extraction.py': (3, 3)
        }
        
        self.assertEqual(result, expected)
    
    def test_parse_numstat_empty(self):
        """空のnumstat出力のテスト"""
        result = self.analyzer._parse_numstat("")
        self.assertEqual(result, {})
    
    def test_parse_numstat_binary_files(self):
        """バイナリファイルを含むnumstat出力のテスト"""
        numstat_with_binary = """5\t2\ttest.py
-\t-\timage.png
3\t1\tother.py"""
        
        result = self.analyzer._parse_numstat(numstat_with_binary)
        
        expected = {
            'test.py': (5, 2),
            'image.png': (0, 0),  # バイナリファイルは0,0として処理
            'other.py': (3, 1)
        }
        
        self.assertEqual(result, expected)
    
    def test_classify_change_type_documentation(self):
        """ドキュメントファイルの分類テスト"""
        change_type = self.analyzer._classify_change_type(
            "docs/README.md", 
            "Some documentation changes"
        )
        
        self.assertEqual(change_type, ChangeType.DOCUMENTATION)
    
    def test_classify_change_type_tracker_id(self):
        """トラッカーID変更の分類テスト"""
        diff_content = """
-    # P1-A001 処理
+    # QUAL-001 処理
"""
        
        change_type = self.analyzer._classify_change_type(
            "features/test.py",
            diff_content
        )
        
        self.assertEqual(change_type, ChangeType.TRACKER_ID)
    
    def test_classify_change_type_format_only(self):
        """フォーマットのみ変更の分類テスト"""
        diff_content = """
-import os
-import sys
+import os
+import sys
+
 def test():
     pass"""
        
        change_type = self.analyzer._classify_change_type(
            "test.py",
            diff_content
        )
        
        # フォーマット変更の比率が高い場合
        self.assertIn(change_type, [ChangeType.FORMAT_ONLY, ChangeType.MIXED])
    
    def test_classify_change_type_functional(self):
        """機能的変更の分類テスト"""
        diff_content = """
 def process_image(image):
-    return old_function(image)
+    return new_enhanced_function(image, quality='high')
"""
        
        change_type = self.analyzer._classify_change_type(
            "processing.py",
            diff_content
        )
        
        self.assertEqual(change_type, ChangeType.FUNCTIONAL)
    
    def test_determine_action_critical_files(self):
        """重要ファイルのアクション決定テスト"""
        # 重要ファイルのフォーマット変更 → DISCARD
        action = self.analyzer._determine_action(
            "features/extraction/commands/extract_character.py",
            ChangeType.FORMAT_ONLY,
            2, 2
        )
        self.assertEqual(action, RecommendedAction.DISCARD)
        
        # 重要ファイルの機能変更 → REVIEW
        action = self.analyzer._determine_action(
            "features/extraction/commands/extract_character.py",
            ChangeType.FUNCTIONAL,
            10, 5
        )
        self.assertEqual(action, RecommendedAction.REVIEW)
    
    def test_determine_action_tracker_id(self):
        """トラッカーID変更のアクション決定テスト"""
        action = self.analyzer._determine_action(
            "some_file.py",
            ChangeType.TRACKER_ID,
            3, 3
        )
        
        self.assertEqual(action, RecommendedAction.INTEGRATE)
    
    def test_determine_action_documentation(self):
        """ドキュメント変更のアクション決定テスト"""
        action = self.analyzer._determine_action(
            "docs/README.md",
            ChangeType.DOCUMENTATION,
            5, 2
        )
        
        self.assertEqual(action, RecommendedAction.INTEGRATE)
    
    def test_calculate_confidence(self):
        """信頼度計算のテスト"""
        # フォーマットのみ変更 → 高い信頼度
        confidence = self.analyzer._calculate_confidence(
            ChangeType.FORMAT_ONLY,
            "normal_file.py",
            "some diff content"
        )
        self.assertGreaterEqual(confidence, 0.9)
        
        # 重要ファイルの機能変更 → 低い信頼度
        confidence = self.analyzer._calculate_confidence(
            ChangeType.FUNCTIONAL,
            "features/extraction/commands/extract_character.py",
            "large diff content" * 100
        )
        self.assertLessEqual(confidence, 0.6)
    
    def test_generate_reasons(self):
        """判定理由生成のテスト"""
        reasons = self.analyzer._generate_reasons(
            ChangeType.FUNCTIONAL,
            "features/extraction/commands/extract_character.py",
            100, 50
        )
        
        self.assertIn("機能的な変更を含む", reasons)
        self.assertIn("重要システムファイル", reasons)
        # added + removed = 150 < 500 なので大規模変更ではない
    
    def test_extract_sample_changes(self):
        """サンプル変更抽出のテスト"""
        diff_content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
-old line 1
+new line 1
-old line 2
+new line 2"""
        
        samples = self.analyzer._extract_sample_changes(diff_content)
        
        self.assertLessEqual(len(samples), 3)
        self.assertTrue(any("-old line 1" in sample for sample in samples))
        self.assertTrue(any("+new line 1" in sample for sample in samples))
    
    @patch('subprocess.run')
    def test_get_file_diff(self, mock_run):
        """ファイル差分取得のテスト"""
        mock_run.return_value.stdout = self.mock_diff_content
        
        result = self.analyzer._get_file_diff("test_file.py")
        
        self.assertEqual(result, self.mock_diff_content)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_get_file_diff_error(self, mock_run):
        """ファイル差分取得エラーのテスト"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        
        result = self.analyzer._get_file_diff("test_file.py")
        
        self.assertEqual(result, "")
    
    def test_analyze_file_changes(self):
        """ファイル変更分析のテスト"""
        with patch.object(self.analyzer, '_get_file_diff') as mock_get_diff:
            mock_get_diff.return_value = self.mock_diff_content
            
            analysis = self.analyzer._analyze_file_changes(
                "features/extraction/commands/extract_character.py", 5, 2
            )
            
            self.assertIsInstance(analysis, FileAnalysis)
            self.assertEqual(analysis.file_path, "features/extraction/commands/extract_character.py")
            self.assertEqual(analysis.lines_added, 5)
            self.assertEqual(analysis.lines_removed, 2)
            self.assertIsInstance(analysis.change_type, ChangeType)
            self.assertIsInstance(analysis.recommended_action, RecommendedAction)
            self.assertIsInstance(analysis.confidence, float)
            self.assertIsInstance(analysis.reasons, list)
            self.assertIsInstance(analysis.sample_changes, list)
    
    @patch('subprocess.run')
    def test_analyze_stash_files_success(self, mock_run):
        """stashファイル分析成功のテスト"""
        mock_run.return_value.stdout = self.mock_numstat_output
        mock_run.return_value.returncode = 0
        
        with patch.object(self.analyzer, '_analyze_file_changes') as mock_analyze:
            mock_analysis = FileAnalysis(
                file_path="test.py",
                lines_added=5,
                lines_removed=2,
                change_type=ChangeType.FUNCTIONAL,
                recommended_action=RecommendedAction.REVIEW,
                confidence=0.8,
                reasons=["test reason"],
                sample_changes=["test change"]
            )
            mock_analyze.return_value = mock_analysis
            
            result = self.analyzer.analyze_stash_files()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 4)  # mock_numstat_output has 4 files
    
    @patch('subprocess.run')
    def test_analyze_stash_files_error(self, mock_run):
        """stashファイル分析エラーのテスト"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        
        result = self.analyzer.analyze_stash_files()
        
        self.assertEqual(result, {})
    
    def test_generate_summary_report(self):
        """サマリーレポート生成のテスト"""
        # テスト用の分析結果を作成
        analyses = {
            "file1.py": FileAnalysis(
                file_path="file1.py",
                lines_added=10,
                lines_removed=5,
                change_type=ChangeType.FUNCTIONAL,
                recommended_action=RecommendedAction.REVIEW,
                confidence=0.8,
                reasons=["機能変更"],
                sample_changes=["+new code"]
            ),
            "file2.py": FileAnalysis(
                file_path="file2.py",
                lines_added=2,
                lines_removed=2,
                change_type=ChangeType.FORMAT_ONLY,
                recommended_action=RecommendedAction.DISCARD,
                confidence=0.9,
                reasons=["フォーマットのみ"],
                sample_changes=["+  spaced code"]
            ),
            "docs/README.md": FileAnalysis(
                file_path="docs/README.md",
                lines_added=3,
                lines_removed=1,
                change_type=ChangeType.DOCUMENTATION,
                recommended_action=RecommendedAction.INTEGRATE,
                confidence=0.85,
                reasons=["ドキュメント更新"],
                sample_changes=["+新しい説明"]
            )
        }
        
        summary = self.analyzer.generate_summary_report(analyses)
        
        # 基本構造のチェック
        self.assertIn('total_files', summary)
        self.assertIn('total_changes', summary)
        self.assertIn('action_recommendations', summary)
        self.assertIn('change_types', summary)
        self.assertIn('confidence_stats', summary)
        
        # 値のチェック
        self.assertEqual(summary['total_files'], 3)
        self.assertEqual(summary['total_changes']['added'], 15)
        self.assertEqual(summary['total_changes']['removed'], 8)
        self.assertEqual(summary['total_changes']['net'], 7)
        
        # アクション分布
        self.assertEqual(summary['action_recommendations']['review'], 1)
        self.assertEqual(summary['action_recommendations']['discard'], 1)
        self.assertEqual(summary['action_recommendations']['integrate'], 1)
        
        # 信頼度統計
        self.assertEqual(summary['confidence_stats']['high_confidence_count'], 2)  # 0.8以上
        self.assertAlmostEqual(summary['confidence_stats']['average_confidence'], 0.85, places=2)
    
    def test_save_analysis_results(self):
        """分析結果保存のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用の分析結果
            analyses = {
                "test.py": FileAnalysis(
                    file_path="test.py",
                    lines_added=5,
                    lines_removed=2,
                    change_type=ChangeType.FUNCTIONAL,
                    recommended_action=RecommendedAction.REVIEW,
                    confidence=0.8,
                    reasons=["テスト"],
                    sample_changes=["+test code"]
                )
            }
            
            # 保存実行
            self.analyzer.save_analysis_results(analyses, temp_dir)
            
            # ファイル存在確認
            output_path = Path(temp_dir)
            self.assertTrue((output_path / "stash_analysis_detailed.json").exists())
            self.assertTrue((output_path / "stash_analysis_summary.json").exists())
            self.assertTrue((output_path / "stash_analysis_report.md").exists())
            
            # JSON内容確認
            with open(output_path / "stash_analysis_detailed.json", 'r', encoding='utf-8') as f:
                detailed = json.load(f)
                self.assertIn("test.py", detailed)
                self.assertEqual(detailed["test.py"]["change_type"], "functional")
            
            with open(output_path / "stash_analysis_summary.json", 'r', encoding='utf-8') as f:
                summary = json.load(f)
                self.assertEqual(summary["total_files"], 1)
    
    def test_text_report_generation(self):
        """テキストレポート生成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyses = {
                "file1.py": FileAnalysis(
                    file_path="file1.py",
                    lines_added=10,
                    lines_removed=5,
                    change_type=ChangeType.FUNCTIONAL,
                    recommended_action=RecommendedAction.REVIEW,
                    confidence=0.8,
                    reasons=["機能変更"],
                    sample_changes=["+new code"]
                )
            }
            
            summary = self.analyzer.generate_summary_report(analyses)
            output_file = Path(temp_dir) / "test_report.md"
            
            self.analyzer._generate_text_report(analyses, summary, output_file)
            
            self.assertTrue(output_file.exists())
            
            # レポート内容確認
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("212ファイル復旧差分分析レポート", content)
                self.assertIn("分析サマリー", content)
                self.assertIn("推奨アクション", content)
                self.assertIn("file1.py", content)


class TestChangeTypeEnum(unittest.TestCase):
    """ChangeType列挙型のテスト"""
    
    def test_change_type_values(self):
        """ChangeType値のテスト"""
        self.assertEqual(ChangeType.FORMAT_ONLY.value, "format_only")
        self.assertEqual(ChangeType.FUNCTIONAL.value, "functional")
        self.assertEqual(ChangeType.TRACKER_ID.value, "tracker_id")
        self.assertEqual(ChangeType.DOCUMENTATION.value, "documentation")
        self.assertEqual(ChangeType.MIXED.value, "mixed")


class TestRecommendedActionEnum(unittest.TestCase):
    """RecommendedAction列挙型のテスト"""
    
    def test_recommended_action_values(self):
        """RecommendedAction値のテスト"""
        self.assertEqual(RecommendedAction.INTEGRATE.value, "integrate")
        self.assertEqual(RecommendedAction.DISCARD.value, "discard")
        self.assertEqual(RecommendedAction.REVIEW.value, "review")
        self.assertEqual(RecommendedAction.PRESERVE.value, "preserve")


class TestFileAnalysisDataclass(unittest.TestCase):
    """FileAnalysis データクラスのテスト"""
    
    def test_file_analysis_creation(self):
        """FileAnalysis作成のテスト"""
        analysis = FileAnalysis(
            file_path="test.py",
            lines_added=10,
            lines_removed=5,
            change_type=ChangeType.FUNCTIONAL,
            recommended_action=RecommendedAction.REVIEW,
            confidence=0.8,
            reasons=["test reason"],
            sample_changes=["test change"]
        )
        
        self.assertEqual(analysis.file_path, "test.py")
        self.assertEqual(analysis.lines_added, 10)
        self.assertEqual(analysis.lines_removed, 5)
        self.assertEqual(analysis.change_type, ChangeType.FUNCTIONAL)
        self.assertEqual(analysis.recommended_action, RecommendedAction.REVIEW)
        self.assertEqual(analysis.confidence, 0.8)
        self.assertEqual(analysis.reasons, ["test reason"])
        self.assertEqual(analysis.sample_changes, ["test change"])


class TestMainFunction(unittest.TestCase):
    """main関数のテスト"""
    
    @patch('tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer.StashDiffAnalyzer')
    def test_main_success(self, mock_analyzer_class):
        """main関数成功のテスト"""
        from tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer import main
        
        # Mock設定
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.analyze_stash_files.return_value = {"file1.py": "mock_analysis"}
        mock_analyzer.generate_summary_report.return_value = {
            'total_files': 1,
            'action_recommendations': {'integrate': 1, 'discard': 0, 'review': 0},
            'confidence_stats': {'average_confidence': 0.8}
        }
        
        result = main()
        
        self.assertTrue(result)
        mock_analyzer.analyze_stash_files.assert_called_once()
        mock_analyzer.save_analysis_results.assert_called_once()
        mock_analyzer.generate_summary_report.assert_called_once()
    
    @patch('tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer.StashDiffAnalyzer')
    def test_main_failure(self, mock_analyzer_class):
        """main関数失敗のテスト"""
        from tools.trackers.QUAL_035_stash_analysis.stash_diff_analyzer import main
        
        # Mock設定（失敗）
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.analyze_stash_files.return_value = {}
        
        result = main()
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()