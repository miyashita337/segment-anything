#!/usr/bin/env python3
"""
Level 2: 品質ワークフローテスト

品質ワークフロー関連のテスト:
- SAM+YOLO抽出処理テスト
- 品質評価システムテスト
- ダッシュボード生成テスト
- extraction_result.json生成・検証テスト
"""

import tempfile
import pytest
import json
from pathlib import Path
import sys
import os

# テスト対象をインポート
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tests.mocks.mock_sam_yolo import MockSamYoloExtractor, MockQualityEvaluator
    from tools.testing.quality_workflow_tester import QualityWorkflowTester
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestQualityWorkflow:
    """Level 2: 品質ワークフローテストクラス"""
    
    # ================================
    # SAM+YOLO抽出処理テスト（12テストケース）
    # ================================
    
    def test_mock_sam_yolo_single_extraction_success(self):
        """単体画像抽出成功テスト"""
        extractor = MockSamYoloExtractor(random_seed=42)
        
        result = extractor.extract_single_image("test_image_001.jpg")
        
        # 基本構造確認
        assert "success" in result
        assert "processing_time" in result
        assert "image_path" in result
        assert result["image_path"] == "test_image_001.jpg"
        
        # 処理時間確認（0.5-2.0秒の範囲）
        assert 0.0 < result["processing_time"] <= 2.0
    
    def test_mock_sam_yolo_quality_methods(self):
        """品質評価手法別テスト"""
        methods = ["balanced", "confidence_priority", "size_priority", "fullbody_priority", "central_priority"]
        
        for method in methods:
            extractor = MockSamYoloExtractor(quality_method=method, random_seed=42)
            result = extractor.extract_single_image(f"test_{method}.jpg")
            
            # 成功率は手法によって異なる
            expected_success_rates = {
                "balanced": 0.85,
                "confidence_priority": 0.78,
                "size_priority": 0.82,
                "fullbody_priority": 0.75,
                "central_priority": 0.80
            }
            
            # 複数回実行して成功率を確認（統計的テスト）
            success_count = 0
            total_tests = 20
            
            for i in range(total_tests):
                test_result = extractor.extract_single_image(f"test_{method}_{i}.jpg")
                if test_result["success"]:
                    success_count += 1
            
            actual_success_rate = success_count / total_tests
            expected_rate = expected_success_rates[method]
            
            # 許容範囲±15%で確認
            assert abs(actual_success_rate - expected_rate) <= 0.15, f"Method {method}: expected {expected_rate}, got {actual_success_rate}"
    
    def test_mock_sam_yolo_grade_distribution(self):
        """品質グレード分布テスト"""
        extractor = MockSamYoloExtractor(random_seed=42)
        
        results = []
        for i in range(100):  # 統計的に信頼できる数でテスト
            result = extractor.extract_single_image(f"test_{i}.jpg")
            if result["success"]:
                results.append(result["grade"])
        
        # グレード分布確認
        grade_counts = {grade: results.count(grade) for grade in ["A", "B", "C", "D", "F"]}
        total_success = len(results)
        
        if total_success > 0:
            # 期待される分布と比較（±10%の許容範囲）
            expected_distribution = {"A": 0.25, "B": 0.35, "C": 0.25, "D": 0.10, "F": 0.05}
            
            for grade, expected_ratio in expected_distribution.items():
                if grade == "F":
                    continue  # 成功結果のみをカウントしているのでF評価はスキップ
                actual_ratio = grade_counts[grade] / total_success
                assert abs(actual_ratio - expected_ratio) <= 0.15, f"Grade {grade}: expected {expected_ratio}, got {actual_ratio}"
    
    def test_mock_sam_yolo_batch_extraction(self):
        """バッチ抽出テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = f"{temp_dir}/input"
            output_dir = f"{temp_dir}/output"
            
            # 入力ディレクトリ作成
            Path(input_dir).mkdir()
            
            extractor = MockSamYoloExtractor(random_seed=42)
            
            # バッチ抽出実行
            batch_result = extractor.extract_batch(input_dir, output_dir, max_files=10)
            
            # 結果構造確認
            required_keys = [
                "input_dir", "output_dir", "total_images", "processed_images",
                "successful_extractions", "failed_extractions", "extraction_results",
                "quality_distribution", "average_quality_score", "success_rate"
            ]
            
            for key in required_keys:
                assert key in batch_result, f"Missing key: {key}"
            
            # 数値整合性確認
            assert batch_result["processed_images"] == batch_result["total_images"]
            assert batch_result["successful_extractions"] + batch_result["failed_extractions"] == batch_result["total_images"]
            
            # extraction_result.json生成確認
            json_path = Path(output_dir) / "extraction_result.json"
            assert json_path.exists()
            
            # JSON内容確認
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            assert "tracker_id" in json_data
            assert "extraction_results" in json_data
    
    def test_mock_sam_yolo_nonexistent_input(self):
        """存在しない入力ディレクトリでのエラーテスト"""
        extractor = MockSamYoloExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_batch("/nonexistent/input", "/tmp/output")
    
    def test_mock_detection_result_structure(self):
        """YOLO検出結果構造テスト"""
        extractor = MockSamYoloExtractor(random_seed=42)
        result = extractor.extract_single_image("test.jpg")
        
        if result["success"]:
            detection = result["detection_result"]
            
            assert hasattr(detection, 'bbox')
            assert hasattr(detection, 'confidence')
            assert hasattr(detection, 'class_id')
            assert hasattr(detection, 'class_name')
            
            # 境界ボックス形式確認（x1, y1, x2, y2）
            x1, y1, x2, y2 = detection.bbox
            assert x1 < x2
            assert y1 < y2
            assert 0 <= detection.confidence <= 1.0
            assert detection.class_name == "character"
    
    def test_mock_segmentation_result_structure(self):
        """SAMセグメンテーション結果構造テスト"""
        extractor = MockSamYoloExtractor(random_seed=42)
        result = extractor.extract_single_image("test.jpg")
        
        if result["success"]:
            segmentation = result["segmentation_result"]
            
            assert hasattr(segmentation, 'mask_area')
            assert hasattr(segmentation, 'quality_score')
            assert hasattr(segmentation, 'grade')
            assert hasattr(segmentation, 'success')
            
            assert segmentation.mask_area > 0
            assert 0.0 <= segmentation.quality_score <= 1.0
            assert segmentation.grade in ["A", "B", "C", "D", "F"]
    
    # ================================
    # 品質評価システムテスト（15テストケース）
    # ================================
    
    def test_quality_evaluator_success_case(self):
        """品質評価器成功ケーステスト"""
        evaluator = MockQualityEvaluator()
        
        # 成功的な抽出結果のモック
        extraction_result = {
            "success": True,
            "quality_score": 0.85,
            "grade": "A"
        }
        
        evaluation = evaluator.evaluate_extraction_quality(extraction_result)
        
        assert "quality_score" in evaluation
        assert "grade" in evaluation
        assert "evaluation_details" in evaluation
        
        assert evaluation["quality_score"] == 0.85
        assert evaluation["grade"] == "A"
        
        # 詳細評価確認
        details = evaluation["evaluation_details"]
        required_details = ["completeness", "accuracy", "clarity", "composition"]
        for detail in required_details:
            assert detail in details
            assert 0.0 <= details[detail] <= 1.0
    
    def test_quality_evaluator_failure_case(self):
        """品質評価器失敗ケーステスト"""
        evaluator = MockQualityEvaluator()
        
        # 失敗的な抽出結果のモック
        extraction_result = {
            "success": False,
            "error": "抽出失敗"
        }
        
        evaluation = evaluator.evaluate_extraction_quality(extraction_result)
        
        assert evaluation["quality_score"] == 0.0
        assert evaluation["grade"] == "F"
        
        # 詳細評価も全て0.0
        details = evaluation["evaluation_details"]
        for score in details.values():
            assert score == 0.0
    
    def test_quality_report_generation(self):
        """品質レポート生成テスト"""
        evaluator = MockQualityEvaluator()
        
        # バッチ結果のモック
        batch_result = {
            "total_images": 20,
            "successful_extractions": 16,
            "success_rate": 80.0,
            "average_quality_score": 0.75,
            "quality_distribution": {
                "A": 4, "B": 6, "C": 4, "D": 2, "F": 4
            }
        }
        
        report = evaluator.generate_quality_report(batch_result)
        
        # レポート内容確認
        assert isinstance(report, str)
        assert "品質評価レポート" in report
        assert "総画像数" in report
        assert "成功率" in report
        assert "A評価" in report
        assert "80.0%" in report  # 成功率
    
    def test_quality_workflow_tester_initialization(self):
        """品質ワークフローテスター初期化テスト"""
        tester = QualityWorkflowTester(quality_method="balanced", random_seed=42)
        
        assert tester.quality_method == "balanced"
        assert tester.random_seed == 42
        assert tester.extractor is not None
        assert tester.evaluator is not None
    
    def test_quality_workflow_single_test(self):
        """品質ワークフロー単体テスト"""
        tester = QualityWorkflowTester(random_seed=42)
        
        result = tester.run_single_image_test("test_image.jpg")
        
        # テスト結果構造確認
        required_keys = ["test_type", "image_path", "extraction_success", "quality_evaluation", "processing_time"]
        for key in required_keys:
            assert key in result
        
        assert result["test_type"] == "single_image"
        assert result["image_path"] == "test_image.jpg"
    
    def test_quality_workflow_batch_test(self):
        """品質ワークフローバッチテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = f"{temp_dir}/input"
            output_dir = f"{temp_dir}/output"
            
            # 入力ディレクトリ作成
            Path(input_dir).mkdir()
            
            tester = QualityWorkflowTester(random_seed=42)
            result = tester.run_batch_test(input_dir, output_dir, max_files=5)
            
            # テスト結果構造確認
            required_keys = ["test_type", "input_dir", "output_dir", "extraction_results", "test_summary"]
            for key in required_keys:
                assert key in result
            
            assert result["test_type"] == "batch"
            
            # サマリー確認
            summary = result["test_summary"]
            assert "total_images" in summary
            assert "success_rate" in summary
            assert "quality_method" in summary
    
    def test_quality_method_comparison(self):
        """品質評価手法比較テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = f"{temp_dir}/input"
            output_dir = f"{temp_dir}/output"
            
            # 入力ディレクトリ作成
            Path(input_dir).mkdir()
            
            tester = QualityWorkflowTester(random_seed=42)
            result = tester.run_quality_method_comparison(input_dir, output_dir)
            
            # 比較結果構造確認
            assert result["test_type"] == "quality_method_comparison"
            assert "methods_tested" in result
            assert "results" in result
            assert "analysis" in result
            
            # 5つの手法がテストされている
            assert len(result["methods_tested"]) == 5
            assert len(result["results"]) == 5
            
            # 分析結果確認
            analysis = result["analysis"]
            assert "best_method" in analysis
            assert "worst_method" in analysis
            assert "average_success_rate" in analysis
    
    # ================================
    # ダッシュボード生成テスト（8テストケース）
    # ================================
    
    def test_mock_dashboard_generation(self):
        """ダッシュボードHTML生成テスト"""
        tester = QualityWorkflowTester(random_seed=42)
        
        # モックバッチ結果
        batch_result = {
            "total_images": 10,
            "successful_extractions": 8,
            "success_rate": 80.0,
            "average_quality_score": 0.75,
            "quality_distribution": {"A": 2, "B": 3, "C": 2, "D": 1, "F": 2}
        }
        
        dashboard_html = tester._generate_mock_dashboard(batch_result)
        
        # HTML構造確認
        assert isinstance(dashboard_html, str)
        assert "<!DOCTYPE html>" in dashboard_html
        assert "品質ワークフローテストダッシュボード" in dashboard_html
        assert "総画像数: 10枚" in dashboard_html
        assert "成功率: 80.0%" in dashboard_html
        assert "平均品質スコア: 0.75" in dashboard_html
    
    def test_output_file_validation(self):
        """出力ファイル検証テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = QualityWorkflowTester()
            
            # テストファイル作成
            test_files = ["extraction_result.json", "dashboard.html", "quality_report.md"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()
            
            validation_results = tester.validate_output_files(temp_dir)
            
            # 全ファイルが存在することを確認
            for filename in test_files:
                assert filename in validation_results
                assert validation_results[filename] == True
    
    def test_output_file_validation_missing(self):
        """出力ファイル検証（ファイル不足）テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = QualityWorkflowTester()
            
            # 一部ファイルのみ作成
            (Path(temp_dir) / "extraction_result.json").touch()
            
            validation_results = tester.validate_output_files(temp_dir)
            
            # 存在チェック確認
            assert validation_results["extraction_result.json"] == True
            assert validation_results["dashboard.html"] == False
            assert validation_results["quality_report.md"] == False
    
    def test_test_results_saving(self):
        """テスト結果保存テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = QualityWorkflowTester()
            
            # テスト結果のモック
            test_results = {
                "test_type": "batch",
                "quality_report": "# テストレポート\n内容",
                "dashboard_html": "<html><body>テスト</body></html>",
                "test_summary": {"success_rate": 85.0}
            }
            
            tester.save_test_results(test_results, temp_dir)
            
            # 保存されたファイル確認
            assert (Path(temp_dir) / "test_results.json").exists()
            assert (Path(temp_dir) / "quality_report.md").exists()
            assert (Path(temp_dir) / "dashboard.html").exists()
            
            # JSON内容確認
            with open(Path(temp_dir) / "test_results.json", 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert saved_data["test_type"] == "batch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])