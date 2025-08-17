#!/usr/bin/env python3
"""
OPT-024: 品質基準統一システム ユニットテスト

統一品質基準システムの動作確認テスト実装
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from tools.core.unified_quality_standard import (
    QualityStandard, 
    UnifiedQualityResult, 
    UnifiedQualityStandardSystem
)


@pytest.fixture
def temp_workspace():
    """テスト用一時ワークスペース"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_quality_standard():
    """サンプル品質基準"""
    return QualityStandard(
        name="test_standard_v1",
        version="1.0.0",
        ab_excellent_threshold=0.9,
        sci_excellent_threshold=0.85
    )


@pytest.fixture
def sample_results_data():
    """サンプル結果データ"""
    return {
        "total_processed": 16,
        "successful_extractions": 15,
        "ab_evaluation_rate": 0.8,
        "sci_score": 0.75,
        "pla_score": 0.8,
        "ple_score": 0.85,
        "avg_fill_ratio": 0.85,
        "avg_compactness": 0.7,
        "avg_coverage": 0.8,
        "grade_distribution": {"A": 5, "B": 7, "C": 3, "D": 1}
    }


class TestQualityStandard:
    """QualityStandardクラステスト"""
    
    def test_quality_standard_creation(self):
        """品質基準作成テスト"""
        standard = QualityStandard(
            name="test_standard",
            version="1.0.0"
        )
        
        assert standard.name == "test_standard"
        assert standard.version == "1.0.0"
        assert standard.ab_excellent_threshold == 0.9
        assert "kana08" in standard.dataset_weights
        assert standard.dataset_weights["kana08"] == 1.0
    
    def test_dataset_weights_default(self):
        """データセット重みデフォルト値テスト"""
        standard = QualityStandard(name="test", version="1.0")
        
        expected_datasets = ["kana03", "kana04", "kana05", "kana06", "kana07", "kana08", "kana09"]
        for dataset in expected_datasets:
            assert dataset in standard.dataset_weights
        
        assert standard.dataset_weights["kana09"] == 0.8  # 新規データセット重み


class TestUnifiedQualityStandardSystem:
    """UnifiedQualityStandardSystemクラステスト"""
    
    def test_system_initialization(self):
        """システム初期化テスト"""
        # 実際のパスでシステム初期化テスト（ディレクトリ作成は許可）
        system = UnifiedQualityStandardSystem()
        
        assert system.project_root is not None
        assert system.standard is not None
        assert system.standard.name == "unified_quality_standard_v1"
        assert system.standard.version == "1.0.0"
        assert system.workspace_dir.exists()  # 実際にディレクトリが作成されることを確認
    
    def test_calculate_unified_score(self, sample_quality_standard):
        """統一スコア計算テスト"""
        system = UnifiedQualityStandardSystem()
        system.standard = sample_quality_standard
        
        # サンプルメトリクス
        ab_rate = 0.8
        sci = 0.75
        pla = 0.8
        ple = 0.85
        success_rate = 0.9
        fill_ratio = 0.85
        compactness = 0.7
        coverage = 0.8
        dataset_name = "kana08"
        
        score = system._calculate_unified_score(
            ab_rate, sci, pla, ple, success_rate, 
            fill_ratio, compactness, coverage, dataset_name
        )
        
        # 期待値計算 (重み: ab_rate=0.3, sci=0.2, pla=0.15, ple=0.15, success_rate=0.1, etc.)
        expected_score = (
            0.8 * 0.3 +     # ab_rate
            0.75 * 0.2 +    # sci
            0.8 * 0.15 +    # pla
            0.85 * 0.15 +   # ple
            0.9 * 0.1 +     # success_rate
            0.85 * 0.04 +   # fill_ratio
            0.7 * 0.03 +    # compactness
            0.8 * 0.03      # coverage
        ) * 1.0  # dataset_weight for kana08
        
        assert abs(score - expected_score) < 0.001
        assert 0 <= score <= 1.0
    
    def test_determine_unified_grade(self):
        """統一グレード判定テスト"""
        system = UnifiedQualityStandardSystem()
        
        test_cases = [
            (0.95, "A+"),
            (0.87, "A"),
            (0.78, "B+"),
            (0.68, "B"),
            (0.58, "C+"),
            (0.48, "C"),
            (0.38, "D"),
            (0.25, "F")
        ]
        
        for score, expected_grade in test_cases:
            actual_grade = system._determine_unified_grade(score)
            assert actual_grade == expected_grade, f"Score {score} should be grade {expected_grade}, got {actual_grade}"
    
    def test_determine_quality_level(self):
        """品質レベル判定テスト"""
        system = UnifiedQualityStandardSystem()
        
        test_cases = [
            (0.9, "EXCELLENT"),
            (0.75, "GOOD"),
            (0.55, "ACCEPTABLE"),
            (0.3, "POOR")
        ]
        
        for score, expected_level in test_cases:
            actual_level = system._determine_quality_level(score)
            assert actual_level == expected_level
    
    def test_evaluate_dataset_quality(self, sample_results_data):
        """データセット品質評価テスト"""
        system = UnifiedQualityStandardSystem()
        
        result = system.evaluate_dataset_quality("kana08", sample_results_data)
        
        assert isinstance(result, UnifiedQualityResult)
        assert result.dataset_name == "kana08"
        assert result.total_processed == 16
        assert result.successful_extractions == 15
        assert result.success_rate == 15/16
        assert result.ab_evaluation_rate == 0.8
        assert result.unified_score > 0
        assert result.unified_grade in ["A+", "A", "B+", "B", "C+", "C", "D", "F"]
        assert result.quality_level in ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"]
    
    def test_compare_datasets(self, sample_results_data):
        """データセット横断比較テスト"""
        system = UnifiedQualityStandardSystem()
        
        # 複数データセット結果作成
        result1 = system.evaluate_dataset_quality("kana08", sample_results_data)
        
        # 異なる品質のデータセット
        sample_results_data2 = sample_results_data.copy()
        sample_results_data2["ab_evaluation_rate"] = 0.9
        sample_results_data2["sci_score"] = 0.85
        result2 = system.evaluate_dataset_quality("kana05", sample_results_data2)
        
        comparison = system.compare_datasets([result1, result2])
        
        assert comparison["total_datasets"] == 2
        assert "overall_statistics" in comparison
        assert "unified_score" in comparison["overall_statistics"]
        assert "dataset_ranking" in comparison
        assert len(comparison["dataset_ranking"]) == 2
        assert "recommendations" in comparison
        
        # ランキング順序確認（高スコア順）
        rankings = comparison["dataset_ranking"]
        assert rankings[0]["unified_score"] >= rankings[1]["unified_score"]
    
    def test_generate_recommendations(self):
        """推奨事項生成テスト"""
        system = UnifiedQualityStandardSystem()
        
        # 低品質データセット作成
        poor_result = UnifiedQualityResult(
            dataset_name="test_poor",
            total_processed=10,
            successful_extractions=5,
            ab_evaluation_rate=0.3,  # 低いA/B評価率
            sci_score=0.4,
            pla_score=0.3,
            ple_score=0.4,
            success_rate=0.5,
            avg_fill_ratio=0.5,
            avg_compactness=0.4,
            avg_coverage=0.5,
            grade_distribution={},
            unified_score=0.35,
            unified_grade="D",
            quality_level="POOR",
            evaluation_timestamp="2025-01-01T00:00:00",
            processing_time=1.0,
            detailed_metrics={}
        )
        
        # 高品質データセット作成
        excellent_result = UnifiedQualityResult(
            dataset_name="test_excellent",
            total_processed=20,
            successful_extractions=19,
            ab_evaluation_rate=0.95,
            sci_score=0.9,
            pla_score=0.88,
            ple_score=0.92,
            success_rate=0.95,
            avg_fill_ratio=0.9,
            avg_compactness=0.85,
            avg_coverage=0.9,
            grade_distribution={},
            unified_score=0.92,
            unified_grade="A+",
            quality_level="EXCELLENT",
            evaluation_timestamp="2025-01-01T00:00:00",
            processing_time=1.0,
            detailed_metrics={}
        )
        
        recommendations = system._generate_recommendations([poor_result, excellent_result])
        
        assert len(recommendations) > 0
        
        # POOR品質データセットの推奨事項確認
        poor_recommendation = [r for r in recommendations if "品質改善が必要" in r]
        assert len(poor_recommendation) > 0
        assert "test_poor" in poor_recommendation[0]
        
        # 低A/B評価率の推奨事項確認
        ab_recommendation = [r for r in recommendations if "A/B評価率改善が必要" in r]
        assert len(ab_recommendation) > 0
        
        # 最高品質データセットの推奨事項確認
        best_recommendation = [r for r in recommendations if "最高品質データセット" in r]
        assert len(best_recommendation) > 0
        assert "test_excellent" in best_recommendation[0]
    
    def test_generate_unified_report(self, sample_results_data):
        """統一品質レポート生成テスト"""
        system = UnifiedQualityStandardSystem()
        
        result1 = system.evaluate_dataset_quality("kana08", sample_results_data)
        result2 = system.evaluate_dataset_quality("kana05", sample_results_data)
        
        report = system.generate_unified_report([result1, result2])
        
        assert "report_id" in report
        assert report["report_id"].startswith("P1A002_unified_quality_")
        assert "generated_at" in report
        assert "standard_version" in report
        assert report["summary"]["total_datasets_evaluated"] == 2
        assert "overall_quality_level" in report["summary"]
        assert "individual_results" in report
        assert len(report["individual_results"]) == 2
        assert "cross_dataset_comparison" in report
        assert "quality_standard_applied" in report
    
    def test_edge_cases(self):
        """エッジケース・境界値テスト"""
        system = UnifiedQualityStandardSystem()
        
        # 空データセット
        empty_data = {
            "total_processed": 0,
            "successful_extractions": 0,
            "ab_evaluation_rate": 0.0,
            "sci_score": 0.0,
            "pla_score": 0.0,
            "ple_score": 0.0
        }
        
        result = system.evaluate_dataset_quality("empty", empty_data)
        assert result.success_rate == 0.0
        assert result.unified_score >= 0.0
        
        # 完璧なデータセット
        perfect_data = {
            "total_processed": 100,
            "successful_extractions": 100,
            "ab_evaluation_rate": 1.0,
            "sci_score": 1.0,
            "pla_score": 1.0,
            "ple_score": 1.0,
            "avg_fill_ratio": 1.0,
            "avg_compactness": 1.0,
            "avg_coverage": 1.0
        }
        
        perfect_result = system.evaluate_dataset_quality("perfect", perfect_data)
        assert perfect_result.success_rate == 1.0
        assert perfect_result.unified_score <= 1.0
        assert perfect_result.quality_level == "EXCELLENT"
        assert perfect_result.unified_grade in ["A+", "A"]


class TestIntegration:
    """統合テスト"""
    
    def test_full_workflow_integration(self, sample_results_data):
        """フルワークフロー統合テスト"""
        system = UnifiedQualityStandardSystem()
        
        # 1. 複数データセット評価
        datasets = ["kana08", "kana05", "kana06"]
        results = []
        
        for i, dataset in enumerate(datasets):
            # データセットごとに異なる品質設定
            data = sample_results_data.copy()
            data["ab_evaluation_rate"] = 0.8 - (i * 0.1)  # 段階的品質低下
            data["sci_score"] = 0.75 - (i * 0.05)
            
            result = system.evaluate_dataset_quality(dataset, data)
            results.append(result)
        
        # 2. データセット比較
        comparison = system.compare_datasets(results)
        assert comparison["total_datasets"] == 3
        
        # 3. 統一品質レポート生成
        report = system.generate_unified_report(results)
        assert report["summary"]["total_datasets_evaluated"] == 3
        
        # 4. 品質レベル分布確認
        quality_levels = [r.quality_level for r in results]
        assert len(set(quality_levels)) >= 1  # 最低1つの品質レベルが存在
        
        # 5. ランキング順序確認
        rankings = comparison["dataset_ranking"]
        for i in range(len(rankings) - 1):
            assert rankings[i]["unified_score"] >= rankings[i+1]["unified_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])