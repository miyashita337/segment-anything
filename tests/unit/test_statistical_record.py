#!/usr/bin/env python3
"""
StatisticalRecord クラスのテストスイート
MetricsRecord 削除後の代替確認テスト
"""

import sys
import os
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.progress_tracker.data_models import (
    StatisticalRecord,
    TaskRecord,
    TaskStatus,
    PriorityLevel,
    ComponentStatus
)


class TestStatisticalRecord:
    """統計分析レコードのテスト"""
    
    def test_statistical_record_initialization(self):
        """StatisticalRecord の初期化テスト"""
        stats = StatisticalRecord()
        
        assert stats.current_score is None
        assert stats.baseline_score is None
        assert stats.p_value is None
        assert stats.effect_size is None
        assert stats.improvement_rate is None
        assert stats.statistical_significance is None
    
    def test_statistical_record_with_values(self):
        """値付きStatisticalRecord の作成テスト"""
        stats = StatisticalRecord(
            current_score=0.857,
            baseline_score=0.742,
            p_value=0.0234,
            effect_size=1.245,
            improvement_rate=15.5,
            statistical_significance="有意"
        )
        
        assert stats.current_score == 0.857
        assert stats.baseline_score == 0.742
        assert stats.p_value == 0.0234
        assert stats.effect_size == 1.245
        assert stats.improvement_rate == 15.5
        assert stats.statistical_significance == "有意"
    
    def test_to_sheets_row(self):
        """Google Sheets行データ変換テスト"""
        stats = StatisticalRecord(
            current_score=0.857,
            baseline_score=0.742,
            p_value=0.0234,
            effect_size=1.245,
            improvement_rate=15.5,
            statistical_significance="有意"
        )
        
        row = stats.to_sheets_row()
        expected = ["0.857", "0.742", "0.0234", "1.245", "15.5%", "有意"]
        
        assert row == expected
    
    def test_to_sheets_row_empty(self):
        """空のStatisticalRecord の行データ変換テスト"""
        stats = StatisticalRecord()
        row = stats.to_sheets_row()
        expected = ["", "", "", "", "", ""]
        
        assert row == expected
    
    def test_from_sheets_row(self):
        """Google Sheets行データからの復元テスト"""
        # X-AC列のテストデータ（インデックス23-28）
        test_row = [""] * 30  # 29列+余裕
        test_row[23] = "0.857"    # X列: current_score
        test_row[24] = "0.742"    # Y列: baseline_score  
        test_row[25] = "0.0234"   # Z列: p_value
        test_row[26] = "1.245"    # AA列: effect_size
        test_row[27] = "15.5%"    # AB列: improvement_rate
        test_row[28] = "有意"      # AC列: statistical_significance
        
        stats = StatisticalRecord.from_sheets_row(test_row, start_col=23)
        
        assert stats.current_score == 0.857
        assert stats.baseline_score == 0.742
        assert stats.p_value == 0.0234
        assert stats.effect_size == 1.245
        assert stats.improvement_rate == 15.5
        assert stats.statistical_significance == "有意"
    
    def test_from_sheets_row_empty(self):
        """空行からのStatisticalRecord 復元テスト"""
        test_row = [""] * 30
        stats = StatisticalRecord.from_sheets_row(test_row, start_col=23)
        
        assert stats.current_score is None
        assert stats.baseline_score is None
        assert stats.p_value is None
        assert stats.effect_size is None
        assert stats.improvement_rate is None
        assert stats.statistical_significance is None


class TestTaskRecordStatisticsIntegration:
    """TaskRecord と StatisticalRecord の統合テスト"""
    
    def test_task_record_with_statistics(self):
        """統計分析付きTaskRecord の作成テスト"""
        stats = StatisticalRecord(
            current_score=0.857,
            baseline_score=0.742,
            p_value=0.0234,
            effect_size=1.245,
            improvement_rate=15.5,
            statistical_significance="有意"
        )
        
        task = TaskRecord(
            tracker_id="METRICS-CLEANUP-001",
            description="旧10指標システム削除テスト",
            statistics=stats
        )
        
        assert task.tracker_id == "METRICS-CLEANUP-001"
        assert task.statistics.current_score == 0.857
        assert task.statistics.statistical_significance == "有意"
    
    def test_task_record_auto_statistics_initialization(self):
        """TaskRecord の統計分析自動初期化テスト"""
        task = TaskRecord(
            tracker_id="TEST-001",
            description="自動初期化テスト"
        )
        
        # __post_init__ で自動初期化されることを確認
        assert task.statistics is not None
        assert isinstance(task.statistics, StatisticalRecord)
        assert task.statistics.current_score is None
    
    def test_task_record_to_sheets_row_with_statistics(self):
        """統計分析付きTaskRecord のGoogle Sheets行データ変換テスト"""
        stats = StatisticalRecord(
            current_score=0.857,
            baseline_score=0.742,
            p_value=0.0234,
            effect_size=1.245,
            improvement_rate=15.5,
            statistical_significance="有意"
        )
        
        task = TaskRecord(
            tracker_id="METRICS-CLEANUP-001",
            priority=PriorityLevel.HIGH,
            status=TaskStatus.RELEASE,
            description="旧10指標システム削除",
            details="完了テスト",
            statistics=stats
        )
        
        row = task.to_sheets_row()
        
        # 基本13列 + 空N-W列10列 + 統計X-AC列6列 = 29列
        assert len(row) == 29
        
        # 基本情報確認
        assert row[0] == "METRICS-CLEANUP-001"
        assert row[1] == "優先度高"
        assert row[2] == "/release"
        assert row[5] == "旧10指標システム削除"
        assert row[6] == "完了テスト"
        
        # N-W列（13-22）が空であることを確認
        for i in range(13, 23):
            assert row[i] == ""
        
        # X-AC列（23-28）の統計データ確認
        assert row[23] == "0.857"    # current_score
        assert row[24] == "0.742"    # baseline_score
        assert row[25] == "0.0234"   # p_value
        assert row[26] == "1.245"    # effect_size
        assert row[27] == "15.5%"    # improvement_rate
        assert row[28] == "有意"      # statistical_significance
    
    def test_task_record_from_sheets_row_with_statistics(self):
        """Google Sheets行データからの統計分析付きTaskRecord 復元テスト"""
        # 29列のテストデータ
        test_row = [
            "METRICS-CLEANUP-001",  # A: tracker_id
            "優先度高",               # B: priority
            "/release",             # C: status
            "2025-08-14 12:30:00",  # D: created_date
            "2025-08-14 13:45:00",  # E: updated_date
            "旧10指標システム削除",     # F: description
            "完了テスト詳細",          # G: details
            "完了",                  # H: operation_check
            "完了",                  # I: unit_test
            "完了",                  # J: quality_evaluation
            "完了",                  # K: integration_script
            "完了",                  # L: dashboard_generation
            "完了",                  # M: extraction_pipeline
        ] + [""] * 10 + [          # N-W列（空）
            "0.857",                # X: current_score
            "0.742",                # Y: baseline_score
            "0.0234",               # Z: p_value
            "1.245",                # AA: effect_size
            "15.5%",                # AB: improvement_rate
            "有意"                   # AC: statistical_significance
        ]
        
        task = TaskRecord.from_sheets_row(test_row)
        
        # 基本情報確認
        assert task.tracker_id == "METRICS-CLEANUP-001"
        assert task.priority == PriorityLevel.HIGH
        assert task.status == TaskStatus.RELEASE
        assert task.description == "旧10指標システム削除"
        assert task.details == "完了テスト詳細"
        
        # 統計分析データ確認
        assert task.statistics is not None
        assert task.statistics.current_score == 0.857
        assert task.statistics.baseline_score == 0.742
        assert task.statistics.p_value == 0.0234
        assert task.statistics.effect_size == 1.245
        assert task.statistics.improvement_rate == 15.5
        assert task.statistics.statistical_significance == "有意"


def run_all_tests():
    """全テスト実行"""
    print("=== StatisticalRecord クラステスト開始 ===")
    
    # テストクラス初期化
    test_stats = TestStatisticalRecord()
    test_task_integration = TestTaskRecordStatisticsIntegration()
    
    tests = [
        ("StatisticalRecord初期化", test_stats.test_statistical_record_initialization),
        ("StatisticalRecord値付き作成", test_stats.test_statistical_record_with_values),
        ("sheets行データ変換", test_stats.test_to_sheets_row),
        ("空sheets行データ変換", test_stats.test_to_sheets_row_empty),
        ("sheets行データ復元", test_stats.test_from_sheets_row),
        ("空行復元", test_stats.test_from_sheets_row_empty),
        ("TaskRecord統計分析統合", test_task_integration.test_task_record_with_statistics),
        ("TaskRecord自動初期化", test_task_integration.test_task_record_auto_statistics_initialization),
        ("TaskRecord統計付きsheets変換", test_task_integration.test_task_record_to_sheets_row_with_statistics),
        ("TaskRecord統計付きsheets復元", test_task_integration.test_task_record_from_sheets_row_with_statistics),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASS")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAIL - {e}")
            failed += 1
    
    print("\n=== テスト結果 ===")
    print(f"合格: {passed}件")
    print(f"失敗: {failed}件")
    print(f"成功率: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)