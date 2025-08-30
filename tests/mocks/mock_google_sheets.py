#!/usr/bin/env python3
"""
Google Sheets連携モックシステム

実際のGoogle Sheets APIをモックで再現
テスト用に予測可能な結果を返す
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid


@dataclass
class MockTrackerEntry:
    """トラッカーエントリのモック"""
    tracker_id: str
    priority: str
    status: str
    created_date: str
    updated_date: str
    description: str
    current_score: Optional[float] = None
    baseline_score: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    improvement_rate: Optional[str] = None
    significance: Optional[str] = None


class MockGoogleSheetsClient:
    """Google Sheets APIのモッククライアント"""
    
    def __init__(self):
        """モックGoogle Sheetsクライアント初期化"""
        self.sheet_data = self._initialize_mock_data()
        self.operation_delay = 0.1  # API遅延シミュレート
    
    def _initialize_mock_data(self) -> Dict[str, MockTrackerEntry]:
        """モックシートデータ初期化"""
        base_date = datetime.now() - timedelta(days=30)
        
        # 完了済みトラッカーのモックデータ
        completed_trackers = [
            ("QUAL-001", "高", 0.552, 0.361, 52.9, 1.490, 0.1105, "非有意"),
            ("QUAL-002", "中", 0.748, 0.712, 5.1, 0.324, 0.2455, "非有意"),  
            ("QUAL-003", "中", 0.834, 0.792, 5.3, 0.401, 0.1876, "非有意"),
            ("INTG-001", "高", 0.672, 0.645, 4.2, 0.278, 0.3234, "非有意"),
            ("OPTM-001", "低", 0.591, 0.578, 2.2, 0.156, 0.4123, "非有意"),
        ]
        
        mock_data = {}
        
        for i, (tracker_id, priority, current, baseline, improvement, cohens_d, p_val, significance) in enumerate(completed_trackers):
            created = base_date + timedelta(days=i*3)
            updated = created + timedelta(days=1, hours=random.randint(1, 12))
            
            mock_data[tracker_id] = MockTrackerEntry(
                tracker_id=tracker_id,
                priority=priority,
                status="/release",
                created_date=created.strftime("%Y-%m-%d %H:%M:%S"),
                updated_date=updated.strftime("%Y-%m-%d %H:%M:%S"),
                description=f"{tracker_id}の説明テキスト",
                current_score=current,
                baseline_score=baseline,
                p_value=p_val,
                effect_size=cohens_d,
                improvement_rate=f"{improvement:.1f}%",
                significance=significance
            )
        
        # 進行中トラッカーのモックデータ
        in_progress_trackers = [
            ("TEST-001", "高", "着手中"),
            ("QUAL-042", "中", "着手前"),
            ("INTG-005", "低", "着手前")
        ]
        
        for j, (tracker_id, priority, status) in enumerate(in_progress_trackers):
            created = base_date + timedelta(days=(len(completed_trackers) + j) * 2)
            
            mock_data[tracker_id] = MockTrackerEntry(
                tracker_id=tracker_id,
                priority=priority,
                status=status,
                created_date=created.strftime("%Y-%m-%d %H:%M:%S"),
                updated_date=created.strftime("%Y-%m-%d %H:%M:%S"),
                description=f"{tracker_id}の説明テキスト"
            )
        
        return mock_data
    
    def get_tracker_data(self, tracker_id: str) -> Optional[MockTrackerEntry]:
        """
        トラッカーデータ取得
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            トラッカーデータまたはNone
        """
        time.sleep(self.operation_delay)
        return self.sheet_data.get(tracker_id)
    
    def update_tracker_status(self, tracker_id: str, new_status: str) -> bool:
        """
        トラッカーステータス更新
        
        Args:
            tracker_id: トラッカーID
            new_status: 新しいステータス
            
        Returns:
            更新成功可否
        """
        time.sleep(self.operation_delay)
        
        if tracker_id in self.sheet_data:
            self.sheet_data[tracker_id].status = new_status
            self.sheet_data[tracker_id].updated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False
    
    def update_statistical_data(
        self,
        tracker_id: str,
        current_score: float,
        baseline_score: float,
        p_value: float,
        effect_size: float,
        improvement_rate: float,
        significance: str
    ) -> bool:
        """
        統計データ更新
        
        Args:
            tracker_id: トラッカーID
            current_score: 現在スコア
            baseline_score: ベースラインスコア
            p_value: p値
            effect_size: 効果サイズ（Cohen's d）
            improvement_rate: 改善率
            significance: 統計的有意性
            
        Returns:
            更新成功可否
        """
        time.sleep(self.operation_delay)
        
        if tracker_id in self.sheet_data:
            entry = self.sheet_data[tracker_id]
            entry.current_score = current_score
            entry.baseline_score = baseline_score
            entry.p_value = p_value
            entry.effect_size = effect_size
            entry.improvement_rate = f"{improvement_rate:.1f}%"
            entry.significance = significance
            entry.updated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False
    
    def get_completed_trackers(self) -> List[MockTrackerEntry]:
        """
        完了済みトラッカー一覧取得
        
        Returns:
            完了済みトラッカーリスト
        """
        time.sleep(self.operation_delay)
        
        completed = []
        for entry in self.sheet_data.values():
            if entry.status == "/release":
                completed.append(entry)
        
        # 更新日時でソート
        completed.sort(key=lambda x: x.updated_date, reverse=True)
        return completed
    
    def get_trackers_with_statistical_data(self) -> List[MockTrackerEntry]:
        """
        統計データありのトラッカー一覧取得
        
        Returns:
            統計データありトラッカーリスト
        """
        time.sleep(self.operation_delay)
        
        statistical_trackers = []
        for entry in self.sheet_data.values():
            if entry.current_score is not None and entry.baseline_score is not None:
                statistical_trackers.append(entry)
        
        return statistical_trackers
    
    def find_baseline_candidate(self, current_tracker_id: str) -> Optional[MockTrackerEntry]:
        """
        ベースライン候補検索
        
        Args:
            current_tracker_id: 現在のトラッカーID
            
        Returns:
            ベースライン候補またはNone
        """
        time.sleep(self.operation_delay)
        
        # 統計データありの完了済みトラッカーを取得
        candidates = []
        for entry in self.sheet_data.values():
            if (entry.tracker_id != current_tracker_id and 
                entry.status == "/release" and 
                entry.current_score is not None):
                candidates.append(entry)
        
        if not candidates:
            return None
        
        # 最新の更新日時のものを返す
        candidates.sort(key=lambda x: x.updated_date, reverse=True)
        return candidates[0]
    
    def create_tracker(self, tracker_id: str, description: str, priority: str = "中") -> bool:
        """
        新規トラッカー作成
        
        Args:
            tracker_id: トラッカーID
            description: 説明
            priority: 優先度
            
        Returns:
            作成成功可否
        """
        time.sleep(self.operation_delay)
        
        if tracker_id in self.sheet_data:
            return False  # 既存
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.sheet_data[tracker_id] = MockTrackerEntry(
            tracker_id=tracker_id,
            priority=priority,
            status="着手前",
            created_date=now,
            updated_date=now,
            description=description
        )
        
        return True
    
    def get_all_trackers(self) -> List[MockTrackerEntry]:
        """
        全トラッカー取得
        
        Returns:
            全トラッカーリスト
        """
        time.sleep(self.operation_delay)
        
        all_trackers = list(self.sheet_data.values())
        all_trackers.sort(key=lambda x: x.updated_date, reverse=True)
        return all_trackers
    
    def export_to_json(self) -> str:
        """
        データをJSON形式でエクスポート
        
        Returns:
            JSON文字列
        """
        export_data = {}
        for tracker_id, entry in self.sheet_data.items():
            export_data[tracker_id] = {
                "tracker_id": entry.tracker_id,
                "priority": entry.priority,
                "status": entry.status,
                "created_date": entry.created_date,
                "updated_date": entry.updated_date,
                "description": entry.description,
                "statistical_data": {
                    "current_score": entry.current_score,
                    "baseline_score": entry.baseline_score,
                    "p_value": entry.p_value,
                    "effect_size": entry.effect_size,
                    "improvement_rate": entry.improvement_rate,
                    "significance": entry.significance
                }
            }
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)


class MockStatisticalAnalyzer:
    """統計分析エンジンのモック"""
    
    @staticmethod
    def calculate_cohens_d(current_data: List[float], baseline_data: List[float]) -> float:
        """
        Cohen's d効果サイズ計算のモック
        
        Args:
            current_data: 現在データ
            baseline_data: ベースラインデータ
            
        Returns:
            Cohen's d値
        """
        if not current_data or not baseline_data:
            return 0.0
        
        # 平均差計算
        current_mean = sum(current_data) / len(current_data)
        baseline_mean = sum(baseline_data) / len(baseline_data)
        mean_diff = current_mean - baseline_mean
        
        # プールされた標準偏差計算（簡易版）
        # 実際はより複雑な計算だが、テスト用に簡略化
        current_var = sum((x - current_mean) ** 2 for x in current_data) / len(current_data)
        baseline_var = sum((x - baseline_mean) ** 2 for x in baseline_data) / len(baseline_data)
        pooled_std = ((current_var + baseline_var) / 2) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    @staticmethod
    def calculate_p_value(current_data: List[float], baseline_data: List[float]) -> float:
        """
        p値計算のモック（ウェルチのt検定近似）
        
        Args:
            current_data: 現在データ
            baseline_data: ベースラインデータ
            
        Returns:
            p値（近似値）
        """
        if not current_data or not baseline_data:
            return 1.0
        
        # t統計量計算（簡易版）
        current_mean = sum(current_data) / len(current_data)
        baseline_mean = sum(baseline_data) / len(baseline_data)
        
        current_var = sum((x - current_mean) ** 2 for x in current_data) / len(current_data)
        baseline_var = sum((x - baseline_mean) ** 2 for x in baseline_data) / len(baseline_data)
        
        se_diff = (current_var / len(current_data) + baseline_var / len(baseline_data)) ** 0.5
        
        if se_diff == 0:
            return 1.0
        
        t_stat = abs(current_mean - baseline_mean) / se_diff
        
        # p値近似（簡易版：t統計量から推定）
        if t_stat > 2.576:  # 99%信頼区間
            return 0.01
        elif t_stat > 1.96:  # 95%信頼区間
            return 0.05
        elif t_stat > 1.645:  # 90%信頼区間
            return 0.10
        else:
            return min(1.0, 0.20 + random.uniform(-0.05, 0.05))
    
    @staticmethod
    def determine_significance(p_value: float, alpha: float = 0.05) -> str:
        """
        統計的有意性判定
        
        Args:
            p_value: p値
            alpha: 有意水準
            
        Returns:
            有意性判定結果
        """
        return "有意" if p_value < alpha else "非有意"
    
    @staticmethod
    def calculate_confidence_interval(
        data: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        信頼区間計算のモック
        
        Args:
            data: データ
            confidence_level: 信頼度
            
        Returns:
            信頼区間（下限、上限）
        """
        if not data:
            return (0.0, 0.0)
        
        mean_val = sum(data) / len(data)
        var_val = sum((x - mean_val) ** 2 for x in data) / len(data)
        std_err = (var_val / len(data)) ** 0.5
        
        # z値近似（簡易版）
        if confidence_level >= 0.99:
            z_score = 2.576
        elif confidence_level >= 0.95:
            z_score = 1.96
        else:
            z_score = 1.645
        
        margin_of_error = z_score * std_err
        
        return (mean_val - margin_of_error, mean_val + margin_of_error)
    
    @staticmethod
    def generate_mock_quality_data(n_samples: int, base_score: float, variation: float = 0.1) -> List[float]:
        """
        品質データ生成（テスト用）
        
        Args:
            n_samples: サンプル数
            base_score: ベーススコア
            variation: バリエーション
            
        Returns:
            品質データリスト
        """
        data = []
        for _ in range(n_samples):
            # 正規分布近似でデータ生成
            score = base_score + random.uniform(-variation, variation)
            score = max(0.0, min(1.0, score))  # 0-1の範囲に制限
            data.append(score)
        
        return data


# モジュールレベル変数で共有インスタンス管理
_mock_sheets_client = None


def get_mock_sheets_client() -> MockGoogleSheetsClient:
    """
    モックGoogle Sheetsクライアント取得
    
    Returns:
        モックGoogle Sheetsクライアントインスタンス
    """
    global _mock_sheets_client
    if _mock_sheets_client is None:
        _mock_sheets_client = MockGoogleSheetsClient()
    return _mock_sheets_client


def reset_mock_sheets_client() -> None:
    """モックGoogle Sheetsクライアントリセット"""
    global _mock_sheets_client
    _mock_sheets_client = None