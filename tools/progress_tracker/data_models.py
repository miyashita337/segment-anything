#!/usr/bin/env python3
"""
進捗管理データモデル
Google Sheets連携用のデータ構造とバリデーション
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class TaskStatus(Enum):
    """タスクステータス定義"""
    NOT_STARTED = "着手前"
    IN_PROGRESS = "着手中"
    IMPLEMENTATION_DONE = "実装完了"
    OPERATION_CHECK = "動作確認"
    UNIT_TEST = "テストUNIT"
    QUALITY_CHECK = "品質チェック"
    EXTRACTION_PIPELINE = "抽出パイプライン"
    RELEASE = "/release"
    COMPLETED = "終了"


class ComponentStatus(Enum):
    """コンポーネントステータス定義"""
    EMPTY = ""
    COMPLETED = "完了"
    FAILED = "失敗"
    SKIPPED = "スキップ"
    IN_PROGRESS = "実行中"


class PriorityLevel(Enum):
    """優先度レベル定義"""
    HIGHEST = "優先度最高"
    HIGH = "優先度高"
    MEDIUM = "優先度中"  # デフォルト
    LOW = "優先度低"


class ProgressColumns:
    """Google Sheets列定義（拡張可能設計）"""
    
    # 固定列（変更不可）
    FIXED_COLUMNS = {
        'A': 'tracker_id',          # トラッカーID
        'B': 'priority',            # 優先度
        'C': 'status',              # ステータス
        'D': 'created_date',        # 登録日付
        'E': 'updated_date',        # 更新日付
        'F': 'description',         # 概要
        'G': 'details'              # 詳細
    }
    
    # 動的列（追加・削除可能）
    DYNAMIC_COLUMNS = {
        'H': 'operation_check',     # 動作確認
        'I': 'unit_test',          # テストUNIT
        'J': 'quality_evaluation', # 品質評価
        'K': 'integration_script', # 統合実行スクリプト
        'L': 'dashboard_generation', # ダッシュボード生成
        'M': 'extraction_pipeline' # 抽出パイプライン
    }
    
    # 統計分析列（X-AC列）
    STATISTICAL_COLUMNS = {
        'X': 'current_score',           # Current品質スコア
        'Y': 'baseline_score',          # BaseLine品質スコア
        'Z': 'p_value',                 # p値（統計的有意性）
        'AA': 'effect_size',            # 効果サイズ（Cohen's d）
        'AB': 'improvement_rate',       # 改善率（%）
        'AC': 'statistical_significance' # 統計的有意性
    }
    
    @classmethod
    def get_all_columns(cls) -> Dict[str, str]:
        """全列定義を取得"""
        return {**cls.FIXED_COLUMNS, **cls.DYNAMIC_COLUMNS, **cls.STATISTICAL_COLUMNS}
    
    @classmethod
    def get_column_letter(cls, field_name: str) -> Optional[str]:
        """フィールド名から列文字を取得"""
        all_columns = cls.get_all_columns()
        for letter, name in all_columns.items():
            if name == field_name:
                return letter
        return None
    
    @classmethod
    def get_field_name(cls, column_letter: str) -> Optional[str]:
        """列文字からフィールド名を取得"""
        all_columns = cls.get_all_columns()
        return all_columns.get(column_letter.upper())


@dataclass 
class StatisticalRecord:
    """統計分析レコード（X-AC列）"""
    current_score: Optional[float] = None           # Current品質スコア
    baseline_score: Optional[float] = None          # BaseLine品質スコア
    p_value: Optional[float] = None                 # p値（統計的有意性）
    effect_size: Optional[float] = None             # 効果サイズ（Cohen's d）
    improvement_rate: Optional[float] = None        # 改善率（%）
    statistical_significance: Optional[str] = None  # 統計的有意性
    
    def to_sheets_row(self) -> List[str]:
        """統計分析データをGoogle Sheets行データに変換"""
        return [
            f"{self.current_score:.3f}" if self.current_score is not None else "",
            f"{self.baseline_score:.3f}" if self.baseline_score is not None else "",
            f"{self.p_value:.4f}" if self.p_value is not None else "",
            f"{self.effect_size:.3f}" if self.effect_size is not None else "",
            f"{self.improvement_rate:.1f}%" if self.improvement_rate is not None else "",
            self.statistical_significance if self.statistical_significance else ""
        ]
    
    @classmethod
    def from_sheets_row(cls, row: List[str], start_col: int = 23) -> 'StatisticalRecord':
        """Google Sheets行データから統計分析データ作成（X-AC列、23-28インデックス）"""
        stats_row = row[start_col:start_col+6] if len(row) > start_col else []
        defaults = [""] * 6
        stats_row = stats_row + defaults[len(stats_row):]
        
        def safe_float(value: str) -> Optional[float]:
            """安全なfloat変換"""
            try:
                # %記号を除去して変換
                clean_value = value.replace('%', '') if value else ''
                return float(clean_value) if clean_value else None
            except ValueError:
                return None
        
        return cls(
            current_score=safe_float(stats_row[0]),
            baseline_score=safe_float(stats_row[1]),
            p_value=safe_float(stats_row[2]),
            effect_size=safe_float(stats_row[3]),
            improvement_rate=safe_float(stats_row[4]),
            statistical_significance=stats_row[5] if stats_row[5] else None
        )


@dataclass
class TaskRecord:
    """タスクレコード（進捗+10指標統合）"""
    tracker_id: str
    priority: PriorityLevel = PriorityLevel.MEDIUM  # デフォルト優先度中
    status: TaskStatus = TaskStatus.NOT_STARTED
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    description: str = ""
    details: str = ""  # 詳細フィールド追加
    
    # コンポーネント別ステータス
    operation_check: ComponentStatus = ComponentStatus.EMPTY
    unit_test: ComponentStatus = ComponentStatus.EMPTY
    quality_evaluation: ComponentStatus = ComponentStatus.EMPTY
    integration_script: ComponentStatus = ComponentStatus.EMPTY
    dashboard_generation: ComponentStatus = ComponentStatus.EMPTY
    extraction_pipeline: ComponentStatus = ComponentStatus.EMPTY
    
    # 統計分析統合
    statistics: Optional[StatisticalRecord] = None
    
    def __post_init__(self):
        """初期化後処理"""
        # 日付フィールドはデフォルトで空文字のまま（手動設定または更新時のみ設定）
        # 統計分析データも空で初期化
        if self.statistics is None:
            self.statistics = StatisticalRecord()
    
    def update_status(self, new_status: TaskStatus) -> None:
        """ステータス更新"""
        self.status = new_status
        self.updated_date = datetime.now()  # 更新時のみ自動設定
    
    def update_component(self, component: str, status: ComponentStatus) -> None:
        """コンポーネントステータス更新"""
        if hasattr(self, component):
            setattr(self, component, status)
            self.updated_date = datetime.now()  # 更新時のみ自動設定
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def to_sheets_row(self) -> List[str]:
        """Google Sheets行データに変換（基本13列：A-M + 統計6列：X-AC）"""
        base_row = [
            self.tracker_id,
            self.priority.value,
            self.status.value,
            self.created_date.strftime('%Y-%m-%d %H:%M:%S') if self.created_date else "",
            self.updated_date.strftime('%Y-%m-%d %H:%M:%S') if self.updated_date else "",
            self.description,
            self.details,  # 詳細フィールド追加
            self.operation_check.value,
            self.unit_test.value,
            self.quality_evaluation.value,
            self.integration_script.value,
            self.dashboard_generation.value,
            self.extraction_pipeline.value
        ]
        
        # N-W列（削除済み領域）は空で埋める
        empty_nw_columns = [""] * 10
        
        # X-AC列統計分析データ追加
        statistics_row = self.statistics.to_sheets_row() if self.statistics else [""] * 6
        
        return base_row + empty_nw_columns + statistics_row
    
    @staticmethod
    def _parse_date_flexible(date_str: str) -> Optional[datetime]:
        """柔軟な日付解析（既存データ対応）"""
        if not date_str:
            return None
        
        # 新フォーマット（yyyy-mm-dd hh:mm:ss）を優先
        try:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
        
        # 旧フォーマット（yyyy-mm-dd）への後方互換
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            pass
        
        # ISO形式への対応
        try:
            return datetime.fromisoformat(date_str.replace('T', ' '))
        except ValueError:
            pass
        
        return None

    @classmethod
    def from_sheets_row(cls, row: List[str]) -> 'TaskRecord':
        """Google Sheets行データから作成（拡張列対応）"""
        # デフォルト値設定（29列分：A-M + N-W(空) + X-AC）
        defaults = [""] * 29
        row = row + defaults[len(row):]
        
        # 統計分析部分を抽出（X-AC列、インデックス23-28）
        statistics = StatisticalRecord.from_sheets_row(row, start_col=23)
        
        # 優先度の安全な変換
        def safe_priority(value: str) -> PriorityLevel:
            """優先度の安全な変換"""
            for priority in PriorityLevel:
                if priority.value == value:
                    return priority
            return PriorityLevel.MEDIUM  # デフォルト
        
        return cls(
            tracker_id=row[0],
            priority=safe_priority(row[1]),
            status=TaskStatus(row[2]) if row[2] else TaskStatus.NOT_STARTED,
            created_date=cls._parse_date_flexible(row[3]) if row[3] else None,
            updated_date=cls._parse_date_flexible(row[4]) if row[4] else None,
            description=row[5],
            details=row[6],  # 詳細フィールド追加
            operation_check=ComponentStatus(row[7]) if row[7] else ComponentStatus.EMPTY,
            unit_test=ComponentStatus(row[8]) if row[8] else ComponentStatus.EMPTY,
            quality_evaluation=ComponentStatus(row[9]) if row[9] else ComponentStatus.EMPTY,
            integration_script=ComponentStatus(row[10]) if row[10] else ComponentStatus.EMPTY,
            dashboard_generation=ComponentStatus(row[11]) if row[11] else ComponentStatus.EMPTY,
            extraction_pipeline=ComponentStatus(row[12]) if row[12] else ComponentStatus.EMPTY,
            statistics=statistics
        )


@dataclass
class ProgressTrackerConfig:
    """進捗管理設定"""
    spreadsheet_id: str
    sheet_name: str = "Progress Tracker"
    auth_file_path: str = "config/google_sheets_auth.json"
    
    # ドロップダウン値定義
    priority_values: List[str] = field(default_factory=lambda: [p.value for p in PriorityLevel])
    status_values: List[str] = field(default_factory=lambda: [s.value for s in TaskStatus])
    component_values: List[str] = field(default_factory=lambda: [s.value for s in ComponentStatus])


class ProgressTrackerError(Exception):
    """進捗管理専用例外"""
    pass


class SheetsAPIError(ProgressTrackerError):
    """Google Sheets API例外"""
    pass


class ValidationError(ProgressTrackerError):
    """バリデーション例外"""
    pass