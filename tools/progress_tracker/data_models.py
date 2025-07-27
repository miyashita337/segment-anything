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
    
    # 10指標列（N-W列）
    METRICS_COLUMNS = {
        'N': 'lca',                 # LCA (バウンディングボックス精度)
        'O': 'ab_evaluation_rate',  # A/B評価率
        'P': 'fps',                 # FPS (処理速度)
        'Q': 'c_plus_rate',         # C以上評価率
        'R': 'avg_coverage_rate',   # 平均カバレッジ率
        'S': 'avg_compactness',     # 平均コンパクトネス
        'T': 'avg_fill_rate',       # 平均フィル率
        'U': 'sci',                 # SCI (Semantic Completeness Index)
        'V': 'pla',                 # PLA (Pixel-Level Accuracy)
        'W': 'ple'                  # PLE (Progressive Learning Efficiency)
    }
    
    @classmethod
    def get_all_columns(cls) -> Dict[str, str]:
        """全列定義を取得"""
        return {**cls.FIXED_COLUMNS, **cls.DYNAMIC_COLUMNS, **cls.METRICS_COLUMNS}
    
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
class MetricsRecord:
    """10指標レコード"""
    lca: Optional[float] = None                 # LCA (バウンディングボックス精度)
    ab_evaluation_rate: Optional[float] = None  # A/B評価率
    fps: Optional[float] = None                 # FPS (処理速度)
    c_plus_rate: Optional[float] = None         # C以上評価率
    avg_coverage_rate: Optional[float] = None   # 平均カバレッジ率
    avg_compactness: Optional[float] = None     # 平均コンパクトネス
    avg_fill_rate: Optional[float] = None       # 平均フィル率
    sci: Optional[float] = None                 # SCI (Semantic Completeness Index)
    pla: Optional[float] = None                 # PLA (Pixel-Level Accuracy)
    ple: Optional[float] = None                 # PLE (Progressive Learning Efficiency)
    
    def to_sheets_row(self) -> List[str]:
        """10指標をGoogle Sheets行データに変換"""
        return [
            f"{self.lca:.3f}" if self.lca is not None else "",
            f"{self.ab_evaluation_rate:.3f}" if self.ab_evaluation_rate is not None else "",
            f"{self.fps:.3f}" if self.fps is not None else "",
            f"{self.c_plus_rate:.3f}" if self.c_plus_rate is not None else "",
            f"{self.avg_coverage_rate:.3f}" if self.avg_coverage_rate is not None else "",
            f"{self.avg_compactness:.3f}" if self.avg_compactness is not None else "",
            f"{self.avg_fill_rate:.3f}" if self.avg_fill_rate is not None else "",
            f"{self.sci:.3f}" if self.sci is not None else "",
            f"{self.pla:.3f}" if self.pla is not None else "",
            f"{self.ple:.3f}" if self.ple is not None else ""
        ]
    
    @classmethod
    def from_sheets_row(cls, row: List[str], start_col: int = 12) -> 'MetricsRecord':
        """Google Sheets行データから10指標作成"""
        # デフォルト値設定（M-V列、12-21インデックス）
        metrics_row = row[start_col:start_col+10] if len(row) > start_col else []
        defaults = [""] * 10
        metrics_row = metrics_row + defaults[len(metrics_row):]
        
        def safe_float(value: str) -> Optional[float]:
            """安全なfloat変換"""
            try:
                return float(value) if value else None
            except ValueError:
                return None
        
        return cls(
            lca=safe_float(metrics_row[0]),
            ab_evaluation_rate=safe_float(metrics_row[1]),
            fps=safe_float(metrics_row[2]),
            c_plus_rate=safe_float(metrics_row[3]),
            avg_coverage_rate=safe_float(metrics_row[4]),
            avg_compactness=safe_float(metrics_row[5]),
            avg_fill_rate=safe_float(metrics_row[6]),
            sci=safe_float(metrics_row[7]),
            pla=safe_float(metrics_row[8]),
            ple=safe_float(metrics_row[9])
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
    
    # 10指標統合
    metrics: Optional[MetricsRecord] = None
    
    def __post_init__(self):
        """初期化後処理"""
        # 日付フィールドはデフォルトで空文字のまま（手動設定または更新時のみ設定）
        # メトリクスも空で初期化
        if self.metrics is None:
            self.metrics = MetricsRecord()
    
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
        """Google Sheets行データに変換（23列：A-W）"""
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
        
        # 10指標追加（N-W列）
        metrics_row = self.metrics.to_sheets_row() if self.metrics else [""] * 10
        
        return base_row + metrics_row
    
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
        """Google Sheets行データから作成（23列対応）"""
        # デフォルト値設定（23列分）
        defaults = [""] * 23
        row = row + defaults[len(row):]
        
        # メトリクス部分を抽出（N-W列、インデックス13-22）
        metrics = MetricsRecord.from_sheets_row(row, start_col=13)
        
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
            metrics=metrics
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