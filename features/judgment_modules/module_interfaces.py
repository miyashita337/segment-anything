"""
KIRO-012: 判定処理モジュール共通インターフェース

判定処理モジュール間の統一されたインターフェースと共通データ構造を定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np


class QualityGrade(Enum):
    """品質評価グレード"""
    A = "A"  # 最高品質
    B = "B"  # 高品質
    C = "C"  # 中品質
    D = "D"  # 低品質
    E = "E"  # 最低品質
    F = "F"  # 失敗


@dataclass
class JudgmentInput:
    """判定処理への入力データ"""
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_context: Optional[Dict[str, Any]] = None


@dataclass
class JudgmentResult:
    """判定処理の結果"""
    quality_grade: QualityGrade
    confidence_score: float  # 0.0-1.0
    numeric_score: float     # 0.0-1.0
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    processing_time: float
    module_version: str


@dataclass
class AggregatedJudgment:
    """複数モジュールの統合判定結果"""
    final_grade: QualityGrade
    overall_confidence: float
    module_results: Dict[str, JudgmentResult]
    consensus_metrics: Dict[str, float]
    conflict_analysis: Dict[str, Any]
    recommendation_summary: List[str]


class JudgmentModule(ABC):
    """判定処理モジュールの基底クラス"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: モジュール固有の設定パラメータ
        """
        self.config = config or {}
        self.module_name = self.__class__.__name__
        self.version = "1.0.0"

    @abstractmethod
    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """
        判定処理のメイン実行メソッド

        Args:
            input_data: 判定対象データ

        Returns:
            JudgmentResult: 判定結果
        """
        pass

    @abstractmethod
    def get_thresholds(self) -> Dict[str, float]:
        """
        現在設定されている閾値を取得

        Returns:
            Dict[str, float]: 閾値の辞書
        """
        pass

    @abstractmethod
    def update_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """
        閾値を更新

        Args:
            thresholds: 新しい閾値の辞書

        Returns:
            bool: 更新成功フラグ
        """
        pass

    def validate_input(self, input_data: JudgmentInput) -> bool:
        """
        入力データの検証

        Args:
            input_data: 検証対象データ

        Returns:
            bool: 検証結果
        """
        if input_data.image is None:
            return False
        if not isinstance(input_data.image, np.ndarray):
            return False
        if len(input_data.image.shape) not in [2, 3]:
            return False
        return True

    def create_result(self,
                     grade: QualityGrade,
                     confidence: float,
                     score: float,
                     issues: List[str],
                     recommendations: List[str],
                     metrics: Dict[str, float],
                     processing_time: float) -> JudgmentResult:
        """
        統一された結果オブジェクトの作成
        """
        return JudgmentResult(
            quality_grade=grade,
            confidence_score=max(0.0, min(1.0, confidence)),
            numeric_score=max(0.0, min(1.0, score)),
            issues=issues,
            recommendations=recommendations,
            metrics=metrics,
            processing_time=processing_time,
            module_version=self.version
        )


class ConfigurableModule(JudgmentModule):
    """設定可能な判定モジュールの基底クラス"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._default_thresholds = self._get_default_thresholds()
        self._current_thresholds = self._default_thresholds.copy()
        if config and 'thresholds' in config:
            self._current_thresholds.update(config['thresholds'])

    @abstractmethod
    def _get_default_thresholds(self) -> Dict[str, float]:
        """デフォルト閾値の定義"""
        pass

    def get_thresholds(self) -> Dict[str, float]:
        """現在の閾値を取得"""
        return self._current_thresholds.copy()

    def update_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """閾値を更新"""
        try:
            for key, value in thresholds.items():
                if key in self._default_thresholds:
                    self._current_thresholds[key] = float(value)
                else:
                    raise KeyError(f"Unknown threshold: {key}")
            return True
        except (ValueError, KeyError) as e:
            return False

    def reset_thresholds(self):
        """閾値をデフォルトにリセット"""
        self._current_thresholds = self._default_thresholds.copy()


class ModuleRegistry:
    """判定モジュールの登録・管理クラス"""

    def __init__(self):
        self._modules: Dict[str, JudgmentModule] = {}
        self._module_order: List[str] = []

    def register_module(self, name: str, module: JudgmentModule, order: Optional[int] = None):
        """
        モジュールを登録

        Args:
            name: モジュール名
            module: モジュールインスタンス
            order: 実行順序（None の場合は末尾に追加）
        """
        self._modules[name] = module
        if order is not None and 0 <= order <= len(self._module_order):
            self._module_order.insert(order, name)
        else:
            self._module_order.append(name)

    def get_module(self, name: str) -> Optional[JudgmentModule]:
        """モジュールを取得"""
        return self._modules.get(name)

    def get_all_modules(self) -> Dict[str, JudgmentModule]:
        """全モジュールを取得"""
        return self._modules.copy()

    def get_execution_order(self) -> List[str]:
        """実行順序を取得"""
        return self._module_order.copy()

    def unregister_module(self, name: str) -> bool:
        """モジュールの登録を解除"""
        if name in self._modules:
            del self._modules[name]
            if name in self._module_order:
                self._module_order.remove(name)
            return True
        return False


# グローバルレジストリインスタンス
default_registry = ModuleRegistry()