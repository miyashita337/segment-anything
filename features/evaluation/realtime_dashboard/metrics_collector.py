"""
メトリクス収集クラス

リアルタイムで抽出処理のメトリクスを収集・管理
"""

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


@dataclass
class ExtractMetrics:
    """抽出メトリクスデータ"""

    timestamp: float
    image_name: str
    status: str  # processing, success, failed
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """集計メトリクス"""

    total_images: int = 0
    processed_images: int = 0
    success_count: int = 0
    failed_count: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    success_rate: float = 0.0
    current_fps: float = 0.0
    memory_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


class MetricsCollector:
    """メトリクス収集クラス"""

    def __init__(self, max_history: int = 1000):
        """
        初期化

        Args:
            max_history: 保持する履歴の最大数
        """
        self._metrics_history: Deque[ExtractMetrics] = deque(maxlen=max_history)
        self._current_metrics: Dict[str, ExtractMetrics] = {}
        self._lock = Lock()
        self._start_time = time.time()

    def start_processing(self, image_name: str) -> None:
        """
        画像処理開始を記録

        Args:
            image_name: 処理中の画像名
        """
        with self._lock:
            metrics = ExtractMetrics(
                timestamp=time.time(), image_name=image_name, status="processing"
            )
            self._current_metrics[image_name] = metrics

    def complete_processing(
        self,
        image_name: str,
        success: bool,
        quality_score: Optional[float] = None,
        memory_usage: Optional[Dict[str, float]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        画像処理完了を記録

        Args:
            image_name: 処理完了した画像名
            success: 成功フラグ
            quality_score: 品質スコア
            memory_usage: メモリ使用量
            error_message: エラーメッセージ（失敗時）
        """
        with self._lock:
            if image_name in self._current_metrics:
                metrics = self._current_metrics[image_name]
                metrics.status = "success" if success else "failed"
                metrics.processing_time = time.time() - metrics.timestamp
                metrics.quality_score = quality_score
                metrics.memory_usage = memory_usage
                metrics.error_message = error_message

                # 履歴に追加
                self._metrics_history.append(metrics)
                del self._current_metrics[image_name]

    def get_current_status(self) -> Dict[str, Any]:
        """
        現在の処理状況を取得

        Returns:
            現在の処理状況
        """
        with self._lock:
            processing_images = list(self._current_metrics.keys())
            return {
                "processing_images": processing_images,
                "processing_count": len(processing_images),
                "timestamp": time.time(),
            }

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """
        集計メトリクスを取得

        Returns:
            集計されたメトリクス
        """
        with self._lock:
            total_images = len(self._metrics_history) + len(self._current_metrics)
            processed_images = len(self._metrics_history)
            success_count = sum(1 for m in self._metrics_history if m.status == "success")
            failed_count = sum(1 for m in self._metrics_history if m.status == "failed")

            # 平均処理時間計算
            processing_times = [
                m.processing_time for m in self._metrics_history if m.processing_time is not None
            ]
            avg_processing_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )

            # 平均品質スコア計算
            quality_scores = [
                m.quality_score for m in self._metrics_history if m.quality_score is not None
            ]
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            # 成功率計算
            success_rate = success_count / processed_images if processed_images > 0 else 0

            # FPS計算（最近10件の平均）
            recent_metrics = list(self._metrics_history)[-10:]
            if recent_metrics and recent_metrics[0].processing_time:
                total_time = sum(m.processing_time for m in recent_metrics if m.processing_time)
                current_fps = len(recent_metrics) / total_time if total_time > 0 else 0
            else:
                current_fps = 0

            # メモリ統計（最新のもの）
            memory_stats = {}
            for m in reversed(self._metrics_history):
                if m.memory_usage:
                    memory_stats = m.memory_usage
                    break

            return AggregatedMetrics(
                total_images=total_images,
                processed_images=processed_images,
                success_count=success_count,
                failed_count=failed_count,
                average_processing_time=avg_processing_time,
                average_quality_score=avg_quality_score,
                success_rate=success_rate,
                current_fps=current_fps,
                memory_stats=memory_stats,
            )

    def get_recent_history(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        最近の処理履歴を取得

        Args:
            count: 取得する履歴数

        Returns:
            最近の処理履歴
        """
        with self._lock:
            recent = list(self._metrics_history)[-count:]
            return [m.to_dict() for m in recent]

    def export_metrics(self, output_path: Path) -> None:
        """
        メトリクスをJSONファイルにエクスポート

        Args:
            output_path: 出力ファイルパス
        """
        with self._lock:
            data = {
                "start_time": self._start_time,
                "export_time": time.time(),
                "aggregated_metrics": self.get_aggregated_metrics().to_dict(),
                "history": [m.to_dict() for m in self._metrics_history],
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
