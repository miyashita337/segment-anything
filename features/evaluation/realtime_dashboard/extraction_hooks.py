"""
抽出処理用フック

extract_character.pyと統合するためのフック関数
"""

import logging
from typing import Optional

from .dashboard_server import DashboardServer
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

# グローバルインスタンス
_metrics_collector: Optional[MetricsCollector] = None
_dashboard_server: Optional[DashboardServer] = None


def initialize_realtime_dashboard(enable_dashboard: bool = True, port: int = 8080) -> Optional[MetricsCollector]:
    """
    リアルタイムダッシュボードを初期化
    
    Args:
        enable_dashboard: ダッシュボードを有効にするか
        port: ダッシュボードのポート番号
        
    Returns:
        MetricsCollectorインスタンス（ダッシュボード無効時もメトリクス収集は可能）
    """
    global _metrics_collector, _dashboard_server
    
    # メトリクス収集器を作成
    _metrics_collector = MetricsCollector()
    
    # ダッシュボードサーバーを開始
    if enable_dashboard:
        try:
            _dashboard_server = DashboardServer(_metrics_collector, port=port)
            _dashboard_server.run_in_thread()
            logger.info(f"Realtime dashboard started at http://localhost:{port}")
        except Exception as e:
            logger.warning(f"Failed to start dashboard server: {e}")
            logger.info("Continuing without realtime dashboard...")
    
    return _metrics_collector


def on_image_start(image_name: str) -> None:
    """
    画像処理開始時のフック
    
    Args:
        image_name: 処理開始する画像名
    """
    if _metrics_collector:
        _metrics_collector.start_processing(image_name)


def on_image_complete(
    image_name: str,
    success: bool,
    quality_score: Optional[float] = None,
    memory_stats: Optional[dict] = None,
    error_message: Optional[str] = None
) -> None:
    """
    画像処理完了時のフック
    
    Args:
        image_name: 処理完了した画像名
        success: 成功フラグ
        quality_score: 品質スコア
        memory_stats: メモリ使用統計
        error_message: エラーメッセージ（失敗時）
    """
    if _metrics_collector:
        _metrics_collector.complete_processing(
            image_name=image_name,
            success=success,
            quality_score=quality_score,
            memory_usage=memory_stats,
            error_message=error_message
        )


def get_metrics_collector() -> Optional[MetricsCollector]:
    """
    メトリクス収集器インスタンスを取得
    
    Returns:
        MetricsCollectorインスタンス
    """
    return _metrics_collector


def shutdown_dashboard() -> None:
    """ダッシュボードをシャットダウン"""
    global _dashboard_server, _metrics_collector
    
    if _dashboard_server:
        # Note: サーバーはデーモンスレッドで実行されているため、
        # プログラム終了時に自動的に終了します
        _dashboard_server = None
    
    _metrics_collector = None