"""
P1-023: VSCode安定性向上 - メモリ監視システム

GPT-4分析に基づくメモリ枯渇対策:
- プロアクティブなメモリ監視
- 閾値ベースのガベージコレクション
- ML模델メモリの最適化
- WSL2環境での安定動作保証
"""

import gc
import logging
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    システムメモリ監視・最適化クラス
    
    WSL2-VSCode環境でのML処理における安定性向上を目的とし、
    メモリ使用量の監視と自動最適化を実行する。
    """
    
    def __init__(
        self,
        memory_threshold: float = 85.0,
        vram_threshold: float = 90.0,
        monitor_interval: float = 30.0,
        enable_auto_cleanup: bool = True
    ):
        """
        Args:
            memory_threshold: RAM使用率警告閾値 (%)
            vram_threshold: VRAM使用率警告閾値 (%)
            monitor_interval: 監視間隔 (秒)
            enable_auto_cleanup: 自動クリーンアップ有効化
        """
        self.memory_threshold = memory_threshold
        self.vram_threshold = vram_threshold
        self.monitor_interval = monitor_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        self.cleanup_callbacks = []
        self.monitoring_active = False
        self.last_cleanup_time = None
        
        logger.info(f"MemoryMonitor初期化: RAM閾値={memory_threshold}%, VRAM閾値={vram_threshold}%")
    
    def get_memory_status(self) -> Dict[str, float]:
        """現在のメモリ使用状況を取得"""
        memory = psutil.virtual_memory()
        
        status = {
            'ram_used_gb': memory.used / (1024**3),
            'ram_total_gb': memory.total / (1024**3),
            'ram_percent': memory.percent,
            'ram_available_gb': memory.available / (1024**3)
        }
        
        # CUDA利用可能時はVRAM情報も取得
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_percent = (vram_used / vram_total) * 100
                
                status.update({
                    'vram_used_gb': vram_used,
                    'vram_total_gb': vram_total,
                    'vram_percent': vram_percent
                })
        except ImportError:
            # PyTorchが利用できない場合はRAM情報のみ
            pass
        
        return status
    
    def check_memory_pressure(self) -> Dict[str, bool]:
        """メモリ圧迫状態をチェック"""
        status = self.get_memory_status()
        
        pressure = {
            'ram_pressure': status['ram_percent'] > self.memory_threshold,
            'vram_pressure': False
        }
        
        if 'vram_percent' in status:
            pressure['vram_pressure'] = status['vram_percent'] > self.vram_threshold
        
        return pressure
    
    def perform_memory_cleanup(self, force: bool = False) -> Dict[str, float]:
        """メモリクリーンアップの実行"""
        if not force and not self.enable_auto_cleanup:
            logger.debug("自動クリーンアップが無効化されています")
            return {}
        
        cleanup_start = time.time()
        before_status = self.get_memory_status()
        
        logger.info("メモリクリーンアップ開始")
        
        # Pythonガベージコレクション
        collected = gc.collect()
        logger.debug(f"ガベージコレクション: {collected}オブジェクト回収")
        
        # CUDA キャッシュクリア
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("CUDAキャッシュクリア完了")
        except ImportError:
            pass
        
        # 登録されたクリーンアップコールバック実行
        for callback in self.cleanup_callbacks:
            try:
                callback()
                logger.debug(f"クリーンアップコールバック実行: {callback.__name__}")
            except Exception as e:
                logger.warning(f"クリーンアップコールバック失敗: {e}")
        
        after_status = self.get_memory_status()
        cleanup_time = time.time() - cleanup_start
        
        # 効果測定
        ram_freed = before_status['ram_used_gb'] - after_status['ram_used_gb']
        vram_freed = 0
        if 'vram_used_gb' in before_status and 'vram_used_gb' in after_status:
            vram_freed = before_status['vram_used_gb'] - after_status['vram_used_gb']
        
        logger.info(
            f"メモリクリーンアップ完了: RAM解放={ram_freed:.2f}GB, "
            f"VRAM解放={vram_freed:.2f}GB, 処理時間={cleanup_time:.3f}s"
        )
        
        self.last_cleanup_time = datetime.now()
        
        return {
            'ram_freed_gb': ram_freed,
            'vram_freed_gb': vram_freed,
            'cleanup_time_s': cleanup_time,
            'before_ram_percent': before_status['ram_percent'],
            'after_ram_percent': after_status['ram_percent']
        }
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """クリーンアップ時に実行するコールバックを登録"""
        if callback not in self.cleanup_callbacks:
            self.cleanup_callbacks.append(callback)
            logger.debug(f"クリーンアップコールバック登録: {callback.__name__}")
    
    def monitor_once(self) -> Optional[Dict[str, float]]:
        """一回のメモリ監視チェックを実行"""
        status = self.get_memory_status()
        pressure = self.check_memory_pressure()
        
        log_msg = f"メモリ状況: RAM={status['ram_percent']:.1f}%"
        if 'vram_percent' in status:
            log_msg += f", VRAM={status['vram_percent']:.1f}%"
        
        logger.debug(log_msg)
        
        # メモリ圧迫時の自動クリーンアップ
        if pressure['ram_pressure'] or pressure['vram_pressure']:
            logger.warning(f"メモリ圧迫検出: {pressure}")
            if self.enable_auto_cleanup:
                return self.perform_memory_cleanup()
        
        return None
    
    def get_monitoring_report(self) -> Dict:
        """監視レポート生成"""
        status = self.get_memory_status()
        pressure = self.check_memory_pressure()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_status': status,
            'pressure_status': pressure,
            'last_cleanup': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
            'monitoring_active': self.monitoring_active,
            'thresholds': {
                'ram_threshold': self.memory_threshold,
                'vram_threshold': self.vram_threshold
            }
        }


# グローバルインスタンス
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """グローバルメモリ監視インスタンスを取得"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def monitor_memory_once() -> Optional[Dict[str, float]]:
    """一回限りのメモリ監視（簡易インターフェース）"""
    monitor = get_memory_monitor()
    return monitor.monitor_once()


def force_memory_cleanup() -> Dict[str, float]:
    """強制メモリクリーンアップ（簡易インターフェース）"""
    monitor = get_memory_monitor()
    return monitor.perform_memory_cleanup(force=True)


def get_memory_status() -> Dict[str, float]:
    """現在のメモリ状況取得（簡易インターフェース）"""
    monitor = get_memory_monitor()
    return monitor.get_memory_status()


# デコレータ: 関数実行前後でメモリ監視
def monitor_memory(cleanup_after: bool = False):
    """
    関数デコレータ: 実行前後でメモリ監視
    
    Args:
        cleanup_after: 実行後に自動クリーンアップを実行
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_memory_monitor()
            
            # 実行前状態
            before_status = monitor.get_memory_status()
            logger.debug(f"{func.__name__} 実行前: RAM={before_status['ram_percent']:.1f}%")
            
            # 関数実行
            try:
                result = func(*args, **kwargs)
                
                # 実行後状態
                after_status = monitor.get_memory_status()
                ram_diff = after_status['ram_percent'] - before_status['ram_percent']
                
                logger.debug(
                    f"{func.__name__} 実行後: RAM={after_status['ram_percent']:.1f}% "
                    f"(差分: {ram_diff:+.1f}%)"
                )
                
                # 必要に応じてクリーンアップ
                if cleanup_after or after_status['ram_percent'] > monitor.memory_threshold:
                    monitor.perform_memory_cleanup()
                
                return result
                
            except Exception as e:
                logger.error(f"{func.__name__} 実行中にエラー: {e}")
                # エラー時もクリーンアップ
                if cleanup_after:
                    monitor.perform_memory_cleanup()
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # テスト実行
    import json
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    monitor = MemoryMonitor()
    
    print("=== P1-023 メモリ監視システム テスト ===")
    print()
    
    # 現在の状況
    status = monitor.get_memory_status()
    print("現在のメモリ状況:")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print()
    
    # 監視実行
    cleanup_result = monitor.monitor_once()
    if cleanup_result:
        print("クリーンアップ結果:")
        print(json.dumps(cleanup_result, indent=2, ensure_ascii=False))
    else:
        print("メモリ圧迫なし - クリーンアップスキップ")
    print()
    
    # 強制クリーンアップテスト
    print("強制クリーンアップテスト:")
    force_result = monitor.perform_memory_cleanup(force=True)
    print(json.dumps(force_result, indent=2, ensure_ascii=False))
    print()
    
    # 監視レポート
    report = monitor.get_monitoring_report()
    print("監視レポート:")
    print(json.dumps(report, indent=2, ensure_ascii=False))