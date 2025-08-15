#!/usr/bin/env python3
"""
PH2-004-RESOURCE: リソース管理最適化システム

【概要】
GPU・CPU・メモリ・ディスク・ネットワークリソースを統合監視し、
パフォーマンス低下を早期検出・自動調整するシステム

【主要機能】
- リアルタイムリソース監視
- 自動メモリ最適化
- GPU温度・使用率監視
- ディスク容量監視・警告
- パフォーマンスアラート機能

【使用方法】
from features.common.resource_monitor import ResourceMonitor
monitor = ResourceMonitor()
monitor.start_monitoring()
"""

import os
import sys
import time
import psutil
import threading
import logging
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import subprocess

try:
    import nvidia_ml_py3 as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceStatus:
    """リソース状態データクラス"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_count: int
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_temperature: List[float]
    gpu_utilization: List[float]
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    python_processes: int
    load_average: List[float]

class ResourceMonitor:
    """リソース監視・最適化システム"""
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 history_size: int = 100,
                 alert_thresholds: Optional[Dict] = None):
        """
        Args:
            monitoring_interval: 監視間隔（秒）
            history_size: 履歴保持数
            alert_thresholds: アラート閾値設定
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.is_monitoring = False
        self.monitor_thread = None
        
        # デフォルトアラート閾値
        default_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 85.0,
            'memory_available_gb': 2.0
        }
        
        if alert_thresholds:
            # カスタム閾値をデフォルトとマージ
            self.alert_thresholds = default_thresholds.copy()
            self.alert_thresholds.update(alert_thresholds)
        else:
            self.alert_thresholds = default_thresholds
        
        # 履歴データ
        self.resource_history: List[ResourceStatus] = []
        
        # アラートコールバック
        self.alert_callbacks: List[Callable] = []
        
        # GPU初期化
        self.gpu_available = self._initialize_gpu()
        
        # 統計データ
        self.optimization_count = 0
        self.alert_count = 0
        self.last_optimization = None
        
        logger.info("🔧 PH2-004-RESOURCE: リソース監視システム初期化完了")

    def _initialize_gpu(self) -> bool:
        """GPU監視初期化"""
        if not NVIDIA_ML_AVAILABLE:
            logger.warning("⚠️ nvidia-ml-py3が利用できません。GPU監視は無効化されます")
            return False
        
        try:
            nvml.nvmlInit()
            gpu_count = nvml.nvmlDeviceGetCount()
            logger.info(f"🎮 GPU監視初期化完了: {gpu_count}台のGPU検出")
            return True
        except Exception as e:
            logger.error(f"❌ GPU監視初期化失敗: {e}")
            return False

    def get_current_status(self) -> ResourceStatus:
        """現在のリソース状態を取得"""
        # CPU監視
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # メモリ監視
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # ディスク監視
        disk = psutil.disk_usage(str(PROJECT_ROOT))
        disk_percent = disk.used / disk.total * 100
        disk_free_gb = disk.free / (1024**3)
        
        # ネットワーク監視
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # プロセス監視
        active_processes = len(psutil.pids())
        python_processes = len([p for p in psutil.process_iter(['name']) 
                              if 'python' in p.info['name'].lower()])
        
        # システム負荷
        load_average = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
        
        # GPU監視
        gpu_count = 0
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_temperature = []
        gpu_utilization = []
        
        if self.gpu_available:
            try:
                gpu_count = nvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU メモリ情報
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used.append(memory_info.used / (1024**3))  # GB
                    gpu_memory_total.append(memory_info.total / (1024**3))  # GB
                    
                    # GPU 温度
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        gpu_temperature.append(temp)
                    except:
                        gpu_temperature.append(0)
                    
                    # GPU 使用率
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization.append(util.gpu)
                    except:
                        gpu_utilization.append(0)
                        
            except Exception as e:
                logger.error(f"GPU監視エラー: {e}")
        
        return ResourceStatus(
            timestamp=datetime.datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            gpu_count=gpu_count,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperature,
            gpu_utilization=gpu_utilization,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_processes=active_processes,
            python_processes=python_processes,
            load_average=load_average
        )

    def check_alerts(self, status: ResourceStatus) -> List[Dict]:
        """アラート条件チェック"""
        alerts = []
        
        # CPU使用率アラート
        if status.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"CPU使用率が高すぎます: {status.cpu_percent:.1f}%",
                'value': status.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
        
        # メモリ使用率アラート
        if status.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"メモリ使用率が高すぎます: {status.memory_percent:.1f}%",
                'value': status.memory_percent,
                'threshold': self.alert_thresholds['memory_percent']
            })
        
        # 利用可能メモリアラート
        if status.memory_available_gb < self.alert_thresholds['memory_available_gb']:
            alerts.append({
                'type': 'memory_low',
                'severity': 'critical',
                'message': f"利用可能メモリが少なすぎます: {status.memory_available_gb:.1f}GB",
                'value': status.memory_available_gb,
                'threshold': self.alert_thresholds['memory_available_gb']
            })
        
        # ディスク容量アラート
        if status.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_full',
                'severity': 'warning',
                'message': f"ディスク使用率が高すぎます: {status.disk_percent:.1f}%",
                'value': status.disk_percent,
                'threshold': self.alert_thresholds['disk_percent']
            })
        
        # GPU関連アラート
        if status.gpu_count > 0:
            for i, (memory_used, memory_total, temp) in enumerate(
                zip(status.gpu_memory_used, status.gpu_memory_total, status.gpu_temperature)
            ):
                # GPU メモリ使用率
                if memory_total > 0:
                    gpu_memory_percent = (memory_used / memory_total) * 100
                    if gpu_memory_percent > self.alert_thresholds['gpu_memory_percent']:
                        alerts.append({
                            'type': 'gpu_memory_high',
                            'severity': 'warning',
                            'message': f"GPU{i} メモリ使用率が高すぎます: {gpu_memory_percent:.1f}%",
                            'value': gpu_memory_percent,
                            'threshold': self.alert_thresholds['gpu_memory_percent'],
                            'gpu_id': i
                        })
                
                # GPU 温度
                if temp > self.alert_thresholds['gpu_temperature']:
                    alerts.append({
                        'type': 'gpu_temperature_high',
                        'severity': 'critical',
                        'message': f"GPU{i} 温度が高すぎます: {temp}°C",
                        'value': temp,
                        'threshold': self.alert_thresholds['gpu_temperature'],
                        'gpu_id': i
                    })
        
        if alerts:
            self.alert_count += len(alerts)
        
        return alerts

    def auto_optimize(self, status: ResourceStatus, alerts: List[Dict]):
        """自動最適化実行"""
        optimization_actions = []
        
        # メモリ最適化
        if any(alert['type'] in ['memory_high', 'memory_low'] for alert in alerts):
            actions = self._optimize_memory()
            optimization_actions.extend(actions)
        
        # GPU メモリ最適化
        gpu_memory_alerts = [a for a in alerts if a['type'] == 'gpu_memory_high']
        if gpu_memory_alerts:
            actions = self._optimize_gpu_memory()
            optimization_actions.extend(actions)
        
        # プロセス最適化
        if status.cpu_percent > 95.0 or status.python_processes > 10:
            actions = self._optimize_processes()
            optimization_actions.extend(actions)
        
        if optimization_actions:
            self.optimization_count += 1
            self.last_optimization = datetime.datetime.now().isoformat()
            logger.info(f"🔧 自動最適化実行: {len(optimization_actions)}件の最適化")
            
            for action in optimization_actions:
                logger.info(f"  ✅ {action}")
        
        return optimization_actions

    def _optimize_memory(self) -> List[str]:
        """メモリ最適化"""
        actions = []
        
        # Python ガベージコレクション
        try:
            import gc
            collected = gc.collect()
            if collected > 0:
                actions.append(f"Python GC実行: {collected}オブジェクト解放")
        except Exception as e:
            logger.error(f"GC最適化エラー: {e}")
        
        # メモリ消費の大きいプロセス特定・警告
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                if proc.info['memory_percent'] > 5.0:  # 5%以上のメモリ使用
                    processes.append(proc.info)
            
            if processes:
                processes.sort(key=lambda x: x['memory_percent'], reverse=True)
                top_process = processes[0]
                actions.append(f"メモリ消費プロセス特定: {top_process['name']} "
                             f"({top_process['memory_percent']:.1f}%)")
        except Exception as e:
            logger.error(f"プロセス分析エラー: {e}")
        
        return actions

    def _optimize_gpu_memory(self) -> List[str]:
        """GPU メモリ最適化"""
        actions = []
        
        try:
            # PyTorch キャッシュクリア（利用可能な場合）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    actions.append("PyTorch GPU キャッシュクリア実行")
            except ImportError:
                pass
            
            # TensorFlow GPU メモリ最適化（利用可能な場合）
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    # GPU メモリ成長を有効化
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    actions.append("TensorFlow GPU メモリ最適化実行")
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"GPU最適化エラー: {e}")
        
        return actions

    def _optimize_processes(self) -> List[str]:
        """プロセス最適化"""
        actions = []
        
        try:
            # 不要な Python プロセス検出
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
                if 'python' in proc.info['name'].lower():
                    python_processes.append(proc.info)
            
            # CPU使用率の高いプロセス特定
            high_cpu_processes = [p for p in python_processes if p['cpu_percent'] > 50.0]
            
            if high_cpu_processes:
                actions.append(f"高CPU使用プロセス特定: {len(high_cpu_processes)}件")
                
            # アイドルプロセス検出（実際の終了は行わない）
            idle_processes = [p for p in python_processes if p.get('cpu_percent', 0) < 0.1]
            if len(idle_processes) > 3:  # 閾値を下げて検出しやすく
                actions.append(f"アイドルプロセス検出: {len(idle_processes)}件（要手動確認）")
                
        except Exception as e:
            logger.error(f"プロセス最適化エラー: {e}")
        
        return actions

    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, alerts: List[Dict]):
        """アラート発火"""
        for alert in alerts:
            logger.warning(f"⚠️ {alert['message']}")
            
            # 登録されたコールバック実行
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"アラートコールバックエラー: {e}")

    def _monitoring_loop(self):
        """監視ループ（別スレッド実行）"""
        logger.info("📊 リソース監視開始")
        
        while self.is_monitoring:
            try:
                # リソース状態取得
                status = self.get_current_status()
                
                # 履歴に追加
                self.resource_history.append(status)
                if len(self.resource_history) > self.history_size:
                    self.resource_history.pop(0)
                
                # アラートチェック
                alerts = self.check_alerts(status)
                
                # アラート処理
                if alerts:
                    self._trigger_alerts(alerts)
                    
                    # 自動最適化実行
                    self.auto_optimize(status, alerts)
                
                # 定期ログ出力（詳細モード）
                if len(self.resource_history) % 12 == 0:  # 1分ごと（5秒間隔×12）
                    self._log_status_summary(status)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(self.monitoring_interval)

    def _log_status_summary(self, status: ResourceStatus):
        """状態サマリーログ出力"""
        gpu_info = ""
        if status.gpu_count > 0:
            gpu_memory_percents = []
            for used, total in zip(status.gpu_memory_used, status.gpu_memory_total):
                if total > 0:
                    gpu_memory_percents.append((used / total) * 100)
            
            if gpu_memory_percents:
                avg_gpu_memory = sum(gpu_memory_percents) / len(gpu_memory_percents)
                avg_gpu_temp = sum(status.gpu_temperature) / len(status.gpu_temperature) if status.gpu_temperature else 0
                gpu_info = f", GPU: {avg_gpu_memory:.1f}% ({avg_gpu_temp:.0f}°C)"
        
        logger.info(f"📊 リソース状況 - CPU: {status.cpu_percent:.1f}%, "
                   f"RAM: {status.memory_percent:.1f}% ({status.memory_available_gb:.1f}GB空き), "
                   f"Disk: {status.disk_percent:.1f}%{gpu_info}")

    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("既に監視中です")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 リソース監視開始")

    def stop_monitoring(self):
        """監視停止"""
        if not self.is_monitoring:
            logger.warning("監視は開始されていません")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("⏹️ リソース監視停止")

    def get_statistics(self) -> Dict:
        """統計情報取得"""
        if not self.resource_history:
            return {}
        
        # 最近の履歴から統計計算
        recent_data = self.resource_history[-20:]  # 最新20件
        
        cpu_values = [s.cpu_percent for s in recent_data]
        memory_values = [s.memory_percent for s in recent_data]
        
        stats = {
            'monitoring_duration_minutes': len(self.resource_history) * self.monitoring_interval / 60,
            'data_points': len(self.resource_history),
            'optimization_count': self.optimization_count,
            'alert_count': self.alert_count,
            'last_optimization': self.last_optimization,
            'current_status': asdict(self.resource_history[-1]) if self.resource_history else None,
            'averages': {
                'cpu_percent': sum(cpu_values) / len(cpu_values),
                'memory_percent': sum(memory_values) / len(memory_values),
            },
            'peaks': {
                'cpu_percent': max(cpu_values),
                'memory_percent': max(memory_values),
            }
        }
        
        return stats

    def export_report(self, output_path: Optional[Path] = None) -> Path:
        """レポート出力"""
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = PROJECT_ROOT / "logs" / "resource_monitoring" / f"resource_report_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'generated_at': datetime.datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'alert_thresholds': self.alert_thresholds,
            'recent_history': [asdict(s) for s in self.resource_history[-10:]],  # 最新10件
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage(str(PROJECT_ROOT)).total / (1024**3),
                'gpu_available': self.gpu_available,
                'gpu_count': self.resource_history[-1].gpu_count if self.resource_history else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 リソース監視レポート出力: {output_path}")
        return output_path


def main():
    """メイン実行関数（テスト用）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PH2-004-RESOURCE: リソース管理最適化システム")
    parser.add_argument('--interval', type=float, default=5.0, help='監視間隔（秒）')
    parser.add_argument('--duration', type=int, default=60, help='監視時間（秒）')
    parser.add_argument('--export', action='store_true', help='レポート出力')
    
    args = parser.parse_args()
    
    # リソース監視開始
    monitor = ResourceMonitor(monitoring_interval=args.interval)
    
    # アラートコールバック設定
    def alert_handler(alert):
        print(f"🚨 アラート: {alert['message']}")
    
    monitor.add_alert_callback(alert_handler)
    
    try:
        monitor.start_monitoring()
        print(f"📊 {args.duration}秒間のリソース監視を開始します...")
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\n⏹️ 監視を停止しています...")
    finally:
        monitor.stop_monitoring()
        
        # 統計表示
        stats = monitor.get_statistics()
        print("\n📊 監視結果:")
        print(f"  監視時間: {stats.get('monitoring_duration_minutes', 0):.1f}分")
        print(f"  データポイント: {stats.get('data_points', 0)}")
        print(f"  最適化実行回数: {stats.get('optimization_count', 0)}")
        print(f"  アラート発生回数: {stats.get('alert_count', 0)}")
        
        if args.export:
            report_path = monitor.export_report()
            print(f"📄 レポート出力: {report_path}")


if __name__ == "__main__":
    main()