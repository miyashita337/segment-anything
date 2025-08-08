#!/usr/bin/env python3
"""
PH2-006: リアルタイム監視システム
性能メトリクス収集・分析・ダッシュボード統合システム
"""

import torch

import asyncio
import json
import logging
import psutil
# プロジェクトルート追加
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from features.common.resource_manager import ResourceManager


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    gpu_available: bool = False
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    load_average: List[float] = field(default_factory=list)


@dataclass
class ProcessingMetrics:
    """処理性能メトリクス"""
    timestamp: float
    task_id: str
    engine_type: str
    duration: float
    success: bool
    throughput: float
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class AlertRule:
    """監視アラートルール"""
    metric_name: str
    threshold: float
    condition: str  # '>', '<', '=='
    severity: str  # 'low', 'medium', 'high', 'critical'
    message_template: str
    enabled: bool = True


@dataclass
class Alert:
    """アラート"""
    timestamp: float
    rule: AlertRule
    current_value: float
    message: str
    acknowledged: bool = False


class MetricsCollector:
    """メトリクス収集器"""
    
    def __init__(self, collection_interval: float = 1.0):
        """
        初期化
        
        Args:
            collection_interval: メトリクス収集間隔（秒）
        """
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        self.resource_manager = ResourceManager()
        
        # メトリクス履歴（最大1000件保持）
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.processing_metrics_history: deque = deque(maxlen=1000)
        
        # 収集スレッド
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_collection = threading.Event()
        
        # ネットワーク統計ベースライン
        self.network_stats_baseline = None
        
        self.logger.info(f"メトリクス収集器初期化: 収集間隔 {collection_interval}秒")
    
    def start_collection(self):
        """メトリクス収集開始"""
        if self.collection_thread and self.collection_thread.is_alive():
            self.logger.warning("メトリクス収集は既に開始されています")
            return
        
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info("メトリクス収集開始")
    
    def stop_collection_process(self):
        """メトリクス収集停止"""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_collection.set()
            self.collection_thread.join(timeout=5.0)
        
        self.logger.info("メトリクス収集停止")
    
    def _collection_loop(self):
        """メトリクス収集ループ"""
        # ネットワーク統計ベースライン設定
        net_io = psutil.net_io_counters()
        self.network_stats_baseline = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'timestamp': time.time()
        }
        
        while not self.stop_collection.wait(self.collection_interval):
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
            except Exception as e:
                self.logger.error(f"メトリクス収集エラー: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        # CPU・メモリ情報
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # プロセス数
        process_count = len(psutil.pids())
        
        # 負荷平均（Linuxのみ）
        load_avg = []
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            # Windows等では利用不可
            pass
        
        # ネットワーク統計
        net_io = psutil.net_io_counters()
        network_sent = 0
        network_recv = 0
        
        if self.network_stats_baseline:
            network_sent = net_io.bytes_sent - self.network_stats_baseline['bytes_sent']
            network_recv = net_io.bytes_recv - self.network_stats_baseline['bytes_recv']
        
        # GPU情報
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        
        if gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            
            # GPU使用率（nvidia-ml-pyが必要、なければ0）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = gpu_util.gpu
            except ImportError:
                gpu_utilization = 0.0
            except Exception:
                gpu_utilization = 0.0
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            gpu_available=gpu_available,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
            process_count=process_count,
            load_average=load_avg
        )
    
    def add_processing_metrics(self, metrics: ProcessingMetrics):
        """処理メトリクス追加"""
        self.processing_metrics_history.append(metrics)
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """最新システムメトリクス取得"""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_metrics_in_range(self, start_time: float, end_time: float) -> Tuple[List[SystemMetrics], List[ProcessingMetrics]]:
        """指定時間範囲のメトリクス取得"""
        system_metrics = [
            m for m in self.system_metrics_history 
            if start_time <= m.timestamp <= end_time
        ]
        
        processing_metrics = [
            m for m in self.processing_metrics_history 
            if start_time <= m.timestamp <= end_time
        ]
        
        return system_metrics, processing_metrics
    
    def get_average_metrics(self, duration_minutes: int = 5) -> Dict[str, float]:
        """指定期間の平均メトリクス計算"""
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)
        
        system_metrics, _ = self.get_metrics_in_range(start_time, end_time)
        
        if not system_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in system_metrics) / len(system_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in system_metrics) / len(system_metrics),
            'avg_gpu_memory_used_mb': sum(m.gpu_memory_used_mb for m in system_metrics) / len(system_metrics),
            'avg_gpu_utilization': sum(m.gpu_utilization for m in system_metrics) / len(system_metrics),
            'total_network_sent_mb': sum(m.network_bytes_sent for m in system_metrics) / (1024**2),
            'total_network_recv_mb': sum(m.network_bytes_recv for m in system_metrics) / (1024**2)
        }


class AlertManager:
    """アラート管理システム"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=500)
        
        # デフォルトアラートルール設定
        self._setup_default_rules()
        
        self.logger.info("アラート管理システム初期化完了")
    
    def _setup_default_rules(self):
        """デフォルトアラートルール設定"""
        default_rules = [
            AlertRule(
                metric_name="cpu_percent",
                threshold=80.0,
                condition=">",
                severity="high",
                message_template="CPU使用率が高い: {current_value:.1f}%"
            ),
            AlertRule(
                metric_name="memory_percent",
                threshold=85.0,
                condition=">",
                severity="high",
                message_template="メモリ使用率が高い: {current_value:.1f}%"
            ),
            AlertRule(
                metric_name="gpu_memory_used_mb",
                threshold=14000.0,  # 14GB (16GB GPU想定)
                condition=">",
                severity="medium",
                message_template="GPU メモリ使用量が多い: {current_value:.0f}MB"
            ),
            AlertRule(
                metric_name="disk_percent",
                threshold=90.0,
                condition=">",
                severity="medium",
                message_template="ディスク使用率が高い: {current_value:.1f}%"
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(self, rule: AlertRule):
        """アラートルール追加"""
        self.alert_rules.append(rule)
        self.logger.info(f"アラートルール追加: {rule.metric_name} {rule.condition} {rule.threshold}")
    
    def check_alerts(self, metrics: SystemMetrics):
        """アラートチェック"""
        new_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # メトリクス値取得
            current_value = getattr(metrics, rule.metric_name, None)
            if current_value is None:
                continue
            
            # 条件チェック
            trigger_alert = False
            if rule.condition == ">" and current_value > rule.threshold:
                trigger_alert = True
            elif rule.condition == "<" and current_value < rule.threshold:
                trigger_alert = True
            elif rule.condition == "==" and current_value == rule.threshold:
                trigger_alert = True
            
            if trigger_alert:
                # 既存アラートチェック（重複防止）
                existing_alert = next(
                    (a for a in self.active_alerts 
                     if a.rule.metric_name == rule.metric_name and not a.acknowledged), 
                    None
                )
                
                if not existing_alert:
                    alert = Alert(
                        timestamp=metrics.timestamp,
                        rule=rule,
                        current_value=current_value,
                        message=rule.message_template.format(current_value=current_value)
                    )
                    
                    new_alerts.append(alert)
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)
        
        if new_alerts:
            for alert in new_alerts:
                self.logger.warning(f"🚨 アラート発生: {alert.message}")
        
        return new_alerts
    
    def acknowledge_alert(self, alert_index: int):
        """アラート確認応答"""
        if 0 <= alert_index < len(self.active_alerts):
            self.active_alerts[alert_index].acknowledged = True
            self.logger.info(f"アラート確認: {self.active_alerts[alert_index].message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        return [a for a in self.active_alerts if not a.acknowledged]
    
    def clear_acknowledged_alerts(self):
        """確認済みアラートクリア"""
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]


class PH2006MonitoringSystem:
    """PH2-006: 統合監視システム"""
    
    def __init__(self, collection_interval: float = 2.0):
        """
        初期化
        
        Args:
            collection_interval: メトリクス収集間隔（秒）
        """
        self.logger = logging.getLogger(__name__)
        
        # コンポーネント初期化
        self.metrics_collector = MetricsCollector(collection_interval)
        self.alert_manager = AlertManager()
        
        # 監視状態
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring_event = threading.Event()
        
        # 出力ディレクトリ
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/PH2-006")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 統計情報
        self.monitoring_stats = {
            'start_time': None,
            'total_alerts': 0,
            'metrics_collected': 0,
            'uptime_seconds': 0
        }
        
        self.logger.info("PH2-006 統合監視システム初期化完了")
    
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に開始されています")
            return
        
        self.monitoring_active = True
        self.monitoring_stats['start_time'] = time.time()
        
        # メトリクス収集開始
        self.metrics_collector.start_collection()
        
        # 監視スレッド開始
        self.stop_monitoring_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("🚀 統合監視システム開始")
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_monitoring_event.set()
        
        # メトリクス収集停止
        self.metrics_collector.stop_collection_process()
        
        # 監視スレッド停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # 統計更新
        if self.monitoring_stats['start_time']:
            self.monitoring_stats['uptime_seconds'] = time.time() - self.monitoring_stats['start_time']
        
        self.logger.info("⏹️  統合監視システム停止")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while not self.stop_monitoring_event.wait(5.0):  # 5秒間隔でアラートチェック
            try:
                # 最新メトリクス取得
                latest_metrics = self.metrics_collector.get_latest_system_metrics()
                if latest_metrics:
                    # アラートチェック
                    new_alerts = self.alert_manager.check_alerts(latest_metrics)
                    self.monitoring_stats['total_alerts'] += len(new_alerts)
                
                # 統計更新
                self.monitoring_stats['metrics_collected'] = len(self.metrics_collector.system_metrics_history)
                
                # 確認済みアラートクリア
                self.alert_manager.clear_acknowledged_alerts()
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態取得"""
        latest_metrics = self.metrics_collector.get_latest_system_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        status = {
            'monitoring_active': self.monitoring_active,
            'uptime_seconds': time.time() - self.monitoring_stats['start_time'] if self.monitoring_stats['start_time'] else 0,
            'metrics_collected': len(self.metrics_collector.system_metrics_history),
            'processing_metrics_collected': len(self.metrics_collector.processing_metrics_history),
            'total_alerts': self.monitoring_stats['total_alerts'],
            'active_alerts_count': len(active_alerts),
            'latest_system_metrics': asdict(latest_metrics) if latest_metrics else None,
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'average_metrics_5min': self.metrics_collector.get_average_metrics(5)
        }
        
        return status
    
    def generate_report(self, duration_hours: int = 1) -> Dict[str, Any]:
        """監視レポート生成"""
        end_time = time.time()
        start_time = end_time - (duration_hours * 3600)
        
        system_metrics, processing_metrics = self.metrics_collector.get_metrics_in_range(start_time, end_time)
        
        # システムメトリクス統計
        system_stats = {}
        if system_metrics:
            system_stats = {
                'cpu_percent': {
                    'avg': sum(m.cpu_percent for m in system_metrics) / len(system_metrics),
                    'max': max(m.cpu_percent for m in system_metrics),
                    'min': min(m.cpu_percent for m in system_metrics)
                },
                'memory_percent': {
                    'avg': sum(m.memory_percent for m in system_metrics) / len(system_metrics),
                    'max': max(m.memory_percent for m in system_metrics),
                    'min': min(m.memory_percent for m in system_metrics)
                },
                'gpu_utilization': {
                    'avg': sum(m.gpu_utilization for m in system_metrics) / len(system_metrics),
                    'max': max(m.gpu_utilization for m in system_metrics),
                    'min': min(m.gpu_utilization for m in system_metrics)
                }
            }
        
        # 処理メトリクス統計
        processing_stats = {}
        if processing_metrics:
            successful_tasks = [m for m in processing_metrics if m.success]
            failed_tasks = [m for m in processing_metrics if not m.success]
            
            processing_stats = {
                'total_tasks': len(processing_metrics),
                'successful_tasks': len(successful_tasks),
                'failed_tasks': len(failed_tasks),
                'success_rate': len(successful_tasks) / len(processing_metrics) if processing_metrics else 0,
                'avg_duration': sum(m.duration for m in processing_metrics) / len(processing_metrics),
                'avg_throughput': sum(m.throughput for m in processing_metrics) / len(processing_metrics)
            }
        
        # アラート統計
        alert_stats = {
            'total_alerts_in_period': len([a for a in self.alert_manager.alert_history 
                                         if start_time <= a.timestamp <= end_time]),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'alert_types': defaultdict(int)
        }
        
        for alert in self.alert_manager.alert_history:
            if start_time <= alert.timestamp <= end_time:
                alert_stats['alert_types'][alert.rule.severity] += 1
        
        report = {
            'report_period': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_hours': duration_hours
            },
            'system_statistics': system_stats,
            'processing_statistics': processing_stats,
            'alert_statistics': dict(alert_stats),
            'monitoring_health': {
                'monitoring_active': self.monitoring_active,
                'metrics_collection_rate': len(system_metrics) / duration_hours if duration_hours > 0 else 0,
                'system_performance': 'healthy' if system_stats.get('cpu_percent', {}).get('avg', 0) < 70 else 'stressed'
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """レポート保存"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_report_{timestamp}.json"
        
        report_path = self.output_dir / "reports" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"監視レポート保存: {report_path}")
        return str(report_path)
    
    def export_metrics(self, format: str = 'json') -> str:
        """メトリクスエクスポート"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            export_data = {
                'system_metrics': [asdict(m) for m in self.metrics_collector.system_metrics_history],
                'processing_metrics': [asdict(m) for m in self.metrics_collector.processing_metrics_history],
                'alert_history': [asdict(a) for a in self.alert_manager.alert_history],
                'export_timestamp': timestamp
            }
            
            export_path = self.output_dir / "exports" / f"metrics_export_{timestamp}.json"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"未対応フォーマット: {format}")
        
        self.logger.info(f"メトリクスエクスポート完了: {export_path}")
        return str(export_path)


# グローバル監視システム
global_monitoring_system = PH2006MonitoringSystem()


def start_global_monitoring(collection_interval: float = 2.0):
    """グローバル監視開始"""
    global global_monitoring_system
    global_monitoring_system = PH2006MonitoringSystem(collection_interval)
    global_monitoring_system.start_monitoring()
    return global_monitoring_system


def stop_global_monitoring():
    """グローバル監視停止"""
    global_monitoring_system.stop_monitoring()


def get_monitoring_status() -> Dict[str, Any]:
    """監視状態取得"""
    return global_monitoring_system.get_monitoring_status()


# 便利関数
def add_processing_metrics(task_id: str, engine_type: str, duration: float, 
                         success: bool, throughput: float, error_message: Optional[str] = None):
    """処理メトリクス追加"""
    metrics = ProcessingMetrics(
        timestamp=time.time(),
        task_id=task_id,
        engine_type=engine_type,
        duration=duration,
        success=success,
        throughput=throughput,
        error_message=error_message
    )
    
    global_monitoring_system.metrics_collector.add_processing_metrics(metrics)