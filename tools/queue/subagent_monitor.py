#!/usr/bin/env python3
"""
SubAgent監視システム
QUAL-044: 同一セッション内でのタスク監視と次アクション実行

Task toolを活用して、同一セッション内でqueue_status.jsonを監視し、
タスク完了時に自動的に次のアクションを実行する
"""

import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import logging
import subprocess

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubAgentMonitor:
    """SubAgent監視クラス"""
    
    def __init__(self, workspace_path: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-044"):
        """
        初期化
        
        Args:
            workspace_path: ワークスペースパス
        """
        self.workspace = Path(workspace_path)
        self.queue_dir = self.workspace / "queue"
        self.status_file = self.queue_dir / "queue_status.json"
        
        # 監視設定
        self.check_interval = 5  # 5秒間隔でチェック
        self.is_monitoring = False
        self.last_status: Optional[Dict[str, Any]] = None

        # INTG-089: 拡張監視機能
        self.gpu_monitoring_enabled = self._check_gpu_availability()
        self.memory_baseline = psutil.virtual_memory().used
        self.anomaly_thresholds = {
            'gpu_memory_usage': 90,  # GPU メモリ使用率 90%
            'system_memory_usage': 85,  # システムメモリ使用率 85%
            'cpu_usage': 95,  # CPU使用率 95%
            'temperature_threshold': 80,  # GPU温度 80°C
            'process_timeout': 3600,  # プロセスタイムアウト 1時間
        }
        self.anomaly_history: List[Dict[str, Any]] = []

        # コールバック登録
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_anomaly_detected: Optional[Callable] = None  # INTG-089追加
        
        logger.info(f"SubAgentMonitor initialized for {self.workspace}")
    
    def read_status_file(self) -> Optional[Dict[str, Any]]:
        """状態ファイル読み込み"""
        if not self.status_file.exists():
            return None
        
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read status file: {e}")
            return None
    
    def start_monitoring(self, task_id: str) -> Dict[str, Any]:
        """
        タスク監視開始（同一セッション内実行）
        
        Args:
            task_id: 監視対象タスクID
            
        Returns:
            監視結果
        """
        logger.info(f"Starting monitoring for task: {task_id}")
        self.is_monitoring = True
        
        monitoring_result = {
            'task_id': task_id,
            'monitoring_started': time.time(),
            'status_changes': [],
            'final_status': None,
            'next_action_executed': False
        }
        
        try:
            while self.is_monitoring:
                status = self.read_status_file()
                
                if not status:
                    time.sleep(self.check_interval)
                    continue
                
                # 状態変化検出
                if self.last_status != status:
                    monitoring_result['status_changes'].append({
                        'timestamp': time.time(),
                        'status': status.get('status'),
                        'task_id': status.get('task_id')
                    })
                    
                    logger.info(f"Status change detected: {status.get('status')}")
                    
                    # タスク完了検出
                    if status.get('status') == 'task_completed' and status.get('task_id') == task_id:
                        logger.info(f"Task completed: {task_id}")
                        monitoring_result['final_status'] = 'completed'
                        
                        # 次アクション実行
                        if self.on_task_complete:
                            next_action_result = self.on_task_complete(status)
                            monitoring_result['next_action_result'] = next_action_result
                            monitoring_result['next_action_executed'] = True
                        
                        self.is_monitoring = False
                        break
                    
                    # タスク失敗検出
                    elif status.get('status') == 'task_failed' and status.get('task_id') == task_id:
                        logger.error(f"Task failed: {task_id}")
                        monitoring_result['final_status'] = 'failed'
                        
                        # 手動レビューが必要
                        if status.get('requires_manual_review'):
                            monitoring_result['requires_manual_review'] = True
                            logger.info("Task requires manual review")
                        
                        if self.on_task_failed:
                            self.on_task_failed(status)
                        
                        self.is_monitoring = False
                        break
                    
                    # エラー検出
                    elif status.get('status') == 'task_error' and status.get('task_id') == task_id:
                        logger.error(f"Task error: {task_id}")
                        monitoring_result['final_status'] = 'error'
                        monitoring_result['error'] = status.get('error')
                        
                        if self.on_task_error:
                            self.on_task_error(status)
                        
                        self.is_monitoring = False
                        break
                    
                    self.last_status = status
                
                time.sleep(self.check_interval)
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            monitoring_result['final_status'] = 'interrupted'
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            monitoring_result['final_status'] = 'error'
            monitoring_result['error'] = str(e)
        
        finally:
            monitoring_result['monitoring_ended'] = time.time()
            monitoring_result['duration'] = monitoring_result['monitoring_ended'] - monitoring_result['monitoring_started']
            
        return monitoring_result
    
    def stop_monitoring(self) -> None:
        """監視停止"""
        logger.info("Stopping monitoring")
        self.is_monitoring = False
    
    def register_callbacks(self, 
                          on_complete: Optional[Callable] = None,
                          on_failed: Optional[Callable] = None,
                          on_error: Optional[Callable] = None) -> None:
        """
        コールバック登録
        
        Args:
            on_complete: 完了時コールバック
            on_failed: 失敗時コールバック
            on_error: エラー時コールバック
        """
        self.on_task_complete = on_complete
        self.on_task_failed = on_failed
        self.on_task_error = on_error
        logger.info("Callbacks registered")
    
    def execute_next_action(self, task_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        次アクション実行（デフォルト実装）
        
        Args:
            task_status: タスク完了状態
            
        Returns:
            実行結果
        """
        logger.info("Executing next action based on task completion")
        
        result = {
            'action': 'next_task',
            'previous_task': task_status.get('task_id'),
            'timestamp': time.time()
        }
        
        # タスクタイプに応じた次アクション判定
        if 'pytest' in str(task_status.get('task_id', '')):
            # pytest完了後は結果分析
            result['next_action'] = 'analyze_test_results'
            result['details'] = 'Analyzing pytest results for failures and performance'
            
        elif 'extract_character' in str(task_status.get('task_id', '')):
            # extract_character完了後は品質評価
            result['next_action'] = 'evaluate_extraction_quality'
            result['details'] = 'Running quality assessment on extracted characters'
        
        else:
            # デフォルトアクション
            result['next_action'] = 'review_output'
            result['details'] = 'Reviewing task output for next steps'
        
        logger.info(f"Next action determined: {result['next_action']}")
        return result

    # INTG-089: 拡張監視機能
    def _check_gpu_availability(self) -> bool:
        """GPU可用性チェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available - GPU monitoring disabled")
            return False

    def detect_gpu_anomalies(self) -> Optional[str]:
        """
        GPU異常検知

        Returns:
            Optional[str]: 異常メッセージ（異常なしの場合はNone）
        """
        if not self.gpu_monitoring_enabled:
            return None

        try:
            import torch

            # GPU メモリ使用率チェック
            allocated = torch.cuda.memory_allocated()
            max_allocated = torch.cuda.max_memory_allocated()

            # ゼロ除算エラーを防止
            if max_allocated == 0:
                # GPU使用実績がない場合は現在のGPUメモリ容量を使用
                try:
                    max_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_allocated = (allocated / max_memory) * 100 if max_memory > 0 else 0
                except Exception:
                    gpu_memory_allocated = 0
            else:
                gpu_memory_allocated = (allocated / max_allocated) * 100

            if gpu_memory_allocated > self.anomaly_thresholds['gpu_memory_usage']:
                anomaly_msg = f"GPU Memory Usage Critical: {gpu_memory_allocated:.1f}%"
                self._record_anomaly('gpu_memory_critical', {
                    'gpu_memory_usage': gpu_memory_allocated,
                    'threshold': self.anomaly_thresholds['gpu_memory_usage']
                })
                return anomaly_msg

            # GPU温度チェック（nvidia-smi使用）
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temp = int(result.stdout.strip())
                    if temp > self.anomaly_thresholds['temperature_threshold']:
                        anomaly_msg = f"GPU Temperature High: {temp}°C"
                        self._record_anomaly('gpu_temperature_high', {
                            'temperature': temp,
                            'threshold': self.anomaly_thresholds['temperature_threshold']
                        })
                        return anomaly_msg
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
                logger.warning("Failed to get GPU temperature")

        except Exception as e:
            logger.error(f"GPU anomaly detection error: {e}")

        return None

    def detect_memory_leaks(self) -> Optional[str]:
        """
        メモリリーク検出（改善版）

        Returns:
            Optional[str]: メモリリークメッセージ（リークなしの場合はNone）
        """
        try:
            current_memory = psutil.virtual_memory()
            memory_usage_percent = current_memory.percent
            current_time = time.time()

            # メモリ履歴管理（最大10エントリ）
            if not hasattr(self, 'memory_history'):
                self.memory_history = []

            # 現在のメモリ情報を履歴に追加
            self.memory_history.append({
                'timestamp': current_time,
                'used_mb': current_memory.used / 1024 / 1024,
                'percent': memory_usage_percent
            })

            # 履歴を最新10件に制限
            if len(self.memory_history) > 10:
                self.memory_history = self.memory_history[-10:]

            # 1. システムメモリ使用率チェック
            if memory_usage_percent > self.anomaly_thresholds['system_memory_usage']:
                anomaly_msg = f"System Memory Usage High: {memory_usage_percent:.1f}%"
                self._record_anomaly('memory_usage_high', {
                    'memory_usage_percent': memory_usage_percent,
                    'threshold': self.anomaly_thresholds['system_memory_usage']
                })
                return anomaly_msg

            # 2. ベースラインからの増加チェック（改善版）
            memory_increase = current_memory.used - self.memory_baseline
            memory_increase_mb = memory_increase / 1024 / 1024
            memory_increase_percent = (memory_increase / self.memory_baseline) * 100 if self.memory_baseline > 0 else 0

            # 閾値を動的に調整（システム総メモリの10%以上、または2GB以上）
            total_memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            dynamic_threshold_mb = min(2048, (total_memory_gb * 0.1) * 1024)  # 10%またはmax 2GB

            if memory_increase_mb > dynamic_threshold_mb:
                anomaly_msg = f"Potential Memory Leak: +{memory_increase_mb:.0f}MB (+{memory_increase_percent:.1f}%) from baseline"
                self._record_anomaly('memory_leak_suspected', {
                    'memory_increase_mb': memory_increase_mb,
                    'memory_increase_percent': memory_increase_percent,
                    'current_used_gb': current_memory.used / 1024 / 1024 / 1024,
                    'threshold_mb': dynamic_threshold_mb
                })
                return anomaly_msg

            # 3. メモリ使用量の急激な増加傾向チェック
            if len(self.memory_history) >= 5:  # 5回以上のサンプルが必要
                recent_samples = self.memory_history[-5:]
                oldest_sample = recent_samples[0]
                newest_sample = recent_samples[-1]

                time_diff_minutes = (newest_sample['timestamp'] - oldest_sample['timestamp']) / 60
                memory_diff_mb = newest_sample['used_mb'] - oldest_sample['used_mb']

                if time_diff_minutes > 0:
                    # 1分あたりのメモリ増加率を計算
                    memory_growth_rate = memory_diff_mb / time_diff_minutes

                    # 1分あたり50MB以上の増加は異常とみなす
                    if memory_growth_rate > 50:
                        anomaly_msg = f"Rapid Memory Growth: {memory_growth_rate:.1f}MB/min over {time_diff_minutes:.1f}min"
                        self._record_anomaly('memory_growth_rapid', {
                            'growth_rate_mb_per_min': memory_growth_rate,
                            'time_window_min': time_diff_minutes,
                            'memory_diff_mb': memory_diff_mb
                        })
                        return anomaly_msg

            # 4. 継続的なメモリ使用量チェック（リーク兆候）
            if len(self.memory_history) >= 8:  # より長期的な傾向分析
                # 最新8サンプルでの平均増加率を計算
                growth_rates = []
                for i in range(1, len(self.memory_history)):
                    prev = self.memory_history[i-1]
                    curr = self.memory_history[i]
                    time_diff = curr['timestamp'] - prev['timestamp']
                    if time_diff > 0:
                        rate = (curr['used_mb'] - prev['used_mb']) / (time_diff / 60)  # MB/min
                        growth_rates.append(rate)

                if growth_rates:
                    avg_growth_rate = sum(growth_rates) / len(growth_rates)
                    # 平均で1分あたり10MB以上の継続的な増加
                    positive_growth_count = sum(1 for rate in growth_rates if rate > 5)

                    if avg_growth_rate > 10 and positive_growth_count >= 6:
                        anomaly_msg = f"Sustained Memory Leak: avg {avg_growth_rate:.1f}MB/min, {positive_growth_count}/{len(growth_rates)} positive trends"
                        self._record_anomaly('memory_leak_sustained', {
                            'avg_growth_rate': avg_growth_rate,
                            'positive_trends': positive_growth_count,
                            'total_samples': len(growth_rates)
                        })
                        return anomaly_msg

        except Exception as e:
            logger.error(f"Memory leak detection error: {e}")

        return None

    def monitor_process_health(self, process_id: Optional[int] = None) -> Optional[str]:
        """
        プロセス健全性監視

        Args:
            process_id: 監視対象プロセスID（Noneの場合は現在のプロセス）

        Returns:
            Optional[str]: プロセス異常メッセージ（異常なしの場合はNone）
        """
        try:
            if process_id:
                process = psutil.Process(process_id)
            else:
                process = psutil.Process()

            # CPU使用率チェック
            cpu_percent = process.cpu_percent(interval=1)
            if cpu_percent > self.anomaly_thresholds['cpu_usage']:
                anomaly_msg = f"Process CPU Usage Critical: {cpu_percent:.1f}%"
                self._record_anomaly('process_cpu_high', {
                    'cpu_percent': cpu_percent,
                    'threshold': self.anomaly_thresholds['cpu_usage'],
                    'process_id': process.pid
                })
                return anomaly_msg

            # プロセス実行時間チェック
            create_time = process.create_time()
            runtime = time.time() - create_time

            if runtime > self.anomaly_thresholds['process_timeout']:
                anomaly_msg = f"Process Running Too Long: {runtime/3600:.1f} hours"
                self._record_anomaly('process_timeout', {
                    'runtime_hours': runtime / 3600,
                    'threshold_hours': self.anomaly_thresholds['process_timeout'] / 3600,
                    'process_id': process.pid
                })
                return anomaly_msg

            # プロセス状態チェック（ゾンビプロセス等）
            if process.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                anomaly_msg = f"Process Status Abnormal: {process.status()}"
                self._record_anomaly('process_status_abnormal', {
                    'status': process.status(),
                    'process_id': process.pid
                })
                return anomaly_msg

        except psutil.NoSuchProcess:
            return "Process Not Found"
        except Exception as e:
            logger.error(f"Process health monitoring error: {e}")

        return None

    def comprehensive_anomaly_check(self, process_id: Optional[int] = None) -> Dict[str, Any]:
        """
        包括的異常チェック

        Args:
            process_id: 監視対象プロセスID

        Returns:
            Dict[str, Any]: 異常検知結果
        """
        anomalies = []

        # GPU異常チェック
        gpu_anomaly = self.detect_gpu_anomalies()
        if gpu_anomaly:
            anomalies.append({'type': 'gpu', 'message': gpu_anomaly})

        # メモリリークチェック
        memory_anomaly = self.detect_memory_leaks()
        if memory_anomaly:
            anomalies.append({'type': 'memory', 'message': memory_anomaly})

        # プロセス健全性チェック
        process_anomaly = self.monitor_process_health(process_id)
        if process_anomaly:
            anomalies.append({'type': 'process', 'message': process_anomaly})

        result = {
            'timestamp': time.time(),
            'anomalies_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'system_stats': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_available': self.gpu_monitoring_enabled
            }
        }

        # 異常検知時のコールバック実行
        if anomalies and self.on_anomaly_detected:
            self.on_anomaly_detected(result)

        return result

    def _record_anomaly(self, anomaly_type: str, details: Dict[str, Any]) -> None:
        """異常履歴記録"""
        anomaly_record = {
            'timestamp': time.time(),
            'type': anomaly_type,
            'details': details
        }
        self.anomaly_history.append(anomaly_record)

        # 履歴を最大100件に制限
        if len(self.anomaly_history) > 100:
            self.anomaly_history = self.anomaly_history[-100:]

        logger.warning(f"Anomaly recorded: {anomaly_type} - {details}")

    def get_anomaly_report(self) -> Dict[str, Any]:
        """
        異常レポート取得

        Returns:
            Dict[str, Any]: 異常統計レポート
        """
        if not self.anomaly_history:
            return {
                'total_anomalies': 0,
                'anomaly_types': {},
                'recent_anomalies': [],
                'trend': 'stable'
            }

        # 異常タイプ別統計
        type_counts = {}
        recent_anomalies = []

        # 過去1時間の異常を最近の異常とする
        recent_threshold = time.time() - 3600

        for anomaly in self.anomaly_history:
            anomaly_type = anomaly['type']
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1

            if anomaly['timestamp'] > recent_threshold:
                recent_anomalies.append(anomaly)

        # トレンド分析
        trend = 'stable'
        if len(recent_anomalies) > 5:
            trend = 'increasing'
        elif len(recent_anomalies) == 0:
            trend = 'improving'

        return {
            'total_anomalies': len(self.anomaly_history),
            'anomaly_types': type_counts,
            'recent_anomalies': recent_anomalies,
            'trend': trend,
            'monitoring_duration': time.time() - (self.anomaly_history[0]['timestamp'] if self.anomaly_history else time.time())
        }


class SubAgentIntegration:
    """SubAgent統合クラス（Claude Code内で使用）"""
    
    def __init__(self):
        """初期化"""
        self.monitor = SubAgentMonitor()
        self.current_context: Dict[str, Any] = {}
        logger.info("SubAgentIntegration initialized")
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        作業コンテキスト設定
        
        Args:
            context: 現在の作業コンテキスト（トラッカーID、Todoリスト等）
        """
        self.current_context = context
        logger.info(f"Context set: {context.get('tracker_id', 'unknown')}")
    
    def monitor_long_task(self, task_id: str, task_command: str) -> Dict[str, Any]:
        """
        長時間タスクの監視（同一セッション内）
        
        Args:
            task_id: タスクID
            task_command: 実行コマンド
            
        Returns:
            監視結果と次アクション
        """
        logger.info(f"Monitoring long task in same session: {task_id}")
        
        # コンテキスト継承の確認
        logger.info(f"Current context: {self.current_context}")
        
        # 完了時の自動アクション設定
        def on_complete(status):
            logger.info(f"Task {task_id} completed successfully")
            
            # コンテキストを保持したまま次アクション実行
            next_action = self.monitor.execute_next_action(status)
            next_action['context'] = self.current_context
            
            # Todoリスト更新（もし存在すれば）
            if 'todo_list' in self.current_context:
                logger.info("Updating Todo list with completion")
                # ここでTodoリスト更新処理
            
            return next_action
        
        # 失敗時の手動レビュー連携
        def on_failed(status):
            logger.error(f"Task {task_id} failed")
            
            if status.get('requires_manual_review'):
                logger.info("Switching to manual review for error analysis")
                # ここで手動レビュー切り替え処理
                return {
                    'action': 'switch_to_manual_review',
                    'reason': 'task_failure',
                    'task_id': task_id,
                    'error': status.get('error')
                }
        
        # コールバック登録
        self.monitor.register_callbacks(
            on_complete=on_complete,
            on_failed=on_failed
        )
        
        # 監視開始
        result = self.monitor.start_monitoring(task_id)
        
        # 結果にコンテキスト情報追加
        result['session_context'] = self.current_context
        result['same_session'] = True
        
        return result


def demonstrate_subagent_monitoring():
    """
    SubAgent監視のデモンストレーション
    実際のClaude Code内での使用例
    """
    print("🎯 SubAgent監視デモンストレーション")
    print("=" * 50)
    
    # SubAgent統合初期化
    integration = SubAgentIntegration()
    
    # 現在の作業コンテキスト設定（Claude Codeから継承）
    integration.set_context({
        'tracker_id': 'QUAL-044',
        'current_task': 'long_task_queue_implementation',
        'todo_list': ['implement_queue', 'test_system', 'deploy'],
        'session_id': 'current_claude_session'
    })
    
    print("📋 コンテキスト設定完了:")
    print(f"   トラッカー: QUAL-044")
    print(f"   セッション: 同一セッション内実行")
    print()
    
    # 長時間タスクの監視例
    print("🔄 長時間タスク監視開始...")
    print("   タスク: pytest実行")
    print("   監視: SubAgentによる同一セッション監視")
    print()
    
    # 実際の監視（デモ用にシミュレート）
    demo_result = {
        'task_id': 'pytest_20250830_161234',
        'monitoring_started': time.time(),
        'status_changes': [
            {'timestamp': time.time(), 'status': 'task_running'},
            {'timestamp': time.time() + 10, 'status': 'task_completed'}
        ],
        'final_status': 'completed',
        'next_action_executed': True,
        'next_action_result': {
            'action': 'analyze_test_results',
            'details': 'Analyzing pytest results for failures and performance',
            'context': {
                'tracker_id': 'QUAL-044',
                'session_id': 'current_claude_session'
            }
        },
        'session_context': {
            'tracker_id': 'QUAL-044',
            'current_task': 'long_task_queue_implementation',
            'todo_list': ['implement_queue', 'test_system', 'deploy'],
            'session_id': 'current_claude_session'
        },
        'same_session': True,
        'duration': 10.0
    }
    
    print("✅ 監視結果:")
    print(f"   タスク状態: {demo_result['final_status']}")
    print(f"   次アクション: {demo_result['next_action_result']['action']}")
    print(f"   セッション継続: {demo_result['same_session']}")
    print(f"   コンテキスト保持: ✅")
    print()
    
    print("🎯 重要な特徴:")
    print("   1. 同一セッション内で監視・実行")
    print("   2. コンテキスト（トラッカーID、Todo）完全継承")
    print("   3. 自動的な次アクション判定・実行")
    print("   4. 手動レビュー連携（エラー時）")
    print()
    
    return demo_result


def main():
    """CLI実行用メイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        # デモンストレーション実行
        demonstrate_subagent_monitoring()
    else:
        task_id = sys.argv[1]
        monitor = SubAgentMonitor()
        
        print(f"📍 Monitoring task: {task_id}")
        print("Press Ctrl+C to stop monitoring...")
        
        result = monitor.start_monitoring(task_id)
        
        print("\n📊 Monitoring Result:")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()