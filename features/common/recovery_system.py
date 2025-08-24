"""
復旧機能システム (PH2-008)
Purpose: プロセス監視・自動復旧・失敗検出システム実装
"""

import json
import logging
import psutil
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RecoveryState:
    """復旧状態管理"""
    process_id: str
    start_time: datetime
    current_phase: str
    retry_count: int = 0
    max_retries: int = 3
    last_checkpoint: Optional[str] = None
    error_history: List[str] = None
    
    def __post_init__(self):
        if self.error_history is None:
            self.error_history = []


@dataclass
class SystemStatus:
    """システム状態"""
    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    gpu_memory_mb: Optional[float]
    process_alive: bool
    timestamp: datetime


class ProcessMonitor:
    """プロセス監視クラス"""
    
    def __init__(self, process_name: str = "python"):
        self.process_name = process_name
        self.logger = logging.getLogger(__name__)
        
    def get_system_status(self) -> SystemStatus:
        """システム状態取得"""
        try:
            # CPU・メモリ使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # ディスク容量
            disk_usage = shutil.disk_usage("/")
            disk_free_gb = disk_usage.free / (1024**3)
            
            # GPU メモリ（利用可能な場合）
            gpu_memory_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            except ImportError:
                pass
            
            # プロセス生存確認
            process_alive = any(proc.name() == self.process_name for proc in psutil.process_iter(['name']))
            
            return SystemStatus(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_free_gb=disk_free_gb,
                gpu_memory_mb=gpu_memory_mb,
                process_alive=process_alive,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"システム状態取得エラー: {str(e)}")
            return SystemStatus(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_free_gb=0.0,
                gpu_memory_mb=None,
                process_alive=False,
                timestamp=datetime.now()
            )
    
    def detect_system_issues(self, status: SystemStatus) -> List[str]:
        """システム問題検出"""
        issues = []
        
        # CPU過負荷
        if status.cpu_percent > 95:
            issues.append(f"CPU使用率過高: {status.cpu_percent:.1f}%")
        
        # メモリ不足
        if status.memory_percent > 90:
            issues.append(f"メモリ使用率過高: {status.memory_percent:.1f}%")
        
        # ディスク容量不足
        if status.disk_free_gb < 1.0:
            issues.append(f"ディスク容量不足: {status.disk_free_gb:.1f}GB")
        
        # プロセス停止
        if not status.process_alive:
            issues.append("対象プロセスが停止中")
        
        return issues


class FailureDetector:
    """失敗検出システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_patterns = [
            "CUDA out of memory",
            "No space left on device",
            "Permission denied",
            "Module not found",
            "Segmentation fault",
            "Connection timed out",
            "Process killed"
        ]
    
    def detect_failure_from_log(self, log_file: Path) -> Tuple[bool, List[str]]:
        """ログファイルから失敗検出"""
        if not log_file.exists():
            return False, []
        
        detected_failures = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                
                for pattern in self.failure_patterns:
                    if pattern.lower() in log_content.lower():
                        detected_failures.append(pattern)
                        
        except Exception as e:
            self.logger.error(f"ログ解析エラー: {str(e)}")
            return False, []
        
        return len(detected_failures) > 0, detected_failures
    
    def detect_output_failure(self, expected_output_dir: Path) -> Tuple[bool, str]:
        """出力失敗検出"""
        if not expected_output_dir.exists():
            return True, "出力ディレクトリが存在しません"
        
        # 画像ファイル存在確認
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(expected_output_dir.glob(f"*{ext}"))
        
        if len(image_files) == 0:
            return True, "抽出された画像が存在しません"
        
        return False, ""


class AutoRecoverySystem:
    """自動復旧システム（指数バックオフ対応）"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.process_monitor = ProcessMonitor()
        self.failure_detector = FailureDetector()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """設定読み込み"""
        default_config = {
            "max_retries": 3,
            "base_delay": 30,  # 基本待機時間（秒）
            "max_delay": 300,  # 最大待機時間（秒）
            "backoff_multiplier": 2.0,
            "memory_threshold": 90,
            "disk_threshold": 1.0,
            "cpu_threshold": 95
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗、デフォルト使用: {str(e)}")
        
        return default_config
    
    def calculate_backoff_delay(self, attempt: int) -> int:
        """指数バックオフ遅延計算"""
        delay = min(
            self.config["base_delay"] * (self.config["backoff_multiplier"] ** attempt),
            self.config["max_delay"]
        )
        return int(delay)
    
    def attempt_recovery(self, recovery_state: RecoveryState, error_context: str) -> bool:
        """復旧試行"""
        if recovery_state.retry_count >= recovery_state.max_retries:
            self.logger.error(f"最大復旧試行回数に達しました: {recovery_state.retry_count}")
            return False
        
        recovery_state.retry_count += 1
        recovery_state.error_history.append(f"[{datetime.now()}] {error_context}")
        
        # 指数バックオフ待機
        delay = self.calculate_backoff_delay(recovery_state.retry_count)
        self.logger.info(f"復旧試行 {recovery_state.retry_count}/{recovery_state.max_retries} - {delay}秒待機")
        
        time.sleep(delay)
        
        # システム状態確認
        status = self.process_monitor.get_system_status()
        issues = self.process_monitor.detect_system_issues(status)
        
        if issues:
            self.logger.warning(f"システム問題検出: {', '.join(issues)}")
            # 重大な問題がある場合は復旧を延期
            if status.memory_percent > self.config["memory_threshold"] or \
               status.disk_free_gb < self.config["disk_threshold"]:
                self.logger.error("重大なシステム問題により復旧を中断")
                return False
        
        self.logger.info(f"復旧試行準備完了: CPU {status.cpu_percent:.1f}%, RAM {status.memory_percent:.1f}%")
        return True
    
    def save_recovery_state(self, state: RecoveryState, state_file: Path):
        """復旧状態保存"""
        try:
            state_data = asdict(state)
            # datetime を文字列に変換
            state_data['start_time'] = state.start_time.isoformat()
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"復旧状態を保存: {state_file}")
            
        except Exception as e:
            self.logger.error(f"復旧状態保存エラー: {str(e)}")
    
    def load_recovery_state(self, state_file: Path) -> Optional[RecoveryState]:
        """復旧状態読み込み"""
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # datetime を復元
            state_data['start_time'] = datetime.fromisoformat(state_data['start_time'])
            
            return RecoveryState(**state_data)
            
        except Exception as e:
            self.logger.error(f"復旧状態読み込みエラー: {str(e)}")
            return None


class RecoveryManager:
    """復旧管理統合クラス"""
    
    def __init__(self, tracker_id: str, workspace_dir: Path):
        self.tracker_id = tracker_id
        self.workspace_dir = workspace_dir
        self.logger = self._setup_logging()
        self.recovery_system = AutoRecoverySystem()
        self.state_file = workspace_dir / "recovery_state.json"
        
        # ワークスペース作成
        workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """ロギング設定"""
        logger = logging.getLogger(f"recovery_manager_{self.tracker_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_recovery_session(self, phase: str) -> RecoveryState:
        """復旧セッション初期化"""
        # 既存状態確認
        existing_state = self.recovery_system.load_recovery_state(self.state_file)
        
        if existing_state:
            self.logger.info(f"既存復旧状態を発見: 試行回数 {existing_state.retry_count}")
            return existing_state
        
        # 新規状態作成
        new_state = RecoveryState(
            process_id=f"{self.tracker_id}_{int(time.time())}",
            start_time=datetime.now(),
            current_phase=phase
        )
        
        self.recovery_system.save_recovery_state(new_state, self.state_file)
        self.logger.info(f"新規復旧セッション開始: {new_state.process_id}")
        
        return new_state
    
    def handle_failure(self, recovery_state: RecoveryState, error_msg: str) -> bool:
        """失敗処理"""
        self.logger.error(f"失敗検出: {error_msg}")
        
        # 復旧試行
        can_recover = self.recovery_system.attempt_recovery(recovery_state, error_msg)
        
        # 状態保存
        self.recovery_system.save_recovery_state(recovery_state, self.state_file)
        
        if not can_recover:
            self.logger.error("復旧不可能と判定")
            return False
        
        self.logger.info("復旧準備完了、処理を再開します")
        return True
    
    def cleanup_recovery_session(self):
        """復旧セッション終了処理"""
        if self.state_file.exists():
            self.state_file.unlink()
            self.logger.info("復旧セッション終了、状態ファイルを削除")


def create_recovery_enhanced_pipeline(tracker_id: str, workspace_dir: Path):
    """復旧機能付きパイプライン作成"""
    recovery_manager = RecoveryManager(tracker_id, workspace_dir)
    
    # 例: 復旧機能使用例
    recovery_state = recovery_manager.initialize_recovery_session("phase4")
    
    return recovery_manager, recovery_state


if __name__ == "__main__":
    # テスト実行
    from pathlib import Path
    
    test_workspace = Path("/tmp/recovery_test")
    recovery_manager, state = create_recovery_enhanced_pipeline("TEST-001", test_workspace)
    
    print(f"復旧システム初期化完了: {state.process_id}")
    
    # システム状態確認テスト
    monitor = ProcessMonitor()
    status = monitor.get_system_status()
    print(f"システム状態: CPU {status.cpu_percent}%, RAM {status.memory_percent}%")
    
    # 失敗検出テスト
    detector = FailureDetector()
    print("失敗検出システム準備完了")
    
    print("PH2-008 復旧機能システム実装完了")