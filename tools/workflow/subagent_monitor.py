#!/usr/bin/env python3
"""
SubAgent状態監視システム - KIRO-011 SubAgent-ワークフロー連携統合
KIRO-010解決策: SubAgentの不完全実行検出・自動対応システム
"""

import logging
import os
import psutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubAgentStatus(Enum):
    """SubAgentプロセス状態定義"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STALLED = "stalled"


class SubAgentProcess:
    """SubAgentプロセス情報"""

    def __init__(
        self,
        tracker_id: str,
        process_type: str,
        command: str,
        workspace_path: str,
        expected_duration: int = 1800,
    ):
        self.tracker_id = tracker_id
        self.process_type = process_type  # 'extraction', 'validation', etc.
        self.command = command
        self.workspace_path = workspace_path
        self.expected_duration = expected_duration  # seconds
        self.start_time = None
        self.end_time = None
        self.pid = None
        self.status = SubAgentStatus.NOT_STARTED
        self.retry_count = 0
        self.max_retries = 3
        self.lock_file_path = os.path.join(workspace_path, f".{process_type}_lock")
        self.progress_file_path = os.path.join(workspace_path, f".{process_type}_progress")
        self.log_file_path = os.path.join(workspace_path, f"logs/{process_type}.log")


class SubAgentMonitor:
    """SubAgent状態監視・管理システム"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = self._find_project_root()
            db_path = os.path.join(project_root, "subagent_monitor.db")

        self.db_path = db_path
        self.lock = threading.Lock()
        self.active_processes: Dict[str, SubAgentProcess] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self._init_database()
        self._load_active_processes()
        logger.info(f"SubAgent監視システムが初期化されました: {db_path}")

    def _find_project_root(self) -> str:
        """プロジェクトルートディレクトリを検索"""
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "CLAUDE.md")):
                return current
            current = os.path.dirname(current)
        return "/mnt/c/AItools/segment-anything"

    def _init_database(self):
        """SQLiteデータベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subagent_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracker_id TEXT NOT NULL,
                    process_type TEXT NOT NULL,
                    command TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    retry_count INTEGER DEFAULT 0,
                    expected_duration INTEGER DEFAULT 1800,
                    actual_duration INTEGER,
                    exit_code INTEGER,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tracker_id, process_type)
                )
            """
            )
            conn.commit()
            logger.info("SubAgent監視データベース初期化完了")

    def _load_active_processes(self):
        """データベースから既存のアクティブプロセスを読み込み"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT tracker_id, process_type, command, workspace_path,
                           start_time, end_time, status, pid, retry_count, expected_duration
                    FROM subagent_executions
                    WHERE status IN ('running', 'failed', 'not_started')
                """
                )

                for row in cursor.fetchall():
                    (
                        tracker_id,
                        process_type,
                        command,
                        workspace_path,
                        start_time,
                        end_time,
                        status,
                        pid,
                        retry_count,
                        expected_duration,
                    ) = row

                    # SubAgentProcessオブジェクトを復元
                    process = SubAgentProcess(
                        tracker_id=tracker_id,
                        process_type=process_type,
                        command=command,
                        workspace_path=workspace_path,
                        expected_duration=expected_duration or 1800,
                    )

                    # 状態を復元
                    process.status = SubAgentStatus(status)
                    process.pid = pid
                    process.retry_count = retry_count or 0

                    if start_time:
                        process.start_time = datetime.fromisoformat(start_time)
                    if end_time:
                        process.end_time = datetime.fromisoformat(end_time)

                    # active_processesに追加
                    process_key = f"{tracker_id}_{process_type}"
                    self.active_processes[process_key] = process

                logger.info(f"アクティブプロセス復元完了: {len(self.active_processes)}件")

        except Exception as e:
            logger.error(f"アクティブプロセス読み込みエラー: {e}")

    def register_subagent(
        self,
        tracker_id: str,
        process_type: str,
        command: str,
        workspace_path: str,
        expected_duration: int = 1800,
    ) -> bool:
        """SubAgentプロセスを監視対象として登録"""
        try:
            # ワークスペースの存在確認・作成
            workspace_dir = Path(workspace_path)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "logs").mkdir(exist_ok=True)

            # プロセス情報を作成
            process = SubAgentProcess(
                tracker_id=tracker_id,
                process_type=process_type,
                command=command,
                workspace_path=workspace_path,
                expected_duration=expected_duration,
            )

            # データベースに登録
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO subagent_executions
                        (tracker_id, process_type, command, workspace_path,
                         status, expected_duration, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (
                            tracker_id,
                            process_type,
                            command,
                            workspace_path,
                            SubAgentStatus.NOT_STARTED.value,
                            expected_duration,
                        ),
                    )
                    conn.commit()

                # メモリ上でも管理
                process_key = f"{tracker_id}_{process_type}"
                self.active_processes[process_key] = process

            logger.info(f"SubAgent登録完了: {tracker_id}/{process_type}")
            return True

        except Exception as e:
            logger.error(f"SubAgent登録エラー: {tracker_id}/{process_type} - {e}")
            return False

    def start_subagent(self, tracker_id: str, process_type: str) -> bool:
        """SubAgentプロセス開始"""
        process_key = f"{tracker_id}_{process_type}"

        if process_key not in self.active_processes:
            logger.error(f"未登録のSubAgent: {process_key}")
            return False

        process = self.active_processes[process_key]

        try:
            # 既存ロックファイル確認（二重起動防止）
            if os.path.exists(process.lock_file_path):
                logger.warning(f"ロックファイル検出: {process.lock_file_path}")
                if self._check_process_running_by_lock(process):
                    logger.error(f"SubAgent既に実行中: {process_key}")
                    return False
                else:
                    # 古いロックファイル削除
                    os.remove(process.lock_file_path)
                    logger.info(f"古いロックファイル削除: {process.lock_file_path}")

            # ロックファイル作成
            with open(process.lock_file_path, "w") as f:
                f.write(f"pid=PENDING\nstart_time={datetime.now().isoformat()}\n")

            # プロセス開始
            import subprocess

            # KIRO-011修正: 相対パス解決のためプロジェクトルートで実行
            project_root = self._find_project_root()

            # KIRO-011修正: シンプルなsubprocess実行（TTY環境制約解決）
            proc = subprocess.Popen(
                process.command,
                shell=True,
                cwd=project_root
                # stdout/stderrパイプを除去してTTY環境制約を解決
            )

            # プロセス情報更新
            process.pid = proc.pid
            process.start_time = datetime.now()
            process.status = SubAgentStatus.RUNNING

            # ロックファイル更新
            with open(process.lock_file_path, "w") as f:
                f.write(f"pid={proc.pid}\nstart_time={process.start_time.isoformat()}\n")

            # KIRO-011修正: プロセス出力をログファイルに保存開始
            self._start_output_logging(process, proc)

            # データベース更新
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE subagent_executions
                        SET status = ?, pid = ?, start_time = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE tracker_id = ? AND process_type = ?
                    """,
                        (
                            SubAgentStatus.RUNNING.value,
                            proc.pid,
                            process.start_time.isoformat(),
                            tracker_id,
                            process_type,
                        ),
                    )
                    conn.commit()

            logger.info(f"SubAgent開始: {process_key} (PID: {proc.pid})")

            # 監視開始
            if not self.monitoring_active:
                self.start_monitoring()

            return True

        except Exception as e:
            logger.error(f"SubAgent開始エラー: {process_key} - {e}")
            process.status = SubAgentStatus.FAILED
            return False

    def _start_output_logging(self, process: SubAgentProcess, proc):
        """基本ログファイル作成（簡素化版）"""
        try:
            # ログファイル準備
            log_file = Path(process.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(process.log_file_path, "w") as f:
                f.write(f"SubAgent実行ログ - {datetime.now().isoformat()}\n")
                f.write(f"コマンド: {process.command}\n")
                f.write(f"PID: {proc.pid}\n")
                f.write("=" * 80 + "\n\n")

        except Exception as e:
            logger.error(f"ログファイル作成エラー: {e}")

    def check_subagent_status(self, tracker_id: str, process_type: str) -> SubAgentStatus:
        """SubAgentプロセス状態確認"""
        process_key = f"{tracker_id}_{process_type}"

        if process_key not in self.active_processes:
            # データベースから状態を確認
            return self._get_status_from_db(tracker_id, process_type)

        process = self.active_processes[process_key]

        # プロセス生存確認
        if process.pid and process.status == SubAgentStatus.RUNNING:
            if self._is_process_alive(process.pid):
                # タイムアウト確認
                if self.is_timeout_reached(tracker_id, process_type):
                    process.status = SubAgentStatus.TIMEOUT
                    self._update_process_status(process)
                else:
                    # 進捗停滞確認
                    if self._is_process_stalled(process):
                        process.status = SubAgentStatus.STALLED
                        self._update_process_status(process)
            else:
                # プロセス終了確認
                process.end_time = datetime.now()
                if self._check_completion_success(process):
                    process.status = SubAgentStatus.COMPLETED
                else:
                    process.status = SubAgentStatus.FAILED
                self._update_process_status(process)
                self._cleanup_process_files(process)

        return process.status

    def is_timeout_reached(self, tracker_id: str, process_type: str) -> bool:
        """タイムアウト判定"""
        process_key = f"{tracker_id}_{process_type}"

        if process_key not in self.active_processes:
            return False

        process = self.active_processes[process_key]

        if not process.start_time:
            return False

        elapsed = datetime.now() - process.start_time
        return elapsed.total_seconds() > process.expected_duration

    def get_all_active_processes(self) -> List[Dict[str, Any]]:
        """全アクティブプロセス一覧取得"""
        processes = []

        with self.lock:
            for process_key, process in self.active_processes.items():
                status = self.check_subagent_status(process.tracker_id, process.process_type)
                processes.append(
                    {
                        "tracker_id": process.tracker_id,
                        "process_type": process.process_type,
                        "status": status.value,
                        "pid": process.pid,
                        "start_time": process.start_time.isoformat()
                        if process.start_time
                        else None,
                        "elapsed_seconds": (datetime.now() - process.start_time).total_seconds()
                        if process.start_time
                        else 0,
                        "expected_duration": process.expected_duration,
                        "retry_count": process.retry_count,
                    }
                )

        return processes

    def terminate_subagent(self, tracker_id: str, process_type: str, force: bool = False) -> bool:
        """SubAgentプロセス終了"""
        process_key = f"{tracker_id}_{process_type}"

        if process_key not in self.active_processes:
            logger.warning(f"未登録のSubAgent終了要求: {process_key}")
            return False

        process = self.active_processes[process_key]

        if not process.pid:
            logger.warning(f"PID不明のSubAgent終了要求: {process_key}")
            return False

        try:
            if self._is_process_alive(process.pid):
                proc = psutil.Process(process.pid)

                if force:
                    proc.kill()
                    logger.info(f"SubAgent強制終了: {process_key} (PID: {process.pid})")
                else:
                    proc.terminate()
                    logger.info(f"SubAgent正常終了: {process_key} (PID: {process.pid})")

                # 終了待機
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    if not force:
                        proc.kill()
                        logger.warning(f"SubAgent強制終了実行: {process_key}")

                process.status = SubAgentStatus.FAILED
                process.end_time = datetime.now()
                self._update_process_status(process)
                self._cleanup_process_files(process)

                return True
            else:
                logger.info(f"SubAgent既に終了済み: {process_key}")
                return True

        except Exception as e:
            logger.error(f"SubAgent終了エラー: {process_key} - {e}")
            return False

    def start_monitoring(self) -> bool:
        """監視スレッド開始"""
        if self.monitoring_active:
            logger.warning("監視システム既に稼働中")
            return False

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("SubAgent監視システム開始")
        return True

    def stop_monitoring(self) -> bool:
        """監視システム停止"""
        if not self.monitoring_active:
            return False

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("SubAgent監視システム停止")
        return True

    def _monitoring_loop(self):
        """監視ループ（バックグラウンド実行）"""
        auto_retry_interval = 0  # 自動再実行チェック間隔カウンター

        while self.monitoring_active:
            try:
                # 全プロセス状態確認
                for process_key in list(self.active_processes.keys()):
                    process = self.active_processes[process_key]
                    status = self.check_subagent_status(process.tracker_id, process.process_type)

                    # 終了状態のプロセスをクリーンアップ
                    if status in [SubAgentStatus.COMPLETED]:
                        del self.active_processes[process_key]
                        logger.info(f"監視終了（完了）: {process_key}")
                    elif status in [SubAgentStatus.FAILED, SubAgentStatus.TIMEOUT]:
                        # リトライ上限チェック
                        if process.retry_count >= process.max_retries:
                            del self.active_processes[process_key]
                            logger.info(f"監視終了（リトライ上限）: {process_key}")

                # 5分ごとに自動再実行チェック
                auto_retry_interval += 1
                if auto_retry_interval >= 10:  # 30秒 x 10 = 5分
                    auto_retry_interval = 0
                    retry_count = self.check_and_auto_retry_all()
                    if retry_count > 0:
                        logger.info(f"定期自動再実行: {retry_count}プロセス")

                time.sleep(30)  # 30秒間隔で監視

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機

    def _is_process_alive(self, pid: int) -> bool:
        """プロセス生存確認"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False

    def _check_process_running_by_lock(self, process: SubAgentProcess) -> bool:
        """ロックファイルからプロセス稼働確認"""
        try:
            if not os.path.exists(process.lock_file_path):
                return False

            with open(process.lock_file_path, "r") as f:
                content = f.read()

            # PID抽出
            for line in content.split("\n"):
                if line.startswith("pid="):
                    pid_str = line.split("=")[1]
                    if pid_str == "PENDING":
                        return True  # 起動中として扱う
                    try:
                        pid = int(pid_str)
                        return self._is_process_alive(pid)
                    except:
                        return False
            return False

        except Exception as e:
            logger.error(f"ロックファイル確認エラー: {e}")
            return False

    def _is_process_stalled(self, process: SubAgentProcess) -> bool:
        """プロセス停滞判定"""
        try:
            if not os.path.exists(process.progress_file_path):
                # 進捗ファイルがない場合、開始から10分経過で停滞とみなす
                if process.start_time:
                    elapsed = datetime.now() - process.start_time
                    return elapsed.total_seconds() > 600
                return False

            # 進捗ファイルの最終更新時刻確認
            mtime = datetime.fromtimestamp(os.path.getmtime(process.progress_file_path))
            elapsed = datetime.now() - mtime

            # 5分間進捗ファイル更新がない場合は停滞
            return elapsed.total_seconds() > 300

        except Exception as e:
            logger.error(f"停滞判定エラー: {e}")
            return False

    def _check_completion_success(self, process: SubAgentProcess) -> bool:
        """完了成功判定"""
        try:
            # 基本的な成功条件
            success_indicators = [
                os.path.exists(
                    os.path.join(process.workspace_path, "extraction", "dashboard.html")
                ),
                os.path.exists(os.path.join(process.workspace_path, "extraction", "index.html")),
                os.path.exists(os.path.join(process.workspace_path, "extraction", "progress.json")),
            ]

            return any(success_indicators)

        except Exception as e:
            logger.error(f"完了判定エラー: {e}")
            return False

    def _update_process_status(self, process: SubAgentProcess):
        """プロセス状態をデータベースに更新"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    duration = None
                    if process.start_time and process.end_time:
                        duration = int((process.end_time - process.start_time).total_seconds())

                    conn.execute(
                        """
                        UPDATE subagent_executions
                        SET status = ?, end_time = ?, actual_duration = ?,
                            retry_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE tracker_id = ? AND process_type = ?
                    """,
                        (
                            process.status.value,
                            process.end_time.isoformat() if process.end_time else None,
                            duration,
                            process.retry_count,
                            process.tracker_id,
                            process.process_type,
                        ),
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"状態更新エラー: {e}")

    def _cleanup_process_files(self, process: SubAgentProcess):
        """プロセス関連ファイルクリーンアップ"""
        try:
            # ロックファイル削除
            if os.path.exists(process.lock_file_path):
                os.remove(process.lock_file_path)
                logger.info(f"ロックファイル削除: {process.lock_file_path}")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")

    def should_auto_retry(self, tracker_id: str, process_type: str = "extraction") -> bool:
        """自動再実行判定"""
        try:
            process_key = f"{tracker_id}_{process_type}"

            if process_key not in self.active_processes:
                logger.warning(f"未登録プロセスの自動再実行判定: {process_key}")
                return False

            process = self.active_processes[process_key]

            # リトライ回数上限確認
            if process.retry_count >= process.max_retries:
                logger.info(
                    f"リトライ回数上限: {process_key} ({process.retry_count}/{process.max_retries})"
                )
                return False

            # 現在の状態確認
            current_status = self.check_subagent_status(tracker_id, process_type)
            if current_status not in [
                SubAgentStatus.FAILED,
                SubAgentStatus.TIMEOUT,
                SubAgentStatus.STALLED,
            ]:
                logger.info(f"自動再実行対象外の状態: {process_key} - {current_status.value}")
                return False

            # 抽出結果確認（0件判定）
            workspace_path = Path(process.workspace_path)
            extraction_dir = workspace_path / "extraction"

            if extraction_dir.exists():
                # 抽出ファイル数確認
                extracted_files = list(extraction_dir.glob("extracted_*.jpg")) + list(
                    extraction_dir.glob("*.png")
                )
                if len(extracted_files) > 0:
                    logger.info(f"部分成功のため再実行しない: {process_key} - {len(extracted_files)}ファイル")
                    return False

                # dashboard.htmlの存在確認
                dashboard_file = extraction_dir / "dashboard.html"
                if dashboard_file.exists():
                    # ファイルサイズ確認（実質的なコンテンツがある場合）
                    if dashboard_file.stat().st_size > 1000:  # 1KB以上
                        logger.info(f"ダッシュボード存在のため再実行しない: {process_key}")
                        return False

            # 最小実行時間確認（無限ループ防止）
            if process.start_time:
                elapsed = datetime.now() - process.start_time
                if elapsed.total_seconds() < 300:  # 5分未満は再実行しない
                    logger.info(f"最小実行時間未満: {process_key} - {elapsed.total_seconds()}秒")
                    return False

            logger.info(f"自動再実行条件を満たす: {process_key}")
            return True

        except Exception as e:
            logger.error(f"自動再実行判定エラー: {tracker_id}/{process_type} - {e}")
            return False

    def auto_retry_subagent(self, tracker_id: str, process_type: str = "extraction") -> bool:
        """SubAgent自動再実行"""
        try:
            if not self.should_auto_retry(tracker_id, process_type):
                return False

            process_key = f"{tracker_id}_{process_type}"
            process = self.active_processes[process_key]

            logger.info(
                f"SubAgent自動再実行開始: {process_key} (リトライ {process.retry_count + 1}/{process.max_retries})"
            )

            # 現在のプロセス終了
            if process.pid and self._is_process_alive(process.pid):
                try:
                    proc = psutil.Process(process.pid)
                    proc.terminate()
                    proc.wait(timeout=10)
                    logger.info(f"旧プロセス終了: PID {process.pid}")
                except:
                    try:
                        proc.kill()
                        logger.warning(f"旧プロセス強制終了: PID {process.pid}")
                    except:
                        pass

            # クリーンアップ
            self._cleanup_process_files(process)

            # リトライ回数増加
            process.retry_count += 1

            # 少し待機（リソース解放待ち）
            time.sleep(5)

            # 再実行
            success = self.start_subagent(tracker_id, process_type)

            if success:
                logger.info(f"SubAgent自動再実行成功: {process_key}")
                return True
            else:
                logger.error(f"SubAgent自動再実行失敗: {process_key}")
                return False

        except Exception as e:
            logger.error(f"SubAgent自動再実行エラー: {tracker_id}/{process_type} - {e}")
            return False

    def check_and_auto_retry_all(self) -> int:
        """全アクティブプロセスの自動再実行チェック・実行"""
        retry_count = 0

        try:
            # アクティブプロセスのコピーを作成（実行中に変更される可能性があるため）
            active_process_keys = list(self.active_processes.keys())

            for process_key in active_process_keys:
                if process_key not in self.active_processes:
                    continue  # 既に削除済み

                process = self.active_processes[process_key]

                # 自動再実行判定・実行
                if self.auto_retry_subagent(process.tracker_id, process.process_type):
                    retry_count += 1

                    # 連続実行防止のための待機
                    time.sleep(2)

            if retry_count > 0:
                logger.info(f"自動再実行バッチ完了: {retry_count}プロセス")

        except Exception as e:
            logger.error(f"自動再実行バッチエラー: {e}")

        return retry_count

    def _get_status_from_db(self, tracker_id: str, process_type: str) -> SubAgentStatus:
        """データベースから状態取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT status FROM subagent_executions
                    WHERE tracker_id = ? AND process_type = ?
                    ORDER BY updated_at DESC LIMIT 1
                """,
                    (tracker_id, process_type),
                )
                row = cursor.fetchone()

                if row:
                    return SubAgentStatus(row[0])
                return SubAgentStatus.NOT_STARTED

        except Exception as e:
            logger.error(f"DB状態取得エラー: {e}")
            return SubAgentStatus.NOT_STARTED


# グローバルインスタンス
_subagent_monitor = None


def get_subagent_monitor() -> SubAgentMonitor:
    """SubAgent監視システムのグローバルインスタンスを取得"""
    global _subagent_monitor
    if _subagent_monitor is None:
        _subagent_monitor = SubAgentMonitor()
    return _subagent_monitor


if __name__ == "__main__":
    # テスト実行
    monitor = SubAgentMonitor()

    # テスト用プロセス登録
    test_tracker = "TEST-001"
    test_command = "python -c 'import time; time.sleep(5); print(\"test complete\")'"
    test_workspace = "/tmp/test_workspace"

    success = monitor.register_subagent(
        tracker_id=test_tracker,
        process_type="extraction",
        command=test_command,
        workspace_path=test_workspace,
        expected_duration=300,
    )

    print(f"登録結果: {success}")

    if success:
        # 監視開始
        monitor.start_monitoring()

        # プロセス状態確認
        status = monitor.check_subagent_status(test_tracker, "extraction")
        print(f"初期状態: {status}")

        # 全プロセス一覧
        processes = monitor.get_all_active_processes()
        print(f"アクティブプロセス: {processes}")
