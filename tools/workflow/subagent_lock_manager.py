#!/usr/bin/env python3
"""
SubAgent二重起動防止システム - KIRO-011 Phase 2実装
ロックファイル管理によるSubAgent重複実行防止
"""

import os
import json
import psutil
import socket
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubAgentLockManager:
    """SubAgent二重起動防止管理クラス"""

    def __init__(self, lock_dir: str = None):
        if lock_dir is None:
            self.lock_dir = Path("/tmp/segment-anything-locks")
        else:
            self.lock_dir = Path(lock_dir)

        # ロックディレクトリ作成
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        self.lock_file = self.lock_dir / "subagent_extraction.lock"
        self.lock = threading.Lock()

        # ロック情報のクリーンアップ
        self._cleanup_stale_locks()

        logger.info(f"SubAgent二重起動防止システム初期化: {self.lock_dir}")

    def acquire_lock(self, tracker_id: str, process_type: str = "extraction") -> bool:
        """SubAgentプロセス用ロック取得"""
        with self.lock:
            try:
                specific_lock_file = self.lock_dir / f"subagent_{process_type}_{tracker_id}.lock"

                # 既存ロック確認
                if specific_lock_file.exists():
                    if self._is_lock_valid(specific_lock_file):
                        existing_lock = self._read_lock_file(specific_lock_file)
                        logger.warning(f"アクティブロック検出: {tracker_id}/{process_type} (PID: {existing_lock.get('pid')})")
                        return False
                    else:
                        # 無効なロック削除
                        specific_lock_file.unlink()
                        logger.info(f"無効ロック削除: {specific_lock_file}")

                # グローバルロック確認
                if self.lock_file.exists():
                    if self._is_lock_valid(self.lock_file):
                        existing_lock = self._read_lock_file(self.lock_file)
                        logger.warning(f"グローバルロック検出 (PID: {existing_lock.get('pid')})")
                        return False
                    else:
                        self.lock_file.unlink()
                        logger.info(f"無効グローバルロック削除: {self.lock_file}")

                # ロック作成
                lock_data = {
                    'tracker_id': tracker_id,
                    'process_type': process_type,
                    'pid': os.getpid(),
                    'created_at': datetime.now().isoformat(),
                    'hostname': socket.gethostname(),
                    'status': 'active',
                    'command': f"extract_character.py {tracker_id}",
                    'workspace_path': f"/mnt/c/AItools/lora/train/*/tracker-workspace/{tracker_id}"
                }

                # 個別ロックファイル作成
                with open(specific_lock_file, 'w') as f:
                    json.dump(lock_data, f, indent=2)

                # グローバルロックファイル作成
                with open(self.lock_file, 'w') as f:
                    json.dump(lock_data, f, indent=2)

                logger.info(f"ロック取得成功: {tracker_id}/{process_type} (PID: {os.getpid()})")
                return True

            except Exception as e:
                logger.error(f"ロック取得エラー: {tracker_id}/{process_type} - {e}")
                return False

    def release_lock(self, tracker_id: str, process_type: str = "extraction") -> bool:
        """SubAgentプロセス用ロック解放"""
        with self.lock:
            try:
                specific_lock_file = self.lock_dir / f"subagent_{process_type}_{tracker_id}.lock"
                success = True

                # 個別ロックファイル削除
                if specific_lock_file.exists():
                    lock_data = self._read_lock_file(specific_lock_file)
                    if lock_data and lock_data.get('pid') == os.getpid():
                        specific_lock_file.unlink()
                        logger.info(f"個別ロック解放: {specific_lock_file}")
                    else:
                        logger.warning(f"PID不一致または無効ロック: {specific_lock_file}")
                        success = False

                # グローバルロック確認・削除
                if self.lock_file.exists():
                    global_lock = self._read_lock_file(self.lock_file)
                    if (global_lock and
                        global_lock.get('tracker_id') == tracker_id and
                        global_lock.get('process_type') == process_type and
                        global_lock.get('pid') == os.getpid()):
                        self.lock_file.unlink()
                        logger.info(f"グローバルロック解放: {self.lock_file}")

                return success

            except Exception as e:
                logger.error(f"ロック解放エラー: {tracker_id}/{process_type} - {e}")
                return False

    def check_existing_locks(self) -> Dict[str, Any]:
        """既存ロック状況確認"""
        locks_info = {
            'global_lock': None,
            'specific_locks': [],
            'stale_locks': [],
            'active_count': 0
        }

        try:
            # グローバルロック確認
            if self.lock_file.exists():
                global_lock = self._read_lock_file(self.lock_file)
                if global_lock:
                    global_lock['file_path'] = str(self.lock_file)
                    global_lock['is_valid'] = self._is_lock_valid(self.lock_file)
                    locks_info['global_lock'] = global_lock
                    if global_lock['is_valid']:
                        locks_info['active_count'] += 1

            # 個別ロック確認
            for lock_file in self.lock_dir.glob("subagent_*.lock"):
                if lock_file.name == "subagent_extraction.lock":
                    continue  # グローバルロックは除外

                lock_data = self._read_lock_file(lock_file)
                if lock_data:
                    lock_data['file_path'] = str(lock_file)
                    lock_data['is_valid'] = self._is_lock_valid(lock_file)

                    if lock_data['is_valid']:
                        locks_info['specific_locks'].append(lock_data)
                        locks_info['active_count'] += 1
                    else:
                        locks_info['stale_locks'].append(lock_data)

        except Exception as e:
            logger.error(f"ロック確認エラー: {e}")

        return locks_info

    def force_cleanup_locks(self, tracker_id: str = None) -> int:
        """強制ロッククリーンアップ"""
        cleaned_count = 0

        try:
            with self.lock:
                # 特定トラッカーのロックのみクリーンアップ
                if tracker_id:
                    pattern = f"subagent_*_{tracker_id}.lock"
                    for lock_file in self.lock_dir.glob(pattern):
                        lock_file.unlink()
                        cleaned_count += 1
                        logger.info(f"強制削除: {lock_file}")

                    # グローバルロック確認
                    if self.lock_file.exists():
                        global_lock = self._read_lock_file(self.lock_file)
                        if global_lock and global_lock.get('tracker_id') == tracker_id:
                            self.lock_file.unlink()
                            cleaned_count += 1
                            logger.info(f"グローバルロック強制削除: {self.lock_file}")
                else:
                    # 全ロッククリーンアップ
                    for lock_file in self.lock_dir.glob("subagent_*.lock"):
                        lock_file.unlink()
                        cleaned_count += 1
                        logger.info(f"全ロック強制削除: {lock_file}")

        except Exception as e:
            logger.error(f"強制クリーンアップエラー: {e}")

        logger.info(f"強制クリーンアップ完了: {cleaned_count}ファイル削除")
        return cleaned_count

    def is_duplicate_execution_risk(self, tracker_id: str, process_type: str = "extraction") -> bool:
        """二重実行リスク判定"""
        try:
            # アクティブロック確認
            specific_lock_file = self.lock_dir / f"subagent_{process_type}_{tracker_id}.lock"
            if specific_lock_file.exists() and self._is_lock_valid(specific_lock_file):
                return True

            # グローバルロック確認
            if self.lock_file.exists() and self._is_lock_valid(self.lock_file):
                global_lock = self._read_lock_file(self.lock_file)
                if (global_lock and
                    global_lock.get('tracker_id') == tracker_id and
                    global_lock.get('process_type') == process_type):
                    return True

            # プロセス確認（追加の安全チェック）
            return self._check_running_subagent_process(tracker_id)

        except Exception as e:
            logger.error(f"二重実行リスク判定エラー: {e}")
            return True  # エラー時は安全側に倒す

    def get_lock_owner_info(self, tracker_id: str, process_type: str = "extraction") -> Optional[Dict[str, Any]]:
        """ロック所有者情報取得"""
        try:
            specific_lock_file = self.lock_dir / f"subagent_{process_type}_{tracker_id}.lock"

            if specific_lock_file.exists():
                lock_data = self._read_lock_file(specific_lock_file)
                if lock_data and self._is_lock_valid(specific_lock_file):
                    return lock_data

            return None

        except Exception as e:
            logger.error(f"ロック所有者情報取得エラー: {e}")
            return None

    def _is_lock_valid(self, lock_file: Path) -> bool:
        """ロックファイル有効性確認"""
        try:
            if not lock_file.exists():
                return False

            lock_data = self._read_lock_file(lock_file)
            if not lock_data:
                return False

            # PID確認
            pid = lock_data.get('pid')
            if not pid:
                return False

            # プロセス生存確認
            if not self._is_process_alive(pid):
                return False

            # 作成時刻確認（24時間以上古い場合は無効）
            created_at = lock_data.get('created_at')
            if created_at:
                try:
                    created_time = datetime.fromisoformat(created_at)
                    age = datetime.now() - created_time
                    if age.total_seconds() > 86400:  # 24時間
                        logger.warning(f"ロック期限切れ: {lock_file} (経過: {age})")
                        return False
                except:
                    return False

            return True

        except Exception as e:
            logger.error(f"ロック有効性確認エラー: {lock_file} - {e}")
            return False

    def _is_process_alive(self, pid: int) -> bool:
        """プロセス生存確認"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False

    def _read_lock_file(self, lock_file: Path) -> Optional[Dict[str, Any]]:
        """ロックファイル読み込み"""
        try:
            with open(lock_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"ロックファイル読み込みエラー: {lock_file} - {e}")
            return None

    def _cleanup_stale_locks(self):
        """古いロックファイルクリーンアップ"""
        try:
            cleaned_count = 0
            for lock_file in self.lock_dir.glob("subagent_*.lock"):
                if not self._is_lock_valid(lock_file):
                    lock_file.unlink()
                    cleaned_count += 1
                    logger.info(f"古いロック削除: {lock_file}")

            if cleaned_count > 0:
                logger.info(f"初期クリーンアップ完了: {cleaned_count}ファイル削除")

        except Exception as e:
            logger.error(f"初期クリーンアップエラー: {e}")

    def _check_running_subagent_process(self, tracker_id: str) -> bool:
        """実行中SubAgentプロセス確認"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if not cmdline:
                        continue

                    # extract_character.pyプロセス確認
                    if any('extract_character.py' in cmd for cmd in cmdline):
                        if tracker_id in ' '.join(cmdline):
                            logger.warning(f"実行中SubAgent検出: PID {proc.info['pid']}, CMD: {' '.join(cmdline[:3])}")
                            return True

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return False

        except Exception as e:
            logger.error(f"実行中プロセス確認エラー: {e}")
            return True  # エラー時は安全側に倒す


# グローバルインスタンス
_lock_manager = None


def get_lock_manager() -> SubAgentLockManager:
    """SubAgent二重起動防止システムのグローバルインスタンスを取得"""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = SubAgentLockManager()
    return _lock_manager


def create_execution_context(tracker_id: str, process_type: str = "extraction"):
    """実行コンテキスト作成（with文での使用）"""
    class ExecutionContext:
        def __init__(self, tracker_id: str, process_type: str):
            self.tracker_id = tracker_id
            self.process_type = process_type
            self.lock_manager = get_lock_manager()
            self.acquired = False

        def __enter__(self):
            self.acquired = self.lock_manager.acquire_lock(self.tracker_id, self.process_type)
            if not self.acquired:
                raise RuntimeError(f"ロック取得失敗: {self.tracker_id}/{self.process_type} - 二重実行防止")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.acquired:
                self.lock_manager.release_lock(self.tracker_id, self.process_type)

    return ExecutionContext(tracker_id, process_type)


if __name__ == "__main__":
    # テスト実行
    lock_manager = SubAgentLockManager()

    test_tracker = "TEST-001"

    # ロック情報確認
    locks = lock_manager.check_existing_locks()
    print(f"既存ロック状況: {locks}")

    # ロック取得テスト
    print(f"\nロック取得テスト: {test_tracker}")
    success1 = lock_manager.acquire_lock(test_tracker, "extraction")
    print(f"初回取得: {success1}")

    # 二重取得テスト
    success2 = lock_manager.acquire_lock(test_tracker, "extraction")
    print(f"二重取得: {success2}")

    # 二重実行リスク確認
    risk = lock_manager.is_duplicate_execution_risk(test_tracker, "extraction")
    print(f"二重実行リスク: {risk}")

    # ロック解放
    success3 = lock_manager.release_lock(test_tracker, "extraction")
    print(f"ロック解放: {success3}")

    # コンテキスト使用例
    print(f"\nコンテキスト使用例:")
    try:
        with create_execution_context(test_tracker, "extraction"):
            print("実行中...")
            time.sleep(1)
        print("正常終了")
    except RuntimeError as e:
        print(f"実行エラー: {e}")