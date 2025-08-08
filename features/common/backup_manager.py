#!/usr/bin/env python3
"""
PH2-007: バックアップマネージャー
処理結果の自動バックアップ機能
"""

import glob
import json
import logging
import os
import shutil
import tarfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .backup_config import BackupConfig

logger = logging.getLogger(__name__)


class BackupManager:
    """バックアップマネージャー"""
    
    def __init__(self, workspace_path: Path, config: Optional[BackupConfig] = None):
        """
        Initialize backup manager
        
        Args:
            workspace_path: ワークスペースディレクトリパス
            config: バックアップ設定（Noneの場合はデフォルト使用）
        """
        self.workspace_path = workspace_path
        self.config = config or BackupConfig()
        self.backup_dir = self.config.get_backup_dir(workspace_path)
        self._lock = threading.Lock()
        
        # バックアップディレクトリ作成
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🗄️ PH2-007 バックアップマネージャー初期化: {self.backup_dir}")
    
    def create_backup(self, backup_name: Optional[str] = None) -> Optional[Path]:
        """
        バックアップ作成
        
        Args:
            backup_name: バックアップ名（Noneの場合は自動生成）
            
        Returns:
            作成されたバックアップファイルパス
        """
        if not self.config.enabled:
            logger.info("バックアップが無効化されています")
            return None
        
        with self._lock:
            try:
                # バックアップ名生成
                if not backup_name:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"backup_{timestamp}"
                
                # バックアップファイルパス
                if self.config.compression_enabled:
                    backup_path = self.backup_dir / f"{backup_name}.tar.gz"
                else:
                    backup_path = self.backup_dir / f"{backup_name}.tar"
                
                # バックアップ対象ファイル収集
                backup_files = self._collect_backup_targets()
                
                if not backup_files:
                    logger.warning("バックアップ対象ファイルが見つかりません")
                    return None
                
                # アーカイブ作成
                self._create_archive(backup_path, backup_files, backup_name)
                
                # メタデータ作成
                if self.config.include_metadata:
                    self._create_backup_metadata(backup_path, backup_files, backup_name)
                
                # 古いバックアップ削除
                self._cleanup_old_backups()
                
                logger.info(f"✅ バックアップ作成完了: {backup_path}")
                return backup_path
                
            except Exception as e:
                logger.error(f"❌ バックアップ作成失敗: {e}")
                return None
    
    def _collect_backup_targets(self) -> List[Path]:
        """バックアップ対象ファイル収集"""
        backup_files = []
        
        for target in self.config.backup_targets:
            if target.endswith('/'):
                # ディレクトリ対象
                dir_path = self.workspace_path / target.rstrip('/')
                if dir_path.exists() and dir_path.is_dir():
                    for file_path in dir_path.rglob('*'):
                        if file_path.is_file():
                            backup_files.append(file_path)
            else:
                # ファイルパターン対象
                for file_path in self.workspace_path.glob(target):
                    if file_path.is_file():
                        backup_files.append(file_path)
        
        # 重複除去とソート
        backup_files = sorted(list(set(backup_files)))
        
        logger.info(f"🎯 バックアップ対象ファイル数: {len(backup_files)}")
        return backup_files
    
    def _create_archive(self, backup_path: Path, files: List[Path], backup_name: str) -> None:
        """アーカイブファイル作成"""
        mode = 'w:gz' if self.config.compression_enabled else 'w'
        
        with tarfile.open(backup_path, mode) as tar:
            for file_path in files:
                # ワークスペースからの相対パス
                arcname = file_path.relative_to(self.workspace_path)
                tar.add(file_path, arcname=arcname)
        
        # ファイルサイズチェック
        file_size_mb = backup_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > self.config.max_backup_size_mb:
            logger.warning(f"⚠️ バックアップサイズが上限超過: {file_size_mb:.1f}MB > {self.config.max_backup_size_mb}MB")
        
        logger.info(f"📦 アーカイブ作成完了: {backup_path} ({file_size_mb:.1f}MB)")
    
    def _create_backup_metadata(self, backup_path: Path, files: List[Path], backup_name: str) -> None:
        """バックアップメタデータ作成"""
        metadata = {
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "workspace_path": str(self.workspace_path),
            "backup_path": str(backup_path),
            "file_count": len(files),
            "files": [str(f.relative_to(self.workspace_path)) for f in files],
            "config": self.config.to_dict(),
            "archive_size_bytes": backup_path.stat().st_size if backup_path.exists() else 0
        }
        
        metadata_path = Path(str(backup_path) + '.meta.json')
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 メタデータ作成完了: {metadata_path}")
    
    def _cleanup_old_backups(self) -> None:
        """古いバックアップファイル削除"""
        cutoff_date = self.config.get_retention_cutoff_date()
        deleted_count = 0
        
        for backup_file in self.backup_dir.glob('backup_*.tar*'):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    # メタデータファイルも削除
                    meta_file = backup_file.with_suffix(backup_file.suffix + '.meta.json')
                    if meta_file.exists():
                        meta_file.unlink()
                    
                    backup_file.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ 古いバックアップ削除: {backup_file}")
                
                except Exception as e:
                    logger.error(f"❌ バックアップ削除失敗: {backup_file} - {e}")
        
        if deleted_count > 0:
            logger.info(f"🧹 古いバックアップ削除完了: {deleted_count}個")
    
    def list_backups(self) -> List[Dict]:
        """バックアップリスト取得"""
        backups = []
        
        # .tar.gz と .tar ファイルのみを対象とする
        # Note: backup_*.tar.gz パターンが効かないため、より単純なパターンを使用
        all_files = list(self.backup_dir.iterdir())
        tar_gz_files = [f for f in all_files if f.name.endswith('.tar.gz') and not f.name.endswith('.meta.json')]
        tar_files = [f for f in all_files if f.name.endswith('.tar') and not f.name.endswith('.tar.gz') and not f.name.endswith('.meta.json')]
        
        for backup_file in sorted(tar_gz_files + tar_files):
            # メタデータファイル(.meta.json)は除外
            if backup_file.name.endswith('.meta.json'):
                continue
                
            # .tar.gz ファイルの場合、stem だけでは正しい名前が取れないため修正
            if backup_file.name.endswith('.tar.gz'):
                backup_name = backup_file.name.replace('.tar.gz', '')
            elif backup_file.name.endswith('.tar'):
                backup_name = backup_file.name.replace('.tar', '')
            else:
                continue  # 対象外のファイル
                
            metadata = self._load_backup_metadata(backup_file)
            
            backup_info = {
                "name": backup_name,
                "path": str(backup_file),
                "size_mb": backup_file.stat().st_size / (1024 * 1024),
                "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                "metadata": metadata
            }
            
            backups.append(backup_info)
        
        return backups
    
    def _load_backup_metadata(self, backup_path: Path) -> Optional[Dict]:
        """バックアップメタデータ読み込み"""
        # .tar.gz の場合のサフィックス処理を正しく行う
        meta_path = Path(str(backup_path) + '.meta.json')
        
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"メタデータ読み込み失敗: {meta_path} - {e}")
        
        return None
    
    def restore_backup(self, backup_name: str, restore_path: Optional[Path] = None) -> bool:
        """
        バックアップ復元
        
        Args:
            backup_name: バックアップ名
            restore_path: 復元先パス（Noneの場合は元の場所）
            
        Returns:
            復元成功フラグ
        """
        backup_file = None
        
        # バックアップファイル検索
        possible_files = [
            self.backup_dir / f'{backup_name}.tar.gz',
            self.backup_dir / f'{backup_name}.tar'
        ]
        
        for backup_path in possible_files:
            if backup_path.exists():
                backup_file = backup_path
                break
        
        if not backup_file:
            logger.error(f"❌ バックアップファイルが見つかりません: {backup_name}")
            return False
        
        try:
            restore_target = restore_path or self.workspace_path
            
            mode = 'r:gz' if backup_file.suffix == '.gz' else 'r'
            
            with tarfile.open(backup_file, mode) as tar:
                tar.extractall(path=restore_target)
            
            logger.info(f"✅ バックアップ復元完了: {backup_file} -> {restore_target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ バックアップ復元失敗: {e}")
            return False
    
    def get_backup_statistics(self) -> Dict:
        """バックアップ統計情報取得"""
        backups = self.list_backups()
        
        if not backups:
            return {
                "total_backups": 0,
                "total_size_mb": 0,
                "oldest_backup": None,
                "newest_backup": None
            }
        
        total_size = sum(b["size_mb"] for b in backups)
        creation_dates = [datetime.fromisoformat(b["created_at"]) for b in backups]
        
        return {
            "total_backups": len(backups),
            "total_size_mb": round(total_size, 2),
            "oldest_backup": min(creation_dates).isoformat(),
            "newest_backup": max(creation_dates).isoformat(),
            "average_size_mb": round(total_size / len(backups), 2)
        }


def create_backup_manager(workspace_path: Path, config: Optional[BackupConfig] = None) -> BackupManager:
    """バックアップマネージャー作成"""
    return BackupManager(workspace_path, config)


# バックアップフック関数（extract_character.py用）
def backup_extraction_results(workspace_path: Path, backup_name: Optional[str] = None) -> Optional[Path]:
    """
    抽出結果バックアップ（フック関数）
    
    Args:
        workspace_path: ワークスペースパス
        backup_name: バックアップ名
        
    Returns:
        バックアップファイルパス
    """
    try:
        manager = create_backup_manager(workspace_path)
        return manager.create_backup(backup_name)
    except Exception as e:
        logger.error(f"❌ 抽出結果バックアップ失敗: {e}")
        return None