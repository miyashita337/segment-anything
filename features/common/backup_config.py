#!/usr/bin/env python3
"""
PH2-007: バックアップ設定管理
処理結果の自動バックアップ機能設定
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


@dataclass
class BackupConfig:
    """バックアップ設定"""

    # 基本設定
    enabled: bool = True
    retention_days: int = 7
    max_backup_size_mb: int = 1000

    # バックアップ対象
    backup_targets: List[str] = None

    # 圧縮設定
    compression_enabled: bool = True
    compression_level: int = 6  # 0-9

    # 保存場所
    backup_base_dir: Optional[str] = None

    # メタデータ
    include_metadata: bool = True

    def __post_init__(self):
        """デフォルト値設定"""
        if self.backup_targets is None:
            self.backup_targets = [
                "extraction/",
                "quality/",
                "dashboard/",
                "tests/",
                "*.log",
                "*.json",
            ]

    @classmethod
    def from_dict(cls, data: dict) -> "BackupConfig":
        """辞書からインスタンス作成"""
        return cls(**data)

    def to_dict(self) -> dict:
        """辞書に変換"""
        return asdict(self)

    @classmethod
    def load_from_file(cls, config_path: Path) -> "BackupConfig":
        """設定ファイルから読み込み"""
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            # デフォルト設定で新規作成
            config = cls()
            config.save_to_file(config_path)
            return config

    def save_to_file(self, config_path: Path) -> None:
        """設定ファイルに保存"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_backup_dir(self, workspace_path: Path) -> Path:
        """バックアップディレクトリ取得"""
        if self.backup_base_dir:
            return Path(self.backup_base_dir)
        else:
            return workspace_path / "backups"

    def get_retention_cutoff_date(self) -> datetime:
        """保持期限日取得"""
        return datetime.now() - timedelta(days=self.retention_days)

    def validate(self) -> List[str]:
        """設定値検証"""
        errors = []

        if self.retention_days < 1:
            errors.append("retention_days must be >= 1")

        if self.max_backup_size_mb < 10:
            errors.append("max_backup_size_mb must be >= 10")

        if not (0 <= self.compression_level <= 9):
            errors.append("compression_level must be 0-9")

        if not self.backup_targets:
            errors.append("backup_targets cannot be empty")

        return errors


def get_default_backup_config() -> BackupConfig:
    """デフォルトバックアップ設定取得"""
    return BackupConfig(
        enabled=True,
        retention_days=7,
        max_backup_size_mb=1000,
        backup_targets=[
            "extraction/",
            "quality/",
            "dashboard/",
            "tests/",
            "*.log",
            "*.json",
            "*.md",
        ],
        compression_enabled=True,
        compression_level=6,
        include_metadata=True,
    )


def create_backup_config_for_tracker(tracker_id: str, workspace_base: Path) -> BackupConfig:
    """トラッカー専用バックアップ設定作成"""
    config = get_default_backup_config()

    # トラッカー固有の設定調整
    if tracker_id.startswith("PH3-"):
        # Phase 3は設定ファイルも含める
        config.backup_targets.extend(["*.yaml", "*.toml", "config/"])
    elif tracker_id.startswith("PH2-"):
        # Phase 2はログを重視
        config.backup_targets.extend(["logs/", "*.txt"])

    return config
