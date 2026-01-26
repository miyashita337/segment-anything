#!/usr/bin/env python3
"""
P1-013: 差分処理最適化システム
変更箇所のみの再処理による効率的な抽出パイプライン
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.workspace_config import WorkspaceConfig
from features.common.output_path_manager import OutputCategory, OutputPathManager

logger = logging.getLogger(__name__)


@dataclass
class FileChangeInfo:
    """ファイル変更情報"""

    file_path: str
    last_modified: float
    file_hash: str
    file_size: int
    change_type: str  # 'added', 'modified', 'deleted'
    dependencies: List[str]  # 依存ファイルリスト


@dataclass
class ProcessingCache:
    """処理キャッシュ情報"""

    input_file: str
    input_hash: str
    output_files: List[str]
    processing_time: float
    success: bool
    timestamp: float
    processing_params: Dict[str, Any]


@dataclass
class DifferentialReport:
    """差分処理レポート"""

    total_files: int
    changed_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    processing_time: float
    cache_hits: int
    cache_misses: int
    change_details: List[FileChangeInfo]


class DifferentialProcessor:
    """差分処理最適化システム"""

    def __init__(self, tracker_id: str, input_dir: str, enable_cache: bool = True):
        """
        初期化

        Args:
            tracker_id: トラッカーID
            input_dir: 入力ディレクトリ
            enable_cache: キャッシュ機能有効フラグ
        """
        self.tracker_id = tracker_id
        self.input_dir = Path(input_dir)
        self.enable_cache = enable_cache

        # パス管理
        self.path_manager = OutputPathManager(tracker_id)
        self.cache_dir = self.path_manager.ensure_output_dir(OutputCategory.TEMP)

        # キャッシュファイル
        self.hash_cache_file = self.cache_dir / f"{tracker_id}_file_hashes.json"
        self.processing_cache_file = self.cache_dir / f"{tracker_id}_processing_cache.json"
        self.dependency_file = self.cache_dir / f"{tracker_id}_dependencies.json"

        # データ構造
        self.file_hashes: Dict[str, FileChangeInfo] = {}
        self.processing_cache: Dict[str, ProcessingCache] = {}
        self.dependency_map: Dict[str, Set[str]] = {}

        # 統計
        self.stats = {
            "total_files": 0,
            "changed_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # 既存キャッシュ読み込み
        self._load_caches()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        ファイルハッシュ計算

        Args:
            file_path: ファイルパス

        Returns:
            SHA256ハッシュ値
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                # 大ファイル対応：チャンクサイズで読み込み
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"ハッシュ計算エラー {file_path}: {e}")
            return ""

    def _get_file_info(self, file_path: Path) -> FileChangeInfo:
        """
        ファイル情報取得

        Args:
            file_path: ファイルパス

        Returns:
            ファイル変更情報
        """
        try:
            stat = file_path.stat()
            file_hash = self._calculate_file_hash(file_path)

            return FileChangeInfo(
                file_path=str(file_path),
                last_modified=stat.st_mtime,
                file_hash=file_hash,
                file_size=stat.st_size,
                change_type="unknown",
                dependencies=[],
            )
        except Exception as e:
            logger.error(f"ファイル情報取得エラー {file_path}: {e}")
            return None

    def _load_caches(self):
        """既存キャッシュ読み込み"""
        try:
            # ファイルハッシュキャッシュ
            if self.hash_cache_file.exists():
                with open(self.hash_cache_file, "r", encoding="utf-8") as f:
                    hash_data = json.load(f)
                    self.file_hashes = {
                        path: FileChangeInfo(**info) for path, info in hash_data.items()
                    }
                logger.info(f"ハッシュキャッシュ読み込み: {len(self.file_hashes)}件")

            # 処理キャッシュ
            if self.processing_cache_file.exists():
                with open(self.processing_cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    self.processing_cache = {
                        path: ProcessingCache(**info) for path, info in cache_data.items()
                    }
                logger.info(f"処理キャッシュ読み込み: {len(self.processing_cache)}件")

            # 依存関係マップ
            if self.dependency_file.exists():
                with open(self.dependency_file, "r", encoding="utf-8") as f:
                    dep_data = json.load(f)
                    self.dependency_map = {path: set(deps) for path, deps in dep_data.items()}
                logger.info(f"依存関係マップ読み込み: {len(self.dependency_map)}件")

        except Exception as e:
            logger.error(f"キャッシュ読み込みエラー: {e}")

    def _save_caches(self):
        """キャッシュ保存"""
        try:
            # ファイルハッシュキャッシュ保存
            hash_data = {path: asdict(info) for path, info in self.file_hashes.items()}
            with open(self.hash_cache_file, "w", encoding="utf-8") as f:
                json.dump(hash_data, f, ensure_ascii=False, indent=2)

            # 処理キャッシュ保存
            cache_data = {path: asdict(info) for path, info in self.processing_cache.items()}
            with open(self.processing_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # 依存関係マップ保存
            dep_data = {path: list(deps) for path, deps in self.dependency_map.items()}
            with open(self.dependency_file, "w", encoding="utf-8") as f:
                json.dump(dep_data, f, ensure_ascii=False, indent=2)

            logger.info("✅ キャッシュ保存完了")

        except Exception as e:
            logger.error(f"キャッシュ保存エラー: {e}")

    def detect_changes(self) -> List[FileChangeInfo]:
        """
        ファイル変更検出

        Returns:
            変更されたファイルのリスト
        """
        logger.info(f"🔍 変更検出開始: {self.input_dir}")

        # 画像ファイル収集
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        current_files = []

        for ext in image_extensions:
            current_files.extend(self.input_dir.glob(f"*{ext}"))
            current_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"📁 対象ファイル数: {len(current_files)}枚")

        changes = []
        current_paths = set()

        # 現在のファイル状態チェック
        for file_path in current_files:
            current_paths.add(str(file_path))
            file_info = self._get_file_info(file_path)

            if file_info is None:
                logger.error(f"❌ ファイル情報取得失敗: {file_path.name}")
                continue

            old_info = self.file_hashes.get(str(file_path))

            if old_info is None:
                # 新規ファイル
                file_info.change_type = "added"
                changes.append(file_info)
                logger.debug(f"➕ 新規: {file_path.name}")

            elif old_info.file_hash != file_info.file_hash:
                # 変更ファイル
                file_info.change_type = "modified"
                changes.append(file_info)
                logger.debug(f"📝 変更: {file_path.name}")

            else:
                # 変更なし
                logger.debug(f"✅ 変更なし: {file_path.name}")

            # 現在の状態を記録
            self.file_hashes[str(file_path)] = file_info

        # 削除されたファイル検出
        old_paths = set(self.file_hashes.keys())
        deleted_paths = old_paths - current_paths

        for deleted_path in deleted_paths:
            old_info = self.file_hashes[deleted_path]
            deleted_info = FileChangeInfo(
                file_path=deleted_path,
                last_modified=0.0,
                file_hash="",
                file_size=0,
                change_type="deleted",
                dependencies=[],
            )
            changes.append(deleted_info)
            logger.debug(f"🗑️ 削除: {Path(deleted_path).name}")

            # 削除されたファイルはキャッシュからも削除
            del self.file_hashes[deleted_path]
            if deleted_path in self.processing_cache:
                del self.processing_cache[deleted_path]

        self.stats["total_files"] = len(current_files)
        self.stats["changed_files"] = len(changes)

        logger.info(f"📊 変更検出完了: {len(changes)}/{len(current_files)}件変更")

        return changes

    def _get_dependencies(self, file_path: str) -> Set[str]:
        """
        ファイル依存関係取得

        Args:
            file_path: ファイルパス

        Returns:
            依存ファイルパスのセット
        """
        # 画像ファイルの場合、設定ファイルや前処理スクリプトとの依存関係を考慮
        dependencies = set()

        # 同ディレクトリの設定ファイル
        file_dir = Path(file_path).parent
        for config_file in ["config.json", "settings.yaml", "processing.conf"]:
            config_path = file_dir / config_file
            if config_path.exists():
                dependencies.add(str(config_path))

        return dependencies

    def _should_process_file(self, file_path: str, change_info: FileChangeInfo) -> bool:
        """
        ファイル処理要否判定

        Args:
            file_path: ファイルパス
            change_info: 変更情報

        Returns:
            処理が必要かどうか
        """
        # 削除されたファイルは処理不要
        if change_info.change_type == "deleted":
            return False

        # キャッシュ無効の場合は常に処理
        if not self.enable_cache:
            self.stats["cache_misses"] += 1
            return True

        # 処理キャッシュ確認
        cache_entry = self.processing_cache.get(file_path)

        if cache_entry is None:
            # キャッシュ未存在
            self.stats["cache_misses"] += 1
            return True

        if cache_entry.input_hash != change_info.file_hash:
            # ハッシュ不一致（ファイル変更済み）
            self.stats["cache_misses"] += 1
            return True

        # 出力ファイル存在確認
        for output_file in cache_entry.output_files:
            if not Path(output_file).exists():
                # 出力ファイル削除済み
                self.stats["cache_misses"] += 1
                return True

        # 依存ファイル変更確認
        dependencies = self._get_dependencies(file_path)
        for dep_path in dependencies:
            if Path(dep_path).exists():
                dep_info = self._get_file_info(Path(dep_path))
                old_dep = self.file_hashes.get(dep_path)
                if old_dep is None or old_dep.file_hash != dep_info.file_hash:
                    # 依存ファイル変更済み
                    self.stats["cache_misses"] += 1
                    return True

        # キャッシュヒット
        self.stats["cache_hits"] += 1
        return False

    def _process_single_file(self, file_path: str, change_info: FileChangeInfo) -> bool:
        """
        単一ファイル処理

        Args:
            file_path: ファイルパス
            change_info: 変更情報

        Returns:
            処理成功フラグ
        """
        logger.info(f"🔧 処理開始: {Path(file_path).name}")

        start_time = time.time()

        try:
            # 実際の抽出処理実行（SAM+YOLO処理）
            output_dir = self.path_manager.ensure_output_dir(OutputCategory.EXTRACTION)

            # 実際のSAM+YOLO抽出処理実行
            # P1-011で成功したコマンドを使用し、単一ファイル処理用に調整

            # 単一ファイル用の一時ディレクトリ作成
            temp_input_dir = output_dir / "temp_input"
            temp_input_dir.mkdir(exist_ok=True)

            # ファイルを一時ディレクトリにコピー
            import shutil

            temp_file_path = temp_input_dir / Path(file_path).name
            shutil.copy2(file_path, temp_file_path)

            # SAM+YOLO抽出コマンド実行
            cmd = [
                "python3",
                "tools/core/sam_yolo_character_segment.py",
                "--mode",
                "reproduce-auto",
                "--input_dir",
                str(temp_input_dir),
                "--output_dir",
                str(output_dir),
                "--score_threshold",
                "0.07",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分タイムアウト（バックグラウンド実行用）
                cwd=Path(__file__).parent.parent.parent,
            )

            processing_time = time.time() - start_time
            success = result.returncode == 0

            # 一時ディレクトリクリーンアップ
            shutil.rmtree(temp_input_dir)

            if success:
                # 出力ファイル検索
                output_files = list(output_dir.glob(f"{Path(file_path).stem}*"))
                output_paths = [str(f) for f in output_files]

                # 処理キャッシュ更新
                if self.enable_cache:
                    cache_entry = ProcessingCache(
                        input_file=file_path,
                        input_hash=change_info.file_hash,
                        output_files=output_paths,
                        processing_time=processing_time,
                        success=True,
                        timestamp=time.time(),
                        processing_params={"method": "sam_yolo_extraction", "timeout": 120},
                    )
                    self.processing_cache[file_path] = cache_entry

                self.stats["processed_files"] += 1
                logger.info(f"✅ 処理成功: {Path(file_path).name} ({processing_time:.1f}秒)")

            else:
                self.stats["failed_files"] += 1
                logger.error(f"❌ 処理失敗: {Path(file_path).name}")
                if result.stderr:
                    logger.error(f"エラー出力: {result.stderr}")

            return success

        except Exception as e:
            self.stats["failed_files"] += 1
            logger.error(f"❌ 処理エラー {Path(file_path).name}: {e}")
            return False

    def process_changes(self, changes: List[FileChangeInfo]) -> DifferentialReport:
        """
        変更されたファイルの処理

        Args:
            changes: 変更されたファイルのリスト

        Returns:
            差分処理レポート
        """
        logger.info(f"🚀 差分処理開始: {len(changes)}件の変更を処理")

        start_time = time.time()

        for change_info in changes:
            if self._should_process_file(change_info.file_path, change_info):
                self._process_single_file(change_info.file_path, change_info)
            else:
                self.stats["skipped_files"] += 1
                logger.info(f"⏭️ スキップ: {Path(change_info.file_path).name} (キャッシュヒット)")

        processing_time = time.time() - start_time

        # レポート生成
        report = DifferentialReport(
            total_files=self.stats["total_files"],
            changed_files=self.stats["changed_files"],
            processed_files=self.stats["processed_files"],
            skipped_files=self.stats["skipped_files"],
            failed_files=self.stats["failed_files"],
            processing_time=processing_time,
            cache_hits=self.stats["cache_hits"],
            cache_misses=self.stats["cache_misses"],
            change_details=changes,
        )

        logger.info(f"✅ 差分処理完了 ({processing_time:.1f}秒)")
        logger.info(
            f"📊 処理済み: {self.stats['processed_files']}, "
            f"スキップ: {self.stats['skipped_files']}, "
            f"失敗: {self.stats['failed_files']}"
        )
        logger.info(
            f"💾 キャッシュヒット率: {self.stats['cache_hits']}/{self.stats['cache_hits'] + self.stats['cache_misses']} "
            f"({self.stats['cache_hits']/(max(self.stats['cache_hits'] + self.stats['cache_misses'], 1))*100:.1f}%)"
        )

        return report

    def save_report(self, report: DifferentialReport) -> Path:
        """
        差分処理レポート保存

        Args:
            report: 差分処理レポート

        Returns:
            レポートファイルパス
        """
        report_dir = self.path_manager.ensure_output_dir(OutputCategory.QUALITY_REPORT)

        # JSON詳細レポート
        json_report = report_dir / f"{self.tracker_id}_differential_report.json"
        report_data = {
            **asdict(report),
            "timestamp": time.time(),
            "tracker_id": self.tracker_id,
            "cache_enabled": self.enable_cache,
        }

        with open(json_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # Markdownサマリー
        md_report = report_dir / f"{self.tracker_id}_differential_summary.md"
        self._generate_markdown_report(report, md_report)

        logger.info(f"📄 差分処理レポート保存: {json_report}")

        return json_report

    def _generate_markdown_report(self, report: DifferentialReport, output_file: Path):
        """
        Markdownレポート生成

        Args:
            report: 差分処理レポート
            output_file: 出力ファイル
        """
        lines = []
        lines.append("# P1-013 差分処理最適化レポート")
        lines.append("")
        lines.append(f"**トラッカーID**: {self.tracker_id}")
        lines.append(f"**処理日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**キャッシュ機能**: {'有効' if self.enable_cache else '無効'}")
        lines.append("")

        # サマリー
        lines.append("## 📊 処理サマリー")
        lines.append(f"- **総ファイル数**: {report.total_files}枚")
        lines.append(f"- **変更ファイル数**: {report.changed_files}枚")
        lines.append(f"- **処理ファイル数**: {report.processed_files}枚")
        lines.append(f"- **スキップファイル数**: {report.skipped_files}枚")
        lines.append(f"- **失敗ファイル数**: {report.failed_files}枚")
        lines.append(f"- **処理時間**: {report.processing_time:.1f}秒")
        lines.append("")

        # キャッシュ効率
        total_cache_ops = report.cache_hits + report.cache_misses
        if total_cache_ops > 0:
            hit_rate = report.cache_hits / total_cache_ops * 100
            lines.append("## 💾 キャッシュ効率")
            lines.append(f"- **キャッシュヒット**: {report.cache_hits}件")
            lines.append(f"- **キャッシュミス**: {report.cache_misses}件")
            lines.append(f"- **ヒット率**: {hit_rate:.1f}%")
            lines.append("")

        # 変更詳細
        if report.change_details:
            lines.append("## 🔍 変更詳細")

            # 変更タイプ別集計
            change_counts = {}
            for change in report.change_details:
                change_type = change.change_type
                change_counts[change_type] = change_counts.get(change_type, 0) + 1

            for change_type, count in change_counts.items():
                emoji = {"added": "➕", "modified": "📝", "deleted": "🗑️"}.get(change_type, "🔧")
                lines.append(f"- **{emoji} {change_type}**: {count}件")

            lines.append("")

            # 主要変更ファイル（最大10件）
            lines.append("### 主要変更ファイル")
            for i, change in enumerate(report.change_details[:10], 1):
                file_name = Path(change.file_path).name
                emoji = {"added": "➕", "modified": "📝", "deleted": "🗑️"}.get(
                    change.change_type, "🔧"
                )
                lines.append(
                    f"{i}. {emoji} **{file_name}** ({change.change_type}, {change.file_size:,} bytes)"
                )

        lines.append("")
        lines.append("---")
        lines.append(
            f"*Generated by P1-013 Differential Processor at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"📄 Markdownレポート保存: {output_file}")

    def cleanup_old_cache(self, max_age_days: int = 30):
        """
        古いキャッシュエントリの削除

        Args:
            max_age_days: 保持日数
        """
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 3600)

        removed_count = 0

        # 処理キャッシュのクリーンアップ
        to_remove = []
        for file_path, cache_entry in self.processing_cache.items():
            if cache_entry.timestamp < cutoff_time:
                to_remove.append(file_path)

        for file_path in to_remove:
            del self.processing_cache[file_path]
            removed_count += 1

        if removed_count > 0:
            logger.info(f"🧹 古いキャッシュエントリ削除: {removed_count}件")

    def run_full_differential_process(self) -> DifferentialReport:
        """
        完全差分処理実行

        Returns:
            差分処理レポート
        """
        logger.info("🚀 P1-013差分処理最適化システム開始")

        # 1. 変更検出
        changes = self.detect_changes()

        # 2. 変更処理
        report = self.process_changes(changes)

        # 3. キャッシュ保存
        self._save_caches()

        # 4. レポート保存
        self.save_report(report)

        # 5. 古いキャッシュクリーンアップ
        self.cleanup_old_cache()

        logger.info(f"✅ P1-013差分処理最適化システム完了")

        return report


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-013: 差分処理最適化システム")
    parser.add_argument("--tracker-id", default="P1-013", help="トラッカーID")
    parser.add_argument("--input-dir", required=True, help="入力ディレクトリ")
    parser.add_argument("--disable-cache", action="store_true", help="キャッシュ無効化")

    args = parser.parse_args()

    # プロセッサ初期化
    processor = DifferentialProcessor(
        tracker_id=args.tracker_id, input_dir=args.input_dir, enable_cache=not args.disable_cache
    )

    # 差分処理実行
    report = processor.run_full_differential_process()

    print(f"🎉 P1-013差分処理最適化完了！")
    print(
        f"📊 処理結果: {report.processed_files}処理, {report.skipped_files}スキップ, {report.failed_files}失敗"
    )
    print(f"⏱️ 処理時間: {report.processing_time:.1f}秒")

    if report.cache_hits + report.cache_misses > 0:
        hit_rate = report.cache_hits / (report.cache_hits + report.cache_misses) * 100
        print(f"💾 キャッシュヒット率: {hit_rate:.1f}%")


if __name__ == "__main__":
    main()
