#!/usr/bin/env python3
"""
P1-021: ソースコード肥大化解決システム - 週次実行スクリプト

【概要】
プロジェクト内の不要なファイル・肥大化要因を週1回自動検出・整理する統合システム

【実行方針】
- ログファイル肥大化: 7日以上の古いログ削除・圧縮
- 一時ファイル除去: *.tmp, *.cache, __pycache__等の一時ファイル
- 画像ファイル管理: プロジェクトルート直下の画像を適切な場所に移動
- ワークスペース整理: 古い実験・テストディレクトリの整理
- Git最適化: .git/objects最適化、unreachable objects除去

【使用方法】
python tools/scripts/weekly_cleanup.py [--dry-run] [--verbose] [--force]
"""

import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートを取得（tools/scripts/から2レベル上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WeeklyCleanupManager:
    """週次クリーンアップ統合管理システム"""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.project_root = PROJECT_ROOT
        self.cleanup_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tasks": [],
            "files_removed": 0,
            "bytes_freed": 0,
            "errors": [],
        }

        # 除外パターン（触ってはいけないファイル・ディレクトリ）
        self.exclude_patterns = {
            "critical_files": [
                ".git/*",
                "sam_vit_*.pth",  # SAMモデルファイル
                "yolov8*.pt",  # YOLOモデルファイル
                "*.safetensors",  # LoRAモデル
                "requirements.txt",
                "setup.py",
                "pyproject.toml",
            ],
            "critical_dirs": [
                ".git",
                "sam-env",
                "venv",
                "node_modules",
                "core/segment_anything",  # Meta原本実装
                "tests/fixtures",  # テストデータ
            ],
        }

        # クリーンアップ対象パターン
        self.cleanup_targets = {
            "log_files": [
                "logs/*.log",
                "logs/*/*.log",
                "*.log",
                "workspace/*/*.log",
                "tracker-workspace/*/*.log",
            ],
            "temp_files": [
                "**/__pycache__",
                "**/*.pyc",
                "**/*.pyo",
                "**/*.tmp",
                "**/*.cache",
                "**/.*cache*",
                "tmp/*",
                "/tmp/extracted_*",
                "/tmp/p1_*",
                "/tmp/test_*",
            ],
            "build_artifacts": [
                "build/*",
                "dist/*",
                "*.egg-info",
                ".coverage",
                "htmlcov/*",
                ".pytest_cache/*",
                ".mypy_cache/*",
            ],
            "image_files_in_root": ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"],
        }

    def run_full_cleanup(self) -> Dict:
        """完全クリーンアップ実行"""
        logger.info("🧹 P1-021: 週次ソースコード肥大化解決開始")

        if self.dry_run:
            logger.info("🔍 DRY-RUN MODE: 実際の削除は行いません")

        # 1. ログファイル整理（最優先）
        self._cleanup_log_files()

        # 2. 一時ファイル・キャッシュ除去
        self._cleanup_temp_files()

        # 3. ビルド成果物除去
        self._cleanup_build_artifacts()

        # 4. 画像ファイル整理（プロジェクトルート直下）
        self._organize_misplaced_images()

        # 5. ワークスペース整理
        self._cleanup_workspace_directories()

        # 6. Git最適化
        self._optimize_git_repository()

        # 7. レポート生成
        self._generate_cleanup_report()

        return self.cleanup_report

    def _cleanup_log_files(self):
        """ログファイル整理・古いログの削除"""
        task_info = {"name": "log_cleanup", "files_processed": 0, "bytes_freed": 0, "actions": []}

        logger.info("📋 ログファイル整理開始...")

        # 7日以上前のログを削除対象とする
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)

        for pattern in self.cleanup_targets["log_files"]:
            for log_file in self.project_root.glob(pattern):
                if not log_file.is_file():
                    continue

                # ファイル年齢チェック
                file_mtime = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                file_size = log_file.stat().st_size

                if file_mtime < cutoff_date and file_size > 0:
                    action = f"削除: {log_file.relative_to(self.project_root)} ({self._format_bytes(file_size)})"
                    task_info["actions"].append(action)
                    task_info["bytes_freed"] += file_size
                    task_info["files_processed"] += 1

                    if self.verbose:
                        logger.info(f"  📄 {action}")

                    if not self.dry_run:
                        try:
                            log_file.unlink()
                        except OSError as e:
                            error_msg = f"ログ削除エラー: {log_file} - {e}"
                            self.cleanup_report["errors"].append(error_msg)
                            logger.error(error_msg)

        self.cleanup_report["tasks"].append(task_info)
        logger.info(
            f"✅ ログファイル整理完了: {task_info['files_processed']}ファイル, "
            f"{self._format_bytes(task_info['bytes_freed'])} 解放"
        )

    def _cleanup_temp_files(self):
        """一時ファイル・キャッシュファイル除去"""
        task_info = {"name": "temp_cleanup", "files_processed": 0, "bytes_freed": 0, "actions": []}

        logger.info("🗂️ 一時ファイル除去開始...")

        for pattern in self.cleanup_targets["temp_files"]:
            for temp_path in self.project_root.glob(pattern):
                if self._is_excluded_path(temp_path):
                    continue

                if temp_path.is_file():
                    file_size = temp_path.stat().st_size
                    task_info["files_processed"] += 1
                    task_info["bytes_freed"] += file_size

                    action = f"削除: {temp_path.relative_to(self.project_root)}"
                    task_info["actions"].append(action)

                    if self.verbose:
                        logger.info(f"  🗑️ {action}")

                    if not self.dry_run:
                        try:
                            temp_path.unlink()
                        except OSError as e:
                            error_msg = f"一時ファイル削除エラー: {temp_path} - {e}"
                            self.cleanup_report["errors"].append(error_msg)

                elif temp_path.is_dir():
                    # ディレクトリサイズ計算
                    dir_size = self._calculate_dir_size(temp_path)
                    task_info["files_processed"] += 1
                    task_info["bytes_freed"] += dir_size

                    action = f"ディレクトリ削除: {temp_path.relative_to(self.project_root)}"
                    task_info["actions"].append(action)

                    if self.verbose:
                        logger.info(f"  📁 {action}")

                    if not self.dry_run:
                        try:
                            shutil.rmtree(temp_path)
                        except OSError as e:
                            error_msg = f"一時ディレクトリ削除エラー: {temp_path} - {e}"
                            self.cleanup_report["errors"].append(error_msg)

        self.cleanup_report["tasks"].append(task_info)
        logger.info(
            f"✅ 一時ファイル除去完了: {task_info['files_processed']}項目, "
            f"{self._format_bytes(task_info['bytes_freed'])} 解放"
        )

    def _cleanup_build_artifacts(self):
        """ビルド成果物・開発成果物の除去"""
        task_info = {"name": "build_cleanup", "files_processed": 0, "bytes_freed": 0, "actions": []}

        logger.info("🔨 ビルド成果物除去開始...")

        for pattern in self.cleanup_targets["build_artifacts"]:
            for artifact_path in self.project_root.glob(pattern):
                if self._is_excluded_path(artifact_path):
                    continue

                if artifact_path.is_file():
                    file_size = artifact_path.stat().st_size
                    task_info["files_processed"] += 1
                    task_info["bytes_freed"] += file_size

                    if not self.dry_run:
                        try:
                            artifact_path.unlink()
                            task_info["actions"].append(
                                f"削除: {artifact_path.relative_to(self.project_root)}"
                            )
                        except OSError as e:
                            error_msg = f"ビルド成果物削除エラー: {artifact_path} - {e}"
                            self.cleanup_report["errors"].append(error_msg)

                elif artifact_path.is_dir():
                    dir_size = self._calculate_dir_size(artifact_path)
                    task_info["files_processed"] += 1
                    task_info["bytes_freed"] += dir_size

                    if not self.dry_run:
                        try:
                            shutil.rmtree(artifact_path)
                            task_info["actions"].append(
                                f"ディレクトリ削除: {artifact_path.relative_to(self.project_root)}"
                            )
                        except OSError as e:
                            error_msg = f"ビルド成果物ディレクトリ削除エラー: {artifact_path} - {e}"
                            self.cleanup_report["errors"].append(error_msg)

        self.cleanup_report["tasks"].append(task_info)
        logger.info(
            f"✅ ビルド成果物除去完了: {task_info['files_processed']}項目, "
            f"{self._format_bytes(task_info['bytes_freed'])} 解放"
        )

    def _organize_misplaced_images(self):
        """プロジェクトルート直下の画像ファイル整理"""
        task_info = {
            "name": "image_organization",
            "files_processed": 0,
            "bytes_freed": 0,
            "actions": [],
        }

        logger.info("🖼️ 画像ファイル整理開始...")

        # 移動先ディレクトリ
        misplaced_dir = self.project_root / "deprecated" / "misplaced_images"

        for pattern in self.cleanup_targets["image_files_in_root"]:
            for image_file in self.project_root.glob(pattern):
                # プロジェクトルート直下の画像のみ対象
                if image_file.parent != self.project_root:
                    continue

                if not image_file.is_file():
                    continue

                file_size = image_file.stat().st_size
                task_info["files_processed"] += 1

                # 移動処理
                if not self.dry_run:
                    try:
                        misplaced_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = misplaced_dir / image_file.name

                        # 重複回避
                        counter = 1
                        while dest_path.exists():
                            stem = image_file.stem
                            suffix = image_file.suffix
                            dest_path = misplaced_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                        shutil.move(str(image_file), str(dest_path))
                        action = f"移動: {image_file.name} → deprecated/misplaced_images/"
                        task_info["actions"].append(action)

                        if self.verbose:
                            logger.info(f"  📸 {action}")

                    except OSError as e:
                        error_msg = f"画像ファイル移動エラー: {image_file} - {e}"
                        self.cleanup_report["errors"].append(error_msg)
                        logger.error(error_msg)
                else:
                    action = f"移動予定: {image_file.name} → deprecated/misplaced_images/"
                    task_info["actions"].append(action)
                    if self.verbose:
                        logger.info(f"  📸 {action}")

        self.cleanup_report["tasks"].append(task_info)
        logger.info(f"✅ 画像ファイル整理完了: {task_info['files_processed']}ファイル移動")

    def _cleanup_workspace_directories(self):
        """ワークスペース・実験ディレクトリの整理"""
        task_info = {
            "name": "workspace_cleanup",
            "files_processed": 0,
            "bytes_freed": 0,
            "actions": [],
        }

        logger.info("📂 ワークスペース整理開始...")

        # 30日以上古い実験ディレクトリを除去対象とする
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)

        workspace_dirs = [
            self.project_root / "workspace",
            self.project_root / "experiments",
            self.project_root / "test_outputs",
            self.project_root / "temp_results",
        ]

        for workspace_dir in workspace_dirs:
            if not workspace_dir.exists():
                continue

            for sub_dir in workspace_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                # ディレクトリの最終更新時刻をチェック
                dir_mtime = datetime.datetime.fromtimestamp(sub_dir.stat().st_mtime)

                if dir_mtime < cutoff_date:
                    dir_size = self._calculate_dir_size(sub_dir)
                    task_info["files_processed"] += 1
                    task_info["bytes_freed"] += dir_size

                    action = f"ワークスペース削除: {sub_dir.relative_to(self.project_root)}"
                    task_info["actions"].append(action)

                    if self.verbose:
                        logger.info(f"  📁 {action}")

                    if not self.dry_run:
                        try:
                            shutil.rmtree(sub_dir)
                        except OSError as e:
                            error_msg = f"ワークスペース削除エラー: {sub_dir} - {e}"
                            self.cleanup_report["errors"].append(error_msg)
                            logger.error(error_msg)

        self.cleanup_report["tasks"].append(task_info)
        logger.info(
            f"✅ ワークスペース整理完了: {task_info['files_processed']}ディレクトリ, "
            f"{self._format_bytes(task_info['bytes_freed'])} 解放"
        )

    def _optimize_git_repository(self):
        """Git リポジトリ最適化"""
        task_info = {
            "name": "git_optimization",
            "files_processed": 0,
            "bytes_freed": 0,
            "actions": [],
        }

        logger.info("🔧 Git リポジトリ最適化開始...")

        git_commands = [
            (["git", "gc", "--prune=now"], "ガベージコレクション実行"),
            (["git", "prune"], "到達不能オブジェクト除去"),
            (["git", "repack", "-ad"], "パックファイル最適化"),
        ]

        for cmd, description in git_commands:
            if not self.dry_run:
                try:
                    result = subprocess.run(
                        cmd, cwd=self.project_root, capture_output=True, text=True, timeout=300
                    )

                    if result.returncode == 0:
                        task_info["actions"].append(f"成功: {description}")
                        if self.verbose:
                            logger.info(f"  ✅ {description}")
                    else:
                        error_msg = f"Git最適化エラー ({description}): {result.stderr}"
                        task_info["actions"].append(f"失敗: {description}")
                        self.cleanup_report["errors"].append(error_msg)
                        logger.error(error_msg)

                except subprocess.TimeoutExpired:
                    error_msg = f"Git最適化タイムアウト: {description}"
                    self.cleanup_report["errors"].append(error_msg)
                    logger.error(error_msg)
                except subprocess.SubprocessError as e:
                    error_msg = f"Git最適化実行エラー ({description}): {e}"
                    self.cleanup_report["errors"].append(error_msg)
                    logger.error(error_msg)
            else:
                task_info["actions"].append(f"実行予定: {description}")
                if self.verbose:
                    logger.info(f"  🔧 実行予定: {description}")

        self.cleanup_report["tasks"].append(task_info)
        logger.info("✅ Git リポジトリ最適化完了")

    def _generate_cleanup_report(self):
        """クリーンアップレポート生成・保存"""
        logger.info("📊 クリーンアップレポート生成中...")

        # 統計集計
        total_files = sum(task["files_processed"] for task in self.cleanup_report["tasks"])
        total_bytes = sum(task["bytes_freed"] for task in self.cleanup_report["tasks"])

        self.cleanup_report["files_removed"] = total_files
        self.cleanup_report["bytes_freed"] = total_bytes

        # レポートファイル保存
        report_dir = self.project_root / "logs" / "weekly_cleanup"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"cleanup_report_{timestamp}.json"

        if not self.dry_run:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(self.cleanup_report, f, indent=2, ensure_ascii=False)

            logger.info(f"📄 レポート保存: {report_file}")

        # サマリー表示
        logger.info("=" * 60)
        logger.info("🧹 P1-021 週次クリーンアップ完了サマリー")
        logger.info("=" * 60)
        logger.info(f"📁 削除ファイル数: {total_files:,}")
        logger.info(f"💾 解放容量: {self._format_bytes(total_bytes)}")
        logger.info(f"⚠️ エラー数: {len(self.cleanup_report['errors'])}")

        for task in self.cleanup_report["tasks"]:
            task_name = task["name"].replace("_", " ").title()
            logger.info(
                f"  📋 {task_name}: {task['files_processed']}項目, "
                f"{self._format_bytes(task['bytes_freed'])}"
            )

        if self.cleanup_report["errors"]:
            logger.warning("⚠️ エラー詳細:")
            for error in self.cleanup_report["errors"]:
                logger.warning(f"  - {error}")

        logger.info("=" * 60)

        return report_file

    def _is_excluded_path(self, path: Path) -> bool:
        """除外パターンチェック"""
        path_str = str(path.relative_to(self.project_root))

        # 重要ディレクトリチェック
        for critical_dir in self.exclude_patterns["critical_dirs"]:
            if path_str.startswith(critical_dir):
                return True

        # 重要ファイルパターンチェック
        for pattern in self.exclude_patterns["critical_files"]:
            if path.match(pattern):
                return True

        return False

    def _calculate_dir_size(self, directory: Path) -> int:
        """ディレクトリサイズ計算"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    try:
                        total_size += filepath.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, FileNotFoundError):
            pass
        return total_size

    def _format_bytes(self, bytes_count: int) -> str:
        """バイト数の人間読みやすい形式変換"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} TB"


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="P1-021: ソースコード肥大化解決システム - 週次実行スクリプト")
    parser.add_argument("--dry-run", action="store_true", help="実際の削除は行わず、削除予定のファイルを表示のみ")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力を有効化")
    parser.add_argument("--force", action="store_true", help="確認なしで実行（cron用）")

    args = parser.parse_args()

    # インタラクティブ確認（--forceでない場合）
    if not args.force and not args.dry_run:
        print("🧹 P1-021: 週次ソースコード肥大化解決システム")
        print("   この操作により、古いログ・一時ファイル・キャッシュが削除されます。")

        response = input("   実行しますか？ [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("   操作をキャンセルしました。")
            return

    # クリーンアップ実行
    cleanup_manager = WeeklyCleanupManager(dry_run=args.dry_run, verbose=args.verbose)

    try:
        report = cleanup_manager.run_full_cleanup()

        if args.dry_run:
            print(
                f"\n🔍 DRY-RUN完了: {report['files_removed']}ファイル削除予定, "
                f"{cleanup_manager._format_bytes(report['bytes_freed'])}解放予定"
            )
        else:
            print(
                f"\n✅ クリーンアップ完了: {report['files_removed']}ファイル削除, "
                f"{cleanup_manager._format_bytes(report['bytes_freed'])}解放"
            )

    except Exception as e:
        logger.error(f"❌ クリーンアップ実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
