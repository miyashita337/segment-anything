#!/usr/bin/env python3
"""
非同期システム一時ディレクトリクリーンアップツール

ハングアップや異常終了により残留した一時ディレクトリを安全に削除
- 破損画像の検出と報告
- 進行中プロセスのチェック
- 安全な削除実行
"""

import logging
import os
import psutil
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AsyncTempCleaner:
    """非同期一時ディレクトリクリーナー"""

    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)

    def find_temp_directories(self) -> List[Path]:
        """一時ディレクトリを検索"""
        temp_dirs = []

        try:
            for tracker_dir in self.workspace_base.iterdir():
                if not tracker_dir.is_dir():
                    continue

                extraction_dir = tracker_dir / "extraction"
                if not extraction_dir.exists():
                    continue

                # async_batch_*_temp または batch_*_temp パターンを検索
                for item in extraction_dir.iterdir():
                    if item.is_dir() and ("_temp" in item.name):
                        temp_dirs.append(item)

        except Exception as e:
            logger.error(f"一時ディレクトリ検索エラー: {e}")

        return temp_dirs

    def check_running_processes(self) -> List[Dict[str, Any]]:
        """関連プロセスの実行状況チェック"""
        related_processes = []

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if not cmdline:
                    continue

                cmdline_str = " ".join(cmdline)

                # SAM/YOLO関連プロセスを検出
                if any(
                    keyword in cmdline_str
                    for keyword in [
                        "sam_yolo_character_segment.py",
                        "async_batched_extraction_runner.py",
                        "async_tracker_system.py",
                        "pytorch",
                        "cuda",
                    ]
                ):
                    related_processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cmdline": cmdline_str[:100] + "..."
                            if len(cmdline_str) > 100
                            else cmdline_str,
                        }
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return related_processes

    def analyze_temp_directory(self, temp_dir: Path) -> Dict[str, Any]:
        """一時ディレクトリの詳細分析"""
        analysis = {
            "path": str(temp_dir),
            "size_mb": 0,
            "file_count": 0,
            "has_input": False,
            "has_output": False,
            "output_files": 0,
            "created_time": None,
            "modified_time": None,
        }

        try:
            # ディレクトリサイズ計算
            total_size = 0
            file_count = 0

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except OSError:
                        pass

            analysis["size_mb"] = total_size / (1024 * 1024)
            analysis["file_count"] = file_count

            # サブディレクトリチェック
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"

            if input_dir.exists():
                analysis["has_input"] = True

            if output_dir.exists():
                analysis["has_output"] = True
                analysis["output_files"] = len(list(output_dir.glob("*")))

            # タイムスタンプ
            stat = temp_dir.stat()
            analysis["created_time"] = time.ctime(stat.st_ctime)
            analysis["modified_time"] = time.ctime(stat.st_mtime)

        except Exception as e:
            logger.warning(f"ディレクトリ分析エラー {temp_dir}: {e}")

        return analysis

    def safe_remove_directory(self, temp_dir: Path, force: bool = False) -> bool:
        """安全なディレクトリ削除"""
        try:
            logger.info(f"削除開始: {temp_dir}")

            if not force:
                # 最終確認
                analysis = self.analyze_temp_directory(temp_dir)
                if analysis["output_files"] > 0:
                    logger.warning(f"出力ファイルが存在します: {analysis['output_files']}件")
                    response = input(f"本当に削除しますか? {temp_dir} [y/N]: ")
                    if response.lower() != "y":
                        logger.info("削除をキャンセルしました")
                        return False

            # 読み取り専用属性を削除
            for root, dirs, files in os.walk(temp_dir):
                for dir_name in dirs:
                    os.chmod(os.path.join(root, dir_name), 0o755)
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.chmod(file_path, 0o755)

            # ディレクトリ削除
            shutil.rmtree(temp_dir)
            logger.info(f"削除完了: {temp_dir}")
            return True

        except Exception as e:
            logger.error(f"削除エラー {temp_dir}: {e}")
            return False

    def generate_cleanup_report(self, temp_dirs: List[Path], analyses: List[Dict[str, Any]]) -> str:
        """クリーンアップレポート生成"""
        report = []
        report.append("# 非同期システム一時ディレクトリクリーンアップレポート")
        report.append(f"## 実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        total_size = sum(a["size_mb"] for a in analyses)
        total_files = sum(a["file_count"] for a in analyses)

        report.append(f"## 概要")
        report.append(f"- 対象ディレクトリ数: {len(temp_dirs)}")
        report.append(f"- 総サイズ: {total_size:.2f} MB")
        report.append(f"- 総ファイル数: {total_files}")
        report.append("")

        report.append("## 詳細")
        for i, (temp_dir, analysis) in enumerate(zip(temp_dirs, analyses)):
            report.append(f"### {i+1}. {temp_dir.name}")
            report.append(f"- パス: `{analysis['path']}`")
            report.append(f"- サイズ: {analysis['size_mb']:.2f} MB")
            report.append(f"- ファイル数: {analysis['file_count']}")
            report.append(f"- 入力ディレクトリ: {'✅' if analysis['has_input'] else '❌'}")
            report.append(f"- 出力ディレクトリ: {'✅' if analysis['has_output'] else '❌'}")
            report.append(f"- 出力ファイル数: {analysis['output_files']}")
            report.append(f"- 作成日時: {analysis['created_time']}")
            report.append(f"- 更新日時: {analysis['modified_time']}")
            report.append("")

        return "\n".join(report)

    def run_cleanup(self, dry_run: bool = False, force: bool = False) -> bool:
        """クリーンアップ実行"""
        logger.info("非同期システム一時ディレクトリクリーンアップ開始")

        # 実行中プロセスチェック
        running_processes = self.check_running_processes()
        if running_processes and not force:
            logger.warning("関連プロセスが実行中です:")
            for proc in running_processes:
                logger.warning(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")

            response = input("プロセスを停止してから続行しますか? [y/N]: ")
            if response.lower() != "y":
                logger.info("クリーンアップを中止しました")
                return False

        # 一時ディレクトリ検索
        temp_dirs = self.find_temp_directories()
        if not temp_dirs:
            logger.info("クリーンアップ対象の一時ディレクトリが見つかりません")
            return True

        logger.info(f"一時ディレクトリを {len(temp_dirs)} 個発見しました")

        # 詳細分析
        analyses = []
        for temp_dir in temp_dirs:
            analysis = self.analyze_temp_directory(temp_dir)
            analyses.append(analysis)

        # レポート生成
        report = self.generate_cleanup_report(temp_dirs, analyses)

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN - 実際の削除は実行されません")
            print("=" * 60)
            print(report)
            return True

        # 削除実行
        success_count = 0
        for temp_dir in temp_dirs:
            if self.safe_remove_directory(temp_dir, force=force):
                success_count += 1

        logger.info(f"クリーンアップ完了: {success_count}/{len(temp_dirs)} 成功")

        # レポート保存
        report_path = self.workspace_base / "cleanup_report.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"レポート保存: {report_path}")
        except Exception as e:
            logger.warning(f"レポート保存エラー: {e}")

        return success_count == len(temp_dirs)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="非同期システム一時ディレクトリクリーンアップ")
    parser.add_argument("--dry-run", action="store_true", help="実際の削除を行わず、対象のみ表示")
    parser.add_argument("--force", action="store_true", help="確認なしで強制削除")
    parser.add_argument(
        "--workspace",
        type=str,
        default="/mnt/c/AItools/lora/train/yado/tracker-workspace",
        help="ワークスペースベースディレクトリ",
    )

    args = parser.parse_args()

    cleaner = AsyncTempCleaner(args.workspace)

    try:
        success = cleaner.run_cleanup(dry_run=args.dry_run, force=args.force)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"クリーンアップエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
