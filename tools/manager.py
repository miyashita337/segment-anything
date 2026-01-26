#!/usr/bin/env python3
"""
Tools Directory 統合管理CLI
TDR-002: 各種ツールを統合した管理インターフェース

使用例:
    # Google Sheetsタスク管理
    python tools/manager.py sheets list --priority 優先度最高
    python tools/manager.py sheets update TDR-002 --status 着手中
    python tools/manager.py sheets read --tracker-id P1-A001
    
    # バッチ処理管理
    python tools/manager.py batch list
    python tools/manager.py batch run kana08_enhanced
    
    # ツール整理・メンテナンス
    python tools/manager.py cleanup --days 30
    python tools/manager.py archive scripts/old_tool.py
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ツールインポート
try:
    from tools.core.google_sheets_updater import GoogleSheetsUpdater
    from tools.progress_tracker.data_models import PriorityLevel, TaskStatus

    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    logging.warning("Google Sheets機能が利用できません")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolsManager:
    """Tools Directory統合管理システム"""

    def __init__(self):
        """初期化"""
        self.tools_dir = Path(__file__).parent
        self.deprecated_dir = project_root / "deprecated" / "tools_archive"
        self.sheets_updater = GoogleSheetsUpdater() if SHEETS_AVAILABLE else None

    # ========== Google Sheets管理機能 ==========

    def sheets_list(
        self, priority: Optional[str] = None, status: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """タスクリスト表示"""
        if not self.sheets_updater:
            logger.error("Google Sheets機能が利用できません")
            return []

        try:
            # フィルタ条件設定
            filters = {}
            if priority:
                filters["priority"] = priority
            if status:
                filters["status"] = status

            # データ取得
            tasks = self.sheets_updater.get_filtered_data(filters, limit)

            # 表示
            print(f"\n{'='*80}")
            print(f"タスク一覧 (フィルタ: {filters or '全て'})")
            print(f"{'='*80}")

            for i, task in enumerate(tasks, 1):
                tracker_id = task[0] if len(task) > 0 else "N/A"
                priority = task[1] if len(task) > 1 else "N/A"
                status = task[2] if len(task) > 2 else "N/A"
                desc = task[5] if len(task) > 5 else "N/A"

                print(f"{i:3d}. [{tracker_id}] {status:<10} {priority:<10} {desc[:50]}")

            print(f"\n合計: {len(tasks)}件")
            return tasks

        except Exception as e:
            logger.error(f"タスクリスト取得エラー: {e}")
            return []

    def sheets_update(
        self, tracker_id: str, status: Optional[str] = None, priority: Optional[str] = None
    ) -> bool:
        """タスク更新"""
        if not self.sheets_updater:
            logger.error("Google Sheets機能が利用できません")
            return False

        try:
            # ステータス更新
            if status:
                if self.sheets_updater.update_task_status(tracker_id, status):
                    logger.info(f"✅ ステータス更新: {tracker_id} → {status}")
                else:
                    logger.error(f"❌ ステータス更新失敗: {tracker_id}")
                    return False

            # 優先度更新（将来実装）
            if priority:
                logger.warning("優先度更新は未実装です")

            return True

        except Exception as e:
            logger.error(f"タスク更新エラー: {e}")
            return False

    def sheets_read(self, tracker_id: str) -> Optional[Dict]:
        """特定タスク詳細表示"""
        if not self.sheets_updater:
            logger.error("Google Sheets機能が利用できません")
            return None

        try:
            task = self.sheets_updater.get_task_by_id(tracker_id)
            if not task:
                logger.error(f"タスクが見つかりません: {tracker_id}")
                return None

            # 詳細表示
            print(f"\n{'='*60}")
            print(f"タスク詳細: {tracker_id}")
            print(f"{'='*60}")
            print(f"優先度: {task[1] if len(task) > 1 else 'N/A'}")
            print(f"ステータス: {task[2] if len(task) > 2 else 'N/A'}")
            print(f"登録日: {task[3] if len(task) > 3 else 'N/A'}")
            print(f"更新日: {task[4] if len(task) > 4 else 'N/A'}")
            print(f"概要: {task[5] if len(task) > 5 else 'N/A'}")
            print(f"詳細: {task[6] if len(task) > 6 else 'N/A'}")

            return task

        except Exception as e:
            logger.error(f"タスク読み取りエラー: {e}")
            return None

    # ========== バッチ処理管理機能 ==========

    def batch_list(self) -> List[Path]:
        """バッチ処理スクリプト一覧"""
        batch_dir = self.tools_dir / "batch"
        if not batch_dir.exists():
            logger.error("batchディレクトリが見つかりません")
            return []

        batch_files = list(batch_dir.glob("*.py"))

        print(f"\n{'='*60}")
        print("バッチ処理スクリプト一覧")
        print(f"{'='*60}")

        for i, file in enumerate(batch_files, 1):
            # ファイル情報取得
            stat = file.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)

            print(f"{i:2d}. {file.name:<40} {size_kb:>8.1f} KB  {mtime:%Y-%m-%d %H:%M}")

        print(f"\n合計: {len(batch_files)}ファイル")
        return batch_files

    def batch_run(self, script_name: str, args: List[str] = None) -> bool:
        """バッチ処理実行"""
        batch_path = self.tools_dir / "batch" / f"{script_name}.py"
        if not batch_path.exists():
            batch_path = self.tools_dir / "batch" / script_name

        if not batch_path.exists():
            logger.error(f"バッチスクリプトが見つかりません: {script_name}")
            return False

        try:
            import subprocess

            cmd = [sys.executable, str(batch_path)]
            if args:
                cmd.extend(args)

            logger.info(f"実行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ バッチ処理完了")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ バッチ処理失敗 (exit code: {result.returncode})")
                if result.stderr:
                    print(result.stderr)
                return False

        except Exception as e:
            logger.error(f"バッチ実行エラー: {e}")
            return False

    # ========== ツール整理・メンテナンス機能 ==========

    def cleanup(self, days: int = 30) -> List[Path]:
        """古いスクリプトの自動クリーンアップ"""
        scripts_dir = self.tools_dir / "scripts"
        if not scripts_dir.exists():
            logger.warning("scriptsディレクトリが見つかりません")
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        old_files = []

        print(f"\n{'='*60}")
        print(f"{days}日以上前のスクリプト検索")
        print(f"{'='*60}")

        for file in scripts_dir.glob("*.py"):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff_date:
                old_files.append(file)
                age_days = (datetime.now() - mtime).days
                print(f"- {file.name:<40} ({age_days}日前)")

        if old_files:
            print(f"\n{len(old_files)}個の古いファイルが見つかりました")
            # 自動実行モードの場合はスキップ
            if os.environ.get("TOOLS_MANAGER_AUTO_MODE"):
                for file in old_files:
                    self.archive(file)
                logger.info(f"✅ {len(old_files)}ファイルを自動アーカイブしました")
            else:
                confirm = input("deprecated/に移動しますか? (y/N): ")

                if confirm.lower() == "y":
                    for file in old_files:
                        self.archive(file)
                    logger.info(f"✅ {len(old_files)}ファイルをアーカイブしました")
        else:
            print("古いファイルは見つかりませんでした")

        return old_files

    def archive(self, file_path: Path) -> bool:
        """ファイルをdeprecated/にアーカイブ"""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"ファイルが見つかりません: {file_path}")
            return False

        try:
            # アーカイブ先確保
            self.deprecated_dir.mkdir(parents=True, exist_ok=True)

            # 移動実行
            dest = self.deprecated_dir / file_path.name
            shutil.move(str(file_path), str(dest))

            logger.info(f"✅ アーカイブ完了: {file_path.name} → deprecated/tools_archive/")

            # README更新
            self._update_archive_readme(file_path.name)

            return True

        except Exception as e:
            logger.error(f"アーカイブエラー: {e}")
            return False

    def _update_archive_readme(self, filename: str):
        """アーカイブREADME更新"""
        readme_path = self.deprecated_dir / "README.md"
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            if readme_path.exists():
                content = readme_path.read_text(encoding="utf-8")
            else:
                content = "# Tools Archive\n\n"

            # 今日の日付セクション追加
            if f"### {today}" not in content:
                content += f"\n### {today} 移動\n"

            content += f"- `{filename}` - 自動アーカイブ\n"

            readme_path.write_text(content, encoding="utf-8")

        except Exception as e:
            logger.warning(f"README更新失敗: {e}")

    # ========== 統計・レポート機能 ==========

    def stats(self) -> Dict[str, Any]:
        """Tools Directory統計情報"""
        stats = {"total_files": 0, "directories": {}, "file_types": Counter(), "total_size_kb": 0}

        print(f"\n{'='*60}")
        print("Tools Directory 統計情報")
        print(f"{'='*60}")

        # 各ディレクトリの統計
        for subdir in ["core", "batch", "testing", "scripts", "utils", "legacy"]:
            dir_path = self.tools_dir / subdir
            if dir_path.exists():
                files = list(dir_path.glob("*.py"))
                stats["directories"][subdir] = len(files)
                stats["total_files"] += len(files)

                # ファイルサイズ計算
                for file in files:
                    stats["total_size_kb"] += file.stat().st_size / 1024
                    stats["file_types"][file.suffix] += 1

                print(f"{subdir:<15}: {len(files):>3} ファイル")

        # progress_tracker統計
        pt_dir = self.tools_dir / "progress_tracker"
        if pt_dir.exists():
            pt_files = list(pt_dir.glob("*.py"))
            stats["directories"]["progress_tracker"] = len(pt_files)
            stats["total_files"] += len(pt_files)
            print(f"{'progress_tracker':<15}: {len(pt_files):>3} ファイル")

        print(f"{'-'*30}")
        print(f"{'合計':<15}: {stats['total_files']:>3} ファイル")
        print(f"{'総サイズ':<15}: {stats['total_size_kb']:>7.1f} KB")

        return stats

    # ========== ガバナンス支援機能 ==========

    def validate_placement(self, file_path: str) -> Dict[str, Any]:
        """ファイル配置妥当性チェック"""
        file_path = Path(file_path)
        filename = file_path.name

        # 推奨ディレクトリ判定
        recommendations = []
        violations = []

        # 命名規則チェック
        if filename.startswith("test_"):
            recommendations.append("testing/")
        elif "batch" in filename.lower():
            recommendations.append("batch/")
        elif (
            filename.endswith("_release.py")
            or filename.startswith("p1_")
            or filename.startswith("tdr_")
        ):
            recommendations.append("scripts/")
        elif "utils" in filename.lower() or filename in ["init_models.py", "cleanup_repository.py"]:
            recommendations.append("utils/")
        elif "legacy" in filename.lower() or "old" in filename.lower():
            recommendations.append("legacy/")
        else:
            recommendations.append("core/")

        # 禁止パターンチェック
        if file_path.parent.name == "segment-anything":  # プロジェクトルート
            violations.append("プロジェクトルートへの配置は禁止")

        if file_path.parent.name == "tools" and filename != "manager.py":
            violations.append("tools/直下への新規ファイル作成は禁止")

        result = {
            "file": filename,
            "current_path": str(file_path),
            "recommendations": recommendations,
            "violations": violations,
            "status": "OK" if not violations else "VIOLATION",
        }

        # 結果表示
        print(f"\n{'='*60}")
        print(f"ファイル配置チェック: {filename}")
        print(f"{'='*60}")
        print(f"現在パス: {file_path}")
        print(f"推奨配置: {', '.join(recommendations)}")

        if violations:
            print(f"⚠️  違反: {', '.join(violations)}")
        else:
            print("✅ 配置OK")

        return result

    def validate_naming(self) -> List[Dict[str, Any]]:
        """命名規則チェック"""
        violations = []

        print(f"\n{'='*60}")
        print("命名規則チェック")
        print(f"{'='*60}")

        # 各ディレクトリの命名規則チェック
        rules = {
            "core": r"^[a-z_]+\.py$",
            "batch": r"^[a-z0-9_]+_batch\.py$",
            "testing": r"^(test_|validate_|evaluate_)[a-z0-9_]+\.py$",
            "scripts": r"^[a-z0-9_]+\.py$",
            "utils": r"^[a-z_]+\.py$",
            "legacy": r"^[a-z0-9_]+\.py$",
        }

        import re

        for dir_name, pattern in rules.items():
            dir_path = self.tools_dir / dir_name
            if dir_path.exists():
                for file in dir_path.glob("*.py"):
                    if not re.match(pattern, file.name):
                        violation = {
                            "file": file.name,
                            "directory": dir_name,
                            "expected_pattern": pattern,
                            "issue": f"命名規則違反: {dir_name}/では{pattern}パターンを推奨",
                        }
                        violations.append(violation)
                        print(f"⚠️  {file.name} ({dir_name}/): {violation['issue']}")

        if not violations:
            print("✅ 全ファイルが命名規則に準拠")

        return violations

    def check_dependencies(self) -> Dict[str, List[str]]:
        """依存関係チェック"""
        dependencies = {}

        print(f"\n{'='*60}")
        print("依存関係チェック")
        print(f"{'='*60}")

        # 各ディレクトリのファイルの依存関係を分析
        for subdir in ["core", "batch", "testing", "scripts", "utils", "legacy"]:
            dir_path = self.tools_dir / subdir
            if dir_path.exists():
                for file in dir_path.glob("*.py"):
                    try:
                        content = file.read_text(encoding="utf-8")
                        deps = []

                        # tools/内のimportを検出
                        import re

                        tool_imports = re.findall(r"from tools\.([a-z_]+)", content)
                        tool_imports.extend(re.findall(r"import tools\.([a-z_]+)", content))

                        if tool_imports:
                            deps = list(set(tool_imports))
                            dependencies[f"{subdir}/{file.name}"] = deps

                    except Exception:
                        pass

        # 依存関係違反チェック
        violations = []
        for file_path, deps in dependencies.items():
            dir_name = file_path.split("/")[0]

            # utils/は他ディレクトリへの依存禁止
            if dir_name == "utils" and any(dep not in ["utils"] for dep in deps):
                violations.append(f"{file_path}: utils/は他ディレクトリへの依存禁止")

        # 結果表示
        if dependencies:
            for file_path, deps in dependencies.items():
                print(f"{file_path}: {', '.join(deps)}")
        else:
            print("依存関係なし")

        if violations:
            print("\n⚠️  違反:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("\n✅ 依存関係OK")

        return dependencies

    def governance_report(self) -> Dict[str, Any]:
        """ガバナンス総合レポート"""
        print(f"\n{'='*80}")
        print("Tools Directory ガバナンス レポート")
        print(f"{'='*80}")

        # 統計情報
        stats = self.stats()

        # 命名規則チェック
        naming_violations = self.validate_naming()

        # 依存関係チェック
        dependencies = self.check_dependencies()

        # scripts/の古いファイルチェック
        scripts_dir = self.tools_dir / "scripts"
        old_scripts = []
        if scripts_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            for file in scripts_dir.glob("*.py"):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff_date:
                    age_days = (datetime.now() - mtime).days
                    old_scripts.append({"file": file.name, "age_days": age_days})

        # 総合判定
        total_violations = len(naming_violations) + len(old_scripts)
        health_score = max(0, 100 - (total_violations * 10))

        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "naming_violations": len(naming_violations),
            "old_scripts": len(old_scripts),
            "total_violations": total_violations,
            "health_score": health_score,
            "dependencies": dependencies,
        }

        # サマリー表示
        print(f"\n{'='*60}")
        print("ガバナンス サマリー")
        print(f"{'='*60}")
        print(f"総ファイル数: {stats['total_files']}")
        print(f"命名規則違反: {len(naming_violations)}件")
        print(f"古いscripts/: {len(old_scripts)}件")
        print(f"健全性スコア: {health_score}/100")

        if health_score >= 90:
            print("✅ 健全な状態")
        elif health_score >= 70:
            print("⚠️  注意が必要")
        else:
            print("🚨 改善が必要")

        return report


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="Tools Directory 統合管理CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Google Sheetsタスク管理
  %(prog)s sheets list --priority 優先度最高
  %(prog)s sheets update TDR-002 --status 着手中
  %(prog)s sheets read P1-A001
  
  # バッチ処理管理
  %(prog)s batch list
  %(prog)s batch run kana08_enhanced
  
  # ツール整理
  %(prog)s cleanup --days 30
  %(prog)s archive scripts/old_tool.py
  %(prog)s stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # Google Sheetsコマンド
    sheets_parser = subparsers.add_parser("sheets", help="Google Sheets管理")
    sheets_sub = sheets_parser.add_subparsers(dest="sheets_cmd")

    # sheets list
    list_parser = sheets_sub.add_parser("list", help="タスク一覧表示")
    list_parser.add_argument("--priority", help="優先度フィルタ")
    list_parser.add_argument("--status", help="ステータスフィルタ")
    list_parser.add_argument("--limit", type=int, default=20, help="表示件数")

    # sheets update
    update_parser = sheets_sub.add_parser("update", help="タスク更新")
    update_parser.add_argument("tracker_id", help="トラッカーID")
    update_parser.add_argument("--status", help="新しいステータス")
    update_parser.add_argument("--priority", help="新しい優先度")

    # sheets read
    read_parser = sheets_sub.add_parser("read", help="タスク詳細表示")
    read_parser.add_argument("tracker_id", help="トラッカーID")

    # バッチ処理コマンド
    batch_parser = subparsers.add_parser("batch", help="バッチ処理管理")
    batch_sub = batch_parser.add_subparsers(dest="batch_cmd")

    # batch list
    batch_sub.add_parser("list", help="バッチスクリプト一覧")

    # batch run
    run_parser = batch_sub.add_parser("run", help="バッチ実行")
    run_parser.add_argument("script", help="スクリプト名")
    run_parser.add_argument("args", nargs="*", help="追加引数")

    # メンテナンスコマンド
    cleanup_parser = subparsers.add_parser("cleanup", help="古いスクリプト整理")
    cleanup_parser.add_argument("--days", type=int, default=30, help="日数閾値")

    archive_parser = subparsers.add_parser("archive", help="ファイルアーカイブ")
    archive_parser.add_argument("file", help="ファイルパス")

    # 統計コマンド
    subparsers.add_parser("stats", help="統計情報表示")

    # ガバナンスコマンド
    governance_parser = subparsers.add_parser("governance", help="ガバナンス管理")
    governance_sub = governance_parser.add_subparsers(dest="governance_cmd")

    # governance validate-placement
    placement_parser = governance_sub.add_parser("validate-placement", help="ファイル配置チェック")
    placement_parser.add_argument("file", help="チェック対象ファイル")

    # governance validate-naming
    governance_sub.add_parser("validate-naming", help="命名規則チェック")

    # governance check-dependencies
    governance_sub.add_parser("check-dependencies", help="依存関係チェック")

    # governance report
    governance_sub.add_parser("report", help="ガバナンス総合レポート")

    args = parser.parse_args()

    # マネージャー初期化
    manager = ToolsManager()

    # コマンド実行
    if args.command == "sheets":
        if args.sheets_cmd == "list":
            manager.sheets_list(args.priority, args.status, args.limit)
        elif args.sheets_cmd == "update":
            manager.sheets_update(args.tracker_id, args.status, args.priority)
        elif args.sheets_cmd == "read":
            manager.sheets_read(args.tracker_id)
        else:
            sheets_parser.print_help()

    elif args.command == "batch":
        if args.batch_cmd == "list":
            manager.batch_list()
        elif args.batch_cmd == "run":
            manager.batch_run(args.script, args.args)
        else:
            batch_parser.print_help()

    elif args.command == "cleanup":
        manager.cleanup(args.days)

    elif args.command == "archive":
        manager.archive(Path(args.file))

    elif args.command == "stats":
        manager.stats()

    elif args.command == "governance":
        if args.governance_cmd == "validate-placement":
            manager.validate_placement(args.file)
        elif args.governance_cmd == "validate-naming":
            manager.validate_naming()
        elif args.governance_cmd == "check-dependencies":
            manager.check_dependencies()
        elif args.governance_cmd == "report":
            manager.governance_report()
        else:
            governance_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
