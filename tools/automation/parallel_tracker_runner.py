#!/usr/bin/env python3
"""
並列トラッカー実行システム
タイムアウト問題解決のための軽量並列処理実装
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.api_config import get_api_config

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskProgress:
    """タスク進捗管理クラス"""

    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def add_task(self, task_id: str, description: str):
        """タスクを追加"""
        with self.lock:
            self.tasks[task_id] = {
                "description": description,
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "result": None,
                "error": None,
            }

    def start_task(self, task_id: str):
        """タスク開始"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "running"
                self.tasks[task_id]["start_time"] = datetime.now()

    def complete_task(self, task_id: str, result: Any = None):
        """タスク完了"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["end_time"] = datetime.now()
                self.tasks[task_id]["result"] = result

    def fail_task(self, task_id: str, error: str):
        """タスク失敗"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["end_time"] = datetime.now()
                self.tasks[task_id]["error"] = error

    def get_progress_summary(self) -> Dict[str, int]:
        """進捗サマリーを取得"""
        with self.lock:
            summary = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
            for task in self.tasks.values():
                summary[task["status"]] += 1
            return summary

    def print_progress(self):
        """進捗を表示"""
        summary = self.get_progress_summary()
        total = sum(summary.values())
        if total == 0:
            return

        completed_pct = (summary["completed"] / total) * 100

        print(f"\n📊 進捗状況: {completed_pct:.1f}% ({summary['completed']}/{total})")
        print(f"   ✅ 完了: {summary['completed']}")
        print(f"   🔄 実行中: {summary['running']}")
        print(f"   ⏳ 待機: {summary['pending']}")
        print(f"   ❌ 失敗: {summary['failed']}")

        # 実行中のタスクを表示
        with self.lock:
            for task_id, task in self.tasks.items():
                if task["status"] == "running":
                    elapsed = (datetime.now() - task["start_time"]).total_seconds()
                    print(f"   🔄 {task['description']}: {elapsed:.1f}秒経過")


class ParallelTrackerRunner:
    """並列トラッカー実行システム"""

    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace_dir = self.workspace_base / tracker_id
        self.progress = TaskProgress()

        # API設定取得
        self.api_config = get_api_config()

        # ログファイル設定
        self.log_file = self.workspace_dir / f"{tracker_id}_parallel_execution.log"

        logger.info(f"並列トラッカー実行システム初期化: {tracker_id}")

    def _run_command(
        self, command: List[str], task_id: str, timeout: int = 300
    ) -> Tuple[bool, str, str]:
        """コマンドを実行"""
        self.progress.start_task(task_id)

        try:
            logger.info(f"コマンド実行開始 [{task_id}]: {' '.join(command)}")

            # 実行ディレクトリをプロジェクトルートに設定
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(os.environ, PYTHONPATH=str(self.project_root)),
            )

            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            # ログファイルに記録
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "command": " ".join(command),
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }

            self._write_log(log_entry)

            if success:
                self.progress.complete_task(task_id, stdout)
                logger.info(f"コマンド実行成功 [{task_id}]")
            else:
                self.progress.fail_task(task_id, stderr)
                logger.warning(f"コマンド実行失敗 [{task_id}]: {stderr}")

            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            error_msg = f"タイムアウト ({timeout}秒)"
            self.progress.fail_task(task_id, error_msg)
            logger.error(f"コマンドタイムアウト [{task_id}]: {error_msg}")
            return False, "", error_msg

        except Exception as e:
            error_msg = f"実行エラー: {str(e)}"
            self.progress.fail_task(task_id, error_msg)
            logger.error(f"コマンド実行エラー [{task_id}]: {error_msg}")
            return False, "", error_msg

    def _write_log(self, log_entry: Dict):
        """ログファイルに記録"""
        try:
            # ログディレクトリ作成
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # ログエントリを追記
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, indent=2, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"ログ書き込みエラー: {e}")

    def prepare_workspace(self) -> bool:
        """ワークスペース準備"""
        task_id = "workspace_prep"
        self.progress.add_task(task_id, "ワークスペース準備")

        try:
            # ディレクトリ作成
            directories = [
                self.workspace_dir / "extraction",
                self.workspace_dir / "quality",
                self.workspace_dir / "dashboard",
                self.workspace_dir / "tests",
            ]

            for dir_path in directories:
                dir_path.mkdir(parents=True, exist_ok=True)

            self.progress.complete_task(task_id, "ワークスペース作成完了")
            logger.info("ワークスペース準備完了")
            return True

        except Exception as e:
            error_msg = f"ワークスペース準備エラー: {str(e)}"
            self.progress.fail_task(task_id, error_msg)
            logger.error(error_msg)
            return False

    def run_tests(self) -> bool:
        """テスト実行"""
        task_id = "unit_tests"
        self.progress.add_task(task_id, "単体テスト実行")

        # トラッカー固有のテストを探す
        test_pattern = f"test_{self.tracker_id.lower().replace('-', '_')}*.py"
        test_files = list(self.project_root.glob(f"tests/unit/{test_pattern}"))

        if test_files:
            # 固有テストがある場合
            command = ["python3", "-m", "pytest"] + [str(f) for f in test_files] + ["-v"]
        else:
            # 汎用テストを実行
            command = ["python3", "-m", "pytest", "tests/unit/test_extract.py", "-v"]

        success, stdout, stderr = self._run_command(command, task_id, timeout=120)

        # 結果をファイルに保存
        result_file = self.workspace_dir / "tests" / "unit_test_results.txt"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {'0' if success else '1'}\n")
                f.write(f"Stdout:\n{stdout}\n")
                if stderr:
                    f.write(f"Stderr:\n{stderr}\n")
        except Exception as e:
            logger.warning(f"テスト結果保存エラー: {e}")

        return success

    def run_quality_check(self) -> bool:
        """品質チェック実行"""
        task_id = "quality_check"
        self.progress.add_task(task_id, "品質チェック実行")

        # linter.sh実行
        linter_script = self.project_root / "bin" / "shell" / "linter.sh"

        if linter_script.exists():
            command = [str(linter_script)]
        else:
            # フォールバック: 個別ツール実行
            command = ["python3", "-m", "flake8", "features/", "tools/"]

        success, stdout, stderr = self._run_command(command, task_id, timeout=180)

        # 結果をファイルに保存
        result_file = self.workspace_dir / "quality" / "linter_results.txt"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {'0' if success else '1'}\n")
                f.write(f"Stdout:\n{stdout}\n")
                if stderr:
                    f.write(f"Stderr:\n{stderr}\n")
        except Exception as e:
            logger.warning(f"品質チェック結果保存エラー: {e}")

        return success

    def run_extraction_pipeline(self) -> bool:
        """抽出パイプライン実行（バッチ分割対応）"""
        task_id = "extraction_pipeline"
        self.progress.add_task(task_id, "バッチ分割抽出パイプライン実行")

        # 入力ディレクトリ確認
        input_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana05")

        if not input_dir.exists():
            error_msg = f"入力ディレクトリが存在しません: {input_dir}"
            self.progress.fail_task(task_id, error_msg)
            logger.error(error_msg)
            return False

        # バッチ分割抽出システムを使用
        command = [
            "python3",
            "tools/automation/batched_extraction_runner.py",
            self.tracker_id,
            "--input-dir",
            str(input_dir),
            "--max-workers",
            "2",
        ]

        # バッチ処理は長時間かかるため、タイムアウトを延長（20分）
        success, stdout, stderr = self._run_command(command, task_id, timeout=1200)

        # 結果をファイルに保存
        result_file = self.workspace_dir / "extraction" / "extraction.log"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {'0' if success else '1'}\n")
                f.write(f"Stdout:\n{stdout}\n")
                if stderr:
                    f.write(f"Stderr:\n{stderr}\n")
        except Exception as e:
            logger.warning(f"抽出結果保存エラー: {e}")

        # 抽出画像数をカウント
        extracted_files = list((self.workspace_dir / "extraction").glob("*.jpg"))
        extracted_files.extend(list((self.workspace_dir / "extraction").glob("*.png")))

        # バッチ処理統計を読み込み
        stats_file = self.workspace_dir / "extraction" / "batch_extraction_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    success_rate = stats.get("success_rate_percent", 0)
                    logger.info(f"バッチ処理完了: {len(extracted_files)}枚出力, 成功率{success_rate:.1f}%")

                    # 60%以上の成功率で成功とみなす
                    if success_rate >= 60:
                        success = True

            except Exception as e:
                logger.warning(f"統計ファイル読み込みエラー: {e}")

        logger.info(f"抽出処理結果: {len(extracted_files)}枚の画像")
        return success

    def generate_dashboard(self) -> bool:
        """ダッシュボード生成"""
        task_id = "dashboard_generation"
        self.progress.add_task(task_id, "ダッシュボード生成")

        # 品質レポート生成
        report_command = [
            "python3",
            "create_phase1_extraction_report.py",
            str(self.workspace_dir / "extraction"),
            str(self.workspace_dir / "quality" / f"{self.tracker_id}_extraction_report"),
        ]

        report_success, _, _ = self._run_command(report_command, f"{task_id}_report", timeout=60)

        # HTMLダッシュボード生成
        dashboard_command = [
            "python3",
            "tools/core/quality_dashboard.py",
            "--results",
            str(self.workspace_dir / "extraction"),
            "--output",
            str(self.workspace_dir / "dashboard" / "dashboard.html"),
        ]

        dashboard_success, _, _ = self._run_command(
            dashboard_command, f"{task_id}_html", timeout=60
        )

        # 部分的成功でも継続
        success = report_success or dashboard_success

        if success:
            self.progress.complete_task(task_id, "ダッシュボード生成完了")
        else:
            self.progress.fail_task(task_id, "ダッシュボード生成失敗")

        return success

    def generate_final_report(self) -> bool:
        """最終レポート生成"""
        task_id = "final_report"
        self.progress.add_task(task_id, "最終レポート生成")

        try:
            report_file = self.workspace_dir / f"{self.tracker_id}_completion_report.md"

            # 統計情報取得
            summary = self.progress.get_progress_summary()
            extracted_files = list((self.workspace_dir / "extraction").glob("*.jpg"))
            extracted_files.extend(list((self.workspace_dir / "extraction").glob("*.png")))

            # レポート生成
            report_content = f"""# {self.tracker_id} 並列実行完了レポート

**生成日時**: {datetime.now().isoformat()}  
**実行方法**: 並列処理システム

## 📊 実行結果サマリー

### ✅ タスク実行状況
- 完了: {summary['completed']}
- 失敗: {summary['failed']}
- 総タスク数: {sum(summary.values())}

### 📁 出力ディレクトリ構造
```
{self.workspace_dir}/
├── extraction/      # 抽出結果 ({len(extracted_files)}枚)
├── quality/        # 品質レポート
├── dashboard/      # HTMLダッシュボード
└── tests/          # テスト結果
```

### 🔍 詳細タスク結果

"""

            # 各タスクの詳細
            with self.progress.lock:
                for task_id, task in self.progress.tasks.items():
                    status_emoji = {
                        "completed": "✅",
                        "failed": "❌",
                        "running": "🔄",
                        "pending": "⏳",
                    }.get(task["status"], "❓")

                    report_content += f"#### {status_emoji} {task['description']}\n"
                    report_content += f"- ステータス: {task['status']}\n"

                    if task["start_time"]:
                        report_content += f"- 開始時刻: {task['start_time'].strftime('%H:%M:%S')}\n"

                    if task["end_time"]:
                        duration = (task["end_time"] - task["start_time"]).total_seconds()
                        report_content += f"- 実行時間: {duration:.1f}秒\n"

                    if task["error"]:
                        report_content += f"- エラー: {task['error']}\n"

                    report_content += "\n"

            report_content += f"""
## 📋 次のステップ

1. ダッシュボードの確認: [{self.tracker_id}/dashboard/dashboard.html](./{self.tracker_id}/dashboard/dashboard.html)
2. 品質レポートの確認: [{self.tracker_id}/quality/](./{self.tracker_id}/quality/)
3. Google Sheetsの更新: `/release` ステータスへ

---
*このレポートは並列処理システムにより自動生成されました*
"""

            # ファイル書き込み
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.progress.complete_task(task_id, str(report_file))
            logger.info(f"最終レポート生成完了: {report_file}")
            return True

        except Exception as e:
            error_msg = f"最終レポート生成エラー: {str(e)}"
            self.progress.fail_task(task_id, error_msg)
            logger.error(error_msg)
            return False

    def run_parallel_workflow(self, skip_extraction: bool = False) -> bool:
        """並列ワークフロー実行"""
        start_time = datetime.now()

        print(f"\n🚀 {self.tracker_id} 並列実行開始")
        print(f"📍 ワークスペース: {self.workspace_dir}")
        print(f"⏰ 開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 進捗表示スレッド開始
        progress_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        progress_thread.start()

        try:
            # 第1段階: 軽量処理を並列実行
            print("\n📋 第1段階: 軽量タスク並列実行")

            light_tasks = [
                ("workspace", self.prepare_workspace),
                ("tests", self.run_tests),
                ("quality", self.run_quality_check),
            ]

            light_results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_name = {executor.submit(func): name for name, func in light_tasks}

                for future in as_completed(future_to_name):
                    task_name = future_to_name[future]
                    try:
                        result = future.result()
                        light_results[task_name] = result
                        print(f"   ✅ {task_name}: {'成功' if result else '失敗'}")
                    except Exception as e:
                        light_results[task_name] = False
                        print(f"   ❌ {task_name}: エラー - {e}")

            # 第2段階: 重い処理を順次実行
            if not skip_extraction:
                print("\n📋 第2段階: 重量タスク実行")
                extraction_result = self.run_extraction_pipeline()
                print(
                    f"   {'✅' if extraction_result else '❌'} 抽出パイプライン: {'成功' if extraction_result else '失敗'}"
                )
            else:
                extraction_result = True
                print("\n📋 抽出パイプラインはスキップされました")

            # 第3段階: 後処理
            print("\n📋 第3段階: 後処理")
            dashboard_result = self.generate_dashboard()
            report_result = self.generate_final_report()

            print(
                f"   {'✅' if dashboard_result else '❌'} ダッシュボード: {'成功' if dashboard_result else '失敗'}"
            )
            print(f"   {'✅' if report_result else '❌'} 最終レポート: {'成功' if report_result else '失敗'}")

            # 結果判定
            critical_success = light_results.get("workspace", False)
            overall_success = critical_success and (
                light_results.get("tests", False) or light_results.get("quality", False)
            )

            # 実行時間計算
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # 結果表示
            print(f"\n🏁 {self.tracker_id} 並列実行完了")
            print(f"⏱️  総実行時間: {execution_time:.1f}秒")
            print(f"📊 結果: {'✅ 成功' if overall_success else '❌ 失敗'}")

            self.progress.print_progress()

            if overall_success:
                print(f"\n📁 出力先: {self.workspace_dir}")
                print(f"📋 レポート: {self.workspace_dir}/{self.tracker_id}_completion_report.md")

            return overall_success

        except Exception as e:
            logger.error(f"並列ワークフロー実行エラー: {e}")
            print(f"\n❌ 並列実行中にエラーが発生しました: {e}")
            return False

    def _progress_monitor(self):
        """進捗監視スレッド"""
        while True:
            time.sleep(10)  # 10秒ごとに進捗表示
            summary = self.progress.get_progress_summary()
            if summary["running"] > 0:
                self.progress.print_progress()
            elif summary["pending"] == 0:
                break  # 全タスク完了


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="並列トラッカー実行システム")
    parser.add_argument("tracker_id", help="トラッカーID (例: PH3-007)")
    parser.add_argument("--skip-extraction", action="store_true", help="抽出パイプラインをスキップ")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="ログレベル"
    )

    args = parser.parse_args()

    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 並列実行システム作成
    runner = ParallelTrackerRunner(args.tracker_id)

    # 実行
    success = runner.run_parallel_workflow(skip_extraction=args.skip_extraction)

    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
