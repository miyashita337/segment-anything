#!/usr/bin/env python3
"""
SubAgent統合ワークフロー実行スクリプト
QUAL-044: 長時間タスクキューシステムを活用した統合ワークフロー

Usage:
    python tools/scripts/run_workflow_with_subagent.py TRACKER_ID INPUT_DIR
    python tools/scripts/run_workflow_with_subagent.py QUAL-044 /mnt/c/AItools/lora/train/yado/org/kana08/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from config.workspace_config import WorkspaceConfig
from tools.queue.subagent_monitor import SubAgentIntegration
from tools.queue.task_integration import TaskOrchestrator

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedWorkflowRunner:
    """統合ワークフロー実行クラス"""

    def __init__(self, tracker_id: str):
        """
        初期化

        Args:
            tracker_id: トラッカーID
        """
        self.tracker_id = tracker_id
        self.orchestrator = TaskOrchestrator(tracker_id)
        self.subagent = SubAgentIntegration()

        # ワークスペース設定
        config = WorkspaceConfig()
        self.workspace_base = config.get_tracker_workspace(tracker_id)

        # コンテキスト設定
        self.subagent.set_context(
            {
                "tracker_id": tracker_id,
                "workflow": "integrated_quality_workflow",
                "session_id": f"workflow_{tracker_id}_{int(time.time())}",
            }
        )

        logger.info(f"IntegratedWorkflowRunner initialized for {tracker_id}")
        logger.info(f"Workspace: {self.workspace_base}")

    def run_phase1_extraction(self, input_dir: str, max_files: int = 10) -> Dict[str, Any]:
        """
        Phase 1: 抽出パイプライン実行

        Args:
            input_dir: 入力ディレクトリ
            max_files: 最大処理ファイル数

        Returns:
            実行結果
        """
        logger.info("=" * 50)
        logger.info("🚀 Phase 1: 抽出パイプライン（SubAgent監視）")
        logger.info("=" * 50)

        # 入力パス検証（QUAL-033準拠）
        input_path = Path(input_dir)
        if not input_path.exists():
            error_msg = f"❌ エラー: 入力ディレクトリが存在しません: {input_dir}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # 画像ファイル確認
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        if not image_files:
            error_msg = f"❌ エラー: 画像ファイルが見つかりません: {input_dir}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        logger.info(f"✅ 入力パス検証成功: {input_dir}")
        logger.info(f"✅ 画像ファイル: {len(image_files)}枚検出")

        # TaskOrchestratorで抽出実行
        output_dir = f"{self.workspace_base}/extraction/"
        logger.info(f"📂 出力先: {output_dir}")

        task_id, result = self.orchestrator.run_extraction_with_monitoring(
            input_dir=input_dir,
            output_dir=output_dir,
            max_files=max_files,
            options=["--verbose", "--strict-validation"],
        )

        logger.info(f"📊 タスクID: {task_id}")
        logger.info(f"📊 処理結果: {result.get('final_status')}")

        # 結果保存
        result_file = Path(self.workspace_base) / "phase1_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def run_phase2_quality_check(self, extraction_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 2: 品質評価実行

        Args:
            extraction_dir: 抽出結果ディレクトリ（省略時は自動判定）

        Returns:
            実行結果
        """
        logger.info("=" * 50)
        logger.info("🎯 Phase 2: 品質評価（SubAgent監視）")
        logger.info("=" * 50)

        if not extraction_dir:
            extraction_dir = f"{self.workspace_base}/extraction/"

        # 抽出結果確認
        extraction_path = Path(extraction_dir)
        if not extraction_path.exists():
            error_msg = f"❌ エラー: 抽出結果ディレクトリが存在しません: {extraction_dir}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # TaskOrchestratorで品質評価実行
        task_id, result = self.orchestrator.run_quality_check_with_monitoring(
            extraction_dir=extraction_dir, options=["--generate-report", "--calculate-metrics"]
        )

        logger.info(f"📊 タスクID: {task_id}")
        logger.info(f"📊 評価結果: {result.get('final_status')}")

        # 結果保存
        result_file = Path(self.workspace_base) / "phase2_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def run_phase3_dashboard(self) -> Dict[str, Any]:
        """
        Phase 3: ダッシュボード生成

        Returns:
            実行結果
        """
        logger.info("=" * 50)
        logger.info("📊 Phase 3: ダッシュボード生成（SubAgent監視）")
        logger.info("=" * 50)

        # TaskOrchestratorでダッシュボード生成
        task_id, result = self.orchestrator.run_dashboard_generation(
            tracker_id=self.tracker_id,
            options=["--with-stats", "--with-images", "--markdown-report"],
        )

        logger.info(f"📊 タスクID: {task_id}")
        logger.info(f"📊 生成結果: {result.get('final_status')}")

        # 結果保存
        result_file = Path(self.workspace_base) / "phase3_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def run_phase4_integration_report(self) -> Dict[str, Any]:
        """
        Phase 4: 統合レポート生成

        Returns:
            実行結果
        """
        logger.info("=" * 50)
        logger.info("📝 Phase 4: 統合レポート生成")
        logger.info("=" * 50)

        # 各フェーズの結果読み込み
        results = {}
        for phase in ["phase1", "phase2", "phase3"]:
            result_file = Path(self.workspace_base) / f"{phase}_result.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    results[phase] = json.load(f)

        # 統合レポート生成
        report = self.orchestrator.generate_final_report(
            tracker_id=self.tracker_id, phase_results=results
        )

        # レポート保存
        report_file = Path(self.workspace_base) / "integration_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"✅ 統合レポート生成完了: {report_file}")

        return {
            "status": "completed",
            "report_path": str(report_file),
            "phases_completed": list(results.keys()),
        }

    def run_full_workflow(self, input_dir: str, max_files: int = 10) -> Dict[str, Any]:
        """
        完全統合ワークフロー実行

        Args:
            input_dir: 入力ディレクトリ
            max_files: 最大処理ファイル数

        Returns:
            全フェーズの実行結果
        """
        logger.info("🎯 SubAgent統合ワークフロー開始")
        logger.info(f"   トラッカーID: {self.tracker_id}")
        logger.info(f"   入力ディレクトリ: {input_dir}")
        logger.info(f"   最大ファイル数: {max_files}")

        workflow_results = {"tracker_id": self.tracker_id, "start_time": time.time(), "phases": {}}

        try:
            # Phase 1: 抽出
            phase1_result = self.run_phase1_extraction(input_dir, max_files)
            workflow_results["phases"]["phase1"] = phase1_result

            if phase1_result.get("final_status") != "completed":
                logger.warning("Phase 1が完了しませんでした。後続フェーズをスキップします。")
                workflow_results["status"] = "partial"
                return workflow_results

            # Phase 2: 品質評価
            phase2_result = self.run_phase2_quality_check()
            workflow_results["phases"]["phase2"] = phase2_result

            # Phase 3: ダッシュボード
            phase3_result = self.run_phase3_dashboard()
            workflow_results["phases"]["phase3"] = phase3_result

            # Phase 4: 統合レポート
            phase4_result = self.run_phase4_integration_report()
            workflow_results["phases"]["phase4"] = phase4_result

            workflow_results["status"] = "completed"

        except Exception as e:
            logger.error(f"ワークフローエラー: {e}")
            workflow_results["status"] = "error"
            workflow_results["error"] = str(e)

        finally:
            workflow_results["end_time"] = time.time()
            workflow_results["duration"] = (
                workflow_results["end_time"] - workflow_results["start_time"]
            )

            # 最終結果保存
            final_result_file = Path(self.workspace_base) / "workflow_result.json"
            with open(final_result_file, "w") as f:
                json.dump(workflow_results, f, indent=2, default=str)

            logger.info("=" * 50)
            logger.info(f"✅ ワークフロー完了")
            logger.info(f"   状態: {workflow_results['status']}")
            logger.info(f"   処理時間: {workflow_results['duration']:.2f}秒")
            logger.info(f"   結果: {final_result_file}")
            logger.info("=" * 50)

        return workflow_results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="SubAgent統合ワークフロー実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python tools/scripts/run_workflow_with_subagent.py QUAL-044 /mnt/c/AItools/lora/train/yado/org/kana08/
  python tools/scripts/run_workflow_with_subagent.py QUAL-044 /mnt/c/AItools/lora/train/yado/org/kana08/ --max-files 5
  python tools/scripts/run_workflow_with_subagent.py QUAL-044 /mnt/c/AItools/lora/train/yado/org/kana08/ --phase phase1
        """,
    )

    parser.add_argument("tracker_id", help="トラッカーID（例：QUAL-044）")
    parser.add_argument("input_dir", help="入力画像ディレクトリパス")
    parser.add_argument("--max-files", type=int, default=10, help="最大処理ファイル数（デフォルト：10）")
    parser.add_argument(
        "--phase",
        choices=["all", "phase1", "phase2", "phase3", "phase4"],
        default="all",
        help="実行フェーズ（デフォルト：all）",
    )

    args = parser.parse_args()

    # ワークフロー実行
    runner = IntegratedWorkflowRunner(args.tracker_id)

    if args.phase == "all":
        result = runner.run_full_workflow(args.input_dir, args.max_files)
    elif args.phase == "phase1":
        result = runner.run_phase1_extraction(args.input_dir, args.max_files)
    elif args.phase == "phase2":
        result = runner.run_phase2_quality_check()
    elif args.phase == "phase3":
        result = runner.run_phase3_dashboard()
    elif args.phase == "phase4":
        result = runner.run_phase4_integration_report()

    # 結果表示
    print(json.dumps(result, indent=2, default=str))

    # 終了コード
    if result.get("status") == "completed":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
