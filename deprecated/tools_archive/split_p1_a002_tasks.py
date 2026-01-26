#!/usr/bin/env python3
"""
P1-A002分割スクリプト
P1-A002を3つのサブタスクに分割し、P1-A002を終了状態に変更
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.google_sheets_updater import GoogleSheetsUpdater
from tools.progress_tracker.data_models import PriorityLevel, TaskRecord, TaskStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class P1A002TaskSplitter:
    """P1-A002タスク分割ツール"""

    def __init__(self):
        """初期化"""
        self.updater = GoogleSheetsUpdater()

        # 3つのサブタスク定義
        self.subtasks = [
            {
                "tracker_id": "P1-A002-1",
                "description": "統一基準による品質評価比較システム",
                "details": "全データセット(kana03-09)を同一基準で処理・比較。実際の抽出結果JSONから横断的品質評価実施。各データセットの成功率・A/B評価率・品質分布を統一基準で測定し、科学的な品質ランキングを確立する。P1-A001で実証された改善効果を基準として、他データセットとの定量的比較を行う。",
            },
            {
                "tracker_id": "P1-A002-2",
                "description": "品質判定アルゴリズム標準化",
                "details": "P1-A001のenhanced判定アルゴリズムを全データセットに適用。公平な品質評価システム確立。現在kana08のみに適用されている_judge_quality_enhanced()関数（confidence*0.3 + sam_score*0.4 + mask_ratio*0.3）を、kana03-07、kana09に展開。データセット特性に応じたパラメータ調整も実施し、統一的な品質判定基盤を構築する。",
            },
            {
                "tracker_id": "P1-A002-3",
                "description": "評価指標運用改善・可視化",
                "details": "10指標の実運用改善。データセット特性把握、比較可視化、精度向上。LCA、SCI、PLA、PLE等の既存指標計算精度向上と、データセット横断的な特性分析を実施。品質ダッシュボード作成により、成果の可視化と継続的な品質監視体制を確立する。統一基準による定量評価結果をステークホルダーが理解しやすい形で提供する。",
            },
        ]

    def split_p1_a002(self) -> bool:
        """
        P1-A002分割実行

        Returns:
            bool: 分割成功/失敗
        """
        try:
            logger.info("P1-A002分割開始")

            # 1. 3つのサブタスク作成
            for subtask in self.subtasks:
                success = self._create_subtask(subtask)
                if not success:
                    logger.error(f"サブタスク作成失敗: {subtask['tracker_id']}")
                    return False

            # 2. P1-A002を終了状態に変更
            success = self.updater.update_task_status("P1-A002", "終了")
            if not success:
                logger.error("P1-A002終了状態変更失敗")
                return False

            logger.info("✅ P1-A002分割完了")
            return True

        except Exception as e:
            logger.error(f"P1-A002分割エラー: {e}")
            return False

    def _create_subtask(self, subtask_data: Dict[str, str]) -> bool:
        """
        サブタスク作成

        Args:
            subtask_data: サブタスクデータ

        Returns:
            bool: 作成成功/失敗
        """
        try:
            # TaskRecord作成
            task = TaskRecord(
                tracker_id=subtask_data["tracker_id"],
                priority=PriorityLevel.HIGH,  # P1-A002と同じ高優先度
                status=TaskStatus.NOT_STARTED,
                created_date=datetime.now(),
                description=subtask_data["description"],
                details=subtask_data["details"],
            )

            # Google Sheetsに追加（TaskRecordを行データに変換して直接追加）
            task_row = task.to_sheets_row()

            # シートに行追加
            sheet_name = "シート1"
            range_name = f"{sheet_name}!A:W"

            body = {"values": [task_row]}

            result = (
                self.updater.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=self.updater.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )

            success = True

            if success:
                logger.info(f"✅ サブタスク作成完了: {subtask_data['tracker_id']}")
                logger.info(f"  概要: {subtask_data['description']}")
            else:
                logger.error(f"❌ サブタスク作成失敗: {subtask_data['tracker_id']}")

            return success

        except Exception as e:
            logger.error(f"サブタスク作成エラー ({subtask_data['tracker_id']}): {e}")
            return False

    def verify_split_result(self) -> bool:
        """分割結果検証"""
        try:
            logger.info("分割結果検証開始")

            # P1-A002状態確認
            p1a002_data = self.updater.get_task_by_id("P1-A002")
            if not p1a002_data:
                logger.error("P1-A002データ取得失敗")
                return False

            current_status = p1a002_data[2]  # C列：ステータス
            if current_status != "終了":
                logger.error(f"P1-A002ステータス未変更: {current_status}")
                return False

            logger.info("✅ P1-A002ステータス: 終了")

            # サブタスク確認
            for subtask in self.subtasks:
                task_data = self.updater.get_task_by_id(subtask["tracker_id"])
                if not task_data:
                    logger.error(f"サブタスク未作成: {subtask['tracker_id']}")
                    return False

                status = task_data[2]  # C列：ステータス
                details = task_data[6]  # G列：詳細

                logger.info(f"✅ {subtask['tracker_id']}: {status}")
                logger.info(f"  詳細文字数: {len(details)}文字")

            logger.info("✅ 分割結果検証完了")
            return True

        except Exception as e:
            logger.error(f"検証エラー: {e}")
            return False


def main():
    """メイン実行"""
    logger.info("P1-A002分割ツール")

    splitter = P1A002TaskSplitter()

    # 分割実行
    success = splitter.split_p1_a002()

    if success:
        # 検証
        verify_success = splitter.verify_split_result()

        if verify_success:
            logger.info("=" * 50)
            logger.info("P1-A002分割完了")
            logger.info("=" * 50)
            logger.info("作成されたサブタスク:")
            for subtask in splitter.subtasks:
                logger.info(f"  - {subtask['tracker_id']}: {subtask['description']}")
            logger.info("")
            logger.info("P1-A002: 着手中 → 終了")
            logger.info("品質基準統一の実装準備が完了しました")

            return 0
        else:
            logger.error("❌ 検証失敗")
            return 1
    else:
        logger.error("❌ 分割失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
