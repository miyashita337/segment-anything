#!/usr/bin/env python3
"""
自動メンテナンススクリプト
TDR-003で確立されたガバナンスルールに基づく定期メンテナンス
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tools.manager import ToolsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AutoMaintenance:
    """自動メンテナンスシステム"""

    def __init__(self):
        """初期化"""
        self.manager = ToolsManager()
        self.maintenance_log = []

    def run_daily_maintenance(self) -> Dict[str, Any]:
        """日次メンテナンス実行"""
        logger.info("=== 日次メンテナンス開始 ===")

        results = {
            "timestamp": datetime.now().isoformat(),
            "stats": None,
            "governance_score": None,
            "cleanup_count": 0,
            "violations": [],
        }

        try:
            # 1. 統計情報取得
            logger.info("統計情報取得中...")
            results["stats"] = self.manager.stats()

            # 2. ガバナンスチェック
            logger.info("ガバナンスチェック実行中...")
            governance_report = self.manager.governance_report()
            results["governance_score"] = governance_report["health_score"]
            results["violations"] = governance_report["naming_violations"]

            # 3. 自動クリーンアップ（7日以上）
            logger.info("自動クリーンアップ実行中...")
            import os

            os.environ["TOOLS_MANAGER_AUTO_MODE"] = "1"
            old_files = self.manager.cleanup(days=7)
            results["cleanup_count"] = len(old_files)

            # 4. 結果サマリー
            self._log_maintenance_summary(results)

            # 5. 警告チェック
            self._check_warnings(results)

            logger.info("=== 日次メンテナンス完了 ===")

        except Exception as e:
            logger.error(f"メンテナンスエラー: {e}")
            results["error"] = str(e)

        return results

    def run_weekly_maintenance(self) -> Dict[str, Any]:
        """週次メンテナンス実行"""
        logger.info("=== 週次メンテナンス開始 ===")

        results = {
            "timestamp": datetime.now().isoformat(),
            "deep_cleanup_count": 0,
            "governance_violations": 0,
            "recommendations": [],
        }

        try:
            # 1. 深度クリーンアップ（30日以上）
            logger.info("深度クリーンアップ実行中...")
            import os

            os.environ["TOOLS_MANAGER_AUTO_MODE"] = "1"
            old_files = self.manager.cleanup(days=30)
            results["deep_cleanup_count"] = len(old_files)

            # 2. 詳細ガバナンスチェック
            logger.info("詳細ガバナンスチェック実行中...")
            naming_violations = self.manager.validate_naming()
            results["governance_violations"] = len(naming_violations)

            # 3. 最適化推奨事項生成
            logger.info("最適化推奨事項生成中...")
            recommendations = self._generate_recommendations()
            results["recommendations"] = recommendations

            # 4. 週次レポート生成
            self._generate_weekly_report(results)

            logger.info("=== 週次メンテナンス完了 ===")

        except Exception as e:
            logger.error(f"週次メンテナンスエラー: {e}")
            results["error"] = str(e)

        return results

    def _log_maintenance_summary(self, results: Dict[str, Any]):
        """メンテナンス結果サマリー出力"""
        stats = results.get("stats", {})
        total_files = stats.get("total_files", 0)
        governance_score = results.get("governance_score", 0)
        cleanup_count = results.get("cleanup_count", 0)

        print(f"\n{'='*60}")
        print("日次メンテナンス サマリー")
        print(f"{'='*60}")
        print(f"総ファイル数: {total_files}")
        print(f"ガバナンススコア: {governance_score}/100")
        print(f"クリーンアップ: {cleanup_count}ファイル")

        if governance_score >= 90:
            print("✅ 健全な状態を維持")
        elif governance_score >= 70:
            print("⚠️  改善の余地あり")
        else:
            print("🚨 至急改善が必要")

    def _check_warnings(self, results: Dict[str, Any]):
        """警告チェックと通知"""
        warnings = []

        # ファイル数警告
        total_files = results.get("stats", {}).get("total_files", 0)
        if total_files > 50:
            warnings.append(f"ファイル数が50を超過: {total_files}ファイル")

        # ガバナンススコア警告
        governance_score = results.get("governance_score", 100)
        if governance_score < 70:
            warnings.append(f"ガバナンススコア低下: {governance_score}/100")

        # 違反件数警告
        violations = results.get("violations", 0)
        if violations > 5:
            warnings.append(f"命名規則違反多数: {violations}件")

        if warnings:
            print(f"\n⚠️  警告:")
            for warning in warnings:
                print(f"  - {warning}")

        return warnings

    def _generate_recommendations(self) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []

        # 統計分析
        stats = self.manager.stats()
        dirs = stats.get("directories", {})

        # scripts/ディレクトリのファイル数チェック
        scripts_count = dirs.get("scripts", 0)
        if scripts_count > 10:
            recommendations.append(f"scripts/ディレクトリに{scripts_count}ファイル - アーカイブを検討")

        # legacy/ディレクトリのファイル数チェック
        legacy_count = dirs.get("legacy", 0)
        if legacy_count > 5:
            recommendations.append(f"legacy/ディレクトリに{legacy_count}ファイル - 削除または統合を検討")

        # core/ディレクトリのファイル数チェック
        core_count = dirs.get("core", 0)
        if core_count > 10:
            recommendations.append(f"core/ディレクトリに{core_count}ファイル - 機能分割を検討")

        return recommendations

    def _generate_weekly_report(self, results: Dict[str, Any]):
        """週次レポート生成"""
        report_path = project_root / "reports" / f"weekly_maintenance_{datetime.now():%Y%m%d}.md"
        report_path.parent.mkdir(exist_ok=True)

        report_content = f"""# 週次メンテナンスレポート

**実行日時**: {results['timestamp']}

## 実行結果

- **深度クリーンアップ**: {results['deep_cleanup_count']}ファイル
- **ガバナンス違反**: {results['governance_violations']}件

## 推奨事項

{chr(10).join(f"- {rec}" for rec in results['recommendations'])}

## 次回アクション

- 命名規則違反の修正
- 不要ファイルの削除検討
- ディレクトリ構造の最適化

---
*自動生成 - TDR-003 Auto Maintenance*
"""

        report_path.write_text(report_content, encoding="utf-8")
        logger.info(f"週次レポート生成: {report_path}")


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Tools Directory 自動メンテナンス")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily", help="メンテナンスモード")

    args = parser.parse_args()

    maintenance = AutoMaintenance()

    if args.mode == "daily":
        results = maintenance.run_daily_maintenance()
    elif args.mode == "weekly":
        results = maintenance.run_weekly_maintenance()

    # 結果をJSONで保存
    log_path = (
        project_root / "logs" / f"maintenance_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    log_path.parent.mkdir(exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"メンテナンスログ保存: {log_path}")


if __name__ == "__main__":
    main()
