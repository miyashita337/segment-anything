#!/usr/bin/env python3
"""
パス準拠性監査ツール
日次20:00自動実行でプロジェクト全体のパス準拠性をチェック・Pushover通知
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.workspace_config import WorkspaceConfig

# Pushover通知（オプション）
try:
    import requests

    PUSHOVER_AVAILABLE = True
except ImportError:
    PUSHOVER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ComplianceIssue:
    """準拠性問題の詳細"""

    file_path: str
    line_number: int
    issue_type: str
    severity: str  # HIGH, MEDIUM, LOW
    description: str
    suggested_fix: Optional[str] = None


class PathComplianceAuditor:
    """パス準拠性監査システム"""

    # 禁止パターン（高優先度）
    HIGH_SEVERITY_PATTERNS = [
        (r'Path\s*\(\s*["\'][^$]*/dashboard_output["\']', "ハードコード相対パス"),
        (r'Path\s*\(\s*["\'][^$]*/results["\']', "ハードコード相対パス"),
        (r'Path\s*\(\s*["\'][^$]*/temp["\']', "ハードコード相対パス"),
        (r'output_dir\s*=\s*["\'][^$]*/', "ハードコード出力パス"),
        (r'"/mnt/c/AItools/[^"]*"', "ハードコード絶対パス"),
    ]

    # 中優先度パターン
    MEDIUM_SEVERITY_PATTERNS = [
        (r'Path\s*\(\s*["\']\./', "カレントディレクトリ基準"),
        (r'Path\s*\(\s*["\']\.\./', "相対パス（親ディレクトリ）"),
        (r'os\.path\.join\s*\([^)]*["\'][^$]*/', "os.pathハードコード"),
    ]

    # 低優先度パターン
    LOW_SEVERITY_PATTERNS = [
        (r'["\'][^"\']*results[^"\']*["\']', "結果ディレクトリ命名"),
        (r'["\'][^"\']*output[^"\']*["\']', "出力ディレクトリ命名"),
    ]

    # 除外ファイルパターン
    EXCLUDE_PATTERNS = [
        r".*\.git/.*",
        r".*__pycache__/.*",
        r".*\.pyc$",
        r".*\.backup$",
        r".*/deprecated/.*",
        r".*/test_.*\.py$",  # テストファイルは除外
        r".*audit_path_compliance\.py$",  # 自分自身は除外
    ]

    def __init__(self, project_root: Optional[Path] = None):
        """
        初期化

        Args:
            project_root: プロジェクトルート（未指定時は自動検出）
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.workspace_config = WorkspaceConfig()
        self.issues: List[ComplianceIssue] = []

    def should_exclude_file(self, file_path: Path) -> bool:
        """
        ファイルが除外対象かチェック

        Args:
            file_path: チェック対象ファイル

        Returns:
            除外対象フラグ
        """
        file_str = str(file_path)
        return any(re.match(pattern, file_str) for pattern in self.EXCLUDE_PATTERNS)

    def scan_file(self, file_path: Path) -> List[ComplianceIssue]:
        """
        ファイル内のパス準拠性をスキャン

        Args:
            file_path: スキャン対象ファイル

        Returns:
            発見された問題リスト
        """
        if self.should_exclude_file(file_path):
            return []

        issues = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # 高優先度パターンチェック
                for pattern, description in self.HIGH_SEVERITY_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(
                            ComplianceIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type="HIGH_PRIORITY_PATH",
                                severity="HIGH",
                                description=f"{description}: {line.strip()}",
                                suggested_fix="WorkspaceConfig またはOutputPathManager使用を検討",
                            )
                        )

                # 中優先度パターンチェック
                for pattern, description in self.MEDIUM_SEVERITY_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(
                            ComplianceIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type="MEDIUM_PRIORITY_PATH",
                                severity="MEDIUM",
                                description=f"{description}: {line.strip()}",
                                suggested_fix="相対パスの見直しを推奨",
                            )
                        )

                # 低優先度パターンチェック
                for pattern, description in self.LOW_SEVERITY_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(
                            ComplianceIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type="LOW_PRIORITY_NAMING",
                                severity="LOW",
                                description=f"{description}: {line.strip()}",
                                suggested_fix="命名規則の統一を検討",
                            )
                        )

        except Exception as e:
            logger.warning(f"ファイルスキャンエラー {file_path}: {e}")

        return issues

    def scan_project(self) -> Dict[str, int]:
        """
        プロジェクト全体をスキャン

        Returns:
            集計結果辞書
        """
        logger.info(f"🔍 プロジェクトスキャン開始: {self.project_root}")

        self.issues = []
        file_count = 0

        # Python ファイルをスキャン
        for py_file in self.project_root.rglob("*.py"):
            if self.should_exclude_file(py_file):
                continue

            file_issues = self.scan_file(py_file)
            self.issues.extend(file_issues)
            file_count += 1

        # シェルスクリプトもスキャン
        for sh_file in self.project_root.rglob("*.sh"):
            if self.should_exclude_file(sh_file):
                continue

            file_issues = self.scan_file(sh_file)
            self.issues.extend(file_issues)
            file_count += 1

        # 集計
        summary = {
            "total_files": file_count,
            "total_issues": len(self.issues),
            "high_issues": len([i for i in self.issues if i.severity == "HIGH"]),
            "medium_issues": len([i for i in self.issues if i.severity == "MEDIUM"]),
            "low_issues": len([i for i in self.issues if i.severity == "LOW"]),
        }

        logger.info(f"✅ スキャン完了: {file_count}ファイル, {len(self.issues)}問題発見")
        return summary

    def generate_report(self) -> str:
        """
        監査レポート生成

        Returns:
            レポートテキスト
        """
        if not self.issues:
            return "🎉 パス準拠性問題は発見されませんでした！"

        lines = []
        lines.append("📋 パス準拠性監査レポート")
        lines.append(f"⏰ 実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 集計サマリー
        high_count = len([i for i in self.issues if i.severity == "HIGH"])
        medium_count = len([i for i in self.issues if i.severity == "MEDIUM"])
        low_count = len([i for i in self.issues if i.severity == "LOW"])

        lines.append("📊 問題集計:")
        lines.append(f"  🚨 高優先度: {high_count}件")
        lines.append(f"  ⚠️  中優先度: {medium_count}件")
        lines.append(f"  ℹ️  低優先度: {low_count}件")
        lines.append(f"  📄 総問題数: {len(self.issues)}件")
        lines.append("")

        # 優先度別詳細
        for severity in ["HIGH", "MEDIUM", "LOW"]:
            severity_issues = [i for i in self.issues if i.severity == severity]
            if not severity_issues:
                continue

            icon = "🚨" if severity == "HIGH" else "⚠️" if severity == "MEDIUM" else "ℹ️"
            lines.append(f"{icon} {severity}優先度問題 ({len(severity_issues)}件):")

            for issue in severity_issues[:10]:  # 最初の10件のみ表示
                relative_path = Path(issue.file_path).relative_to(self.project_root)
                lines.append(f"  📁 {relative_path}:{issue.line_number}")
                lines.append(f"    💡 {issue.description}")
                if issue.suggested_fix:
                    lines.append(f"    🔧 {issue.suggested_fix}")
                lines.append("")

            if len(severity_issues) > 10:
                lines.append(f"    ... その他{len(severity_issues) - 10}件")
                lines.append("")

        return "\n".join(lines)

    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """
        レポート保存

        Args:
            output_path: 保存先（未指定時は自動生成）

        Returns:
            保存先パス
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.project_root / f"audit_report_{timestamp}.md"

        report_content = self.generate_report()
        output_path.write_text(report_content, encoding="utf-8")

        logger.info(f"📄 レポート保存: {output_path}")
        return output_path

    def notify_success(self, summary: Dict[str, int]) -> bool:
        """
        Pushover通知送信

        Args:
            summary: 集計結果

        Returns:
            送信成功フラグ
        """
        if not PUSHOVER_AVAILABLE:
            logger.warning("Pushover通知：requestsライブラリが利用できません")
            return False

        # Pushover設定読み込み
        config_path = self.project_root / "config" / "pushover.json"
        if not config_path.exists():
            logger.warning(f"Pushover設定ファイルが見つかりません: {config_path}")
            return False

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # 通知メッセージ作成
            high_issues = summary["high_issues"]
            total_issues = summary["total_issues"]

            if total_issues == 0:
                title = "✅ パス準拠性監査 - 問題なし"
                message = f"スキャン完了: {summary['total_files']}ファイル\n問題は発見されませんでした！"
                priority = 0
            elif high_issues > 0:
                title = "🚨 パス準拠性監査 - 高優先度問題あり"
                message = f"""スキャン結果:
📁 {summary['total_files']}ファイル
🚨 高優先度: {high_issues}件
⚠️ 中優先度: {summary['medium_issues']}件
ℹ️ 低優先度: {summary['low_issues']}件

詳細レポートを確認してください。"""
                priority = 1
            else:
                title = "⚠️ パス準拠性監査 - 軽微な問題"
                message = f"""スキャン結果:
📁 {summary['total_files']}ファイル
⚠️ 中優先度: {summary['medium_issues']}件
ℹ️ 低優先度: {summary['low_issues']}件"""
                priority = 0

            # Pushover API リクエスト
            # TODO: global_pushover.pyに移行必要
            try:
                from features.common.notification.global_pushover import send_pushover_notification

                success = send_pushover_notification(
                    message=message, title=title, priority=priority
                )
                if success:
                    logger.info("✅ Pushover通知送信成功")
                    return True
                else:
                    logger.error("❌ Pushover通知送信失敗")
                    return False
            except ImportError:
                logger.warning("⚠️ Pushover通知モジュールが見つかりません")
                return False

        except Exception as e:
            logger.error(f"❌ Pushover通知送信エラー: {e}")
            return False


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="パス準拠性監査ツール")
    parser.add_argument("--project-root", type=Path, help="プロジェクトルート指定")
    parser.add_argument("--output", type=Path, help="レポート出力先")
    parser.add_argument("--no-pushover", action="store_true", help="Pushover通知無効化")
    parser.add_argument("--severity", choices=["HIGH", "MEDIUM", "LOW"], help="指定優先度のみ表示")

    args = parser.parse_args()

    # 監査実行
    auditor = PathComplianceAuditor(args.project_root)

    print("🔍 パス準拠性監査開始...")
    start_time = time.time()

    summary = auditor.scan_project()

    elapsed = time.time() - start_time
    print(f"✅ 監査完了 ({elapsed:.2f}秒)")

    # レポート生成・表示
    report = auditor.generate_report()
    print("\n" + report)

    # レポート保存
    if args.output or summary["total_issues"] > 0:
        report_path = auditor.save_report(args.output)
        print(f"\n📄 詳細レポート: {report_path}")

    # Pushover通知
    if not args.no_pushover and summary["total_issues"] > 0:
        auditor.notify_success(summary)

    # 終了コード設定
    if summary["high_issues"] > 0:
        sys.exit(2)  # 高優先度問題あり
    elif summary["medium_issues"] > 0:
        sys.exit(1)  # 中優先度問題あり
    else:
        sys.exit(0)  # 問題なし


if __name__ == "__main__":
    main()
