#!/usr/bin/env python3
"""
重要ファイル保護チェックリストツール
ファイル移動・整理前の必須確認プロセスを自動化
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FileProtectionChecker:
    """重要ファイル保護チェッカー"""

    def __init__(self):
        self.critical_patterns = [
            # Week N成果物
            r".*week\d+.*\.py",
            r".*stable.*\.py",
            r".*final.*\.py",
            # 品質改善機能
            r".*quality.*\.py",
            r".*evaluation.*\.py",
            r".*sci.*\.py",
            # Core実装
            r"core/.*\.py",
            r"features/.*\.py",
            # 最新実装
            r".*_v\d+.*\.py",
            r".*improved.*\.py",
            r".*enhanced.*\.py",
        ]

        self.protection_levels = {
            "CRITICAL": ["week", "stable", "core/", "features/"],
            "HIGH": ["quality", "evaluation", "improved", "enhanced"],
            "MEDIUM": ["tools/", "config/", "utils/"],
            "LOW": [".log", ".pid", ".json", "temp_", "test_"],
        }

    def classify_file_importance(self, file_path: str) -> str:
        """ファイル重要度分類"""
        file_path = file_path.lower()

        for level, patterns in self.protection_levels.items():
            for pattern in patterns:
                if pattern in file_path:
                    return level

        return "UNKNOWN"

    def check_dependencies(self, file_path: str) -> List[str]:
        """依存関係チェック"""
        dependencies = []

        try:
            # gitで参照しているファイルを検索
            result = subprocess.run(
                ["grep", "-r", "--include=*.py", os.path.basename(file_path), "."],
                capture_output=True,
                text=True,
                cwd=Path(file_path).parent.parent,
            )

            if result.stdout:
                dependencies = result.stdout.strip().split("\n")

        except Exception as e:
            print(f"依存関係チェックエラー: {e}")

        return dependencies

    def analyze_file_content(self, file_path: str) -> Dict[str, any]:
        """ファイル内容分析"""
        analysis = {
            "has_classes": False,
            "has_functions": False,
            "has_imports": False,
            "line_count": 0,
            "key_features": [],
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                analysis["line_count"] = len(lines)
                analysis["has_classes"] = "class " in content
                analysis["has_functions"] = "def " in content
                analysis["has_imports"] = any(
                    line.strip().startswith("import") or line.strip().startswith("from")
                    for line in lines
                )

                # 重要キーワード検出
                important_keywords = [
                    "quality",
                    "evaluation",
                    "stable",
                    "improved",
                    "sam",
                    "yolo",
                    "extraction",
                    "segment",
                ]

                for keyword in important_keywords:
                    if keyword.lower() in content.lower():
                        analysis["key_features"].append(keyword)

        except Exception as e:
            print(f"ファイル分析エラー: {e}")

        return analysis

    def generate_protection_report(self, files: List[str]) -> Dict[str, any]:
        """保護チェックレポート生成"""
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(files),
            "files_analysis": [],
            "protection_summary": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0},
            "critical_files_requiring_confirmation": [],
        }

        for file_path in files:
            if not os.path.exists(file_path):
                continue

            importance = self.classify_file_importance(file_path)
            dependencies = self.check_dependencies(file_path)
            content_analysis = self.analyze_file_content(file_path)

            file_analysis = {
                "path": file_path,
                "importance_level": importance,
                "dependencies_count": len(dependencies),
                "dependencies": dependencies[:5],  # 最初の5件のみ
                "content_analysis": content_analysis,
                "requires_user_confirmation": importance in ["CRITICAL", "HIGH"],
            }

            report["files_analysis"].append(file_analysis)
            report["protection_summary"][importance] += 1

            if importance in ["CRITICAL", "HIGH"]:
                report["critical_files_requiring_confirmation"].append(
                    {
                        "path": file_path,
                        "importance": importance,
                        "reason": self._get_protection_reason(file_path, content_analysis),
                    }
                )

        return report

    def _get_protection_reason(self, file_path: str, content_analysis: Dict) -> str:
        """保護理由の生成"""
        reasons = []

        if "stable" in file_path.lower():
            reasons.append("安定版実装")
        if "quality" in content_analysis.get("key_features", []):
            reasons.append("品質改善機能")
        if "improved" in file_path.lower():
            reasons.append("改善実装")
        if content_analysis.get("has_classes") and content_analysis.get("line_count", 0) > 100:
            reasons.append("大規模実装")
        if "week" in file_path.lower():
            reasons.append("Week成果物")

        return ", ".join(reasons) if reasons else "要確認"

    def interactive_confirmation(self, critical_files: List[Dict]) -> Dict[str, bool]:
        """対話的確認プロセス"""
        confirmations = {}

        if not critical_files:
            print("✅ 重要ファイルは検出されませんでした")
            return confirmations

        print("\n🚨 重要ファイル保護確認が必要です")
        print("=" * 60)

        for file_info in critical_files:
            path = file_info["path"]
            importance = file_info["importance"]
            reason = file_info["reason"]

            print(f"\n📁 ファイル: {path}")
            print(f"⚠️  重要度: {importance}")
            print(f"💡 理由: {reason}")

            while True:
                response = input(f"\nこのファイルを移動・削除してもよろしいですか? (y/N): ").strip().lower()
                if response in ["y", "yes"]:
                    confirmations[path] = True
                    print("✅ 移動許可")
                    break
                elif response in ["n", "no", ""]:
                    confirmations[path] = False
                    print("🛡️  保護対象")
                    break
                else:
                    print("y または n で回答してください")

        return confirmations

    def save_report(self, report: Dict, output_path: str):
        """レポート保存"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 保護チェックレポート保存: {output_path}")


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="重要ファイル保護チェック")
    parser.add_argument("files", nargs="+", help="チェック対象ファイル")
    parser.add_argument("--output", default="file_protection_report.json", help="レポート出力パス")
    parser.add_argument("--interactive", action="store_true", help="対話的確認モード")

    args = parser.parse_args()

    checker = FileProtectionChecker()

    print("🔍 重要ファイル保護チェック開始")
    print(f"📊 対象ファイル数: {len(args.files)}")

    # レポート生成
    report = checker.generate_protection_report(args.files)

    # サマリー表示
    print("\n📊 保護レベル分析:")
    for level, count in report["protection_summary"].items():
        if count > 0:
            print(f"  {level}: {count}ファイル")

    # 重要ファイル確認
    critical_files = report["critical_files_requiring_confirmation"]
    if critical_files:
        print(f"\n⚠️  ユーザー確認必要: {len(critical_files)}ファイル")

        if args.interactive:
            confirmations = checker.interactive_confirmation(critical_files)
            report["user_confirmations"] = confirmations

            # 保護対象ファイル表示
            protected_files = [path for path, allowed in confirmations.items() if not allowed]
            if protected_files:
                print(f"\n🛡️  保護対象ファイル: {len(protected_files)}件")
                for path in protected_files:
                    print(f"    - {path}")

    # レポート保存
    checker.save_report(report, args.output)

    print("\n✅ 重要ファイル保護チェック完了")


if __name__ == "__main__":
    main()
