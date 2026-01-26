#!/usr/bin/env python3
"""
統合テストランナー

Level 1-4の全ワークフローテストを統合実行するためのランナー
"""

import json
import os
import pytest
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TestLevel(Enum):
    """テストレベル"""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    ALL = "all"


class TestResult(Enum):
    """テスト結果"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestSummary:
    """テスト実行結果サマリー"""

    level: str
    test_file: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    result: TestResult
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返却"""
        data = asdict(self)
        data["result"] = self.result.value
        return data


class IntegratedTestRunner:
    """統合テストランナークラス"""

    def __init__(self):
        self.test_root = Path(__file__).parent
        self.test_results: List[TestSummary] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # テストレベル別設定
        self.test_levels = {
            TestLevel.LEVEL_1: {
                "name": "基本ワークフローテスト",
                "file": "tests/workflow/test_basic_workflow.py",
                "description": "入力パス検証、トラッカーID検証、ワークスペース作成",
            },
            TestLevel.LEVEL_2: {
                "name": "品質ワークフローテスト",
                "file": "tests/workflow/test_quality_workflow.py",
                "description": "SAM+YOLO抽出、品質評価、ダッシュボード生成",
            },
            TestLevel.LEVEL_3: {
                "name": "統計分析ワークフローテスト",
                "file": "tests/workflow/test_statistical_workflow.py",
                "description": "Cohen's d計算、p値検定、Google Sheets統合",
            },
            TestLevel.LEVEL_4: {
                "name": "承認ワークフローテスト",
                "file": "tests/workflow/test_approval_workflow.py",
                "description": "Pushover通知、承認プロセス、進捗管理",
            },
        }

    def run_tests(
        self,
        level: TestLevel = TestLevel.ALL,
        verbose: bool = True,
        stop_on_failure: bool = False,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        テスト実行

        Args:
            level: 実行するテストレベル
            verbose: 詳細出力フラグ
            stop_on_failure: 失敗時停止フラグ
            output_file: 結果出力ファイルパス

        Returns:
            テスト実行結果の辞書
        """
        self.start_time = datetime.now()
        self.test_results.clear()

        if verbose:
            print(f"🚀 統合テスト開始: {self.start_time}")
            print("=" * 60)

        # 実行対象テストレベル決定
        levels_to_run = []
        if level == TestLevel.ALL:
            levels_to_run = [
                TestLevel.LEVEL_1,
                TestLevel.LEVEL_2,
                TestLevel.LEVEL_3,
                TestLevel.LEVEL_4,
            ]
        else:
            levels_to_run = [level]

        # テスト実行
        overall_success = True
        for test_level in levels_to_run:
            if verbose:
                level_info = self.test_levels[test_level]
                print(f"\n📋 {level_info['name']} 実行中...")
                print(f"   📄 ファイル: {level_info['file']}")
                print(f"   📝 概要: {level_info['description']}")

            # 個別テスト実行
            result = self._run_single_test_level(test_level, verbose)
            self.test_results.append(result)

            if result.result == TestResult.FAILED:
                overall_success = False
                if stop_on_failure:
                    if verbose:
                        print(f"❌ {level_info['name']} 失敗により停止")
                    break

        self.end_time = datetime.now()

        # 結果サマリー生成
        summary = self._generate_summary(overall_success, verbose)

        # 結果ファイル出力
        if output_file:
            self._save_results_to_file(summary, output_file)

        return summary

    def _run_single_test_level(self, level: TestLevel, verbose: bool) -> TestSummary:
        """
        個別テストレベル実行

        Args:
            level: テストレベル
            verbose: 詳細出力フラグ

        Returns:
            テスト結果サマリー
        """
        level_info = self.test_levels[level]
        test_file = level_info["file"]

        # テストファイル存在確認
        test_path = self.test_root / ".." / test_file
        if not test_path.exists():
            return TestSummary(
                level=level.value,
                test_file=test_file,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration_seconds=0.0,
                result=TestResult.ERROR,
                error_message=f"テストファイルが見つかりません: {test_file}",
            )

        # pytest実行
        start_time = time.time()

        # pytest引数構築
        pytest_args = [str(test_path), "-v", "--tb=short", "--disable-warnings"]

        if not verbose:
            pytest_args.append("-q")

        # pytest実行（結果をキャプチャ）
        try:
            # pytestの結果を取得（0=成功, 1=失敗, 2=中断, 3=内部エラー, 4=pytest使用エラー, 5=テスト未発見）
            exit_code = pytest.main(pytest_args)

            duration = time.time() - start_time

            # 結果解析（簡易版 - 実際のテスト数は別途取得が必要）
            # この実装では概算値を使用
            estimated_test_count = self._estimate_test_count(test_path)

            if exit_code == 0:
                # 全テスト成功
                return TestSummary(
                    level=level.value,
                    test_file=test_file,
                    total_tests=estimated_test_count,
                    passed=estimated_test_count,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=duration,
                    result=TestResult.PASSED,
                )
            elif exit_code == 1:
                # テスト失敗あり
                return TestSummary(
                    level=level.value,
                    test_file=test_file,
                    total_tests=estimated_test_count,
                    passed=max(0, estimated_test_count - 1),  # 概算
                    failed=1,  # 概算
                    skipped=0,
                    errors=0,
                    duration_seconds=duration,
                    result=TestResult.FAILED,
                    error_message="1つ以上のテストが失敗しました",
                )
            else:
                # エラー発生
                return TestSummary(
                    level=level.value,
                    test_file=test_file,
                    total_tests=estimated_test_count,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    duration_seconds=duration,
                    result=TestResult.ERROR,
                    error_message=f"pytest実行エラー (exit code: {exit_code})",
                )

        except Exception as e:
            duration = time.time() - start_time
            return TestSummary(
                level=level.value,
                test_file=test_file,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration_seconds=duration,
                result=TestResult.ERROR,
                error_message=f"テスト実行例外: {str(e)}",
            )

    def _estimate_test_count(self, test_file_path: Path) -> int:
        """
        テストファイルのテスト数概算

        Args:
            test_file_path: テストファイルパス

        Returns:
            推定テスト数
        """
        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # def test_ で始まる行をカウント
                test_count = content.count("def test_")
                return max(test_count, 1)
        except Exception:
            return 1  # デフォルト値

    def _generate_summary(self, overall_success: bool, verbose: bool) -> Dict[str, Any]:
        """
        テスト結果サマリー生成

        Args:
            overall_success: 全体成功フラグ
            verbose: 詳細出力フラグ

        Returns:
            サマリー辞書
        """
        total_duration = (self.end_time - self.start_time).total_seconds()

        # 集計値計算
        total_tests = sum(result.total_tests for result in self.test_results)
        total_passed = sum(result.passed for result in self.test_results)
        total_failed = sum(result.failed for result in self.test_results)
        total_skipped = sum(result.skipped for result in self.test_results)
        total_errors = sum(result.errors for result in self.test_results)

        # レベル別成功/失敗カウント
        levels_passed = sum(1 for result in self.test_results if result.result == TestResult.PASSED)
        levels_failed = sum(1 for result in self.test_results if result.result == TestResult.FAILED)
        levels_errors = sum(1 for result in self.test_results if result.result == TestResult.ERROR)

        summary = {
            "overall_result": "PASSED" if overall_success else "FAILED",
            "execution_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "levels_executed": len(self.test_results),
            },
            "aggregate_stats": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0.0,
            },
            "level_stats": {
                "levels_passed": levels_passed,
                "levels_failed": levels_failed,
                "levels_errors": levels_errors,
                "level_success_rate": (levels_passed / len(self.test_results) * 100)
                if self.test_results
                else 0.0,
            },
            "detailed_results": [result.to_dict() for result in self.test_results],
        }

        # コンソール出力
        if verbose:
            self._print_summary(summary)

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """
        コンソールサマリー出力

        Args:
            summary: サマリー辞書
        """
        print("\n" + "=" * 60)
        print("📊 統合テスト結果サマリー")
        print("=" * 60)

        overall = summary["overall_result"]
        emoji = "✅" if overall == "PASSED" else "❌"
        print(f"{emoji} 全体結果: {overall}")

        exec_info = summary["execution_info"]
        print(f"⏰ 実行時間: {exec_info['total_duration_seconds']:.2f}秒")
        print(f"📋 実行レベル数: {exec_info['levels_executed']}")

        agg_stats = summary["aggregate_stats"]
        print(f"🧪 総テスト数: {agg_stats['total_tests']}")
        print(f"✅ 成功: {agg_stats['total_passed']}")
        print(f"❌ 失敗: {agg_stats['total_failed']}")
        print(f"⚠️  エラー: {agg_stats['total_errors']}")
        print(f"📈 成功率: {agg_stats['success_rate']:.1f}%")

        level_stats = summary["level_stats"]
        print(
            f"📊 レベル成功率: {level_stats['level_success_rate']:.1f}% ({level_stats['levels_passed']}/{exec_info['levels_executed']})"
        )

        print("\n📋 レベル別詳細:")
        for result in summary["detailed_results"]:
            level_name = self.test_levels[TestLevel(result["level"])]["name"]
            result_emoji = (
                "✅"
                if result["result"] == "passed"
                else ("❌" if result["result"] == "failed" else "⚠️")
            )

            print(f"  {result_emoji} {level_name}")
            print(f"     📄 {result['test_file']}")
            print(f"     🧪 テスト数: {result['total_tests']}")
            print(f"     ⏰ 実行時間: {result['duration_seconds']:.2f}秒")

            if result.get("error_message"):
                print(f"     ❌ エラー: {result['error_message']}")

        print("=" * 60)

    def _save_results_to_file(self, summary: Dict[str, Any], output_file: str):
        """
        結果ファイル保存

        Args:
            summary: サマリー辞書
            output_file: 出力ファイルパス
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"📁 結果ファイル保存完了: {output_path}")

        except Exception as e:
            print(f"❌ 結果ファイル保存失敗: {e}")


def main():
    """CLI メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="統合ワークフローテストランナー")
    parser.add_argument(
        "--level",
        choices=["level_1", "level_2", "level_3", "level_4", "all"],
        default="all",
        help="実行するテストレベル",
    )
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    parser.add_argument("--stop-on-failure", action="store_true", help="失敗時に停止")
    parser.add_argument("--output", help="結果出力ファイルパス")

    args = parser.parse_args()

    # テストランナー実行
    runner = IntegratedTestRunner()

    level = TestLevel(args.level)

    try:
        summary = runner.run_tests(
            level=level,
            verbose=args.verbose,
            stop_on_failure=args.stop_on_failure,
            output_file=args.output,
        )

        # 終了コード設定
        if summary["overall_result"] == "PASSED":
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによってテストが中断されました")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
