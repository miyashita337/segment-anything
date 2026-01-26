#!/usr/bin/env python3
"""
ダッシュボード品質チェック自動化スクリプト
仕様書: docs/checklists/dashboard_quality_checklist.md

INTG-087: 品質保証冪等性システム統合版
- 複数実行での一貫性チェック
- 冪等性保証テスト
- 統計分析結果検証

実行例:
python tools/testing/dashboard_quality_validator.py TEST-001
python tools/testing/dashboard_quality_validator.py QUAL-042 --server-url http://100.123.241.106:8088
python tools/testing/dashboard_quality_validator.py INTG-087 --idempotency-test 3
"""

import argparse
import json
import os
import re
import requests
import subprocess
import sys
from pathlib import Path
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Optional, Tuple


class DashboardQualityValidator:
    """ダッシュボード品質検証自動化クラス"""

    def __init__(self, tracker_id: str, server_url: str = "http://100.123.241.106:8088"):
        self.tracker_id = tracker_id
        self.server_url = server_url
        self.auth = HTTPBasicAuth("admin", "secure_track_2025_q3_8f9a")
        self.workspace_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}")
        self.dashboard_html_path = self.workspace_dir / "dashboard" / "dashboard.html"
        self.extraction_result_path = self.workspace_dir / "extraction_result.json"

        self.validation_results = {
            "section_a": {"passed": False, "details": []},
            "section_b": {"passed": False, "details": []},
            "section_c": {"passed": False, "details": []},
            "section_d": {"passed": False, "details": []},
        }

    def run_full_validation(self) -> Dict:
        """完全品質チェック実行"""
        print(f"🔍 {self.tracker_id} ダッシュボード品質チェック開始")
        print("=" * 60)

        # Section A: データ構造検証
        self._validate_section_a()

        # Section B: ダッシュボードHTML生成検証
        self._validate_section_b()

        # Section C: サーバーアクセス検証
        self._validate_section_c()

        # Section D: 品質保証・最終確認
        self._validate_section_d()

        # 総合判定
        self._generate_final_report()

        return self.validation_results

    def _validate_section_a(self) -> None:
        """Section A: データ構造検証実行"""
        print("📊 Section A: データ構造検証実行中...")
        results = []

        # 1. extraction_result.json存在確認
        if not self.extraction_result_path.exists():
            results.append("❌ extraction_result.json が存在しません")
            self.validation_results["section_a"] = {"passed": False, "details": results}
            return

        # 2. JSONファイル読み込み・構造確認
        try:
            with open(self.extraction_result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append("✅ extraction_result.json 正常読み込み")
        except Exception as e:
            results.append(f"❌ extraction_result.json 読み込みエラー: {e}")
            self.validation_results["section_a"] = {"passed": False, "details": results}
            return

        # 3. 必須キー存在確認
        required_keys = [
            "tracker_id",
            "total_images",
            "successful_extractions",
            "average_quality_score",
            "extraction_results",
        ]

        missing_keys = []
        for key in required_keys:
            if key not in data:
                missing_keys.append(key)

        if missing_keys:
            results.append(f"❌ 必須キー不足: {missing_keys}")
            self.validation_results["section_a"] = {"passed": False, "details": results}
            return
        else:
            results.append(f"✅ 必須キー全て存在: {required_keys}")

        # 4. 統計分析セクション確認
        if "statistical_analysis" not in data:
            results.append("❌ statistical_analysis セクション不足")
        else:
            stats = data["statistical_analysis"]
            required_stats = ["p_value", "effect_size", "improvement_rate", "significance"]
            missing_stats = [key for key in required_stats if key not in stats]

            if missing_stats:
                results.append(f"❌ 統計分析項目不足: {missing_stats}")
            else:
                results.append(f"✅ 統計分析項目完備: {required_stats}")

        # 5. データ値妥当性確認
        total_images = data.get("total_images", 0)
        successful = data.get("successful_extractions", 0)
        avg_score = data.get("average_quality_score", 0)

        if total_images <= 0:
            results.append("❌ total_images が0以下")
        if successful < 0 or successful > total_images:
            results.append("❌ successful_extractions が無効値")
        if not (0 <= avg_score <= 1):
            results.append("❌ average_quality_score が範囲外")

        if len([r for r in results if "❌" in r]) == 0:
            results.append("✅ データ値妥当性確認完了")

        # Section A判定
        section_a_passed = all("❌" not in result for result in results)
        self.validation_results["section_a"] = {"passed": section_a_passed, "details": results}

        if section_a_passed:
            print("✅ Section A: データ構造検証 - PASS")
        else:
            print("❌ Section A: データ構造検証 - FAIL")

    def _validate_section_b(self) -> None:
        """Section B: ダッシュボードHTML生成検証"""
        print("📊 Section B: ダッシュボードHTML生成検証中...")
        results = []

        # 1. dashboard.html存在確認
        if not self.dashboard_html_path.exists():
            results.append("❌ dashboard.html が存在しません")
            self.validation_results["section_b"] = {"passed": False, "details": results}
            return

        # 2. HTMLファイル読み込み
        try:
            with open(self.dashboard_html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            results.append("✅ dashboard.html 正常読み込み")
        except Exception as e:
            results.append(f"❌ dashboard.html 読み込みエラー: {e}")
            self.validation_results["section_b"] = {"passed": False, "details": results}
            return

        # 3. 統計分析7項目表示確認
        required_stats = [
            "Current(平均品質スコア)",
            "BaseLine",
            "p値",
            "効果サイズ、Cohen's d",
            "改善率",
            "統計的有意性",
            "信頼区間",
        ]

        missing_stats = []
        for stat in required_stats:
            if stat not in html_content:
                missing_stats.append(stat)

        if missing_stats:
            results.append(f"❌ 統計分析項目表示不足: {missing_stats}")
        else:
            results.append(f"✅ 統計分析7項目表示確認: {required_stats}")

        # 4. 基本品質指標表示確認
        required_metrics = ["総画像数", "平均品質スコア", "成功画像数", "要改善数"]
        missing_metrics = []
        for metric in required_metrics:
            if metric not in html_content:
                missing_metrics.append(metric)

        if missing_metrics:
            results.append(f"❌ 基本品質指標表示不足: {missing_metrics}")
        else:
            results.append(f"✅ 基本品質指標表示確認: {required_metrics}")

        # 5. 品質分布表示確認
        quality_labels = ["高品質", "中品質", "低品質", "要改善"]
        missing_labels = []
        for label in quality_labels:
            if label not in html_content:
                missing_labels.append(label)

        if missing_labels:
            results.append(f"❌ 品質分布ラベル不足: {missing_labels}")
        else:
            results.append(f"✅ 品質分布表示確認: {quality_labels}")

        # 6. N/A表示バグチェック（重要バグ）
        na_count = html_content.count(">N/A<")
        if na_count > 0:
            results.append(f"❌ N/A表示バグ検出: {na_count}箇所")
        else:
            results.append("✅ N/A表示バグなし")

        # Section B判定
        section_b_passed = all("❌" not in result for result in results)
        self.validation_results["section_b"] = {"passed": section_b_passed, "details": results}

        if section_b_passed:
            print("✅ Section B: ダッシュボードHTML生成検証 - PASS")
        else:
            print("❌ Section B: ダッシュボードHTML生成検証 - FAIL")

    def _validate_section_c(self) -> None:
        """Section C: サーバーアクセス検証"""
        print("📊 Section C: サーバーアクセス検証中...")
        results = []

        # 1. API経由ダッシュボードリスト確認
        try:
            api_url = f"{self.server_url}/api/dashboards"
            response = requests.get(api_url, auth=self.auth, timeout=10)

            if response.status_code == 200:
                results.append("✅ API接続成功")
                dashboards = response.json().get("dashboards", [])

                # 対象トラッカー存在確認
                tracker_found = any(d.get("tracker") == self.tracker_id for d in dashboards)
                if tracker_found:
                    results.append(f"✅ {self.tracker_id} API経由で認識")
                else:
                    results.append(f"❌ {self.tracker_id} API経由で未認識")
            else:
                results.append(f"❌ API接続失敗: {response.status_code}")

        except Exception as e:
            results.append(f"❌ API接続エラー: {e}")

        # 2. 統合UIアクセス確認
        try:
            ui_url = f"{self.server_url}/tracker/{self.tracker_id}"
            response = requests.get(ui_url, auth=self.auth, timeout=10)

            if response.status_code == 200:
                results.append("✅ 統合UI正常アクセス")

                # フレーム内容確認
                if f'src="/{self.tracker_id}/dashboard/dashboard.html"' in response.text:
                    results.append("✅ 統合UIフレーム正常設定")
                else:
                    results.append("❌ 統合UIフレーム設定異常")
            else:
                results.append(f"❌ 統合UIアクセス失敗: {response.status_code}")

        except Exception as e:
            results.append(f"❌ 統合UIアクセスエラー: {e}")

        # 3. 直接ダッシュボードアクセス確認
        try:
            direct_url = f"{self.server_url}/{self.tracker_id}/dashboard/dashboard.html"
            response = requests.get(direct_url, auth=self.auth, timeout=10)

            if response.status_code == 200:
                results.append("✅ 直接ダッシュボードアクセス成功")

                # ファイルサイズ確認（異常に小さくないか）
                content_length = len(response.content)
                if content_length > 1000:  # 1KB以上
                    results.append(f"✅ レスポンスサイズ正常: {content_length:,} bytes")
                else:
                    results.append(f"❌ レスポンスサイズ異常: {content_length} bytes")
            else:
                results.append(f"❌ 直接ダッシュボードアクセス失敗: {response.status_code}")

        except Exception as e:
            results.append(f"❌ 直接ダッシュボードアクセスエラー: {e}")

        # Section C判定
        section_c_passed = all("❌" not in result for result in results)
        self.validation_results["section_c"] = {"passed": section_c_passed, "details": results}

        if section_c_passed:
            print("✅ Section C: サーバーアクセス検証 - PASS")
        else:
            print("❌ Section C: サーバーアクセス検証 - FAIL")

    def _validate_section_d(self) -> None:
        """Section D: 品質保証・最終確認"""
        print("📊 Section D: 品質保証・最終確認中...")
        results = []

        # 1. 全セクション通過確認
        sections_passed = sum(
            1
            for section in ["section_a", "section_b", "section_c"]
            if self.validation_results[section]["passed"]
        )

        if sections_passed == 3:
            results.append("✅ Section A-C全て通過")
        else:
            results.append(f"❌ Section A-C未通過: {3-sections_passed}セクション失敗")

        # 2. Level S（仕様書完全準拠）判定
        level_s_criteria = [
            self.validation_results["section_a"]["passed"],
            self.validation_results["section_b"]["passed"],
            self.validation_results["section_c"]["passed"],
        ]

        if all(level_s_criteria):
            results.append("🏆 Level S（仕様書完全準拠）達成")
        else:
            results.append("❌ Level S（仕様書完全準拠）未達成")

        # Section D判定
        section_d_passed = all("❌" not in result for result in results)
        self.validation_results["section_d"] = {"passed": section_d_passed, "details": results}

        if section_d_passed:
            print("✅ Section D: 品質保証・最終確認 - PASS")
        else:
            print("❌ Section D: 品質保証・最終確認 - FAIL")

    def _generate_final_report(self) -> None:
        """最終レポート生成"""
        print("\n" + "=" * 60)
        print(f"📊 {self.tracker_id} ダッシュボード品質チェック結果")
        print("=" * 60)

        # セクション別結果表示
        section_names = {
            "section_a": "Section A: データ構造検証",
            "section_b": "Section B: ダッシュボードHTML生成検証",
            "section_c": "Section C: サーバーアクセス検証",
            "section_d": "Section D: 品質保証・最終確認",
        }

        for section_key, section_name in section_names.items():
            section_data = self.validation_results[section_key]
            status = "✅ PASS" if section_data["passed"] else "❌ FAIL"
            print(f"\n{section_name}: {status}")

            for detail in section_data["details"]:
                print(f"  {detail}")

        # 総合判定
        total_passed = sum(1 for section in self.validation_results.values() if section["passed"])
        print(f"\n📈 総合結果: {total_passed}/4 セクション通過")

        if total_passed == 4:
            print("🎉 全セクション通過！ダッシュボード品質保証完了")
            print("🏆 Level S（仕様書完全準拠）達成")
        else:
            print(f"⚠️  {4-total_passed}セクション未通過 - 修正が必要です")

    def save_report(self, output_path: Optional[str] = None) -> None:
        """レポートをファイル保存"""
        if output_path is None:
            output_path = f"/tmp/dashboard_quality_report_{self.tracker_id}.json"

        report_data = {
            "tracker_id": self.tracker_id,
            "server_url": self.server_url,
            "validation_timestamp": subprocess.check_output(["date"], text=True).strip(),
            "results": self.validation_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 レポート保存: {output_path}")

    def run_idempotency_test(self, iterations: int = 3) -> Dict:
        """
        冪等性テスト実行

        Args:
            iterations: テスト実行回数

        Returns:
            Dict: 冪等性テスト結果
        """
        print(f"\n🔄 INTG-087 冪等性テスト開始 (実行回数: {iterations})")
        print("=" * 60)

        # 複数回実行して結果を収集
        results = []
        extraction_results = []

        for i in range(iterations):
            print(f"\n📊 実行 {i+1}/{iterations}")

            # 品質ワークフロー実行
            try:
                cmd = f"bash tools/scripts/run_quality_workflow.sh {self.tracker_id}"
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd="/mnt/c/AItools/segment-anything",
                )

                if result.returncode == 0:
                    print(f"✅ 品質ワークフロー実行成功 (実行 {i+1})")
                else:
                    print(f"❌ 品質ワークフロー実行失敗 (実行 {i+1}): {result.stderr}")
                    continue

            except Exception as e:
                print(f"❌ 品質ワークフロー実行例外 (実行 {i+1}): {e}")
                continue

            # 実行結果読み込み
            if self.extraction_result_path.exists():
                with open(self.extraction_result_path, "r", encoding="utf-8") as f:
                    extraction_data = json.load(f)
                    extraction_results.append(extraction_data)

            # 品質チェック実行
            validation_result = self.run_full_validation()
            results.append(
                {
                    "iteration": i + 1,
                    "timestamp": subprocess.check_output(
                        ["date", "+%Y-%m-%d %H:%M:%S"], text=True
                    ).strip(),
                    "validation_result": validation_result,
                    "extraction_stats": extraction_data
                    if self.extraction_result_path.exists()
                    else None,
                }
            )

            # 間隔を空ける
            import time

            time.sleep(2)

        # 冪等性分析
        idempotency_analysis = self._analyze_idempotency(results, extraction_results)

        print("\n📊 冪等性テスト結果")
        print("=" * 40)
        print(f"実行回数: {len(results)}")
        print(f"一貫性スコア: {idempotency_analysis['consistency_score']:.2f}%")
        print(f"冪等性判定: {'✅ 合格' if idempotency_analysis['is_idempotent'] else '❌ 不合格'}")

        if idempotency_analysis["inconsistencies"]:
            print("\n⚠️ 検出された非一貫性:")
            for inconsistency in idempotency_analysis["inconsistencies"]:
                print(f"  - {inconsistency}")

        return {
            "iterations": len(results),
            "results": results,
            "idempotency_analysis": idempotency_analysis,
        }

    def _analyze_idempotency(self, results: List[Dict], extraction_results: List[Dict]) -> Dict:
        """冪等性分析"""
        if len(results) < 2:
            return {"consistency_score": 100.0, "is_idempotent": True, "inconsistencies": []}

        inconsistencies = []
        total_checks = 0
        consistent_checks = 0

        # 抽出結果の一貫性チェック
        if extraction_results:
            first_stats = extraction_results[0]

            for i, stats in enumerate(extraction_results[1:], 2):
                # 主要統計の比較
                key_metrics = ["total_images", "successful_extractions", "average_quality_score"]

                for metric in key_metrics:
                    total_checks += 1
                    if first_stats.get(metric) == stats.get(metric):
                        consistent_checks += 1
                    else:
                        inconsistencies.append(
                            f"実行{i}: {metric} = {stats.get(metric)} (実行1: {first_stats.get(metric)})"
                        )

        # バリデーション結果の一貫性チェック
        first_validation = results[0]["validation_result"]

        for i, result in enumerate(results[1:], 2):
            validation = result["validation_result"]

            for section, section_result in validation.items():
                total_checks += 1
                if section_result["passed"] == first_validation[section]["passed"]:
                    consistent_checks += 1
                else:
                    inconsistencies.append(
                        f"実行{i}: {section} = {'PASS' if section_result['passed'] else 'FAIL'} "
                        f"(実行1: {'PASS' if first_validation[section]['passed'] else 'FAIL'})"
                    )

        # 一貫性スコア計算
        consistency_score = (consistent_checks / max(total_checks, 1)) * 100
        is_idempotent = consistency_score >= 95.0  # 95%以上で冪等と判定

        return {
            "consistency_score": consistency_score,
            "is_idempotent": is_idempotent,
            "inconsistencies": inconsistencies,
            "total_checks": total_checks,
            "consistent_checks": consistent_checks,
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="ダッシュボード品質チェック自動化スクリプト",
        epilog="例: python tools/testing/dashboard_quality_validator.py TEST-001",
    )

    parser.add_argument("tracker_id", help="対象トラッカーID (例: TEST-001, QUAL-042)")
    parser.add_argument(
        "--server-url", default="http://100.123.241.106:8088", help="ダッシュボードサーバーURL"
    )
    parser.add_argument("--save-report", help="レポート保存パス")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    parser.add_argument(
        "--idempotency-test", type=int, metavar="N", help="INTG-087: 冪等性テスト実行（N回繰り返し、デフォルト3回）"
    )

    args = parser.parse_args()

    # バリデーター初期化
    validator = DashboardQualityValidator(args.tracker_id, args.server_url)

    # 冪等性テスト実行
    if args.idempotency_test:
        iterations = args.idempotency_test if args.idempotency_test > 0 else 3
        idempotency_results = validator.run_idempotency_test(iterations)

        # 冪等性テストレポート保存
        if args.save_report:
            idempotency_report_path = args.save_report.replace(".json", "_idempotency.json")
            with open(idempotency_report_path, "w", encoding="utf-8") as f:
                json.dump(idempotency_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 冪等性テストレポート保存: {idempotency_report_path}")

        # 冪等性テスト結果による終了コード
        exit_code = 0 if idempotency_results["idempotency_analysis"]["is_idempotent"] else 1
    else:
        # 通常の品質チェック実行
        results = validator.run_full_validation()

        # レポート保存
        if args.save_report:
            validator.save_report(args.save_report)

        # 終了コード設定（全セクション通過時は0、そうでなければ1）
        exit_code = 0 if all(section["passed"] for section in results.values()) else 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
