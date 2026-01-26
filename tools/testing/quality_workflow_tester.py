#!/usr/bin/env python3
"""
Level 2: 品質ワークフローテスター

品質ワークフロー全体のテスト実行・検証を行う
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# テスト対象とモックをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.mocks.mock_sam_yolo import MockQualityEvaluator, MockSamYoloExtractor


class QualityWorkflowTester:
    """品質ワークフローテスタークラス"""

    def __init__(self, quality_method: str = "balanced", random_seed: Optional[int] = None):
        """
        品質ワークフローテスター初期化

        Args:
            quality_method: 品質評価手法
            random_seed: ランダムシード
        """
        self.quality_method = quality_method
        self.random_seed = random_seed
        self.extractor = MockSamYoloExtractor(quality_method, random_seed)
        self.evaluator = MockQualityEvaluator()

    def run_single_image_test(self, image_path: str) -> Dict[str, Any]:
        """
        単体画像テスト実行

        Args:
            image_path: 画像パス

        Returns:
            テスト結果
        """
        print(f"🔬 単体画像テスト実行: {image_path}")

        # 抽出実行
        extraction_result = self.extractor.extract_single_image(image_path)

        # 品質評価
        quality_result = self.evaluator.evaluate_extraction_quality(extraction_result)

        # 結果統合
        test_result = {
            "test_type": "single_image",
            "image_path": image_path,
            "extraction_success": extraction_result["success"],
            "quality_evaluation": quality_result,
            "processing_time": extraction_result["processing_time"],
        }

        if extraction_result["success"]:
            test_result["grade"] = quality_result["grade"]
            test_result["quality_score"] = quality_result["quality_score"]
        else:
            test_result["error"] = extraction_result["error"]

        return test_result

    def run_batch_test(
        self, input_dir: str, output_dir: str, max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        バッチ処理テスト実行

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            max_files: 最大処理ファイル数

        Returns:
            バッチテスト結果
        """
        print(f"🔬 バッチ処理テスト実行")
        print(f"📁 入力: {input_dir}")
        print(f"📁 出力: {output_dir}")
        print(f"📊 最大ファイル数: {max_files or '制限なし'}")

        # バッチ抽出実行
        batch_result = self.extractor.extract_batch(input_dir, output_dir, max_files)

        # 品質レポート生成
        quality_report = self.evaluator.generate_quality_report(batch_result)

        # ダッシュボードHTML生成のモック
        dashboard_html = self._generate_mock_dashboard(batch_result)

        # テスト結果統合
        test_result = {
            "test_type": "batch",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "extraction_results": batch_result,
            "quality_report": quality_report,
            "dashboard_html": dashboard_html,
            "test_summary": {
                "total_images": batch_result["total_images"],
                "success_rate": batch_result["success_rate"],
                "average_quality": batch_result["average_quality_score"],
                "processing_time": batch_result["total_processing_time"],
                "quality_method": self.quality_method,
            },
        }

        return test_result

    def run_quality_method_comparison(self, input_dir: str, output_base_dir: str) -> Dict[str, Any]:
        """
        品質評価手法比較テスト

        Args:
            input_dir: 入力ディレクトリ
            output_base_dir: 出力ベースディレクトリ

        Returns:
            比較テスト結果
        """
        print("🔬 品質評価手法比較テスト実行")

        methods = [
            "balanced",
            "confidence_priority",
            "size_priority",
            "fullbody_priority",
            "central_priority",
        ]
        comparison_results = {}

        for method in methods:
            print(f"📊 テスト中: {method}")

            # 手法別出力ディレクトリ
            method_output_dir = f"{output_base_dir}/{method}"

            # 手法別テスター作成
            method_tester = QualityWorkflowTester(method, self.random_seed)

            # バッチテスト実行
            method_result = method_tester.run_batch_test(input_dir, method_output_dir, max_files=10)

            comparison_results[method] = method_result["test_summary"]

        # 比較分析
        best_method = max(
            comparison_results.keys(), key=lambda m: comparison_results[m]["success_rate"]
        )
        worst_method = min(
            comparison_results.keys(), key=lambda m: comparison_results[m]["success_rate"]
        )

        comparison_summary = {
            "test_type": "quality_method_comparison",
            "methods_tested": methods,
            "results": comparison_results,
            "analysis": {
                "best_method": best_method,
                "best_success_rate": comparison_results[best_method]["success_rate"],
                "worst_method": worst_method,
                "worst_success_rate": comparison_results[worst_method]["success_rate"],
                "average_success_rate": sum(r["success_rate"] for r in comparison_results.values())
                / len(methods),
            },
        }

        return comparison_summary

    def _generate_mock_dashboard(self, batch_result: Dict[str, Any]) -> str:
        """
        ダッシュボードHTML生成のモック

        Args:
            batch_result: バッチ処理結果

        Returns:
            ダッシュボードHTML
        """
        total = batch_result["total_images"]
        successful = batch_result["successful_extractions"]
        success_rate = batch_result["success_rate"]
        avg_quality = batch_result["average_quality_score"]

        # 簡易HTMLダッシュボード
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>品質ダッシュボード（テスト用）</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .success-rate {{ color: #28a745; font-weight: bold; }}
        .quality-score {{ color: #007bff; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎯 品質ワークフローテストダッシュボード</h1>
    
    <div class="summary">
        <h2>📊 処理概要</h2>
        <div class="metric">総画像数: {total}枚</div>
        <div class="metric">成功数: {successful}枚</div>
        <div class="metric success-rate">成功率: {success_rate:.1f}%</div>
        <div class="metric quality-score">平均品質スコア: {avg_quality:.3f}</div>
        <div class="metric">品質評価手法: {self.quality_method}</div>
    </div>
    
    <div class="quality-distribution">
        <h2>📈 品質分布</h2>
        <ul>
"""

        for grade, count in batch_result["quality_distribution"].items():
            percentage = (count / total * 100) if total > 0 else 0
            html += f"            <li>{grade}評価: {count}枚 ({percentage:.1f}%)</li>\n"

        html += """        </ul>
    </div>
    
    <div class="test-info">
        <h2>🧪 テスト情報</h2>
        <p>これはモック品質ワークフローテストの結果です。</p>
        <p>実際のSAM+YOLO処理は実行されていません。</p>
    </div>
</body>
</html>"""

        return html

    def validate_output_files(self, output_dir: str) -> Dict[str, bool]:
        """
        出力ファイル検証

        Args:
            output_dir: 出力ディレクトリ

        Returns:
            ファイル存在確認結果
        """
        output_path = Path(output_dir)

        expected_files = {
            "extraction_result.json": output_path / "extraction_result.json",
            "dashboard.html": output_path / "dashboard.html",
            "quality_report.md": output_path / "quality_report.md",
        }

        validation_results = {}
        for file_name, file_path in expected_files.items():
            validation_results[file_name] = file_path.exists()

        return validation_results

    def save_test_results(self, test_results: Dict[str, Any], output_dir: str) -> None:
        """
        テスト結果保存

        Args:
            test_results: テスト結果
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # テスト結果JSON保存
        with open(output_path / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

        # 品質レポート保存
        if "quality_report" in test_results:
            with open(output_path / "quality_report.md", "w", encoding="utf-8") as f:
                f.write(test_results["quality_report"])

        # ダッシュボードHTML保存
        if "dashboard_html" in test_results:
            with open(output_path / "dashboard.html", "w", encoding="utf-8") as f:
                f.write(test_results["dashboard_html"])

        print(f"✅ テスト結果保存完了: {output_dir}")


def main():
    """CLI実行用メイン関数"""
    if len(sys.argv) < 4:
        print(
            "Usage: python quality_workflow_tester.py <test_type> <input_dir> <output_dir> [options]"
        )
        print("Test types: single, batch, comparison")
        sys.exit(1)

    test_type = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # テスター初期化
    tester = QualityWorkflowTester(random_seed=42)  # 再現可能性のため固定シード

    try:
        if test_type == "single":
            # 単体テスト
            result = tester.run_single_image_test(input_dir)  # input_dirを画像パスとして使用
        elif test_type == "batch":
            # バッチテスト
            result = tester.run_batch_test(input_dir, output_dir, max_files=10)
            tester.save_test_results(result, output_dir)
        elif test_type == "comparison":
            # 比較テスト
            result = tester.run_quality_method_comparison(input_dir, output_dir)
            tester.save_test_results(result, output_dir)
        else:
            print(f"❌ 未サポートテストタイプ: {test_type}")
            sys.exit(1)

        print("✅ 品質ワークフローテスト完了")
        print(json.dumps(result.get("test_summary", result), ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
