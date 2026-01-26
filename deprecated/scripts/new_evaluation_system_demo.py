#!/usr/bin/env python3
"""
新評価システムのデモンストレーション
GPT-4O設計による座標+内容統合評価の実演
"""

import numpy as np

import json
import logging
import time
from pathlib import Path

# 新評価システムのインポート（CLIP依存関係チェック付き）
try:
    from evaluation import EvaluationConfig, EvaluationOrchestrator

    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    print(f"⚠️  新評価システムが利用できません: {e}")
    print("必要な依存関係をインストールしてください:")
    print("pip install git+https://github.com/openai/CLIP.git")
    print("pip install torch torchvision")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewEvaluationDemo:
    """新評価システムのデモクラス"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_file = (
            project_root
            / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        )
        self.labels_file = project_root / "segment-anything/extracted_labels.json"

        # 設定
        self.config = EvaluationConfig() if EVALUATION_AVAILABLE else None
        if self.config:
            self.config.alpha = 0.6  # IoU重視（60%）+ 内容類似度40%
            self.config.iou_threshold = 0.3
            self.config.content_threshold = 0.25
            self.config.success_threshold = 0.5
            self.config.use_fp16 = False  # FP16エラー回避

        self.orchestrator = None
        if EVALUATION_AVAILABLE and self.config:
            try:
                self.orchestrator = EvaluationOrchestrator(self.config)
            except Exception as e:
                logger.warning(f"Orchestrator initialization failed: {e}")
                self.orchestrator = None

        # データ読み込み
        self.load_data()

    def load_data(self):
        """データ読み込み"""
        with open(self.results_file, "r", encoding="utf-8") as f:
            self.ai_results = json.load(f)

        with open(self.labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        self.human_labels = {}
        for item in labels_data:
            if item.get("red_boxes"):
                image_id = item["filename"].rsplit(".", 1)[0]
                self.human_labels[image_id] = item

        logger.info(f"データ読み込み完了: {len(self.ai_results)}件の結果, {len(self.human_labels)}件の人間ラベル")

    def prepare_evaluation_data(self, image_id: str) -> tuple:
        """評価用データの準備"""
        # AI結果を検索
        ai_result = None
        for result in self.ai_results:
            if result["image_id"] == image_id:
                ai_result = result
                break

        if not ai_result:
            return None, None

        # 予測データ準備
        predictions = {
            "image_path": ai_result["image_path"],
            "bboxes": [ai_result["final_bbox"]] if ai_result["final_bbox"] else [],
        }

        # Ground truth データ準備
        ground_truth = {"image_path": ai_result["image_path"], "bboxes": [ai_result["human_bbox"]]}

        return predictions, ground_truth

    def demo_single_case(self, image_id: str = "kana07_0023"):
        """単一ケースのデモ"""
        if not EVALUATION_AVAILABLE or not self.orchestrator:
            print("❌ 新評価システムが利用できません")
            return

        print(f"\n🔍 【{image_id}】の新評価システムデモ")

        # データ準備
        predictions, ground_truth = self.prepare_evaluation_data(image_id)
        if not predictions or not ground_truth:
            print(f"❌ {image_id} のデータが見つかりません")
            return

        # 従来評価結果
        old_result = None
        for result in self.ai_results:
            if result["image_id"] == image_id:
                old_result = result
                break

        print(f"📊 従来評価:")
        print(f"   IoU: {old_result['iou_score']:.3f}")
        print(f"   成功判定: {'✅' if old_result['extraction_success'] else '❌'}")

        # 新評価システム実行
        print(f"\n🔄 新評価システム実行中...")
        try:
            start_time = time.time()
            new_result = self.orchestrator.run_single_image(predictions, ground_truth)
            processing_time = time.time() - start_time

            print(f"📊 新評価結果:")
            print(f"   統合スコア: {new_result.get('integrated_score', 0):.3f}")
            print(f"   空間スコア: {new_result.get('spatial_score', 0):.3f}")
            print(f"   内容スコア: {new_result.get('content_score', 0):.3f}")
            print(f"   成功判定: {'✅' if new_result.get('success', False) else '❌'}")
            print(f"   処理時間: {processing_time:.3f}秒")

            # 比較分析
            if old_result["extraction_success"] != new_result.get("success", False):
                print(f"\n⚠️  判定変更:")
                print(f"   従来: {'成功' if old_result['extraction_success'] else '失敗'}")
                print(f"   新方式: {'成功' if new_result.get('success', False) else '失敗'}")
                print(
                    f"   原因: 内容類似度が {new_result.get('content_score', 0):.3f} (閾値: {self.config.content_threshold})"
                )

        except Exception as e:
            print(f"❌ 新評価システムエラー: {e}")

    def demo_batch_comparison(self, sample_size: int = 10):
        """バッチ比較デモ"""
        if not EVALUATION_AVAILABLE or not self.orchestrator:
            print("❌ 新評価システムが利用できません")
            return

        print(f"\n📊 バッチ比較デモ（{sample_size}件サンプル）")

        # サンプル選択（高IoUケース優先）
        high_iou_cases = [
            r for r in self.ai_results if r["extraction_success"] and r["iou_score"] > 0.8
        ]
        sample_cases = (
            high_iou_cases[:sample_size]
            if len(high_iou_cases) >= sample_size
            else self.ai_results[:sample_size]
        )

        predictions_batch = []
        ground_truths_batch = []

        for result in sample_cases:
            pred, gt = self.prepare_evaluation_data(result["image_id"])
            if pred and gt:
                predictions_batch.append(pred)
                ground_truths_batch.append(gt)

        if not predictions_batch:
            print("❌ 評価可能なデータがありません")
            return

        print(f"🔄 {len(predictions_batch)}件のバッチ評価実行中...")

        try:
            batch_result = self.orchestrator.run_batch(predictions_batch, ground_truths_batch)

            # 従来評価の統計
            old_success_count = sum(
                1 for r in sample_cases[: len(predictions_batch)] if r["extraction_success"]
            )
            old_success_rate = old_success_count / len(predictions_batch)
            old_mean_iou = np.mean([r["iou_score"] for r in sample_cases[: len(predictions_batch)]])

            print(f"\n📈 比較結果:")
            print(f"従来評価:")
            print(f"   成功率: {old_success_rate:.1%} ({old_success_count}/{len(predictions_batch)})")
            print(f"   平均IoU: {old_mean_iou:.3f}")

            print(f"新評価システム:")
            print(
                f"   成功率: {batch_result['success_rate']:.1%} ({batch_result['successful_images']}/{batch_result['total_images']})"
            )
            print(f"   平均統合スコア: {batch_result['mean_integrated_score']:.3f}")
            print(f"   平均空間スコア: {batch_result['mean_spatial_score']:.3f}")
            print(f"   平均内容スコア: {batch_result['mean_content_score']:.3f}")
            print(f"   平均処理時間: {batch_result['avg_time_per_image']:.3f}秒/画像")

            # 判定変更の分析
            judgment_changes = 0
            for i, result in enumerate(batch_result["individual_results"]):
                old_success = sample_cases[i]["extraction_success"]
                new_success = result.get("success", False)
                if old_success != new_success:
                    judgment_changes += 1

            print(f"\n🔄 判定変更: {judgment_changes}件 ({judgment_changes/len(predictions_batch):.1%})")

        except Exception as e:
            print(f"❌ バッチ評価エラー: {e}")

    def show_system_info(self):
        """システム情報表示"""
        print("🤖 GPT-4O設計による新評価システム")
        print("=" * 50)
        print("特徴:")
        print("• 座標一致 + 内容類似度の統合評価")
        print("• CLIP/DINOv2による視覚的特徴抽出")
        print("• ハンガリアン法による最適マッチング")
        print("• マルチキャラクター画像対応")
        print("• リアルタイム処理（0.5秒/画像以内）")

        if EVALUATION_AVAILABLE:
            print(f"\n設定:")
            print(f"• α (IoU:内容) = {self.config.alpha}:{1-self.config.alpha}")
            print(f"• IoU閾値: {self.config.iou_threshold}")
            print(f"• 内容類似度閾値: {self.config.content_threshold}")
            print(f"• 成功判定閾値: {self.config.success_threshold}")
        else:
            print("\n⚠️  依存関係が不足しています。CLIPをインストールしてください。")


def main():
    """メイン実行"""
    project_root = Path("/mnt/c/AItools")
    demo = NewEvaluationDemo(project_root)

    # システム情報表示
    demo.show_system_info()

    if EVALUATION_AVAILABLE:
        # 単一ケースデモ（問題のkana07_0023）
        demo.demo_single_case("kana07_0023")

        # バッチ比較デモ
        demo.demo_batch_comparison(5)  # 5件のサンプル

        print(f"\n✅ 新評価システムデモ完了")
        print(f"座標の一致≠内容の一致 問題を解決する統合評価システムを実装しました。")
    else:
        print(f"\n❌ 新評価システムを使用するには以下をインストールしてください:")
        print(f"pip install git+https://github.com/openai/CLIP.git")
        print(f"pip install torch torchvision")


if __name__ == "__main__":
    main()
