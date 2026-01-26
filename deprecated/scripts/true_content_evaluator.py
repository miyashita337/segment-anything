#!/usr/bin/env python3
"""
真の内容評価器
実際の抽出画像と人間意図領域の比較
"""

import numpy as np
import cv2

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

# CLIP import with error handling
try:
    import torch

    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueContentEvaluator:
    """実際の抽出画像による内容評価"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_file = (
            project_root
            / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        )

        # CLIP初期化
        self.model = None
        self.preprocess = None
        if CLIP_AVAILABLE:
            self.model, self.preprocess = clip.load(
                "ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("CLIP model loaded successfully")

        # データ読み込み
        self.load_data()

    def load_data(self):
        """AIベンチマーク結果読み込み"""
        with open(self.results_file, "r", encoding="utf-8") as f:
            self.ai_results = json.load(f)
        logger.info(f"AI結果読み込み: {len(self.ai_results)}件")

    def find_actual_extraction(self, image_id: str) -> Optional[Path]:
        """実際の抽出画像ファイルを検索"""
        candidate_paths = [
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana07/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana05/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana08/{image_id}.jpg",
        ]

        for path in candidate_paths:
            if path.exists():
                return path
        return None

    def get_human_intended_crop(self, image_id: str) -> Optional[np.ndarray]:
        """人間が意図した領域のクロップ取得"""
        # AI結果から情報取得
        ai_result = None
        for result in self.ai_results:
            if result["image_id"] == image_id:
                ai_result = result
                break

        if not ai_result:
            return None

        # オリジナル画像読み込み
        image_path = Path(ai_result["image_path"])
        if not image_path.exists():
            return None

        original_img = cv2.imread(str(image_path))
        if original_img is None:
            return None

        # 人間ラベル領域をクロップ
        hx, hy, hw, hh = ai_result["human_bbox"]
        human_crop = original_img[hy : hy + hh, hx : hx + hw]

        return human_crop

    def calculate_clip_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """CLIP特徴量による類似度計算"""
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available, returning 0.0")
            return 0.0

        try:
            # 画像をPIL形式に変換
            from PIL import Image

            img1_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            img2_pil = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

            # CLIP前処理
            img1_tensor = self.preprocess(img1_pil).unsqueeze(0)
            img2_tensor = self.preprocess(img2_pil).unsqueeze(0)

            # GPU移動
            device = next(self.model.parameters()).device
            img1_tensor = img1_tensor.to(device)
            img2_tensor = img2_tensor.to(device)

            # 特徴抽出
            with torch.no_grad():
                features1 = self.model.encode_image(img1_tensor)
                features2 = self.model.encode_image(img2_tensor)

                # 正規化
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)

                # コサイン類似度
                similarity = torch.cosine_similarity(features1, features2, dim=-1)
                return float(similarity.item())

        except Exception as e:
            logger.error(f"CLIP similarity calculation failed: {e}")
            return 0.0

    def evaluate_true_content(self, image_id: str) -> dict:
        """真の内容評価実行"""
        result = {
            "image_id": image_id,
            "actual_extraction_found": False,
            "human_crop_extracted": False,
            "clip_similarity": 0.0,
            "visual_match": False,
            "evaluation_method": "true_content",
        }

        # 実際の抽出画像取得
        extracted_path = self.find_actual_extraction(image_id)
        if not extracted_path:
            logger.warning(f"実際の抽出画像が見つかりません: {image_id}")
            return result

        extracted_img = cv2.imread(str(extracted_path))
        if extracted_img is None:
            logger.warning(f"抽出画像読み込み失敗: {extracted_path}")
            return result

        result["actual_extraction_found"] = True

        # 人間意図領域取得
        human_crop = self.get_human_intended_crop(image_id)
        if human_crop is None:
            logger.warning(f"人間意図領域取得失敗: {image_id}")
            return result

        result["human_crop_extracted"] = True

        # CLIP類似度計算
        if CLIP_AVAILABLE:
            similarity = self.calculate_clip_similarity(extracted_img, human_crop)
            result["clip_similarity"] = similarity
            result["visual_match"] = similarity > 0.7  # 類似度閾値

            logger.info(f"{image_id}: CLIP類似度 {similarity:.3f}")

        return result

    def test_problem_cases(self, cases: list = None) -> dict:
        """問題ケースのテスト"""
        if cases is None:
            cases = ["kana07_0023", "kana05_0001", "kana05_0002"]

        results = {}

        print("🔍 真の内容評価テスト")
        print("=" * 50)

        for case in cases:
            print(f"\n📊 {case} の評価:")

            # 従来の報告値取得
            ai_result = None
            for result in self.ai_results:
                if result["image_id"] == case:
                    ai_result = result
                    break

            if ai_result:
                print(f"従来IoU: {ai_result['iou_score']:.3f}")
                print(f"従来判定: {'✅成功' if ai_result['extraction_success'] else '❌失敗'}")

            # 真の内容評価
            true_result = self.evaluate_true_content(case)
            results[case] = true_result

            print(f"実際抽出: {'✅' if true_result['actual_extraction_found'] else '❌'}")
            print(f"人間領域: {'✅' if true_result['human_crop_extracted'] else '❌'}")
            if CLIP_AVAILABLE:
                print(f"CLIP類似度: {true_result['clip_similarity']:.3f}")
                print(f"真の判定: {'✅成功' if true_result['visual_match'] else '❌失敗'}")

                # 判定比較
                if ai_result and ai_result["extraction_success"] != true_result["visual_match"]:
                    change = "成功→失敗" if ai_result["extraction_success"] else "失敗→成功"
                    print(f"⚠️  判定変更: {change}")

        return results


def main():
    """メイン実行"""
    project_root = Path("/mnt/c/AItools")
    evaluator = TrueContentEvaluator(project_root)

    if not CLIP_AVAILABLE:
        print("❌ CLIP が利用できません。以下をインストールしてください:")
        print("pip install git+https://github.com/openai/CLIP.git")
        return

    # 問題ケースのテスト
    results = evaluator.test_problem_cases(["kana07_0023"])

    print(f"\n✅ 真の内容評価完了")
    print(f"実際の抽出画像と人間意図領域の比較により、真の一致度を測定しました。")


if __name__ == "__main__":
    main()
