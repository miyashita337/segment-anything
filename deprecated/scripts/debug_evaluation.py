#!/usr/bin/env python3
"""
新評価システムのデバッグ
kana07_0023の詳細確認
"""

import numpy as np
import cv2

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_kana07_0023():
    """kana07_0023の詳細デバッグ"""
    project_root = Path("/mnt/c/AItools")

    # データファイル読み込み
    results_file = (
        project_root
        / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
    )
    with open(results_file, "r", encoding="utf-8") as f:
        ai_results = json.load(f)

    # kana07_0023検索
    target_result = None
    for result in ai_results:
        if result["image_id"] == "kana07_0023":
            target_result = result
            break

    if not target_result:
        print("❌ kana07_0023が見つかりません")
        return

    print("🔍 kana07_0023詳細情報:")
    print(f"画像パス: {target_result['image_path']}")
    print(f"人間ラベル: {target_result['human_bbox']}")
    print(f"AI抽出: {target_result['final_bbox']}")
    print(f"IoU: {target_result['iou_score']}")

    # 画像の存在確認
    image_path = Path(target_result["image_path"])
    print(f"\n📁 ファイル存在確認:")
    print(f"オリジナル画像: {'✅' if image_path.exists() else '❌'} {image_path}")

    # 抽出画像の確認
    extracted_paths = [
        project_root / f"lora/train/yado/clipped_boundingbox/kana07/kana07_0023.jpg",
        project_root / f"lora/train/yado/clipped_boundingbox/kana05/kana07_0023.jpg",
        project_root / f"lora/train/yado/clipped_boundingbox/kana08/kana07_0023.jpg",
    ]

    print(f"\n📷 抽出画像確認:")
    for i, path in enumerate(extracted_paths):
        exists = path.exists()
        print(f"候補{i+1}: {'✅' if exists else '❌'} {path}")
        if exists:
            img = cv2.imread(str(path))
            if img is not None:
                print(f"  サイズ: {img.shape}")

    # オリジナル画像から境界ボックス領域を確認
    if image_path.exists():
        original_img = cv2.imread(str(image_path))
        if original_img is not None:
            print(f"\n🖼️  オリジナル画像情報:")
            print(f"サイズ: {original_img.shape}")

            # 人間ラベル領域
            hx, hy, hw, hh = target_result["human_bbox"]
            print(f"人間ラベル領域: ({hx}, {hy}, {hw}, {hh})")
            print(f"人間ラベル範囲: 左上({hx},{hy}) 右下({hx+hw},{hy+hh})")

            # AI抽出領域
            if target_result["final_bbox"]:
                ax, ay, aw, ah = target_result["final_bbox"]
                print(f"AI抽出領域: ({ax}, {ay}, {aw}, {ah})")
                print(f"AI抽出範囲: 左上({ax},{ay}) 右下({ax+aw},{ay+ah})")

                # 領域の重複確認
                overlap_x = max(0, min(hx + hw, ax + aw) - max(hx, ax))
                overlap_y = max(0, min(hy + hh, ay + ah) - max(hy, ay))
                overlap_area = overlap_x * overlap_y

                human_area = hw * hh
                ai_area = aw * ah
                union_area = human_area + ai_area - overlap_area
                actual_iou = overlap_area / union_area if union_area > 0 else 0

                print(f"\n📊 領域分析:")
                print(f"重複面積: {overlap_area}")
                print(f"人間ラベル面積: {human_area}")
                print(f"AI抽出面積: {ai_area}")
                print(f"実際のIoU: {actual_iou:.6f}")
                print(f"報告IoU: {target_result['iou_score']:.6f}")

                # 境界ボックスのクロップ作成（テスト用）
                human_crop = original_img[hy : hy + hh, hx : hx + hw]
                ai_crop = original_img[ay : ay + ah, ax : ax + aw]

                print(f"\n🔪 クロップ情報:")
                print(f"人間ラベルクロップ: {human_crop.shape}")
                print(f"AI抽出クロップ: {ai_crop.shape}")

                # クロップが同じかどうか確認
                if human_crop.shape == ai_crop.shape:
                    diff = np.abs(human_crop.astype(float) - ai_crop.astype(float))
                    mean_diff = np.mean(diff)
                    print(f"クロップ差分平均: {mean_diff:.6f}")

                    if mean_diff < 1.0:
                        print("⚠️  クロップがほぼ同一 → 内容類似度1.0の原因")
                    else:
                        print("✅ クロップは異なる")


if __name__ == "__main__":
    debug_kana07_0023()
