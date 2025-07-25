#!/usr/bin/env python3
"""
Phase A ボックス拡張効果の可視化デモ
GPT-4O推奨のボックス拡張が実際に動作しているか確認
"""

import numpy as np
import cv2

import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import YOLO wrapper with expansion support
from features.extraction.models.yolo_wrapper import YOLOModelWrapper


def visualize_box_expansion():
    """ボックス拡張効果を可視化"""
    
    print("🎯 Phase A ボックス拡張効果の可視化")
    print("=" * 70)
    
    # テスト画像
    test_image = "kaname09_001.jpg"
    input_path = Path("/mnt/c/AItools/lora/train/yado/org/kaname09") / test_image
    
    print(f"📂 テスト画像: {input_path}")
    
    # 画像読み込み
    image = cv2.imread(str(input_path))
    if image is None:
        print("❌ 画像読み込み失敗")
        return
    
    h, w = image.shape[:2]
    print(f"📊 画像サイズ: {w}x{h}")
    
    # YOLOモデル初期化（アニメ特化）
    print("\n🔧 アニメYOLOモデル初期化...")
    yolo_model = YOLOModelWrapper(model_path="yolov8x6_animeface.pt")
    yolo_model.load_model()
    
    # 1. 通常のYOLO検出
    print("\n📦 1. 通常のYOLO検出")
    persons = yolo_model.detect_persons(image)
    print(f"   検出数: {len(persons)}個")
    
    for i, person in enumerate(persons):
        score = person.get('score', person.get('confidence', 0.0))
        print(f"   人物{i+1}: bbox={person['bbox']}, score={score:.3f}")
    
    # 2. ボックス拡張のシミュレーション
    print("\n📦 2. GPT-4O推奨ボックス拡張シミュレーション")
    print("   （score_masks_with_detectionsメソッド内で適用）")
    
    if len(persons) > 0:
        # 拡張シミュレーション
        from features.extraction.utils.box_expansion import apply_gpt4o_expansion_strategy
        
        print("\n   拡張前後の比較:")
        expanded_persons = apply_gpt4o_expansion_strategy(
            persons.copy(), 
            image.shape[:2], 
            'balanced'
        )
        
        for i, (orig, exp) in enumerate(zip(persons, expanded_persons)):
            print(f"\n   人物{i+1}:")
            print(f"     元: bbox={orig['bbox']}")
            print(f"     拡張後: bbox={exp['bbox']}")
            if exp.get('expansion_applied'):
                print(f"     🔍 拡張タイプ: {exp.get('expansion_type', 'unknown')}")
                ox, oy, ow, oh = orig['bbox']
                ex, ey, ew, eh = exp['bbox']
                print(f"     拡張率: 水平{ew/ow:.1f}倍 × 垂直{eh/oh:.1f}倍")
    
    # 3. 可視化画像作成
    print("\n🎨 可視化画像作成中...")
    vis_image = image.copy()
    
    if len(persons) > 0:
        # 通常検出（赤）
        for person in persons:
            x, y, w, h = [int(v) for v in person['bbox']]
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            score = person.get('score', person.get('confidence', 0.0))
            cv2.putText(vis_image, f"Normal: {score:.2f}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 拡張シミュレーション結果（緑）
        if 'expanded_persons' in locals():
            for person in expanded_persons:
                if person.get('expansion_applied'):
                    x, y, w, h = [int(v) for v in person['bbox']]
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(vis_image, f"Expanded", 
                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kaname09_box_expansion_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"box_expansion_demo_{test_image}"
    
    cv2.imwrite(str(output_path), vis_image)
    print(f"\n💾 可視化画像保存: {output_path}")
    
    # 拡張効果の分析
    print("\n📊 拡張効果分析:")
    if len(persons) > 0:
        if 'expanded_persons' in locals() and expanded_persons[0].get('expansion_applied'):
            print("  ✅ ボックス拡張機能は正常に動作します")
            print("  📐 GPT-4O推奨の拡張率が適用されています")
        print("  🔍 ただし、アニメYOLOは顔検出特化のため:")
        print("     - 検出されるのは顔のみ")
        print("     - 拡張しても顔が大きくなるだけ")
        print("     - 全身は含まれません")
    else:
        print("  ❌ YOLO検出なし")
    
    # デバッグ情報
    print("\n🔍 デバッグ情報:")
    print(f"  YOLOモデル: {yolo_model.model_path}")
    print(f"  閾値: {yolo_model.min_score}")
    print(f"  拡張可能: {hasattr(yolo_model, 'detect_persons')}")
    
    # 結論
    print("\n📋 結論:")
    print("  アニメYOLO (yolov8x6_animeface.pt) は顔検出に特化")
    print("  → 顔のみを検出し、体は検出されない")
    print("  → ボックス拡張しても、元が顔なので全身にはならない")
    print("\n💡 解決策:")
    print("  1. 通常のYOLO (yolov8x.pt) に戻す")
    print("  2. または、顔検出→SAMで全身セグメンテーション")
    print("  3. または、専用の全身検出モデルを探す")


if __name__ == "__main__":
    visualize_box_expansion()