#!/usr/bin/env python3
"""
Phase 4.1システムのテストとバッチ処理検証
"""

import os
import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Phase 4.1システムのインポート
sys.path.append(os.path.dirname(__file__))
from utils.phase41_integrated_system import Phase41IntegratedSystem, Phase41Result
from utils.multi_character_handler import SelectionCriteria

def setup_logging(enable_debug: bool = False):
    """ログ設定"""
    level = logging.DEBUG if enable_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_images(input_dir: str) -> List[tuple]:
    """テスト画像を読み込み"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image = cv2.imread(str(file_path))
            if image is not None:
                image_files.append((str(file_path), image))
    
    return sorted(image_files)

def simulate_yolo_results(image: np.ndarray, scenario: str = "single") -> List[Dict[str, Any]]:
    """YOLO結果をシミュレート"""
    height, width = image.shape[:2]
    
    if scenario == "single":
        # 単一キャラクター
        return [{
            "bbox": (width//4, height//6, 3*width//4, 5*height//6),
            "confidence": 0.85
        }]
    elif scenario == "multi":
        # 複数キャラクター
        return [
            {
                "bbox": (width//6, height//4, width//2, 3*height//4),
                "confidence": 0.75
            },
            {
                "bbox": (width//2, height//3, 5*width//6, 4*height//5),
                "confidence": 0.65
            },
            {
                "bbox": (width//8, height//8, width//3, height//2),
                "confidence": 0.55
            }
        ]
    elif scenario == "none":
        # 検出なし
        return []
    else:
        # ランダム
        import random
        num_detections = random.randint(0, 3)
        results = []
        
        for _ in range(num_detections):
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height//2)
            x2 = random.randint(x1 + 50, width)
            y2 = random.randint(y1 + 100, height)
            confidence = random.uniform(0.3, 0.9)
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence
            })
        
        return results

def test_single_image(system: Phase41IntegratedSystem, 
                     image_path: str, 
                     image: np.ndarray,
                     scenario: str = "auto") -> Phase41Result:
    """単一画像テスト"""
    print(f"\n=== テスト: {os.path.basename(image_path)} ===")
    
    # YOLOシミュレーション
    if scenario == "auto":
        # 実際のYOLO検出（システムに依存）
        yolo_results = None
    else:
        yolo_results = simulate_yolo_results(image, scenario)
        print(f"シミュレートされたYOLO結果: {len(yolo_results)}個の検出")
    
    # Phase 4.1処理
    result = system.extract_character(image, yolo_results)
    
    # 結果表示
    print(f"処理結果: {'成功' if result.success else '失敗'}")
    if result.success:
        print(f"  品質スコア: {result.quality_score:.3f}")
        print(f"  選択エンジン: {result.selected_engine.value}")
        print(f"  処理時間: {result.processing_time:.2f}秒")
        print(f"  YOLO検出数: {result.yolo_detections}")
        print(f"  最終キャラクター数: {result.final_character_count}")
        
        if result.complexity_analysis:
            print(f"  複雑度: {result.complexity_analysis.level.value}")
            print(f"  判定理由: {result.complexity_analysis.reasoning}")
        
        if result.multi_character_analysis and result.multi_character_analysis.success:
            print(f"  複数キャラクター選択: 成功")
            selected = result.multi_character_analysis.selected_character
            if selected:
                print(f"    選択理由: {', '.join(selected.selection_reasons)}")
                print(f"    スコア: {selected.total_score:.3f}")
        
        if result.adjustments_made:
            print(f"  実行された調整: {', '.join(result.adjustments_made)}")
        
        if result.warnings:
            print(f"  警告: {'; '.join(result.warnings)}")
    else:
        print(f"  エラー: {result.error_message}")
    
    return result

def run_batch_test(input_dir: str, 
                  output_dir: str, 
                  criteria: str = "balanced",
                  max_images: int = None,
                  scenario: str = "auto") -> List[Phase41Result]:
    """バッチテスト実行"""
    print(f"\n=== Phase 4.1 バッチテスト開始 ===")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"選択基準: {criteria}")
    print(f"シナリオ: {scenario}")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # システム初期化
    selection_criteria = SelectionCriteria(criteria)
    system = Phase41IntegratedSystem(
        multi_character_criteria=selection_criteria,
        enable_detailed_logging=True
    )
    
    # テスト画像読み込み
    image_files = load_test_images(input_dir)
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"処理対象: {len(image_files)}枚の画像")
    
    # バッチ処理
    results = []
    
    def progress_callback(current, total, result):
        print(f"進捗: {current}/{total} ({current/total*100:.1f}%) - "
              f"{'成功' if result.success else '失敗'} - "
              f"品質: {result.quality_score:.3f}")
    
    for i, (image_path, image) in enumerate(image_files):
        try:
            print(f"\n処理中: {os.path.basename(image_path)} ({i+1}/{len(image_files)})")
            
            # YOLOシミュレーション
            if scenario == "auto":
                yolo_results = None
            else:
                yolo_results = simulate_yolo_results(image, scenario)
            
            # Phase 4.1処理
            result = system.extract_character(image, yolo_results)
            results.append(result)
            
            # 結果画像保存（成功時のみ）
            if result.success and result.final_mask is not None:
                output_filename = f"phase41_{os.path.basename(image_path)}"
                output_path = os.path.join(output_dir, output_filename)
                
                # マスク適用
                masked_image = cv2.bitwise_and(image, image, mask=result.final_mask)
                cv2.imwrite(output_path, masked_image)
            
            progress_callback(i + 1, len(image_files), result)
            
        except Exception as e:
            print(f"エラー: {image_path} - {e}")
            results.append(None)
    
    # システム統計
    system_stats = system.get_system_statistics()
    print(f"\n=== システム統計 ===")
    print(f"総処理数: {system_stats['total_processed']}")
    print(f"成功率: {system_stats['success_rate']:.1%}")
    print(f"Phase 0.0.3使用率: {system_stats['phase_003_usage_rate']:.1%}")
    print(f"Phase 0.0.4使用率: {system_stats['phase_004_usage_rate']:.1%}")
    print(f"複数キャラクター率: {system_stats['multi_character_rate']:.1%}")
    
    # レポート保存
    report_path = os.path.join(output_dir, "phase41_test_report.json")
    system.save_processing_report(results, report_path)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Phase 4.1システムのテスト")
    parser.add_argument("input_dir", help="入力画像ディレクトリ")
    parser.add_argument("-o", "--output", default="./phase41_test_output", 
                       help="出力ディレクトリ")
    parser.add_argument("--criteria", choices=["balanced", "size_priority", "fullbody_priority", 
                                              "central_priority", "confidence_priority"],
                       default="balanced", help="複数キャラクター選択基準")
    parser.add_argument("--scenario", choices=["auto", "single", "multi", "none", "random"],
                       default="auto", help="YOLOシミュレーションシナリオ")
    parser.add_argument("--max-images", type=int, help="最大処理画像数")
    parser.add_argument("--single-test", help="単一画像テスト")
    parser.add_argument("--debug", action="store_true", help="デバッグログ有効化")
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.debug)
    
    if args.single_test:
        # 単一画像テスト
        image = cv2.imread(args.single_test)
        if image is None:
            print(f"画像が読み込めません: {args.single_test}")
            return 1
        
        system = Phase41IntegratedSystem(
            multi_character_criteria=SelectionCriteria(args.criteria),
            enable_detailed_logging=True
        )
        
        result = test_single_image(system, args.single_test, image, args.scenario)
        return 0 if result.success else 1
    else:
        # バッチテスト
        try:
            results = run_batch_test(
                args.input_dir, 
                args.output,
                args.criteria,
                args.max_images,
                args.scenario
            )
            
            successful_results = [r for r in results if r and r.success]
            success_rate = len(successful_results) / len(results) if results else 0
            
            print(f"\n=== 最終結果 ===")
            print(f"成功率: {success_rate:.1%} ({len(successful_results)}/{len(results)})")
            
            return 0 if success_rate > 0.5 else 1
            
        except Exception as e:
            print(f"バッチテストエラー: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())