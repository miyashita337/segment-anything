#!/usr/bin/env python3
"""
アニメ特化YOLOモデルテストスクリプト
yolov8x6_animeface.ptでの抽出効果を検証

目的: v0.3.5失敗画像でのアニメYOLO性能測定
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Character extraction import
from features.extraction.commands.extract_character import CharacterExtractor


def test_anime_yolo_model():
    """アニメ特化YOLOモデルでのテスト実行"""
    
    print("🎌 アニメ特化YOLOモデルテスト開始")
    print("=" * 60)
    
    # テスト画像: v0.3.5失敗の代表例
    test_image = "kaname09_001.jpg"
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kaname09")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/test_anime_yolo")
    output_dir.mkdir(exist_ok=True)
    
    input_path = input_dir / test_image
    output_path = output_dir / test_image
    
    print(f"📋 テスト画像: {test_image}")
    print(f"📂 入力パス: {input_path}")
    print(f"📂 出力パス: {output_path}")
    
    # キャラクター抽出システム初期化
    try:
        extractor = CharacterExtractor()
        print("✅ CharacterExtractor初期化完了")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # テスト結果記録
    test_results = {
        'test_info': {
            'model': 'yolov8x6_animeface.pt',
            'test_image': test_image,
            'timestamp': datetime.now().isoformat()
        },
        'results': []
    }
    
    print(f"\n🎯 アニメYOLOモデルテスト実行開始")
    
    start_time = time.time()
    
    try:
        # アニメ特化モデルでの抽出実行
        result = extractor.extract(
            str(input_path),
            str(output_path),
            save_mask=True,
            save_transparent=True,
            verbose=True,
            high_quality=True,
            min_yolo_score=0.01,  # 標準閾値で開始
            # アニメモデル強制指定（将来の拡張用）
            # anime_model="yolov8x6_animeface.pt"
        )
        
        processing_time = time.time() - start_time
        
        if result.get('success', False):
            print(f"  🎉 成功！ (処理時間: {processing_time:.1f}秒)")
            print(f"     品質スコア: {result.get('quality_score', 'N/A')}")
            status = "success"
        else:
            print(f"  ❌ 失敗 (処理時間: {processing_time:.1f}秒)")
            print(f"     エラー: {result.get('error', 'Unknown error')}")
            status = "failed"
        
        # 結果記録
        test_results['results'].append({
            'success': result.get('success', False),
            'processing_time': processing_time,
            'error': result.get('error'),
            'quality_score': result.get('quality_score'),
            'status': status
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        print(f"  ❌ 例外エラー (処理時間: {processing_time:.1f}秒)")
        print(f"     例外: {str(e)}")
        
        test_results['results'].append({
            'success': False,
            'processing_time': processing_time,
            'error': str(e),
            'status': 'exception'
        })
    
    # 結果サマリー
    result_data = test_results['results'][0]
    print(f"\n📈 アニメYOLOテスト結果")
    print(f"=" * 60)
    print(f"モデル: yolov8x6_animeface.pt")
    print(f"画像: {test_image}")
    print(f"結果: {'✅ 成功' if result_data['success'] else '❌ 失敗'}")
    print(f"処理時間: {result_data['processing_time']:.1f}秒")
    
    if result_data['success']:
        print(f"品質スコア: {result_data.get('quality_score', 'N/A')}")
        print(f"🎯 アニメYOLOモデルで抽出成功！")
    else:
        print(f"エラー: {result_data.get('error', 'Unknown')}")
        print(f"🔍 さらなる調査が必要")
    
    # 比較データ
    print(f"\n📊 v0.3.5標準モデルとの比較:")
    print(f"  標準YOLO: ❌ 失敗 (83.0秒)")
    print(f"  アニメYOLO: {'✅ 成功' if result_data['success'] else '❌ 失敗'} ({result_data['processing_time']:.1f}秒)")
    
    # 結果をJSONファイルに保存
    results_file = output_dir / "anime_yolo_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果保存: {results_file}")
    
    return test_results


if __name__ == "__main__":
    import torch
    test_anime_yolo_model()