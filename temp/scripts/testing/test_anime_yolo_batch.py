#!/usr/bin/env python3
"""
アニメYOLO 5枚バッチテストスクリプト
v0.3.5失敗画像でのアニメYOLO性能測定

目的: アニメYOLOモデルの実際の成功率測定
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


def get_failed_images():
    """v0.3.5失敗画像リストを取得"""
    failed_images = [
        "kaname09_001.jpg",
        "kaname09_006.jpg", 
        "kaname09_013.jpg",
        "kaname09_017.jpg",
        "kaname09_022.jpg"
    ]
    return failed_images


def test_anime_yolo_batch():
    """アニメYOLO 5枚バッチテスト実行"""
    
    print("🎌 アニメYOLO 5枚バッチテスト開始")
    print("=" * 60)
    
    # テスト画像取得
    test_images = get_failed_images()
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kaname09")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/test_anime_yolo_batch")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📋 テスト画像数: {len(test_images)}枚")
    for i, image in enumerate(test_images, 1):
        print(f"  {i}. {image}")
    
    # CharacterExtractor初期化
    try:
        extractor = CharacterExtractor()
        print("✅ CharacterExtractor初期化完了")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # バッチテスト結果記録
    batch_results = {
        'test_info': {
            'model': 'yolov8x6_animeface.pt',
            'total_images': len(test_images),
            'timestamp': datetime.now().isoformat(),
            'test_type': 'v0.3.5失敗画像でのアニメYOLOテスト'
        },
        'results': []
    }
    
    success_count = 0
    total_time = 0
    
    print(f"\\n🎯 アニメYOLOバッチテスト開始")
    print("=" * 40)
    
    for i, image_name in enumerate(test_images, 1):
        print(f"\\n[{i}/{len(test_images)}] 処理中: {image_name}")
        
        input_path = input_dir / image_name
        output_path = output_dir / image_name
        
        start_time = time.time()
        
        try:
            # アニメYOLOでの抽出実行
            result = extractor.extract(
                str(input_path),
                str(output_path),
                save_mask=True,
                save_transparent=True,
                verbose=True,
                high_quality=True,
                min_yolo_score=0.01
            )
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if result.get('success', False):
                success_count += 1
                print(f"  ✅ 成功 (処理時間: {processing_time:.1f}秒)")
                if result.get('quality_score'):
                    print(f"     品質スコア: {result.get('quality_score'):.3f}")
                status = "success"
            else:
                print(f"  ❌ 失敗 (処理時間: {processing_time:.1f}秒)")
                print(f"     エラー: {result.get('error', 'Unknown error')}")
                status = "failed"
            
            # 結果記録
            batch_results['results'].append({
                'filename': image_name,
                'success': result.get('success', False),
                'processing_time': processing_time,
                'error': result.get('error'),
                'quality_score': result.get('quality_score'),
                'status': status
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            total_time += processing_time
            
            print(f"  ❌ 例外エラー (処理時間: {processing_time:.1f}秒)")
            print(f"     例外: {str(e)}")
            
            batch_results['results'].append({
                'filename': image_name,
                'success': False,
                'processing_time': processing_time,
                'error': str(e),
                'status': 'exception'
            })
    
    # 統計計算
    success_rate = (success_count / len(test_images)) * 100
    avg_time = total_time / len(test_images)
    
    batch_results['test_info'].update({
        'success_count': success_count,
        'success_rate': success_rate,
        'total_time': total_time,
        'average_time': avg_time
    })
    
    # 結果サマリー表示
    print(f"\\n📈 アニメYOLOバッチテスト結果")
    print("=" * 60)
    print(f"テスト画像: v0.3.5失敗画像 5枚")
    print(f"使用モデル: yolov8x6_animeface.pt")
    print(f"成功: {success_count}枚")
    print(f"失敗: {len(test_images) - success_count}枚")
    print(f"成功率: {success_rate:.1f}%")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"平均処理時間: {avg_time:.1f}秒/枚")
    
    # v0.3.5との比較
    v035_success_rate = 0.0  # v0.3.5では全失敗
    v035_avg_time = 83.0  # v0.3.5での平均処理時間
    
    print(f"\\n📊 v0.3.5標準YOLOとの比較:")
    print(f"  成功率: {v035_success_rate:.1f}% → {success_rate:.1f}% (+{success_rate:.1f}%)")
    print(f"  平均処理時間: {v035_avg_time:.1f}秒 → {avg_time:.1f}秒 ({(avg_time/v035_avg_time-1)*100:+.1f}%)")
    
    # 詳細結果
    print(f"\\n📋 詳細結果:")
    for i, result in enumerate(batch_results['results'], 1):
        status_emoji = "✅" if result['success'] else "❌"
        print(f"  {i}. {result['filename']}: {status_emoji} {result['status']} ({result['processing_time']:.1f}秒)")
    
    # JSON保存
    results_file = output_dir / "anime_yolo_batch_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n💾 結果保存: {results_file}")
    
    # 最終評価
    if success_rate >= 80:
        print(f"\\n🎉 優秀な結果！アニメYOLOで大幅改善達成")
    elif success_rate >= 60:
        print(f"\\n👍 良好な結果！さらなる改善の余地あり")
    elif success_rate >= 40:
        print(f"\\n📈 改善傾向！追加対策で向上可能")
    else:
        print(f"\\n🔍 限定的改善。他の戦略検討が必要")
    
    return batch_results


if __name__ == "__main__":
    import torch
    test_anime_yolo_batch()