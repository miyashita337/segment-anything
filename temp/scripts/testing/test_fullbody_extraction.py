#!/usr/bin/env python3
"""
Phase A実装：全身キャラクター抽出テスト
GPT-4O推奨の段階的強化戦略に基づく実装

目的: 顔検出 → ボックス拡張 → SAM全身抽出
アプローチ: yolov8x6_animeface.pt + fullbody_priority_enhanced
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


def test_fullbody_extraction():
    """GPT-4O推奨アプローチでの全身抽出テスト"""
    
    print("🎌 Phase A: 全身キャラクター抽出テスト開始")
    print("📋 GPT-4O推奨の段階的強化戦略実装")
    print("=" * 60)
    
    # テスト画像: v0.3.5失敗画像5枚
    test_images = [
        "kaname09_001.jpg",
        "kaname09_006.jpg",
        "kaname09_013.jpg", 
        "kaname09_017.jpg",
        "kaname09_022.jpg"
    ]
    
    # 標準パス構成（GPT-4O推奨 + ユーザー要求）
    input_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/org/kaname09")
    output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname09_anime_fullbody")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 入力ディレクトリ: {input_dir}")
    print(f"📂 出力ディレクトリ: {output_dir}")
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
    
    # Phase A テスト結果記録
    test_results = {
        'test_info': {
            'phase': 'Phase A - 段階的強化戦略',
            'approach': 'アニメYOLO + fullbody_priority_enhanced',
            'model': 'yolov8x6_animeface.pt',
            'selection_criteria': 'fullbody_priority_enhanced',
            'total_images': len(test_images),
            'timestamp': datetime.now().isoformat(),
            'gpt4o_recommendation': True
        },
        'results': []
    }
    
    success_count = 0
    total_time = 0
    
    print(f"\\n🎯 Phase A実装テスト開始")
    print("📋 設定:")
    print("  - モデル: yolov8x6_animeface.pt (アニメ顔検出)")
    print("  - 基準: fullbody_priority_enhanced (全身優先)")
    print("  - 拡張: GPT-4O推奨パラメータ適用")
    print("=" * 40)
    
    for i, image_name in enumerate(test_images, 1):
        print(f"\\n[{i}/{len(test_images)}] 処理中: {image_name}")
        
        input_path = input_dir / image_name
        output_path = output_dir / image_name
        
        start_time = time.time()
        
        try:
            # Phase A: 全身優先基準での抽出実行
            result = extractor.extract(
                str(input_path),
                str(output_path),
                save_mask=True,
                save_transparent=True,
                verbose=True,
                high_quality=True,
                min_yolo_score=0.01,
                multi_character_criteria='fullbody_priority_enhanced'  # 🎯 Phase A核心
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
            test_results['results'].append({
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
            
            test_results['results'].append({
                'filename': image_name,
                'success': False,
                'processing_time': processing_time,
                'error': str(e),
                'status': 'exception'
            })
    
    # 統計計算
    success_rate = (success_count / len(test_images)) * 100
    avg_time = total_time / len(test_images)
    
    test_results['test_info'].update({
        'success_count': success_count,
        'success_rate': success_rate,
        'total_time': total_time,
        'average_time': avg_time
    })
    
    # 結果サマリー表示
    print(f"\\n📈 Phase A実装テスト結果")
    print("=" * 60)
    print(f"戦略: GPT-4O推奨段階的強化")
    print(f"実装: アニメYOLO + fullbody_priority_enhanced")
    print(f"成功: {success_count}枚")
    print(f"失敗: {len(test_images) - success_count}枚")
    print(f"成功率: {success_rate:.1f}%")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"平均処理時間: {avg_time:.1f}秒/枚")
    
    # 前回（顔のみ）との比較
    prev_success_rate = 100.0  # 顔のみ抽出での成功率
    prev_avg_time = 25.1  # 前回の平均処理時間
    
    print(f"\\n📊 前回（顔のみ抽出）との比較:")
    print(f"  成功率: {prev_success_rate:.1f}% → {success_rate:.1f}% ({success_rate-prev_success_rate:+.1f}%)")
    print(f"  平均処理時間: {prev_avg_time:.1f}秒 → {avg_time:.1f}秒 ({(avg_time/prev_avg_time-1)*100:+.1f}%)")
    
    # GPT-4O予測との比較
    print(f"\\n🤖 GPT-4O予測との比較:")
    print(f"  予測成功率: 80-90%")
    print(f"  実際成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(f"  → ✅ GPT-4O予測範囲内の良好な結果")
    elif success_rate >= 60:
        print(f"  → 📈 予測よりやや低いが改善傾向")
    else:
        print(f"  → 🔍 追加調整が必要")
    
    # 詳細結果
    print(f"\\n📋 詳細結果:")
    for i, result in enumerate(test_results['results'], 1):
        status_emoji = "✅" if result['success'] else "❌"
        print(f"  {i}. {result['filename']}: {status_emoji} {result['status']} ({result['processing_time']:.1f}秒)")
    
    # JSON保存
    results_file = output_dir / "phase_a_fullbody_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n💾 結果保存: {results_file}")
    
    # Phase A評価
    print(f"\\n🎯 Phase A実装評価:")
    if success_rate >= 80:
        print(f"✅ 優秀！GPT-4O戦略の有効性確認")
        print(f"   → Phase B（negative prompt）準備可能")
    elif success_rate >= 60:
        print(f"📈 良好！さらなる改善余地あり")
        print(f"   → パラメータ調整で改善可能")
    else:
        print(f"🔍 要改善。代替アプローチ検討必要")
        print(f"   → GPT-4O提案の追加実装が必要")
    
    print(f"\\n📁 出力ファイル確認:")
    print(f"   ls {output_dir}/*.jpg")
    
    return test_results


if __name__ == "__main__":
    import torch
    test_fullbody_extraction()