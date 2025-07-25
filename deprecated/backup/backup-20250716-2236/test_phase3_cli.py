#!/usr/bin/env python3
"""
Phase 3: インタラクティブ機能のCLIテスト
GUIなしでのインタラクティブ機能テスト
"""

import os
import sys

sys.path.append('.')

def test_interactive_features():
    """インタラクティブ機能のコアをテスト"""
    
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了\n")
    
    from hooks.start import get_sam_model, get_yolo_model
    from utils.interactive_core import InteractiveAssistant

    # アシスタント初期化
    assistant = InteractiveAssistant()
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    assistant.set_models(sam_model, yolo_model)
    
    # 失敗画像でテスト
    test_images = [
        {
            'path': '/mnt/c/AItools/lora/train/yado/org/kana03/21_kana03_0020.jpg',
            'name': '21_kana03_0020.jpg',
            'description': 'ダイナミックなポーズ + エフェクト線',
            'seed_points': [(750, 1000, True), (800, 1200, True), (700, 800, True)]  # キャラクター中心部
        },
        {
            'path': '/mnt/c/AItools/lora/train/yado/org/kana03/16_kana03_0015.jpg',
            'name': '16_kana03_0015.jpg', 
            'description': 'マルチコマ構成',
            'seed_points': [(400, 1800, True), (500, 1900, True), (350, 1700, True)]  # 下部のキャラクター
        }
    ]
    
    results = []
    
    print("🧪 Phase 3インタラクティブ機能テスト開始")
    print("=" * 60)
    
    for i, image_config in enumerate(test_images, 1):
        print(f"\n📸 失敗画像 {i}/2: {image_config['name']}")
        print(f"   説明: {image_config['description']}")
        print("-" * 40)
        
        # 画像読み込み
        success = assistant.load_image(image_config['path'])
        if not success:
            print(f"   ❌ 画像読み込み失敗")
            continue
        
        print(f"   ✅ 画像読み込み成功")
        
        # テスト1: シードポイント方式
        print(f"\n🎯 テスト1: シードポイント方式")
        assistant.clear_seed_points()
        
        # 予め設定されたシードポイントを追加
        for x, y, is_positive in image_config['seed_points']:
            assistant.add_seed_point(x, y, is_positive)
        
        print(f"   シードポイント数: {len(assistant.seed_points)}")
        
        try:
            output_path = f"/tmp/phase3_seeds_{image_config['name'].replace('.jpg', '')}"
            result = assistant.extract_character_interactive(output_path)
            
            if result['success']:
                print(f"   ✅ シードポイント方式成功! 出力: {result['output_path']}")
            else:
                print(f"   ❌ シードポイント方式失敗: {result['error']}")
            
            results.append({
                'image': image_config['name'],
                'method': 'seed_points',
                'success': result['success'],
                'error': result.get('error', '')
            })
            
        except Exception as e:
            print(f"   💥 シードポイント方式で例外: {e}")
            results.append({
                'image': image_config['name'],
                'method': 'seed_points',
                'success': False,
                'error': f"Exception: {e}"
            })
        
        # テスト2: バウンディングボックス方式
        print(f"\n📦 テスト2: バウンディングボックス方式")
        assistant.clear_seed_points()
        
        # 画像に応じた領域設定
        if '21_kana03_0020' in image_config['name']:
            # エフェクト線画像: 中央のキャラクター領域
            assistant.set_region(600, 800, 400, 600)
        elif '16_kana03_0015' in image_config['name']:
            # マルチコマ画像: 下部パネルのキャラクター
            assistant.set_region(200, 1600, 600, 400)
        
        print(f"   設定領域: {assistant.selected_region}")
        
        try:
            output_path = f"/tmp/phase3_bbox_{image_config['name'].replace('.jpg', '')}"
            result = assistant.extract_character_interactive(output_path)
            
            if result['success']:
                print(f"   ✅ バウンディングボックス方式成功! 出力: {result['output_path']}")
            else:
                print(f"   ❌ バウンディングボックス方式失敗: {result['error']}")
            
            results.append({
                'image': image_config['name'],
                'method': 'bounding_box',
                'success': result['success'],
                'error': result.get('error', '')
            })
            
        except Exception as e:
            print(f"   💥 バウンディングボックス方式で例外: {e}")
            results.append({
                'image': image_config['name'],
                'method': 'bounding_box',
                'success': False,
                'error': f"Exception: {e}"
            })
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 Phase 3インタラクティブ機能テスト結果")
    print("=" * 60)
    
    for image_name in [img['name'] for img in test_images]:
        print(f"\n📸 {image_name}:")
        
        image_results = [r for r in results if r['image'] == image_name]
        success_count = sum(1 for r in image_results if r['success'])
        
        for result in image_results:
            status = "✅" if result['success'] else "❌"
            method = result['method'].replace('_', ' ').title()
            error = f" - {result['error']}" if result['error'] and not result['success'] else ""
            print(f"   {status} {method:<20} {error}")
        
        success_rate = success_count / len(image_results) * 100 if image_results else 0
        print(f"   📈 成功率: {success_count}/{len(image_results)} ({success_rate:.0f}%)")
    
    # 全体統計
    total_tests = len(results)
    total_successes = sum(1 for r in results if r['success'])
    overall_success_rate = total_successes / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 Phase 3全体成績:")
    print(f"   総テスト数: {total_tests}")
    print(f"   成功数: {total_successes}")
    print(f"   全体成功率: {overall_success_rate:.1f}%")
    
    # 改善効果
    print(f"\n📈 自動処理からの改善:")
    print(f"   Phase 1+2: 0% (2画像とも全失敗)")
    print(f"   Phase 3: {overall_success_rate:.1f}%")
    
    if overall_success_rate > 75:
        print("🎉 Phase 3で大幅改善！インタラクティブ機能により失敗画像の抽出に成功しています。")
    elif overall_success_rate > 50:
        print("✅ Phase 3で顕著な改善。手動介入により抽出が可能になりました。")
    elif overall_success_rate > 25:
        print("🔧 Phase 3で部分的改善。一部の方法で抽出が可能です。")
    else:
        print("⚠️ Phase 3でも限定的。画像が非常に複雑です。")
    
    return results


if __name__ == "__main__":
    test_interactive_features()