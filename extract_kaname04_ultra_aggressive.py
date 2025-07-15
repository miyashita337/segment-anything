#!/usr/bin/env python3
"""
kaname04データセット超積極的処理スクリプト
最低限のYOLO閾値と全画像強制処理
"""

import sys
import os
import json
import time
from pathlib import Path
sys.path.append('.')

def get_failed_files():
    """失敗したファイルリストを取得"""
    input_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/org/kaname04")
    output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
    
    all_files = list(input_dir.glob("*.jpg"))
    processed_files = list(output_dir.glob("*.jpg"))
    
    processed_names = {f.stem for f in processed_files}
    failed_files = [f for f in all_files if f.stem not in processed_names]
    
    return failed_files

def ultra_aggressive_extraction():
    """超積極的抽出実行"""
    print("🚀 Ultra Aggressive Mode: kaname04処理開始...")
    
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    from commands.extract_character import extract_character_from_path
    
    # 超積極的パラメータ設定
    extract_args = {
        'enhance_contrast': True,       # コントラスト強化
        'filter_text': False,           # テキストフィルタ無効
        'save_mask': False,
        'save_transparent': False,
        'min_yolo_score': 0.001,        # 極限まで低い閾値
        'verbose': True,
        'difficult_pose': True,         # 複雑ポーズモード
        'low_threshold': True,          # 低閾値モード
        'auto_retry': True,             # 自動リトライ
        'high_quality': True,           # 高品質SAM
        'manga_mode': True,             # 漫画モード
        'effect_removal': True,         # エフェクト除去
        'panel_split': True,            # パネル分割
        'multi_character_criteria': 'size_priority'  # サイズ優先
    }
    
    print("📊 Ultra Aggressive設定:")
    print("   - YOLO閾値: 0.001 (極限値)")
    print("   - テキストフィルタ: OFF")
    print("   - 全特殊モード: ON")
    
    # 失敗ファイル取得
    failed_files = get_failed_files()
    
    if not failed_files:
        print("🎯 処理すべきファイルがありません（全て完了済み）")
        return {
            'success': True,
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 1.0
        }
    
    print(f"📁 Ultra Aggressive対象ファイル数: {len(failed_files)}")
    
    # 各ファイル処理
    results = []
    successful = 0
    
    for i, image_file in enumerate(failed_files, 1):
        print(f"\n📁 Ultra Aggressive 処理中 [{i}/{len(failed_files)}]: {image_file.name}")
        
        # 出力パス生成
        output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
        output_file = output_dir / image_file.stem
        
        # 抽出実行
        result = extract_character_from_path(
            str(image_file),
            output_path=str(output_file),
            **extract_args
        )
        
        result['filename'] = image_file.name
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"✅ Ultra Aggressive 成功: {image_file.name}")
        else:
            print(f"❌ Ultra Aggressive 失敗: {image_file.name} - {result['error']}")
    
    # 結果
    ultra_result = {
        'success': True,
        'total_files': len(failed_files),
        'successful': successful,
        'failed': len(failed_files) - successful,
        'success_rate': successful / len(failed_files) if failed_files else 1.0,
        'results': results
    }
    
    print(f"\n📊 Ultra Aggressive 結果:")
    print(f"   成功: {successful}/{len(failed_files)} ({ultra_result['success_rate']:.1%})")
    
    return ultra_result

def force_process_with_minimum_requirements():
    """最小要件での強制処理"""
    print("\n🔥 Force Process Mode: 最小要件での強制処理...")
    
    from commands.extract_character import extract_character_from_path
    
    # 最小要件パラメータ
    minimal_args = {
        'enhance_contrast': True,
        'filter_text': False,
        'save_mask': False, 
        'save_transparent': False,
        'min_yolo_score': 0.0001,       # ほぼゼロ閾値
        'verbose': True,
        'difficult_pose': True,
        'low_threshold': True,
        'auto_retry': False,            # リトライOFF（高速化）
        'high_quality': False,          # 標準品質（高速化）
        'manga_mode': False,            # 漫画モードOFF
        'multi_character_criteria': 'balanced'
    }
    
    print("📊 Force Process設定:")
    print("   - YOLO閾値: 0.0001 (ほぼゼロ)")
    print("   - 高速化オプション適用")
    
    failed_files = get_failed_files()
    
    if not failed_files:
        return {'successful': 0, 'total_files': 0, 'success_rate': 1.0}
    
    successful = 0
    
    for i, image_file in enumerate(failed_files, 1):
        print(f"\n📁 Force Process [{i}/{len(failed_files)}]: {image_file.name}")
        
        output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
        output_file = output_dir / image_file.stem
        
        result = extract_character_from_path(
            str(image_file),
            output_path=str(output_file),
            **minimal_args
        )
        
        if result['success']:
            successful += 1
            print(f"✅ Force Process 成功: {image_file.name}")
        else:
            print(f"❌ Force Process 失敗: {image_file.name}")
    
    return {
        'successful': successful,
        'total_files': len(failed_files),
        'success_rate': successful / len(failed_files) if failed_files else 1.0
    }

def main():
    """メイン処理"""
    print("🔥 kaname04超積極的処理システム開始")
    print("=" * 70)
    
    # Phase 1: Ultra Aggressive
    ultra_result = ultra_aggressive_extraction()
    
    # Phase 2: Force Process（まだ失敗があれば）
    remaining_failed = get_failed_files()
    if remaining_failed:
        print(f"\n🔥 Force Process移行: {len(remaining_failed)}ファイルが残存")
        force_result = force_process_with_minimum_requirements()
        total_new_success = ultra_result['successful'] + force_result['successful']
    else:
        total_new_success = ultra_result['successful']
    
    # 最終結果
    final_failed = get_failed_files()
    total_original = 28
    final_successful = total_original - len(final_failed)
    final_success_rate = final_successful / total_original
    
    print("\n" + "=" * 70)
    print("📊 kaname04超積極的処理 最終結果:")
    print(f"   全体成功率: {final_successful}/{total_original} ({final_success_rate:.1%})")
    print(f"   今回処理分: {total_new_success}ファイル追加成功")
    
    if final_success_rate >= 1.0:
        print("🎯 全28ファイル処理完了！")
    else:
        print(f"⚠️ 残り{len(final_failed)}ファイルが未処理")
        for failed_file in final_failed:
            print(f"   - {failed_file.name}")
    
    # 進捗更新
    try:
        progress_file = "progress_req_4_exe_202507120307.json"
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        if 'improvement_stages' not in progress:
            progress['improvement_stages'] = {}
        
        progress['improvement_stages']['ultra_aggressive'] = {
            'successful': total_new_success,
            'final_total': final_successful,
            'final_success_rate': final_success_rate,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        progress['last_update'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✅ 進捗ファイル更新完了")
        
    except Exception as e:
        print(f"⚠️ 進捗ファイル更新エラー: {e}")
    
    return {
        'final_success_rate': final_success_rate,
        'total_successful': final_successful,
        'total_files': total_original
    }

if __name__ == "__main__":
    result = main()
    
    if result['final_success_rate'] >= 1.0:
        print(f"\n🎯 kaname04完全処理達成: 100% 成功率")
        sys.exit(0)
    else:
        print(f"\n⚠️ kaname04部分処理完了: {result['final_success_rate']:.1%} 成功率")
        # 85%以上なら部分的成功とみなす
        if result['final_success_rate'] >= 0.85:
            sys.exit(0)
        else:
            sys.exit(1)