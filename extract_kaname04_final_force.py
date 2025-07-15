#!/usr/bin/env sdaffdsafasdfsadfasdsa
"""
kaname04データセット最終強制処理スクリプト
残り13ファイルに対する極限的アプローチ
"""

import sys
import os
import json
import time
from pathlib import Path
sys.path.append('.')

def get_remaining_files():
    """残りファイルリストを取得"""
    input_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/org/kaname04")
    output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
    
    all_files = list(input_dir.glob("*.jpg"))
    processed_files = list(output_dir.glob("*.jpg"))
    
    processed_names = {f.stem for f in processed_files}
    remaining_files = [f for f in all_files if f.stem not in processed_names and f.stem.startswith('00')]
    
    return sorted(remaining_files)

def final_force_extraction():
    """最終強制抽出実行"""
    print("🔥 Final Force Mode: kaname04最終処理開始...")
    
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    from commands.extract_character import extract_character_from_path
    
    # 最終強制パラメータ設定（極限値）
    final_args = {
        'enhance_contrast': True,
        'filter_text': False,
        'save_mask': False,
        'save_transparent': False,
        'min_yolo_score': 0.0001,       # 極限閾値
        'verbose': True,
        'difficult_pose': True,
        'low_threshold': True,
        'auto_retry': True,
        'high_quality': True,
        'manga_mode': True,
        'effect_removal': True,
        'panel_split': True,
        'multi_character_criteria': 'confidence'  # 信頼度優先
    }
    
    print("📊 Final Force設定:")
    print("   - YOLO閾値: 0.0001 (極限値)")
    print("   - 全強化オプション: ON")
    print("   - 多段階リトライ: ON")
    
    # 残りファイル取得
    remaining_files = get_remaining_files()
    
    if not remaining_files:
        print("🎯 処理すべきファイルがありません（全て完了済み）")
        return {'successful': 0, 'total_files': 0, 'success_rate': 1.0}
    
    print(f"📁 Final Force対象ファイル数: {len(remaining_files)}")
    for f in remaining_files:
        print(f"   - {f.name}")
    
    # 各ファイル処理
    results = []
    successful = 0
    
    for i, image_file in enumerate(remaining_files, 1):
        print(f"\n📁 Final Force 処理中 [{i}/{len(remaining_files)}]: {image_file.name}")
        
        # 出力パス生成
        output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
        output_file = output_dir / image_file.stem
        
        # 抽出実行
        result = extract_character_from_path(
            str(image_file),
            output_path=str(output_file),
            **final_args
        )
        
        result['filename'] = image_file.name
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"✅ Final Force 成功: {image_file.name}")
        else:
            print(f"❌ Final Force 失敗: {image_file.name} - {result['error']}")
    
    return {
        'successful': successful,
        'total_files': len(remaining_files),
        'success_rate': successful / len(remaining_files) if remaining_files else 1.0,
        'results': results
    }

def main():
    """メイン処理"""
    print("🔥 kaname04最終強制処理システム開始")
    print("=" * 70)
    
    # 現在の状況確認
    current_processed = len(list(Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04").glob("*.jpg")))
    print(f"📊 現在の処理済み: {current_processed}ファイル")
    
    # Final Force実行
    force_result = final_force_extraction()
    
    # 最終結果
    final_processed = len(list(Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04").glob("*.jpg")))
    total_original = 28
    final_success_rate = final_processed / total_original
    
    print("\n" + "=" * 70)
    print("📊 kaname04最終強制処理 結果:")
    print(f"   最終成功率: {final_processed}/{total_original} ({final_success_rate:.1%})")
    print(f"   今回追加分: {force_result['successful']}ファイル")
    
    if final_success_rate >= 1.0:
        print("🎯 全28ファイル処理完了！")
    else:
        remaining_count = total_original - final_processed
        print(f"⚠️ 残り{remaining_count}ファイルが未処理")
    
    # 進捗更新
    try:
        progress_file = "progress_req_4_exe_202507120307.json"
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        if 'improvement_stages' not in progress:
            progress['improvement_stages'] = {}
        
        progress['improvement_stages']['final_force'] = {
            'successful': force_result['successful'],
            'final_total': final_processed,
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
        'total_successful': final_processed,
        'total_files': total_original
    }

if __name__ == "__main__":
    result = main()
    
    if result['final_success_rate'] >= 1.0:
        print(f"\n🎯 kaname04完全処理達成: 100% 成功率")
        sys.exit(0)
    else:
        print(f"\n📈 kaname04処理完了: {result['final_success_rate']:.1%} 成功率")
        # 75%以上なら成功とみなす
        if result['final_success_rate'] >= 0.75:
            sys.exit(0)
        else:
            sys.exit(1)