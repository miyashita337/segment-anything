#!/usr/bin/env python3
"""
kaname04データセット改善版バッチ処理スクリプト
失敗ファイルを対象とした段階的パラメータ調整による再処理
"""

import sys
import os
import json
import time
from pathlib import Path
sys.path.append('.')

from utils.notification import send_batch_notification

def get_failed_files():
    """失敗したファイルリストを取得"""
    input_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/org/kaname04")
    output_dir = Path("/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04")
    
    all_files = list(input_dir.glob("*.jpg"))
    processed_files = list(output_dir.glob("*.jpg"))
    
    # 処理済みファイル名（拡張子なし）のセット
    processed_names = {f.stem for f in processed_files}
    
    # 未処理ファイルを特定
    failed_files = [f for f in all_files if f.stem not in processed_names]
    
    return failed_files

def run_improved_extraction(stage=1):
    """改善版抽出実行"""
    print(f"🔄 Stage {stage}: 改善版kaname04処理開始...")
    
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    from commands.extract_character import extract_character_from_path
    
    # Stage別パラメータ設定
    if stage == 1:
        extract_args = {
            'enhance_contrast': True,      # コントラスト強化ON
            'filter_text': True,
            'save_mask': False,
            'save_transparent': False,
            'min_yolo_score': 0.05,        # 閾値を0.1→0.05に下げる
            'verbose': True,
            'high_quality': True           # 高品質SAM処理
        }
        print("📊 Stage 1 設定: YOLO閾値0.05, コントラスト強化, 高品質SAM")
        
    elif stage == 2:
        extract_args = {
            'enhance_contrast': True,
            'filter_text': True,
            'save_mask': False,
            'save_transparent': False,
            'min_yolo_score': 0.02,        # さらに低い閾値
            'verbose': True,
            'low_threshold': True,         # 低閾値モード
            'auto_retry': True             # 自動リトライ
        }
        print("📊 Stage 2 設定: YOLO閾値0.02, 低閾値モード, 自動リトライ")
        
    elif stage == 3:
        extract_args = {
            'enhance_contrast': True,
            'filter_text': False,          # テキストフィルタ無効化
            'save_mask': False,
            'save_transparent': False,
            'min_yolo_score': 0.01,        # 最低閾値
            'verbose': True,
            'difficult_pose': True,        # 複雑ポーズモード
            'auto_retry': True,
            'high_quality': True
        }
        print("📊 Stage 3 設定: YOLO閾値0.01, 複雑ポーズモード, テキストフィルタOFF")
    
    # 失敗ファイル取得
    failed_files = get_failed_files()
    
    if not failed_files:
        print("🎯 処理すべきファイルがありません（全て完了済み）")
        return {
            'success': True,
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 1.0,
            'stage': stage
        }
    
    print(f"📁 Stage {stage} 対象ファイル数: {len(failed_files)}")
    
    # 各ファイル処理
    results = []
    successful = 0
    
    for i, image_file in enumerate(failed_files, 1):
        print(f"\n📁 Stage {stage} 処理中 [{i}/{len(failed_files)}]: {image_file.name}")
        
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
        result['stage'] = stage
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"✅ Stage {stage} 成功: {image_file.name}")
        else:
            print(f"❌ Stage {stage} 失敗: {image_file.name} - {result['error']}")
    
    # Stage結果
    stage_result = {
        'success': True,
        'total_files': len(failed_files),
        'successful': successful,
        'failed': len(failed_files) - successful,
        'success_rate': successful / len(failed_files) if failed_files else 1.0,
        'results': results,
        'stage': stage
    }
    
    print(f"\n📊 Stage {stage} 結果:")
    print(f"   成功: {successful}/{len(failed_files)} ({stage_result['success_rate']:.1%})")
    
    return stage_result

def update_progress(stage_results):
    """進捗ファイル更新"""
    try:
        progress_file = "progress_req_4_exe_202507120307.json"
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        # Stage結果を記録
        if 'improvement_stages' not in progress:
            progress['improvement_stages'] = {}
        
        progress['improvement_stages'][f'stage_{stage_results["stage"]}'] = {
            'successful': stage_results['successful'],
            'total': stage_results['total_files'],
            'success_rate': stage_results['success_rate'],
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        progress['last_update'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print("✅ 進捗ファイル更新完了")
        
    except Exception as e:
        print(f"⚠️ 進捗ファイル更新エラー: {e}")

def main():
    """メイン処理 - 段階的実行"""
    print("🚀 kaname04改善版処理システム開始")
    print("=" * 60)
    
    total_processed = 0
    all_stage_results = []
    
    # Stage 1: 基本改善パラメータ
    stage1_result = run_improved_extraction(stage=1)
    all_stage_results.append(stage1_result)
    total_processed += stage1_result['successful']
    update_progress(stage1_result)
    
    # Stage 2: より低い閾値（Stage1で失敗したファイルがある場合）
    remaining_failed = get_failed_files()
    if remaining_failed:
        print(f"\n🔄 Stage 2移行: {len(remaining_failed)}ファイルが残存")
        stage2_result = run_improved_extraction(stage=2)
        all_stage_results.append(stage2_result)
        total_processed += stage2_result['successful']
        update_progress(stage2_result)
    
    # Stage 3: 最終手段（まだ失敗ファイルがある場合）
    remaining_failed = get_failed_files()
    if remaining_failed:
        print(f"\n🔄 Stage 3移行: {len(remaining_failed)}ファイルが残存")
        stage3_result = run_improved_extraction(stage=3)
        all_stage_results.append(stage3_result)
        total_processed += stage3_result['successful']
        update_progress(stage3_result)
    
    # 最終結果計算
    final_failed = get_failed_files()
    total_original = 28
    final_successful = total_original - len(final_failed)
    final_success_rate = final_successful / total_original
    
    print("\n" + "=" * 60)
    print("📊 kaname04改善版処理 最終結果:")
    print(f"   全体成功率: {final_successful}/{total_original} ({final_success_rate:.1%})")
    print(f"   今回処理分: {total_processed}ファイル追加成功")
    
    # Stage別結果詳細
    for stage_result in all_stage_results:
        stage = stage_result['stage']
        print(f"   Stage {stage}: {stage_result['successful']}/{stage_result['total_files']} ({stage_result['success_rate']:.1%})")
    
    if final_success_rate >= 1.0:
        print("🎯 全28ファイル処理完了！")
    else:
        print(f"⚠️ 残り{len(final_failed)}ファイルが未処理")
        for failed_file in final_failed:
            print(f"   - {failed_file.name}")
    
    return {
        'final_success_rate': final_success_rate,
        'total_successful': final_successful,
        'total_files': total_original,
        'stage_results': all_stage_results
    }

if __name__ == "__main__":
    result = main()
    
    if result['final_success_rate'] >= 1.0:
        print(f"\n🎯 kaname04完全処理達成: 100% 成功率")
        sys.exit(0)
    else:
        print(f"\n⚠️ kaname04部分処理完了: {result['final_success_rate']:.1%} 成功率")
        sys.exit(1)