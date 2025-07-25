#!/usr/bin/env python3
"""
Enhanced Filtering System Full Batch - 全26枚で最終テスト
"""
import os
import sys
import time
from pathlib import Path

# プロジェクトルートに追加
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
from features.common.notification.notification import PushoverNotifier


def main():
    # 入力・出力ディレクトリ
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_enhanced_system_final")
    
    # 出力ディレクトリを確保
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルリスト取得
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    total = len(image_files)
    
    print(f"🚀 Enhanced Filtering System - 最終バッチテスト")
    print(f"📁 入力: {input_dir}")
    print(f"📁 出力: {output_dir}")
    print(f"📊 総数: {total}枚")
    print(f"🔧 システム: Phase1(非キャラクター除外) + Phase3(品質向上)")
    
    # カウンター
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 統計情報
    filter_stats = {
        'original_masks': [],
        'filtered_masks': [],
        'face_detections': 0,
        'quality_improvements': 0
    }
    
    # 各画像を処理
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"📸 処理中 [{i}/{total}]: {image_path.name}")
        
        # 出力パス（元のファイル名を保持）
        output_path = output_dir / image_path.name
        
        try:
            # CLI経由で抽出実行
            cmd = [
                'python3', '-m', 'features.extraction.commands.extract_character',
                str(image_path),
                '-o', str(output_path),
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), timeout=180)
            
            if output_path.exists():
                successful += 1
                print(f"✅ 成功: {output_path.name}")
                
                # 統計情報収集
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines:
                    if 'Final masks for selection:' in line:
                        final_count = int(line.split(':')[1].strip())
                        filter_stats['filtered_masks'].append(final_count)
                        print(f"   🔧 フィルタ後マスク数: {final_count}")
                    elif 'Selected mask validation: faces=' in line:
                        if 'faces=1' in line or 'faces=2' in line:
                            filter_stats['face_detections'] += 1
                        print(f"   👤 {line.strip().split('Selected mask validation:')[1]}")
                    elif 'Mask quality:' in line:
                        quality_line = line.strip()
                        if 'Needs improvement: True' in quality_line:
                            filter_stats['quality_improvements'] += 1
                        print(f"   📊 {quality_line.split('Mask quality:')[1]}")
                    elif 'Character extracted:' in line and 'size:' in line:
                        size_info = line.split('size:')[1].strip().rstrip(')')
                        print(f"   📏 サイズ: {size_info}")
            else:
                failed += 1
                print(f"❌ 失敗: {output_path.name}")
                if result.stderr:
                    print(f"   エラー: {result.stderr.strip()[-100:]}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"❌ タイムアウト: {image_path.name}")
        except Exception as e:
            failed += 1
            print(f"❌ エラー: {image_path.name} - {e}")
    
    # 処理時間計算
    total_time = time.time() - start_time
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"🎯 Enhanced System 最終バッチ完了")
    print(f"✅ 成功: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ 失敗: {failed}")
    print(f"⏱️  処理時間: {total_time:.1f}秒 (平均: {total_time/total:.1f}秒/画像)")
    
    # システム効果分析
    if filter_stats['filtered_masks']:
        avg_filtered = sum(filter_stats['filtered_masks']) / len(filter_stats['filtered_masks'])
        print(f"\n📊 Enhanced System 効果分析:")
        print(f"   🔧 平均フィルタ後マスク数: {avg_filtered:.1f}")
        print(f"   👤 顔検出成功: {filter_stats['face_detections']}/{successful} "
              f"({filter_stats['face_detections']/max(successful,1)*100:.1f}%)")
        print(f"   🛠️ 品質改善実行: {filter_stats['quality_improvements']}/{successful} "
              f"({filter_stats['quality_improvements']/max(successful,1)*100:.1f}%)")
    
    # 期待される改善効果
    if successful >= 20:  # 77%以上成功
        print(f"🎉 優秀！Enhanced Systemは大幅な改善を達成しました")
        print(f"   予想される評価改善: 19.2% → 70%+")
    elif successful >= 15:  # 58%以上成功
        print(f"🔧 良好！Enhanced Systemは顕著な改善を示しました")
        print(f"   予想される評価改善: 19.2% → 50-70%")
    else:
        print(f"⚠️ 部分的改善。追加の調整が必要です")
    
    # Pushover通知
    try:
        notifier = PushoverNotifier()
        notifier.send_batch_complete_with_images(
            successful=successful,
            total=total,
            failed=failed,
            total_time=total_time,
            image_dir=output_dir
        )
        print("📱 Pushover通知送信完了")
    except Exception as e:
        print(f"⚠️ Pushover通知失敗: {e}")

if __name__ == "__main__":
    main()