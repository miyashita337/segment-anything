#!/usr/bin/env python3
"""
境界強調システム - 全26枚バッチテスト
Phase A実装効果の包括的評価
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
    output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_boundary_enhanced_full")
    
    # 出力ディレクトリを確保
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルリスト取得
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    total = len(image_files)
    
    print(f"🚀 Phase A: 境界強調システム - 全バッチテスト")
    print(f"📁 入力: {input_dir}")
    print(f"📁 出力: {output_dir}")
    print(f"📊 総数: {total}枚")
    print(f"🔧 システム: 肌色・衣装境界強調処理 + Enhanced Filtering")
    
    # カウンター
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 統計情報
    enhancement_stats = {
        'contrast_improvements': [],
        'edge_improvements': [],
        'total_enhancement_factor': 0.0
    }
    
    # 各画像を処理
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"📸 処理中 [{i}/{total}]: {image_path.name}")
        
        # 出力パス（元のファイル名を保持）
        output_path = output_dir / image_path.name
        
        try:
            # CLI経由で境界強調版抽出実行
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
                
                # 境界強調統計情報収集
                stdout_lines = result.stdout.split('\\n')
                for line in stdout_lines:
                    if '境界強調統計' in line:
                        print(f"   📊 {line.strip()}")
                        # 統計値抽出
                        if 'コントラスト改善=' in line:
                            try:
                                contrast_val = float(line.split('コントラスト改善=')[1].split('x')[0])
                                enhancement_stats['contrast_improvements'].append(contrast_val)
                            except:
                                pass
                        if 'エッジ改善=' in line:
                            try:
                                edge_val = float(line.split('エッジ改善=')[1].split('x')[0])
                                enhancement_stats['edge_improvements'].append(edge_val)
                            except:
                                pass
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
    print(f"\\n{'='*60}")
    print(f"🎯 Phase A: 境界強調システム バッチ完了")
    print(f"✅ 成功: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ 失敗: {failed}")
    print(f"⏱️  処理時間: {total_time:.1f}秒 (平均: {total_time/total:.1f}秒/画像)")
    
    # Phase A 境界強調効果分析
    if enhancement_stats['contrast_improvements'] and enhancement_stats['edge_improvements']:
        avg_contrast = sum(enhancement_stats['contrast_improvements']) / len(enhancement_stats['contrast_improvements'])
        avg_edge = sum(enhancement_stats['edge_improvements']) / len(enhancement_stats['edge_improvements'])
        
        print(f"\\n📊 Phase A: 境界強調効果分析")
        print(f"   📈 平均コントラスト改善: {avg_contrast:.2f}x")
        print(f"   🔍 平均エッジ改善: {avg_edge:.2f}x")
        print(f"   🎯 境界認識向上度: {((avg_contrast + avg_edge) / 2):.2f}x")
        
        # 境界強調システムの評価
        total_improvement = (avg_contrast + avg_edge) / 2
        if total_improvement >= 1.5:
            print(f"🎉 Phase A成功！境界強調システムが大幅な改善を達成")
            print(f"   予想される評価改善: 20% → 50-70%")
        elif total_improvement >= 1.2:
            print(f"🔧 Phase A効果あり！境界強調システムが改善を示しました")
            print(f"   予想される評価改善: 20% → 35-50%")
        elif total_improvement >= 1.0:
            print(f"📈 Phase A軽微改善。追加の最適化が有効です")
            print(f"   予想される評価改善: 20% → 25-35%")
        else:
            print(f"⚠️ Phase A効果限定的。Phase Bへの移行を推奨")
    
    # 比較結果の提示
    print(f"\\n📋 前回結果との比較")
    print(f"   前回(Enhanced System): 25/26成功 (96.2%), 評価20%")
    print(f"   今回(Phase A): {successful}/{total}成功 ({successful/total*100:.1f}%)")
    
    success_rate_comparison = successful/total if total > 0 else 0
    if success_rate_comparison >= 0.96:
        print(f"   📊 処理成功率: 維持 (優秀)")
    elif success_rate_comparison >= 0.90:
        print(f"   📊 処理成功率: 軽微低下 (許容範囲)")
    else:
        print(f"   ⚠️ 処理成功率: 大幅低下 (要調整)")
    
    # Pushover通知
    try:
        notifier = PushoverNotifier()
        notifier.send_batch_complete_with_images(
            successful=successful,
            total=total,
            failed=failed,
            total_time=total_time,
            image_dir=output_dir,
            title="Phase A: 境界強調システム完了"
        )
        print("📱 Pushover通知送信完了")
    except Exception as e:
        print(f"⚠️ Pushover通知失敗: {e}")

if __name__ == "__main__":
    main()