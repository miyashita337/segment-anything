#!/usr/bin/env python3
"""
kaname07最高品質キャラクター抽出バッチ実行
Phase 1完了版 - 全強化機能統合実行
"""

import sys
import os
import time
from pathlib import Path
import traceback

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_kaname07_highest_quality_batch():
    """kaname07最高品質バッチ抽出実行（Phase 1全機能統合）"""
    
    # パス設定（ユーザー指定）
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    print("🚀 kaname07最高品質キャラクター抽出バッチ実行開始（Phase 1全機能統合）")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    
    # 入力パス検証
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)
    
    # 画像ファイル取得
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    
    if not image_files:
        print(f"❌ エラー: 入力パスに画像ファイルがありません: {input_path}")
        sys.exit(1)
    
    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Phase 1統合抽出実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        # Phase 1品質メトリクス
        quality_scores = []
        processing_times = []
        enhancement_results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")
            
            try:
                image_start = time.time()
                
                # 出力ファイル名設定（番号付きで整理）
                output_filename = f"{i:05d}_{image_file.stem}.jpg"
                output_file_path = Path(output_path) / output_filename
                
                # Phase 1最高品質設定での抽出実行
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file_path),
                    multi_character_criteria='fullbody_priority_enhanced',  # P1-003改良版全身検出
                    enhance_contrast=True,   # コントラスト強化
                    filter_text=True,        # テキストフィルタリング
                    save_mask=True,          # マスク保存
                    save_transparent=True,   # 透明背景保存
                    verbose=False,           # バッチ処理なので詳細ログは抑制
                    high_quality=True,       # 高品質処理
                    difficult_pose=True,     # 困難姿勢対応
                    adaptive_learning=True,  # 適応学習
                    manga_mode=True,         # 漫画モード
                    effect_removal=True,     # エフェクト除去
                    min_yolo_score=0.05,     # YOLO閾値を緩めに設定
                    # Phase 1追加オプション
                    use_enhanced_screentone=True,   # P1-004強化スクリーントーン検出
                    use_mosaic_boundary=True,       # P1-005モザイク境界処理
                    use_solid_fill_enhancement=True, # P1-006ベタ塗り領域改善
                    partial_extraction_check=True,  # P1-002部分抽出検出
                )
                
                image_time = time.time() - image_start
                processing_times.append(image_time)
                
                if result.get('success', False):
                    success_count += 1
                    print(f"✅ 成功: {output_filename}")
                    
                    # Phase 1品質情報表示
                    if 'quality_score' in result:
                        quality_score = result['quality_score']
                        quality_scores.append(quality_score)
                        print(f"   品質スコア: {quality_score:.3f}")
                    
                    # P1-002部分抽出分析結果
                    if 'extraction_analysis' in result:
                        analysis = result['extraction_analysis']
                        completeness = analysis.get('completeness_score', 0)
                        print(f"   完全性スコア: {completeness:.3f}")
                        
                        if analysis.get('has_face', False):
                            print("   ✓ 顔検出")
                        if analysis.get('has_torso', False):
                            print("   ✓ 胴体検出")
                        if analysis.get('has_limbs', False):
                            print("   ✓ 手足検出")
                    
                    # P1-003強化全身検出結果
                    if 'enhanced_fullbody_score' in result:
                        enhanced_score = result['enhanced_fullbody_score']
                        print(f"   強化全身スコア: {enhanced_score:.3f}")
                    
                    # P1-004スクリーントーン検出結果
                    if 'screentone_detected' in result:
                        if result['screentone_detected']:
                            print(f"   🎨 スクリーントーン検出: {result.get('screentone_confidence', 0):.3f}")
                    
                    # P1-005モザイク境界処理結果
                    if 'mosaic_detected' in result:
                        if result['mosaic_detected']:
                            print(f"   🧩 モザイク検出: {result.get('mosaic_type', 'unknown')}")
                    
                    # P1-006ベタ塗り処理結果
                    if 'solid_fill_detected' in result:
                        if result['solid_fill_detected']:
                            regions = result.get('solid_fill_regions', 0)
                            print(f"   🎨 ベタ塗り領域: {regions}個")
                    
                    enhancement_results.append({
                        'file': image_file.name,
                        'quality': result.get('quality_score', 0),
                        'completeness': analysis.get('completeness_score', 0) if 'extraction_analysis' in result else 0,
                        'processing_time': image_time
                    })
                    
                else:
                    error_count += 1
                    print(f"❌ 失敗: {image_file.name}")
                    if 'error' in result:
                        print(f"   エラー: {result['error']}")
                        
            except Exception as e:
                error_count += 1
                print(f"❌ 例外: {image_file.name} - {str(e)}")
                print(f"   詳細: {traceback.format_exc()}")
            
            # 進捗表示
            elapsed = time.time() - start_time
            remaining = len(image_files) - i
            if i > 0:
                avg_time = elapsed / i
                eta = avg_time * remaining
                print(f"   進捗: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%), "
                      f"経過: {elapsed/60:.1f}分, 残り推定: {eta/60:.1f}分")
                print(f"   処理時間: {image_time:.2f}秒")
        
        # Phase 1統合結果サマリー
        total_time = time.time() - start_time
        print(f"\n🎉 Phase 1最高品質バッチ処理完了!")
        print(f"📊 Phase 1統合結果サマリー:")
        print(f"   総処理数: {len(image_files)}")
        print(f"   成功: {success_count}")
        print(f"   失敗: {error_count}")
        print(f"   成功率: {success_count/len(image_files)*100:.1f}%")
        print(f"   総処理時間: {total_time/60:.1f}分")
        print(f"   平均処理時間: {total_time/len(image_files):.1f}秒/画像")
        
        # Phase 1品質メトリクス分析
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            print(f"\n📈 Phase 1品質メトリクス:")
            print(f"   平均品質スコア: {avg_quality:.3f}")
            print(f"   品質範囲: {min_quality:.3f} - {max_quality:.3f}")
            
            # 高品質画像の比率
            high_quality_count = sum(1 for q in quality_scores if q >= 0.8)
            print(f"   高品質画像(≥0.8): {high_quality_count}/{len(quality_scores)} ({high_quality_count/len(quality_scores)*100:.1f}%)")
        
        if processing_times:
            avg_proc_time = sum(processing_times) / len(processing_times)
            print(f"   平均処理時間: {avg_proc_time:.2f}秒/画像")
        
        print(f"\n📁 出力ディレクトリ: {output_path}")
        
        if error_count > 0:
            print(f"⚠️  {error_count}個のファイルでエラーが発生しました")
        
        # Phase 1機能別統計（概算）
        print(f"\n🔬 Phase 1機能動作統計:")
        print(f"   - P1-002 部分抽出検出: 実装済み・統合済み")
        print(f"   - P1-003 強化全身検出: fullbody_priority_enhanced使用")
        print(f"   - P1-004 スクリーントーン検出: 統合実行")
        print(f"   - P1-005 モザイク境界処理: 統合実行")
        print(f"   - P1-006 ベタ塗り処理: 統合実行")
        
        return success_count, error_count, len(image_files)
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("extract_character モジュールが見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        print(f"詳細: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        success, error, total = run_kaname07_highest_quality_batch()
        print(f"\n✨ 最終結果: {success}/{total} 成功 ({success/total*100:.1f}%)")
    except KeyboardInterrupt:
        print(f"\n⏹️  ユーザーによる中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 致命的エラー: {e}")
        sys.exit(1)