#!/usr/bin/env python3
"""
kaname07 10枚バッチ処理スクリプト
v0.1.0適応学習システムによる最高品質抽出
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """kaname07の10枚バッチ処理実行"""
    
    # パス定義
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # 処理対象画像（10枚）
    target_images = [
        "kaname07_0000_cover.jpg",
        "kaname07_0001.jpg", 
        "kaname07_0002.jpg",
        "kaname07_0003.jpg",
        "kaname07_0004.jpg",
        "kaname07_0005.jpg",
        "kaname07_0006.jpg",
        "kaname07_0007.jpg",
        "kaname07_0008.jpg",
        "kaname07_0009.jpg"
    ]
    
    print(f"🚀 kaname07 10枚バッチ処理開始")
    print(f"📂 入力パス: {input_path}")
    print(f"📂 出力パス: {output_path}")
    print(f"📊 処理対象: {len(target_images)}枚")
    
    # パス存在確認
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 入力画像の存在確認
    missing_images = []
    for image_name in target_images:
        image_path = Path(input_path) / image_name
        if not image_path.exists():
            missing_images.append(image_name)
    
    if missing_images:
        print(f"❌ エラー: 以下の画像が見つかりません:")
        for img in missing_images:
            print(f"   - {img}")
        sys.exit(1)
    
    print(f"✅ 入力画像確認完了: {len(target_images)}枚")
    
    # モデル初期化
    print(f"🔄 v0.1.0システム初期化中...")
    from hooks.start import start
    start()
    print(f"✅ v0.1.0システム初期化完了")
    
    # バッチ処理実行
    from commands.extract_character import extract_character_from_path
    
    success_count = 0
    failed_count = 0
    total_start_time = time.time()
    results = []
    
    for i, image_name in enumerate(target_images, 1):
        print(f"\\n🔄 [{i}/{len(target_images)}] 処理中: {image_name}")
        
        # 入力・出力パス構築
        input_image_path = Path(input_path) / image_name
        output_file_path = Path(output_path) / f"batch10_{i:02d}_{Path(image_name).stem}.jpg"
        
        try:
            start_time = time.time()
            
            # v0.1.0最高品質設定
            result = extract_character_from_path(
                str(input_image_path),
                output_path=str(output_file_path),
                adaptive_learning=True,        # 適応学習システム
                high_quality=True,             # 高品質SAM処理
                manga_mode=True,               # 漫画前処理
                effect_removal=True,           # エフェクト除去
                difficult_pose=True,           # 困難姿勢対応
                multi_character_criteria='size_priority',  # 適応学習推奨
                save_mask=False,               # マスク保存無効（高速化）
                save_transparent=False,        # 透明背景無効（高速化）
                verbose=True                   # 詳細出力
            )
            
            processing_time = time.time() - start_time
            
            if result.get('success', False):
                success_count += 1
                print(f"✅ 成功: {output_file_path.name} ({processing_time:.1f}秒)")
                
                # 適応学習情報表示
                if result.get('adaptive_learning_info'):
                    info = result['adaptive_learning_info']
                    method = info.get('recommended_method', 'N/A')
                    quality = info.get('predicted_quality', 'N/A')
                    confidence = info.get('confidence', 'N/A')
                    print(f"   🧠 推奨手法: {method}")
                    print(f"   📊 予測品質: {quality:.3f}")
                    print(f"   🎯 信頼度: {confidence:.3f}")
                
                # 品質メトリクス表示
                if 'mask_quality' in result:
                    quality_metrics = result['mask_quality']
                    coverage = quality_metrics.get('coverage', 0)
                    compactness = quality_metrics.get('compactness', 0)
                    print(f"   📐 品質: coverage={coverage:.3f}, compactness={compactness:.3f}")
                
                status = "SUCCESS"
            else:
                failed_count += 1
                error_msg = result.get('error', '不明なエラー')
                print(f"❌ 失敗: {image_name} - {error_msg}")
                status = "FAILED"
            
            results.append({
                'image': image_name,
                'status': status,
                'processing_time': processing_time,
                'output_file': output_file_path.name if result.get('success') else None
            })
            
        except Exception as e:
            failed_count += 1
            processing_time = time.time() - start_time
            print(f"❌ 例外発生: {image_name} - {str(e)}")
            results.append({
                'image': image_name,
                'status': "EXCEPTION",
                'processing_time': processing_time,
                'output_file': None
            })
    
    total_time = time.time() - total_start_time
    success_rate = (success_count / len(target_images)) * 100
    
    print(f"\\n📊 kaname07 10枚バッチ処理結果:")
    print(f"   処理枚数: {len(target_images)}")
    print(f"   成功: {success_count}枚")
    print(f"   失敗: {failed_count}枚")
    print(f"   成功率: {success_rate:.1f}%")
    print(f"   総処理時間: {total_time:.1f}秒")
    print(f"   平均処理時間: {total_time/len(target_images):.1f}秒/枚")
    
    # 詳細結果表示
    print(f"\\n📋 詳細結果:")
    for result in results:
        status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
        print(f"   {status_symbol} {result['image']:<25} {result['status']:<10} {result['processing_time']:.1f}秒")
    
    # 生成ファイル確認
    generated_files = list(Path(output_path).glob("batch10_*"))
    print(f"\\n💾 生成ファイル: {len(generated_files)}個")
    
    # 品質判定
    if success_rate >= 80:
        print(f"\\n🎉 バッチ処理成功!")
        print(f"   v0.1.0適応学習システムによる最高品質抽出完了")
        print(f"   成功率 {success_rate:.1f}% は期待値を満たしています")
    else:
        print(f"\\n🚨 バッチ処理で品質問題発生")
        print(f"   成功率 {success_rate:.1f}% は期待値80%を下回ります")
        print(f"   ユーザー要求「何かしらおかしかったらエラーで終了」に従い終了")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\\n⏹️ ユーザー中断")
        sys.exit(0)
    except Exception as e:
        print(f"\\n💥 致命的エラー: {str(e)}")
        sys.exit(1)