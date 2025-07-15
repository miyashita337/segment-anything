#!/usr/bin/env python3
"""
統合バッチ処理スクリプト
モデル初期化とバッチ処理を一度に実行

Usage:
    python3 batch_extract.py INPUT_DIR OUTPUT_DIR
    
Example:
    python3 batch_extract.py /path/to/input /path/to/output
"""

import sys
import os
import argparse
import time
import torch
import gc
from pathlib import Path

sys.path.append('.')

from utils.notification import send_batch_notification

def gpu_memory_cleanup():
    """GPU メモリクリーンアップ (TDR対策)"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    except Exception as e:
        print(f"⚠️ GPU メモリクリーンアップ失敗: {e}")

def validate_directories(input_dir, output_dir):
    """ディレクトリの検証"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"入力ディレクトリが存在しません: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"入力パスがディレクトリではありません: {input_dir}")
    
    # 出力ディレクトリを作成
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"出力ディレクトリの作成に失敗: {e}")
    
    return str(input_path), str(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="キャラクター抽出バッチ処理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 batch_extract.py /path/to/input /path/to/output
  python3 batch_extract.py "C:\\Images\\Input" "C:\\Images\\Output"
        """
    )
    
    parser.add_argument('input_dir', help='入力ディレクトリパス')
    parser.add_argument('output_dir', help='出力ディレクトリパス')
    parser.add_argument('--min-yolo-score', type=float, default=0.1, 
                       help='YOLO最小信頼度スコア (デフォルト: 0.1)')
    parser.add_argument('--enhance-contrast', action='store_true',
                       help='コントラスト強化を有効化')
    parser.add_argument('--save-mask', action='store_true',
                       help='マスクファイルを保存')
    parser.add_argument('--save-transparent', action='store_true',
                       help='透明背景画像を保存')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細ログを出力')
    
    args = parser.parse_args()
    
    try:
        # ディレクトリ検証
        input_dir, output_dir = validate_directories(args.input_dir, args.output_dir)
        
        print(f"📁 入力ディレクトリ: {input_dir}")
        print(f"📁 出力ディレクトリ: {output_dir}")
        
        # GPU メモリクリーンアップ
        print("🧹 GPU メモリクリーンアップ...")
        gpu_memory_cleanup()
        
        # モデル初期化
        print("🔄 モデル初期化中...")
        from hooks.start import start
        start()
        print("✅ モデル初期化完了")
        
        # バッチ処理実行
        print("🚀 バッチ処理開始...")
        from commands.extract_character import batch_extract_characters
        
        # 設定
        extract_args = {
            'enhance_contrast': args.enhance_contrast,
            'filter_text': True,  # 常に有効
            'save_mask': args.save_mask,
            'save_transparent': args.save_transparent,
            'min_yolo_score': args.min_yolo_score,
            'verbose': args.verbose
        }
        
        print(f"⚙️ 設定:")
        print(f"   YOLO閾値: {extract_args['min_yolo_score']}")
        print(f"   コントラスト強化: {'有効' if extract_args['enhance_contrast'] else '無効'}")
        print(f"   マスク保存: {'有効' if extract_args['save_mask'] else '無効'}")
        print(f"   透明背景: {'有効' if extract_args['save_transparent'] else '無効'}")
        
        start_time = time.time()
        
        result = batch_extract_characters(input_dir, output_dir, **extract_args)
        
        processing_time = time.time() - start_time
        result['total_time'] = processing_time
        
        print(f"\n📊 最終結果:")
        print(f"   成功: {result['successful']}/{result['total_files']} ({result['success_rate']:.1%})")
        print(f"   失敗: {result['failed']}")
        print(f"   処理時間: {processing_time:.2f}秒")
        
        # 最終GPU メモリクリーンアップ
        gpu_memory_cleanup()
        
        # Pushover通知送信
        print("\n📱 通知送信中...")
        try:
            notification_sent = send_batch_notification(
                successful=result['successful'],
                total=result['total_files'],
                failed=result['failed'],
                total_time=processing_time
            )
            
            if notification_sent:
                print("✅ Pushover通知送信完了")
            else:
                print("⚠️ Pushover通知送信失敗またはスキップ")
        except Exception as e:
            print(f"⚠️ 通知送信エラー: {e}")
        
        # 終了コード
        if result['successful'] > 0:
            print(f"\n🎉 バッチ処理完了! {result['successful']}ファイル処理成功")
            sys.exit(0)
        else:
            print(f"\n❌ 処理可能なファイルがありませんでした")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 処理が中断されました")
        gpu_memory_cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        gpu_memory_cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()