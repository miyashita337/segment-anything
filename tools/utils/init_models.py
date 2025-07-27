#!/usr/bin/env python3
"""
統合モデル初期化スクリプト
Phase 0リファクタリング後の新構造対応版

このスクリプトは以下の機能を提供:
1. SAM/YOLOモデルの統合初期化
2. Phase 0新構造での完全対応
3. 詳細なログ出力とエラーハンドリング
4. 独立実行可能なスタンドアロン版

使用方法:
    python3 init_models.py
    
オプション:
    --verbose : 詳細ログ出力
    --test    : テストモード（モデル動作確認）
"""

import argparse
import os
import sys
from pathlib import Path

# Phase 0新構造対応のパス設定
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="統合モデル初期化スクリプト")
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    parser.add_argument('--test', action='store_true', help='テストモード')
    args = parser.parse_args()
    
    print("🚀 統合モデル初期化スクリプト開始")
    print("=" * 50)
    
    if args.verbose:
        print("📋 Phase 0新構造対応版")
        print("📁 プロジェクト構造:")
        print("  ├── core/            # 元Facebook実装")
        print("  ├── features/        # 自作機能")
        print("  ├── tests/           # 統合テスト")
        print("  └── tools/           # 実行スクリプト")
        print()
    
    success = False
    
    try:
        # Phase 0新構造でのモデル初期化
        from features.common.hooks.start import initialize_models
        
        if args.verbose:
            print("📦 新構造モジュールインポート成功")
        
        print("🔄 モデル初期化実行中...")
        success = initialize_models()
        
        if success:
            print("✅ モデル初期化成功")
            
            if args.test:
                print("\n🧪 テストモード実行中...")
                success = run_test_mode(args.verbose)
                
        else:
            print("❌ モデル初期化失敗")
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("💡 Phase 0構造が正しく配置されていない可能性があります")
        print("   確認項目:")
        print("   - features/common/hooks/start.py の存在")
        print("   - features/extraction/models/ の存在")
        print("   - 必要な依存関係のインストール")
        
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 統合モデル初期化完了")
        print("💡 キャラクター抽出の実行が可能になりました")
        print("   例: python3 run_batch_extraction.py")
    else:
        print("💔 統合モデル初期化失敗")
        print("🔧 トラブルシューティング:")
        print("   1. 依存関係確認: pip install -r requirements.txt")
        print("   2. GPU利用可能性確認: python3 -c \"import torch; print(torch.cuda.is_available())\"")
        print("   3. モデルファイル確認: ls -la *.pth *.pt")
        print("   4. 詳細ログ: python3 init_models.py --verbose")
    
    return 0 if success else 1

def run_test_mode(verbose=False):
    """テストモード実行"""
    try:
        from features.common.hooks.start import (
            get_performance_monitor,
            get_sam_model,
            get_yolo_model,
        )
        
        print("🔍 初期化されたモデルの確認...")
        
        # SAMモデル確認
        sam_model = get_sam_model()
        if sam_model:
            print("✅ SAM model: 正常に初期化済み")
            if verbose:
                print(f"   Type: {type(sam_model)}")
        else:
            print("❌ SAM model: 初期化失敗")
            return False
        
        # YOLOモデル確認
        yolo_model = get_yolo_model()
        if yolo_model:
            print("✅ YOLO model: 正常に初期化済み")
            if verbose:
                print(f"   Type: {type(yolo_model)}")
        else:
            print("❌ YOLO model: 初期化失敗")
            return False
        
        # パフォーマンスモニター確認
        performance_monitor = get_performance_monitor()
        if performance_monitor:
            print("✅ Performance monitor: 正常に初期化済み")
            if verbose:
                print(f"   Type: {type(performance_monitor)}")
        else:
            print("❌ Performance monitor: 初期化失敗")
            return False
        
        print("🎯 全モデルテスト完了")
        return True
        
    except Exception as e:
        print(f"❌ テストモードエラー: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(main())