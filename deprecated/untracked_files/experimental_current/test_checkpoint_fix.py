#!/usr/bin/env python3
"""
チェックポイントシステム根本修正テスト
INTG-063-v2.0の修正内容を検証
"""

import os
import sys
from pathlib import Path

# パス設定
sys.path.insert(0, str(Path(__file__).parent))

import json
import tempfile
from features.common.stable_batch_processor import CheckpointManager, StableBatchProcessor


def test_processing_function(file_path: str) -> tuple:
    """テスト処理関数（50%確率で成功）"""
    import random
    import time
    
    filename = os.path.basename(file_path)
    
    # 特定ファイルは必ず成功させる
    if filename in ["test_001.jpg", "test_002.jpg", "test_003.jpg"]:
        time.sleep(0.1)  # 処理時間シミュレーション
        return True, f"✅ {filename} 処理成功"
    
    # その他は50%確率
    if random.random() > 0.5:
        time.sleep(0.1)
        return True, f"✅ {filename} 処理成功"
    else:
        return False, f"❌ {filename} 処理失敗（テスト）"

def main():
    print("🧪 チェックポイントシステム根本修正テスト開始")
    
    # テスト用一時ディレクトリ
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = os.path.join(temp_dir, "checkpoint")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # テスト用ファイルリスト（10個）
        test_files = [f"test_{i:03d}.jpg" for i in range(1, 11)]
        print(f"📂 テストファイル数: {len(test_files)} 件")
        
        # StableBatchProcessor初期化
        processor = StableBatchProcessor(
            checkpoint_dir=checkpoint_dir,
            micro_batch_size=3,
            max_retries=1
        )
        
        print("\n=== フェーズ1: 初回実行（中断シミュレーション） ===")
        
        # 初回実行（3件処理後に中断シミュレーション）
        result1 = processor.process_with_checkpoint(
            files=test_files[:3],  # 最初の3件のみ処理
            process_function=test_processing_function,
            output_dir=output_dir,
            resume=False
        )
        
        print(f"初回実行結果: {result1['success']}")
        print(f"処理済み: {result1['stats']['processed_files']} 件")
        
        print("\n=== フェーズ2: チェックポイント検証 ===")
        
        # チェックポイントファイル確認
        checkpoint_file = Path(checkpoint_dir) / "processing_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            print(f"✅ チェックポイント保存確認")
            print(f"   バージョン: {checkpoint_data.get('version')}")
            print(f"   処理済みファイル数: {len(checkpoint_data['processed_files'])}")
            print(f"   完全ファイルリスト数: {len(checkpoint_data.get('original_total_files', []))}")
            print(f"   残りファイル数: {len(checkpoint_data['remaining_files'])}")
            
            if "original_total_files" in checkpoint_data:
                print("🔧 v2.0形式: original_total_files存在確認")
                
                # 完全リストが保持されているかチェック
                expected_total = 10  # test_files全体
                actual_original = len(checkpoint_data['original_total_files']) if 'original_total_files' in checkpoint_data else 0
                
                if actual_original == expected_total:
                    print(f"✅ 根本修正成功: 完全ファイルリスト {expected_total} 件保持")
                else:
                    print(f"❌ 根本修正失敗: 期待 {expected_total} 件、実際 {actual_original} 件")
            else:
                print("❌ v2.0形式なし: original_total_files不在")
        
        print("\n=== フェーズ3: レジューム実行（修正版） ===")
        
        # 新しいprocessorでレジューム実行
        processor2 = StableBatchProcessor(
            checkpoint_dir=checkpoint_dir,
            micro_batch_size=3,
            max_retries=1
        )
        
        # 全10件のファイルリストでレジューム
        result2 = processor2.process_with_checkpoint(
            files=test_files,  # 🔧 全10件を渡す（修正版では完全リストを復元するはず）
            process_function=test_processing_function,
            output_dir=output_dir,
            resume=True
        )
        
        print(f"レジューム実行結果: {result2['success']}")
        print(f"最終処理済み: {result2['stats']['processed_files']} 件")
        print(f"成功: {result2['stats']['success_count']} 件")
        print(f"エラー: {result2['stats']['error_count']} 件")
        
        print("\n=== フェーズ4: 修正効果検証 ===")
        
        total_expected = len(test_files)
        total_actual = result2['stats']['processed_files']
        
        if total_actual == total_expected:
            print(f"🎯 修正成功: {total_actual}/{total_expected} 件完了")
            print("✅ 43枚制限問題が解決されました")
        else:
            print(f"❌ 修正不完全: {total_actual}/{total_expected} 件完了")
            print("⚠️ まだ制限問題が残存している可能性")
        
        print("\n🧪 テスト完了")

if __name__ == "__main__":
    main()