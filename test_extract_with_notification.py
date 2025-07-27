#!/usr/bin/env python3
"""
抽出コマンドの通知機能テスト
"""

import os
import sys
from pathlib import Path

# プロジェクトルート追加
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 必要な環境設定
os.chdir(current_dir)

def test_extraction_with_notification():
    """通知機能付き抽出のテスト"""
    
    try:
        from click.testing import CliRunner
        from features.extraction.commands.extract_character import extract_character
        
        print("✅ 抽出コマンドimport成功")
        
        # テスト用データパス
        test_input = "/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0001.jpg"
        test_output = "/tmp/test_extract_notification.png"
        
        if not Path(test_input).exists():
            print(f"❌ テスト画像が見つかりません: {test_input}")
            return False
        
        print(f"📝 テスト画像: {test_input}")
        print(f"📁 出力先: {test_output}")
        
        # CLIランナーでテスト実行
        runner = CliRunner()
        
        # 1. 通知無効でテスト
        print("\n🧪 テスト1: 通知無効")
        result = runner.invoke(extract_character, [
            test_input,
            '-o', test_output,
            '--verbose',
            '--no-notify'
        ])
        
        print(f"終了コード: {result.exit_code}")
        print(f"出力:\n{result.output}")
        
        if result.exit_code == 0:
            print("✅ テスト1成功")
        else:
            print("❌ テスト1失敗")
            return False
        
        # 2. 通知有効・画像なし
        print("\n🧪 テスト2: 通知有効・画像なし")
        result2 = runner.invoke(extract_character, [
            test_input,
            '-o', test_output.replace('.png', '_2.png'),
            '--verbose',
            '--no-images'
        ])
        
        print(f"終了コード: {result2.exit_code}")
        print(f"出力:\n{result2.output}")
        
        if result2.exit_code == 0:
            print("✅ テスト2成功")
        else:
            print("❌ テスト2失敗")
        
        print("\n🎯 抽出通知テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_extraction_with_notification()