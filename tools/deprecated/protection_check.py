#!/usr/bin/env python3
"""
Deprecated File Protection System
sam_yolo_character_segment.py使用防止システム

使用方法:
    python tools/deprecated/protection_check.py <target_script>
"""

import sys
import os
from pathlib import Path

DEPRECATED_FILES = {
    "sam_yolo_character_segment.py": {
        "reason": "ユーザー指定により非推奨化",
        "replacement": "features/extraction/commands/extract_character.py",
        "moved_date": "2025-08-11"
    }
}

def check_deprecated_usage(script_path: str) -> bool:
    """
    スクリプトがdeprecatedファイルを使用しようとしているかチェック
    """
    script_name = os.path.basename(script_path)
    
    if script_name in DEPRECATED_FILES:
        info = DEPRECATED_FILES[script_name]
        
        print("🚫 =============== 実行停止警告 ===============")
        print(f"❌ 非推奨ファイルの使用が検出されました: {script_name}")
        print(f"📅 非推奨化日: {info['moved_date']}")
        print(f"🔍 理由: {info['reason']}")
        print(f"✅ 代替手段: {info['replacement']}")
        print()
        print("🛡️ このファイルは意図的に非推奨化されています。")
        print("⚠️ 使用を継続する場合は、明示的にユーザー確認が必要です。")
        print("===============================================")
        
        return False
    
    return True

def main():
    if len(sys.argv) != 2:
        print("使用方法: python protection_check.py <target_script>")
        sys.exit(1)
    
    target_script = sys.argv[1]
    
    if not check_deprecated_usage(target_script):
        print("\n🛑 実行を停止しました。ユーザーに確認してください。")
        sys.exit(1)
    
    print("✅ 実行可能: deprecated チェックを通過しました")

if __name__ == "__main__":
    main()