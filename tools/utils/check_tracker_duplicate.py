#!/usr/bin/env python3
"""
トラッカーID重複チェックスクリプト
Google SheetsでのトラッカーID衝突を防ぐためのユーティリティ
"""

import sys
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.sheets_client import GoogleSheetsClient  # noqa: E402
from tools.progress_tracker.config import get_default_config  # noqa: E402


def check_tracker_exists(tracker_id: str) -> tuple[bool, dict]:
    """
    指定されたトラッカーIDが既に存在するかチェック
    
    Args:
        tracker_id: チェックするトラッカーID
        
    Returns:
        tuple: (exists, info_dict)
            exists: 存在する場合True
            info_dict: 存在する場合のトラッカー情報
    """
    try:
        config = get_default_config()
        client = GoogleSheetsClient(config)
        
        # A列のトラッカーIDを全て取得
        all_values = client.get_sheet_values('A:G')
        if not all_values:
            return False, {}
        
        # ヘッダーをスキップして検索
        for i, row in enumerate(all_values[1:], 2):
            if row and len(row) > 0 and row[0] == tracker_id:
                return True, {
                    'row': i,
                    'tracker_id': row[0] if len(row) > 0 else '',
                    'priority': row[1] if len(row) > 1 else '',
                    'status': row[2] if len(row) > 2 else '',
                    'reg_date': row[3] if len(row) > 3 else '',
                    'update_date': row[4] if len(row) > 4 else '',
                    'summary': row[5] if len(row) > 5 else '',
                    'details': row[6] if len(row) > 6 else '',
                }
        
        return False, {}
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False, {}


def get_next_tracker_id(prefix: str = "QUAL") -> str:
    """
    指定されたプレフィックスで次のトラッカーIDを取得
    
    Args:
        prefix: トラッカーIDプレフィックス（例: QUAL, INCI, INTG）
        
    Returns:
        str: 次のトラッカーID
    """
    try:
        config = get_default_config()
        client = GoogleSheetsClient(config)
        
        # A列のトラッカーIDを全て取得
        all_values = client.get_sheet_values('A:A')
        if not all_values:
            return f"{prefix}-001"
        
        # 指定されたプレフィックスの番号を収集
        numbers = []
        for row in all_values[1:]:  # ヘッダーをスキップ
            if row and len(row) > 0 and row[0].startswith(f"{prefix}-"):
                try:
                    # QUAL-XXXからXXX部分を抽出
                    num_part = row[0].split('-')[1]
                    if num_part.isdigit():
                        numbers.append(int(num_part))
                except:
                    continue
        
        if numbers:
            next_num = max(numbers) + 1
            return f"{prefix}-{next_num:03d}"
        else:
            return f"{prefix}-001"
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return f"{prefix}-001"


def main():
    """メイン実行関数"""
    if len(sys.argv) < 2:
        print("""使用法: python check_tracker_duplicate.py <コマンド> [引数]

コマンド:
    check <tracker_id>     : トラッカーID存在確認
    next [prefix]          : 次のトラッカーID取得（デフォルト: QUAL）

例:
    python check_tracker_duplicate.py check QUAL-036
    python check_tracker_duplicate.py next QUAL
    python check_tracker_duplicate.py next INCI
""")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "check":
        if len(sys.argv) < 3:
            print("❌ トラッカーIDを指定してください")
            sys.exit(1)
            
        tracker_id = sys.argv[2]
        exists, info = check_tracker_exists(tracker_id)
        
        if exists:
            print(f"⚠️  {tracker_id} は既に存在します")
            print(f"   行番号: {info['row']}")
            print(f"   ステータス: {info['status']}")
            print(f"   概要: {info['summary']}")
            print(f"   詳細: {info['details'][:100]}...")
            sys.exit(1)
        else:
            print(f"✅ {tracker_id} は使用可能です")
            
    elif command == "next":
        prefix = sys.argv[2] if len(sys.argv) > 2 else "QUAL"
        next_id = get_next_tracker_id(prefix)
        print(f"✅ 次のトラッカーID: {next_id}")
        
    else:
        print(f"❌ 不明なコマンド: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()