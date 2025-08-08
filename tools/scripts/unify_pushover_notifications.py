#!/usr/bin/env python3
"""
Pushover通知システム統一化スクリプト
全ファイルをglobal_pushover.pyに移行
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_files_with_pushover() -> List[Path]:
    """Pushover関連のコードを含むファイルを検索"""
    files = []
    
    # 検索パターン
    patterns = [
        r'api\.pushover\.net',
        r'requests\.post.*pushover',
        r'PushoverNotifier',
        r'ExtractionNotifier',
        r'send_pushover',
        r'from.*notification import',
        r'from.*extraction_notifier import'
    ]
    
    # Pythonファイルを検索
    for py_file in PROJECT_ROOT.rglob("*.py"):
        # 除外ディレクトリ
        if any(skip in str(py_file) for skip in [
            "global_pushover.py",  # 統一先は除外
            "__pycache__",
            ".git",
            "venv",
            "sam-env",
            ".pytest_cache"
        ]):
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8')
            for pattern in patterns:
                if re.search(pattern, content):
                    files.append(py_file)
                    break
        except Exception:
            pass
            
    return files


def migrate_file(file_path: Path) -> bool:
    """ファイルをglobal_pushoverに移行"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # 1. 旧インポートを新インポートに置換
        replacements = [
            # extraction_notifier関連
            (r'from features\.extraction\.extraction_notifier import.*?\n(?:.*?\n)*?(?:except ImportError:.*?\n)?',
             'from features.common.notification.global_pushover import (\n    notify_success,\n    notify_error,\n    notify_process_complete\n)\nPUSHOVER_AVAILABLE = True\n'),
            
            # notification.py関連
            (r'from features\.common\.notification\.notification import.*?\n',
             'from features.common.notification.global_pushover import notify_success, notify_error, notify_process_complete\n'),
            
            # 直接API実装の検出と警告コメント追加
            (r'(requests\.post\(["\']https://api\.pushover\.net)',
             r'# TODO: global_pushover.pyに移行必要\n# \1'),
             
            # PushoverNotifier()の置換
            (r'notifier = (?:PushoverNotifier|ExtractionNotifier)\(\)',
             '# 統一通知システムを使用（インスタンス化不要）'),
             
            # send_pushover_notification呼び出しの置換
            (r'send_pushover_notification\((.*?)\)',
             r'notify_success(\1)'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # 2. メソッド呼び出しの置換
        method_replacements = [
            (r'notifier\.send_extraction_completion_notification\([^)]+\)',
             'notify_process_complete(title="抽出完了", successful=success_count, total=total_count, duration=duration)'),
            (r'notifier\.send_notification\((.*?)\)',
             r'notify_success(\1)'),
            (r'notifier\.send_error\((.*?)\)',
             r'notify_error(\1)'),
        ]
        
        for pattern, replacement in method_replacements:
            content = re.sub(pattern, replacement, content)
        
        # 変更があった場合のみ保存
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
            
    except Exception as e:
        print(f"❌ エラー: {file_path}: {e}")
        
    return False


def main():
    """メイン処理"""
    print("🔍 Pushover実装を含むファイルを検索中...")
    files = find_files_with_pushover()
    
    print(f"📊 {len(files)} ファイル発見")
    
    # 重要度順にソート
    priority_patterns = [
        "extract_character.py",
        "sam_yolo_character_segment.py",
        "integrated_quality_pipeline.py",
        "tools/core/",
        "features/extraction/",
        "features/common/",
    ]
    
    def get_priority(path: Path) -> int:
        path_str = str(path)
        for i, pattern in enumerate(priority_patterns):
            if pattern in path_str:
                return i
        return len(priority_patterns)
    
    files.sort(key=get_priority)
    
    # 移行実行
    migrated = 0
    for file_path in files:
        relative_path = file_path.relative_to(PROJECT_ROOT)
        print(f"📝 処理中: {relative_path}")
        
        if migrate_file(file_path):
            print(f"  ✅ 移行完了")
            migrated += 1
        else:
            print(f"  ⏭️  変更なし")
    
    print(f"\n📊 移行結果:")
    print(f"  ✅ 移行済み: {migrated}/{len(files)} ファイル")
    
    # 旧モジュールの無効化
    obsolete_files = [
        PROJECT_ROOT / "features/extraction/extraction_notifier.py",
        PROJECT_ROOT / "features/common/notification/notification.py",
    ]
    
    for obsolete_file in obsolete_files:
        if obsolete_file.exists() and obsolete_file.name != "global_pushover.py":
            # ファイル先頭に廃止警告を追加
            try:
                content = obsolete_file.read_text(encoding='utf-8')
                if "DEPRECATED" not in content[:100]:
                    warning = '''"""
⚠️ DEPRECATED: このモジュールは廃止されました
代わりに features.common.notification.global_pushover を使用してください

from features.common.notification.global_pushover import (
    notify_success,
    notify_error,
    notify_process_complete
)
"""

'''
                    obsolete_file.write_text(warning + content, encoding='utf-8')
                    print(f"⚠️  廃止警告追加: {obsolete_file.name}")
            except Exception as e:
                print(f"❌ 警告追加失敗: {obsolete_file.name}: {e}")
    
    print("\n✅ Pushover通知システムの統一化が完了しました！")
    print("📝 次のステップ:")
    print("  1. テスト実行: python -m pytest tests/")
    print("  2. 動作確認: Pushover通知が正常に届くことを確認")
    print("  3. 旧ファイル削除: 動作確認後、obsoleteファイルを削除")


if __name__ == "__main__":
    main()