#!/usr/bin/env python3
"""
段階的トラッカーID置換実行スクリプト
小さなバッチに分けて安全に置換実行
"""

import json
import os
import shutil
from datetime import datetime
from tracker_replacement_engine import TrackerReplacementEngine


def staged_replacement():
    """段階的置換実行"""
    print("🔄 段階的トラッカーID置換開始")

    # 置換エンジン初期化
    engine = TrackerReplacementEngine()

    # バックアップ作成
    print("📦 バックアップ作成中...")
    backup_path = engine.create_backup()

    # 対象スキャン（軽量版）
    print("🔍 置換対象スキャン（segment-anythingのみ）...")
    segment_anything_dir = "/mnt/c/AItools/segment-anything"

    # 重要ファイルのみ先に処理
    priority_files = ["CLAUDE.md", "CHANGELOG.md", "PROGRESS_TRACKER.md", "README.md"]

    replacements_made = 0

    # Priority files処理
    for filename in priority_files:
        filepath = os.path.join(segment_anything_dir, filename)
        if os.path.exists(filepath):
            print(f"📄 処理中: {filename}")
            result = engine.replace_in_file(filepath, dry_run=False)
            if result:
                replacements_made += len(result)
                print(f"   置換完了: {len(result)}箇所")

    # tools/ディレクトリ処理
    tools_dir = os.path.join(segment_anything_dir, "tools")
    if os.path.exists(tools_dir):
        print("📁 tools/ディレクトリ処理中...")
        for root, dirs, files in os.walk(tools_dir):
            # .gitやpycacheを除外
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for filename in files:
                if filename.endswith((".py", ".md", ".json")):
                    filepath = os.path.join(root, filename)
                    result = engine.replace_in_file(filepath, dry_run=False)
                    if result:
                        replacements_made += len(result)

    print(f"\n✅ 段階的置換完了")
    print(f"   総置換数: {replacements_made}箇所")
    print(f"   バックアップ: {backup_path}")

    return backup_path, replacements_made


if __name__ == "__main__":
    try:
        backup_path, count = staged_replacement()
        print(f"🎉 成功! 置換数: {count}, バックアップ: {backup_path}")
    except Exception as e:
        print(f"❌ エラー: {e}")
