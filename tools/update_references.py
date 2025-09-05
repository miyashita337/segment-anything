#!/usr/bin/env python3
"""
KIRO-001 Phase 3: 参照リンク一括更新スクリプト

中リスクファイル移行時の参照更新を自動化
"""

import os
import re
import glob
from typing import Dict, List, Tuple

# 移行マッピング定義
FILE_MAPPINGS = {
    "spec.md": "docs/technical/specifications/system_spec.md",
    "PROJECT_SETTINGS.md": "docs/development/quality/standards.md", 
    "QC_COMPREHENSIVE_REPORT.md": "docs/reports/quality/qc_comprehensive_report.md",
}

def find_markdown_files(root_dir: str = ".") -> List[str]:
    """Markdownファイルを再帰的に検索"""
    patterns = ["**/*.md", "**/*.py"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    
    # バックアップディレクトリを除外
    excluded_patterns = [
        "docs_backup_",
        "deprecated/",
        "__pycache__",
        ".git/"
    ]
    
    filtered_files = []
    for file_path in files:
        should_exclude = any(excluded in file_path for excluded in excluded_patterns)
        if not should_exclude:
            filtered_files.append(file_path)
    
    return filtered_files

def update_file_references(file_path: str, mappings: Dict[str, str]) -> Tuple[bool, int]:
    """ファイル内の参照を更新"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        replacements = 0
        
        for old_path, new_path in mappings.items():
            # 様々なパターンの参照を更新
            patterns = [
                # Markdown リンク: [text](old_path)
                rf'\[([^\]]*)\]\(({re.escape(old_path)})\)',
                # 直接参照: old_path
                rf'\b{re.escape(old_path)}\b',
                # docs/old_path の形式
                rf'docs/{re.escape(old_path)}',
                # ./old_path の形式  
                rf'\./{re.escape(old_path)}',
            ]
            
            for pattern in patterns:
                if re.search(pattern, content):
                    if '[' in pattern and ']' in pattern:
                        # Markdown リンクの場合
                        content = re.sub(pattern, rf'[\1]({new_path})', content)
                    else:
                        # 直接参照の場合
                        content = re.sub(pattern, new_path, content)
                    replacements += re.subn(pattern, new_path, original_content)[1] - re.subn(pattern, new_path, content)[1]
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, replacements
        
        return False, 0
        
    except Exception as e:
        print(f"エラー: {file_path} の処理中にエラー: {e}")
        return False, 0

def main():
    """メイン実行関数"""
    print("🔄 KIRO-001 Phase 3: 参照リンク一括更新開始")
    print(f"📋 移行マッピング:")
    for old, new in FILE_MAPPINGS.items():
        print(f"  {old} → {new}")
    
    # Markdownファイルを検索
    files = find_markdown_files()
    print(f"📁 対象ファイル: {len(files)}個")
    
    # 統計情報
    updated_files = 0
    total_replacements = 0
    
    # 各ファイルを処理
    for file_path in files:
        was_updated, replacements = update_file_references(file_path, FILE_MAPPINGS)
        if was_updated:
            updated_files += 1
            total_replacements += replacements
            print(f"✅ {file_path}: {replacements}箇所更新")
    
    print(f"\n📊 更新完了:")
    print(f"  - 更新ファイル数: {updated_files}")
    print(f"  - 総更新箇所数: {total_replacements}")
    
    if updated_files > 0:
        print(f"\n⚠️  次の手順:")
        print(f"  1. git add で更新ファイルをステージング")
        print(f"  2. git commit でコミット作成")
        print(f"  3. 動作確認テスト実行")

if __name__ == "__main__":
    main()