#!/usr/bin/env python3
"""
Phase 4参照リンク更新スクリプト
高リスクファイル（トラッカーワークフロー中核ファイル）の参照を一括更新

対象ファイル:
- tracker_workflow_checklist.md (16参照) 
- unified_tracker_template.md (6参照)
- google_sheets_reference.md (18参照)
総計: 40参照箇所
"""

import os
import re
import sys
from pathlib import Path

# Phase 4ファイルマッピング（旧パス → 新パス）
PHASE4_MAPPINGS = {
    # トラッカーワークフローチェックリスト (最重要)
    "docs/checklists/tracker_workflow_checklist.md": "docs/workflows/checklists/tracker_workflow_checklist.md",
    "checklists/tracker_workflow_checklist.md": "docs/workflows/checklists/tracker_workflow_checklist.md",
    
    # 統合テンプレート (SOW機能含む)  
    "docs/templates/unified_tracker_template.md": "docs/workflows/templates/unified_tracker_template.md",
    "templates/unified_tracker_template.md": "docs/workflows/templates/unified_tracker_template.md",
    
    # Google Sheets連携 (外部サービス依存)
    "docs/google_sheets_reference.md": "docs/integrations/external/google_sheets_reference.md",
    "google_sheets_reference.md": "docs/integrations/external/google_sheets_reference.md",
}

def update_references_in_file(file_path):
    """ファイル内のPhase 4参照を更新"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        updates_made = 0
        
        for old_path, new_path in PHASE4_MAPPINGS.items():
            # マークダウンリンク形式: [text](path)
            pattern1 = rf'\[([^\]]*)\]\({re.escape(old_path)}\)'
            replacement1 = rf'[\1]({new_path})'
            content, count1 = re.subn(pattern1, replacement1, content)
            updates_made += count1
            
            # 直接パス参照形式: `path`
            pattern2 = rf'`{re.escape(old_path)}`'
            replacement2 = f'`{new_path}`'
            content, count2 = re.subn(pattern2, replacement2, content)
            updates_made += count2
            
            # 相対パス形式: ../path や ./path
            for prefix in ['../', './', '']:
                if old_path.startswith('docs/') and prefix:
                    continue  # docs/で始まるパスにプレフィックスは不要
                    
                old_with_prefix = prefix + old_path
                new_with_prefix = prefix + new_path
                
                # マークダウンリンク
                pattern3 = rf'\[([^\]]*)\]\({re.escape(old_with_prefix)}\)'
                replacement3 = rf'[\1]({new_with_prefix})'
                content, count3 = re.subn(pattern3, replacement3, content)
                updates_made += count3
                
                # 直接参照
                pattern4 = rf'`{re.escape(old_with_prefix)}`'
                replacement4 = f'`{new_with_prefix}`'
                content, count4 = re.subn(pattern4, replacement4, content)
                updates_made += count4
        
        if updates_made > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path}: {updates_made}箇所更新")
        
        return updates_made
        
    except Exception as e:
        print(f"❌ エラー - {file_path}: {e}")
        return 0

def main():
    """Phase 4参照更新メイン処理"""
    print("🚀 Phase 4参照リンク更新開始")
    print("対象: トラッカーワークフロー中核ファイル (40参照予定)")
    print("-" * 50)
    
    total_updates = 0
    processed_files = 0
    
    # .mdファイルを再帰的に検索して処理
    for md_file in Path('.').rglob('*.md'):
        if '.git' in str(md_file) or 'deprecated' in str(md_file):
            continue  # Git管理ファイルと廃止ファイルをスキップ
            
        updates = update_references_in_file(md_file)
        if updates > 0:
            total_updates += updates
            processed_files += 1
    
    print("-" * 50)
    print(f"🎯 Phase 4更新完了")
    print(f"📊 更新ファイル数: {processed_files}")
    print(f"📊 総更新箇所数: {total_updates}")
    
    if total_updates > 0:
        print("✅ Phase 4参照更新成功")
    else:
        print("⚠️ 更新対象が見つかりませんでした")

if __name__ == "__main__":
    main()