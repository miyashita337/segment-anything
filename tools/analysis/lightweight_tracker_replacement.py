#!/usr/bin/env python3
"""
軽量トラッカーID置換エンジン
Git stash + 選択的バックアップ方式
"""

import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class LightweightTrackerReplacement:
    """軽量版トラッカーID置換システム"""
    
    def __init__(self):
        self.mapping = {}
        self.load_mapping()
        
        # 軽量バックアップ対象（重要ファイルのみ）
        self.important_files = [
            "CLAUDE.md", "CHANGELOG.md", "README.md", "PROGRESS_TRACKER.md",
            "requirements.txt", "setup.py", "pyproject.toml"
        ]
        
        # 置換対象ディレクトリ（限定版）
        self.target_dirs = [
            "tools/", "docs/", "config/", "bin/shell/",
            "features/", "tests/"
        ]
        
        # 除外する大容量ディレクトリ
        self.exclude_heavy_dirs = {
            'sam-env', '.git', '__pycache__', '.pytest_cache',
            'deprecated/untracked_files/experimental_current',
            'node_modules', '.venv'
        }
    
    def load_mapping(self):
        """マッピング読み込み"""
        with open("tools/analysis/tracker_id_mapping.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.mapping = data['id_mapping']
        print(f"✅ マッピング読み込み: {len(self.mapping)}件")
    
    def create_git_stash_backup(self) -> str:
        """Git stashで軽量バックアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stash_message = f"tracker-id-standardization-backup-{timestamp}"
        
        try:
            # 現在の変更をstash
            result = subprocess.run([
                'git', 'stash', 'push', '-m', stash_message
            ], cwd='/mnt/c/AItools/segment-anything', capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Git stashバックアップ完了: {stash_message}")
                return stash_message
            else:
                print(f"⚠️  Git stash結果: {result.stderr}")
                return stash_message
                
        except Exception as e:
            print(f"⚠️  Git stashエラー: {e}")
            return "manual-backup-required"
    
    def create_selective_backup(self) -> str:
        """選択的軽量バックアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"/tmp/tracker_lightweight_backup_{timestamp}"
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup_count = 0
            
            # 重要ファイルのバックアップ
            base_dir = "/mnt/c/AItools/segment-anything"
            for filename in self.important_files:
                src = os.path.join(base_dir, filename)
                if os.path.exists(src):
                    dst = os.path.join(backup_dir, filename)
                    shutil.copy2(src, dst)
                    backup_count += 1
            
            # 重要ディレクトリの軽量バックアップ
            for dirname in self.target_dirs:
                src_dir = os.path.join(base_dir, dirname)
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(backup_dir, dirname)
                    
                    # 軽量コピー（.pyと.mdのみ）
                    for root, dirs, files in os.walk(src_dir):
                        # 除外ディレクトリをスキップ
                        dirs[:] = [d for d in dirs if d not in self.exclude_heavy_dirs]
                        
                        for file in files:
                            if file.endswith(('.py', '.md', '.json', '.yaml', '.yml')):
                                src_file = os.path.join(root, file)
                                rel_path = os.path.relpath(src_file, base_dir)
                                dst_file = os.path.join(backup_dir, rel_path)
                                
                                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                                shutil.copy2(src_file, dst_file)
                                backup_count += 1
            
            print(f"✅ 選択的バックアップ完了: {backup_count}ファイル → {backup_dir}")
            return backup_dir
            
        except Exception as e:
            print(f"❌ 選択的バックアップ失敗: {e}")
            raise
    
    def replace_in_file(self, filepath: str) -> int:
        """ファイル内のトラッカーID置換"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            replacement_count = 0
            
            # 各トラッカーIDを置換
            for old_id, new_id in self.mapping.items():
                before_count = content.count(old_id)
                if before_count > 0:
                    content = content.replace(old_id, new_id)
                    after_count = content.count(old_id)
                    replacement_count += (before_count - after_count)
            
            # 変更があった場合のみ書き込み
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return replacement_count
            
            return 0
            
        except Exception as e:
            print(f"⚠️  ファイル処理エラー: {filepath} - {e}")
            return 0
    
    def process_important_files(self) -> int:
        """重要ファイルの処理"""
        total_replacements = 0
        base_dir = "/mnt/c/AItools/segment-anything"
        
        print("📄 重要ファイル処理中...")
        for filename in self.important_files:
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                count = self.replace_in_file(filepath)
                if count > 0:
                    print(f"   ✅ {filename}: {count}箇所置換")
                    total_replacements += count
        
        return total_replacements
    
    def process_target_directories(self) -> int:
        """対象ディレクトリの処理"""
        total_replacements = 0
        base_dir = "/mnt/c/AItools/segment-anything"
        
        print("📁 対象ディレクトリ処理中...")
        for dirname in self.target_dirs:
            dir_path = os.path.join(base_dir, dirname)
            if not os.path.exists(dir_path):
                continue
                
            dir_replacements = 0
            print(f"   🔄 {dirname}")
            
            for root, dirs, files in os.walk(dir_path):
                # 除外ディレクトリをスキップ
                dirs[:] = [d for d in dirs if d not in self.exclude_heavy_dirs]
                
                for filename in files:
                    if filename.endswith(('.py', '.md', '.json', '.yaml', '.yml', '.txt')):
                        filepath = os.path.join(root, filename)
                        count = self.replace_in_file(filepath)
                        if count > 0:
                            dir_replacements += count
            
            if dir_replacements > 0:
                print(f"   ✅ {dirname}: {dir_replacements}箇所置換")
                total_replacements += dir_replacements
        
        return total_replacements
    
    def execute_lightweight_replacement(self) -> Dict[str, any]:
        """軽量置換実行"""
        print("🚀 軽量トラッカーID置換開始")
        
        # Git stashバックアップ
        stash_name = self.create_git_stash_backup()
        
        # 選択的バックアップ
        backup_dir = self.create_selective_backup()
        
        # 置換実行
        important_replacements = self.process_important_files()
        directory_replacements = self.process_target_directories()
        
        total_replacements = important_replacements + directory_replacements
        
        result = {
            "success": True,
            "stash_backup": stash_name,
            "file_backup": backup_dir,
            "replacements": {
                "important_files": important_replacements,
                "directories": directory_replacements,
                "total": total_replacements
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✅ 軽量置換完了:")
        print(f"   総置換数: {total_replacements}箇所")
        print(f"   Git stash: {stash_name}")
        print(f"   ファイルバックアップ: {backup_dir}")
        
        return result

def main():
    """メイン実行"""
    try:
        replacer = LightweightTrackerReplacement()
        result = replacer.execute_lightweight_replacement()
        
        # 結果保存
        with open("tools/analysis/lightweight_replacement_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 結果保存: tools/analysis/lightweight_replacement_result.json")
        return True
        
    except Exception as e:
        print(f"❌ 軽量置換エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)