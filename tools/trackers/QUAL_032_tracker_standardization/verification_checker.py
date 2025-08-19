#!/usr/bin/env python3
"""
トラッカーID置換結果の整合性確認
残存チェック・動作確認・リンク検証
"""

import os
import json
import subprocess
from typing import Dict, List, Set

class ReplacementVerificationChecker:
    """置換結果検証システム"""
    
    def __init__(self):
        self.load_mapping()
        self.base_dir = "/mnt/c/AItools/segment-anything"
        
    def load_mapping(self):
        """マッピング読み込み"""
        with open("tools/trackers/QUAL_032_tracker_standardization/tracker_id_mapping.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.old_ids = set(data['id_mapping'].keys())
            self.new_ids = set(data['id_mapping'].values())
            self.mapping = data['id_mapping']
        print(f"✅ 検証対象: {len(self.old_ids)}個の旧ID → {len(self.new_ids)}個の新ID")
    
    def check_remaining_old_ids(self) -> Dict[str, List[str]]:
        """残存する旧トラッカーIDをチェック"""
        remaining = {}
        search_dirs = ["tools/", "docs/", "config/", "features/", "tests/"]
        
        print("🔍 残存旧トラッカーIDチェック中...")
        
        for dirname in search_dirs:
            dir_path = os.path.join(self.base_dir, dirname)
            if not os.path.exists(dir_path):
                continue
                
            print(f"   📁 {dirname}")
            dir_remaining = []
            
            for root, dirs, files in os.walk(dir_path):
                # 除外ディレクトリ
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if filename.endswith(('.py', '.md', '.json', '.yaml', '.txt')):
                        filepath = os.path.join(root, filename)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # 各旧IDの残存チェック
                            for old_id in self.old_ids:
                                if old_id in content:
                                    dir_remaining.append(f"{old_id} in {os.path.relpath(filepath, self.base_dir)}")
                                    
                        except Exception:
                            continue
            
            if dir_remaining:
                remaining[dirname] = dir_remaining
        
        return remaining
    
    def check_new_ids_presence(self) -> Dict[str, int]:
        """新IDの存在確認"""
        new_id_counts = {}
        search_dirs = ["tools/", "docs/", "config/", "features/", "tests/"]
        
        print("🔍 新トラッカーID存在確認中...")
        
        for dirname in search_dirs:
            dir_path = os.path.join(self.base_dir, dirname)
            if not os.path.exists(dir_path):
                continue
                
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if filename.endswith(('.py', '.md', '.json', '.yaml', '.txt')):
                        filepath = os.path.join(root, filename)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # 各新IDの出現数カウント
                            for new_id in self.new_ids:
                                count = content.count(new_id)
                                if count > 0:
                                    new_id_counts[new_id] = new_id_counts.get(new_id, 0) + count
                                    
                        except Exception:
                            continue
        
        return new_id_counts
    
    def check_important_files(self) -> Dict[str, any]:
        """重要ファイルの状態確認"""
        important_files = ["CLAUDE.md", "CHANGELOG.md", "README.md", "PROGRESS_TRACKER.md"]
        results = {}
        
        print("📄 重要ファイル状態確認中...")
        
        for filename in important_files:
            filepath = os.path.join(self.base_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 旧ID残存チェック
                    remaining_old = [old_id for old_id in self.old_ids if old_id in content]
                    
                    # 新ID存在チェック  
                    present_new = [new_id for new_id in self.new_ids if new_id in content]
                    
                    results[filename] = {
                        "file_size": len(content),
                        "remaining_old_ids": remaining_old,
                        "present_new_ids": len(present_new),
                        "status": "✅ OK" if not remaining_old else f"⚠️  旧ID残存: {len(remaining_old)}個"
                    }
                    
                except Exception as e:
                    results[filename] = {"error": str(e)}
        
        return results
    
    def test_basic_functionality(self) -> Dict[str, any]:
        """基本機能動作テスト"""
        print("🧪 基本機能動作テスト中...")
        
        tests = {}
        
        # Python import テスト
        try:
            result = subprocess.run([
                'python3', '-c', 
                'import sys; sys.path.insert(0, "."); from tools.trackers.QUAL_032_tracker_standardization.tracker_id_mapping_generator import generate_tracker_mapping; print("✅ Import OK")'
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=30)
            
            tests["python_import"] = {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None
            }
        except Exception as e:
            tests["python_import"] = {"success": False, "error": str(e)}
        
        # Git状態確認
        try:
            result = subprocess.run([
                'git', 'status', '--porcelain'
            ], cwd=self.base_dir, capture_output=True, text=True)
            
            changed_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            tests["git_status"] = {
                "changed_files_count": len(changed_files),
                "changed_files": changed_files[:10],  # 最初の10ファイルのみ
                "status": "✅ Git OK" if result.returncode == 0 else "⚠️  Git Error"
            }
        except Exception as e:
            tests["git_status"] = {"error": str(e)}
        
        return tests
    
    def run_full_verification(self) -> Dict[str, any]:
        """完全検証実行"""
        print("🔍 トラッカーID置換結果の完全検証開始")
        print("=" * 50)
        
        # 各種チェック実行
        remaining_old = self.check_remaining_old_ids()
        new_id_counts = self.check_new_ids_presence()
        important_files = self.check_important_files()
        functionality = self.test_basic_functionality()
        
        # 結果統計
        total_remaining = sum(len(items) for items in remaining_old.values())
        total_new_occurrences = sum(new_id_counts.values())
        
        verification_result = {
            "timestamp": "2025-08-17T17:41:45",
            "overall_success": total_remaining == 0,
            "statistics": {
                "total_old_ids_remaining": total_remaining,
                "total_new_id_occurrences": total_new_occurrences,
                "new_ids_found": len(new_id_counts),
                "expected_new_ids": len(self.new_ids)
            },
            "detailed_results": {
                "remaining_old_ids": remaining_old,
                "new_id_counts": new_id_counts,
                "important_files": important_files,
                "functionality_tests": functionality
            }
        }
        
        # 結果サマリー表示
        print("\n📊 検証結果サマリー:")
        print(f"   残存旧ID: {total_remaining}個")
        print(f"   新ID出現: {total_new_occurrences}箇所")
        print(f"   検出新ID: {len(new_id_counts)}/{len(self.new_ids)}種類")
        
        if total_remaining == 0:
            print("✅ 検証成功: 旧トラッカーIDの残存なし")
        else:
            print(f"⚠️  検証警告: {total_remaining}個の旧IDが残存")
        
        # 重要ファイル状況
        print("\n📄 重要ファイル状況:")
        for filename, status in important_files.items():
            print(f"   {filename}: {status.get('status', 'エラー')}")
        
        return verification_result

def main():
    """メイン実行"""
    checker = ReplacementVerificationChecker()
    result = checker.run_full_verification()
    
    # 結果保存
    with open("tools/analysis/verification_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 検証結果保存: tools/analysis/verification_result.json")
    
    return result["overall_success"]

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)