#!/usr/bin/env python3
"""
リポジトリ構造整理スクリプト

⚠️ 重要: この情報は project_structure_reference.md に統合されました。
最新情報については: docs/project_structure_reference.md を参照してください。

Usage: python tools/cleanup_repository.py --dry-run
"""
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


class RepositoryCleanup:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_date = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 整理対象の定義
        self.delete_candidates = [
            "*.pid", "*.tmp", "auto_execution_log.json", "auto_progress_system.log",
            "phase2_batch*.log", "kana*_batch.log", "run_batch_*.log",
            "create_all_26_files.py", "debug_evaluation.py", "benchmark_*.py"
        ]
        
        self.move_to_deprecated = [
            "true_content_evaluator.py", "true_success_analyzer.py", 
            "visual_intent_analyzer.py", "visual_verification_system.py",
            "backup_analysis_report.md", "improvement_report.html",
            "final_report_dangerous_mode.md", "extract_kana08_batch.py",
            "kana08_*.py", "run_phase2_*.py"
        ]
        
        self.move_to_logs = [
            "*.log", "phase2_completion_message.txt", 
            "phase2_fixed_completion_message.txt", "*_progress.json",
            "batch_evaluation_results*.json"
        ]
    
    def create_directories(self):
        """必要なディレクトリを作成"""
        dirs_to_create = [
            "deprecated", "logs", "config", 
            "docs/migration", "docs/issues",
            "tests/unit", "tests/integration", "tests/fixtures"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ ディレクトリ作成: {dir_path}")
    
    def backup_current_state(self):
        """現在の状態をバックアップ"""
        backup_dir = self.project_root / f"backup_before_cleanup_{self.backup_date}"
        
        # 重要ファイルのバックアップ
        important_files = [
            "spec.md", "PRINCIPLE.md", "README.md",
            "docs/workflows/", "features/", "core/", "tools/"
        ]
        
        backup_dir.mkdir(exist_ok=True)
        
        for item in important_files:
            source = self.project_root / item
            if source.exists():
                if source.is_file():
                    shutil.copy2(source, backup_dir / item)
                else:
                    shutil.copytree(source, backup_dir / item, dirs_exist_ok=True)
        
        print(f"✅ バックアップ作成: {backup_dir}")
    
    def move_files_to_deprecated(self, dry_run=True):
        """廃止予定ファイルをdeprecated/に移動"""
        deprecated_dir = self.project_root / "deprecated"
        moved_files = []
        
        for pattern in self.move_to_deprecated:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    target_path = deprecated_dir / file_path.name
                    
                    if not dry_run:
                        shutil.move(str(file_path), str(target_path))
                    
                    moved_files.append(str(file_path.relative_to(self.project_root)))
                    print(f"📦 deprecated/に移動: {file_path.name}")
        
        return moved_files
    
    def move_files_to_logs(self, dry_run=True):
        """ログファイルをlogs/に移動"""
        logs_dir = self.project_root / "logs"
        moved_files = []
        
        for pattern in self.move_to_logs:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and "logs/" not in str(file_path):
                    target_path = logs_dir / file_path.name
                    
                    if not dry_run:
                        shutil.move(str(file_path), str(target_path))
                    
                    moved_files.append(str(file_path.relative_to(self.project_root)))
                    print(f"📋 logs/に移動: {file_path.name}")
        
        return moved_files
    
    def delete_temporary_files(self, dry_run=True):
        """一時ファイルの削除"""
        deleted_files = []
        
        for pattern in self.delete_candidates:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    if not dry_run:
                        file_path.unlink()
                    
                    deleted_files.append(str(file_path.relative_to(self.project_root)))
                    print(f"🗑️ 削除: {file_path.name}")
        
        return deleted_files
    
    def generate_cleanup_report(self, moved_deprecated, moved_logs, deleted_files):
        """整理レポートの生成"""
        report = {
            "cleanup_date": datetime.now().isoformat(),
            "backup_location": f"backup_before_cleanup_{self.backup_date}",
            "moved_to_deprecated": moved_deprecated,
            "moved_to_logs": moved_logs,
            "deleted_files": deleted_files,
            "new_structure": {
                "deprecated/": "廃止予定ファイル",
                "logs/": "ログファイル集約",
                "config/": "設定ファイル",
                "docs/migration/": "移行ガイド",
                "docs/issues/": "問題追跡"
            }
        }
        
        report_path = self.project_root / "docs" / "cleanup_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 整理レポート作成: {report_path}")
        return report
    
    def execute_cleanup(self, dry_run=True):
        """整理実行"""
        print(f"{'🔍 DRY RUN モード' if dry_run else '🚀 実行モード'}")
        print("="*50)
        
        # 1. バックアップ作成
        if not dry_run:
            self.backup_current_state()
        
        # 2. ディレクトリ作成
        self.create_directories()
        
        # 3. ファイル移動・削除
        moved_deprecated = self.move_files_to_deprecated(dry_run)
        moved_logs = self.move_files_to_logs(dry_run)
        deleted_files = self.delete_temporary_files(dry_run)
        
        # 4. レポート生成
        if not dry_run:
            report = self.generate_cleanup_report(moved_deprecated, moved_logs, deleted_files)
        
        print("="*50)
        print(f"📊 整理完了")
        print(f"   deprecated/移動: {len(moved_deprecated)}ファイル")
        print(f"   logs/移動: {len(moved_logs)}ファイル") 
        print(f"   削除: {len(deleted_files)}ファイル")
        
        if dry_run:
            print("\n💡 実際に実行するには --execute オプションを使用してください")

def main():
    parser = argparse.ArgumentParser(description="リポジトリ構造整理スクリプト")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="実際には変更せず、何が起こるかだけを表示")
    parser.add_argument("--execute", action="store_true", 
                       help="実際に整理を実行")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    cleanup = RepositoryCleanup(str(project_root))
    
    cleanup.execute_cleanup(dry_run=not args.execute)

if __name__ == "__main__":
    main()