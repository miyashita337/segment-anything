#!/usr/bin/env python3
"""
トラッカーID置換エンジン（安全性重視）
バックアップ・復元・検証機能付きの段階的置換システム
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


class TrackerReplacementEngine:
    """安全なトラッカーID置換エンジン"""

    def __init__(
        self,
        mapping_file: str = "tools/trackers/QUAL_032_tracker_standardization/tracker_id_mapping.json",
    ):
        self.mapping_file = mapping_file
        self.mapping = {}
        self.reverse_mapping = {}
        self.backup_dir = None
        self.load_mapping()

        # 置換対象ディレクトリ
        self.target_dirs = ["/mnt/c/AItools/segment-anything", "/mnt/c/AItools/lora/train/yado"]

        # 置換対象ファイル拡張子
        self.target_extensions = {
            ".py",
            ".md",
            ".json",
            ".html",
            ".txt",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
            ".bat",
        }

        # 除外パターン
        self.exclude_patterns = {
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            "node_modules/",
            ".venv/",
            "sam-env/",
            ".idea/",
            ".vscode/",
            "deprecated/untracked_files/experimental_current/",
        }

    def load_mapping(self):
        """マッピングファイル読み込み"""
        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.mapping = data["id_mapping"]
                self.reverse_mapping = data["reverse_mapping"]
                print(f"✅ マッピング読み込み完了: {len(self.mapping)}件")
        except Exception as e:
            print(f"❌ マッピング読み込み失敗: {e}")
            raise

    def create_backup(self) -> str:
        """完全バックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"/tmp/tracker_id_backup_{timestamp}"

        print(f"🔄 バックアップ作成中: {self.backup_dir}")

        try:
            os.makedirs(self.backup_dir, exist_ok=True)

            # 各対象ディレクトリのバックアップ
            for target_dir in self.target_dirs:
                if os.path.exists(target_dir):
                    backup_target = os.path.join(self.backup_dir, os.path.basename(target_dir))
                    print(f"  📦 {target_dir} → {backup_target}")
                    shutil.copytree(target_dir, backup_target, symlinks=True)

            # バックアップメタデータ保存
            metadata = {
                "backup_timestamp": timestamp,
                "target_directories": self.target_dirs,
                "mapping_count": len(self.mapping),
                "git_commit": self.get_git_commit(),
                "git_branch": self.get_git_branch(),
            }

            with open(os.path.join(self.backup_dir, "backup_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ バックアップ完了: {self.backup_dir}")
            return self.backup_dir

        except Exception as e:
            print(f"❌ バックアップ失敗: {e}")
            raise

    def get_git_commit(self) -> str:
        """現在のGitコミットハッシュ取得"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd="/mnt/c/AItools/segment-anything",
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def get_git_branch(self) -> str:
        """現在のGitブランチ取得"""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd="/mnt/c/AItools/segment-anything",
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def scan_replacement_targets(self) -> Dict[str, List[str]]:
        """置換対象ファイル・ディレクトリスキャン"""
        targets = {"files": [], "directories": []}

        print("🔍 置換対象スキャン中...")

        for target_dir in self.target_dirs:
            if not os.path.exists(target_dir):
                print(f"⚠️  ディレクトリが存在しません: {target_dir}")
                continue

            for root, dirs, files in os.walk(target_dir):
                # 除外パターンのチェック
                if any(pattern in root for pattern in self.exclude_patterns):
                    continue

                # ディレクトリ名チェック
                for dirname in dirs[:]:  # スライスコピーで安全に変更
                    if any(old_id in dirname for old_id in self.mapping.keys()):
                        targets["directories"].append(os.path.join(root, dirname))

                    # 除外ディレクトリはスキップ
                    if any(pattern.rstrip("/") in dirname for pattern in self.exclude_patterns):
                        dirs.remove(dirname)

                # ファイル名・内容チェック
                for filename in files:
                    if Path(filename).suffix in self.target_extensions:
                        filepath = os.path.join(root, filename)

                        # ファイル名にトラッカーIDが含まれるか
                        if any(old_id in filename for old_id in self.mapping.keys()):
                            targets["files"].append(filepath)
                        else:
                            # ファイル内容にトラッカーIDが含まれるかチェック
                            if self.file_contains_tracker_ids(filepath):
                                targets["files"].append(filepath)

        print(f"📊 スキャン結果:")
        print(f"   対象ファイル: {len(targets['files'])}件")
        print(f"   対象ディレクトリ: {len(targets['directories'])}件")

        return targets

    def file_contains_tracker_ids(self, filepath: str) -> bool:
        """ファイル内容にトラッカーIDが含まれるかチェック"""
        try:
            # バイナリファイルの検出
            with open(filepath, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:  # バイナリファイルの可能性
                    return False

            # テキストファイルとして読み込み
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return any(old_id in content for old_id in self.mapping.keys())

        except Exception:
            return False

    def replace_in_file(self, filepath: str, dry_run: bool = True) -> Dict[str, int]:
        """ファイル内のトラッカーID置換"""
        replacements = {}

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            original_content = content

            # 各トラッカーIDを置換
            for old_id, new_id in self.mapping.items():
                if old_id in content:
                    content = content.replace(old_id, new_id)
                    replacements[old_id] = (
                        replacements.get(old_id, 0)
                        + content.count(new_id)
                        - original_content.count(old_id)
                    )

            # 実際に書き込み（dry_runでない場合）
            if not dry_run and content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            return replacements

        except Exception as e:
            print(f"⚠️  ファイル処理エラー: {filepath} - {e}")
            return {}

    def replace_directory_names(
        self, directories: List[str], dry_run: bool = True
    ) -> Dict[str, str]:
        """ディレクトリ名置換"""
        renamed = {}

        # 深い階層から処理（子ディレクトリから先に処理）
        directories_sorted = sorted(directories, key=lambda x: x.count("/"), reverse=True)

        for old_dir in directories_sorted:
            new_dir = old_dir

            # ディレクトリ名中のトラッカーIDを置換
            for old_id, new_id in self.mapping.items():
                if old_id in os.path.basename(old_dir):
                    new_dir = old_dir.replace(old_id, new_id)
                    break

            if new_dir != old_dir:
                renamed[old_dir] = new_dir

                if not dry_run:
                    try:
                        os.rename(old_dir, new_dir)
                        print(f"📁 {old_dir} → {new_dir}")
                    except Exception as e:
                        print(f"⚠️  ディレクトリ名変更失敗: {e}")

        return renamed

    def verify_replacement(self) -> Dict[str, any]:
        """置換結果の検証"""
        verification = {
            "success": True,
            "errors": [],
            "warnings": [],
            "statistics": {"total_replacements": 0, "files_modified": 0, "directories_renamed": 0},
        }

        print("🔍 置換結果検証中...")

        # 残存する古いトラッカーIDをチェック
        remaining_old_ids = set()

        for target_dir in self.target_dirs:
            if not os.path.exists(target_dir):
                continue

            for root, dirs, files in os.walk(target_dir):
                # 除外パターンのチェック
                if any(pattern in root for pattern in self.exclude_patterns):
                    continue

                # ディレクトリ名チェック
                for dirname in dirs:
                    for old_id in self.mapping.keys():
                        if old_id in dirname:
                            remaining_old_ids.add(old_id)
                            verification["warnings"].append(
                                f"ディレクトリ名に残存: {old_id} in {os.path.join(root, dirname)}"
                            )

                # ファイルチェック
                for filename in files:
                    if Path(filename).suffix in self.target_extensions:
                        filepath = os.path.join(root, filename)

                        # ファイル名チェック
                        for old_id in self.mapping.keys():
                            if old_id in filename:
                                remaining_old_ids.add(old_id)
                                verification["warnings"].append(f"ファイル名に残存: {old_id} in {filepath}")

        if remaining_old_ids:
            verification["success"] = False
            verification["errors"].append(f"置換未完了のトラッカーID: {', '.join(remaining_old_ids)}")

        return verification

    def execute_replacement(self, dry_run: bool = True) -> Dict[str, any]:
        """置換実行（メインプロセス）"""
        print(f"🚀 トラッカーID置換実行開始 {'(DRY RUN)' if dry_run else '(LIVE RUN)'}")

        # バックアップ作成
        if not dry_run:
            backup_path = self.create_backup()

        # 対象スキャン
        targets = self.scan_replacement_targets()

        # 置換実行
        results = {
            "dry_run": dry_run,
            "backup_path": self.backup_dir if not dry_run else None,
            "targets": targets,
            "file_replacements": {},
            "directory_renames": {},
            "verification": {},
        }

        # ディレクトリ名置換
        print("📁 ディレクトリ名置換中...")
        results["directory_renames"] = self.replace_directory_names(targets["directories"], dry_run)

        # ファイル内容置換
        print("📄 ファイル内容置換中...")
        for filepath in targets["files"]:
            file_results = self.replace_in_file(filepath, dry_run)
            if file_results:
                results["file_replacements"][filepath] = file_results

        # 検証実行
        if not dry_run:
            results["verification"] = self.verify_replacement()

        # 結果サマリー
        print(f"\n📊 置換実行結果:")
        print(f"   対象ファイル: {len(targets['files'])}件")
        print(f"   対象ディレクトリ: {len(targets['directories'])}件")
        print(f"   ディレクトリ名変更: {len(results['directory_renames'])}件")
        print(f"   ファイル内容変更: {len(results['file_replacements'])}件")

        if not dry_run and results["verification"]["success"]:
            print("✅ 置換実行完了・検証成功")
        elif not dry_run:
            print("⚠️  置換実行完了・検証で警告あり")

        return results


def main():
    """メイン実行関数"""
    engine = TrackerReplacementEngine()

    # まずDRY RUNで実行
    print("=" * 60)
    print("フェーズ2.2: 安全性確保システム - DRY RUN実行")
    print("=" * 60)

    dry_results = engine.execute_replacement(dry_run=True)

    # 結果をJSONで保存
    with open("tools/analysis/replacement_dry_run_results.json", "w", encoding="utf-8") as f:
        json.dump(dry_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 DRY RUN結果保存: tools/analysis/replacement_dry_run_results.json")
    print(f"🎯 次ステップ: フェーズ3.1で実際の置換実行")


if __name__ == "__main__":
    main()
