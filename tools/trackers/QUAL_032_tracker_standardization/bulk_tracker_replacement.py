#!/usr/bin/env python3
"""
トラッカーID全件一括置換スクリプト
ユーザー指定B戦略: 137件を QUAL/OPTM/TEST/INTG に一括置換

対象範囲:
- /mnt/c/AItools/segment-anything/
- /mnt/c/AItools/lora/train/yado/

対象ファイル:
- *.py, *.md, *.json, *.html, ディレクトリパス
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class BulkTrackerReplacer:
    def __init__(self):
        self.mapping_file = (
            "tools/trackers/QUAL_032_tracker_standardization/tracker_function_mapping.json"
        )
        self.target_dirs = ["/mnt/c/AItools/segment-anything/", "/mnt/c/AItools/lora/train/yado/"]
        self.target_extensions = [".py", ".md", ".json", ".html"]
        self.replacement_map = {}
        self.stats = {
            "files_processed": 0,
            "directories_renamed": 0,
            "total_replacements": 0,
            "errors": [],
        }

    def load_mapping(self):
        """マッピングデータの読み込み"""
        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 置換マップ作成
            for category, trackers in data["mapping"].items():
                for tracker in trackers:
                    old_id = tracker["original_id"]
                    new_id = tracker["new_id"]
                    self.replacement_map[old_id] = new_id

            print(f"✅ マッピング読み込み完了: {len(self.replacement_map)}件")
            return True

        except Exception as e:
            print(f"❌ マッピング読み込み失敗: {e}")
            return False

    def replace_in_file(self, file_path: Path) -> int:
        """ファイル内のトラッカーID置換"""
        try:
            # ファイル読み込み
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            replacements = 0

            # 各トラッカーIDを置換
            for old_id, new_id in self.replacement_map.items():
                # 文字境界を考慮した置換
                pattern = r"\b" + re.escape(old_id) + r"\b"
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    content = re.sub(pattern, new_id, content)
                    replacements += matches
                    print(f"   📝 {file_path.name}: {old_id} → {new_id} ({matches}箇所)")

            # 変更があった場合のみファイル更新
            if replacements > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return replacements

        except Exception as e:
            error_msg = f"ファイル置換エラー {file_path}: {e}"
            self.stats["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return 0

    def rename_directory(self, dir_path: Path) -> bool:
        """ディレクトリ名の置換"""
        try:
            dir_name = dir_path.name

            # トラッカーID含有チェック
            for old_id, new_id in self.replacement_map.items():
                if old_id in dir_name:
                    new_dir_name = dir_name.replace(old_id, new_id)
                    new_dir_path = dir_path.parent / new_dir_name

                    # ディレクトリリネーム
                    shutil.move(str(dir_path), str(new_dir_path))
                    print(f"📁 ディレクトリリネーム: {dir_name} → {new_dir_name}")
                    return True

            return False

        except Exception as e:
            error_msg = f"ディレクトリリネームエラー {dir_path}: {e}"
            self.stats["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return False

    def process_directory(self, base_dir: str):
        """ディレクトリ処理（再帰）"""
        print(f"\n🔍 処理中: {base_dir}")

        try:
            base_path = Path(base_dir)
            if not base_path.exists():
                print(f"⚠️  ディレクトリが存在しません: {base_dir}")
                return

            # ファイル処理
            for root, dirs, files in os.walk(base_path):
                root_path = Path(root)

                # ファイル内容の置換
                for file_name in files:
                    file_path = root_path / file_name

                    # 対象拡張子チェック
                    if file_path.suffix in self.target_extensions:
                        replacements = self.replace_in_file(file_path)
                        if replacements > 0:
                            self.stats["files_processed"] += 1
                            self.stats["total_replacements"] += replacements

                # ディレクトリ名の置換（後処理で実行）
                for dir_name in dirs[:]:  # コピーを作成して安全に操作
                    dir_path = root_path / dir_name
                    if self.rename_directory(dir_path):
                        self.stats["directories_renamed"] += 1
                        # リストからも削除（os.walkの整合性確保）
                        dirs.remove(dir_name)

        except Exception as e:
            error_msg = f"ディレクトリ処理エラー {base_dir}: {e}"
            self.stats["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    def run_bulk_replacement(self):
        """全件一括置換の実行"""
        print("🚀 トラッカーID全件一括置換開始")
        print("=" * 50)

        # マッピング読み込み
        if not self.load_mapping():
            return False

        # 各対象ディレクトリの処理
        for target_dir in self.target_dirs:
            self.process_directory(target_dir)

        # 結果レポート
        self.print_summary()
        return True

    def print_summary(self):
        """処理結果サマリー"""
        print("\n" + "=" * 50)
        print("📊 全件一括置換完了レポート")
        print("=" * 50)
        print(f"✅ 処理ファイル数: {self.stats['files_processed']}")
        print(f"📁 リネームディレクトリ数: {self.stats['directories_renamed']}")
        print(f"🔄 総置換回数: {self.stats['total_replacements']}")

        if self.stats["errors"]:
            print(f"⚠️  エラー数: {len(self.stats['errors'])}")
            for error in self.stats["errors"][:5]:  # 最初の5件表示
                print(f"   - {error}")
            if len(self.stats["errors"]) > 5:
                print(f"   ... 他{len(self.stats['errors'])-5}件")
        else:
            print("✅ エラー: なし")

    def dry_run(self):
        """ドライラン（実際の変更なしで確認）"""
        print("🔍 ドライラン実行（変更なし）")
        print("=" * 30)

        if not self.load_mapping():
            return False

        total_files = 0
        total_matches = 0

        for target_dir in self.target_dirs:
            base_path = Path(target_dir)
            if not base_path.exists():
                continue

            print(f"\n📁 {target_dir}")

            for root, dirs, files in os.walk(base_path):
                root_path = Path(root)

                for file_name in files:
                    file_path = root_path / file_name

                    if file_path.suffix in self.target_extensions:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            file_matches = 0
                            for old_id in self.replacement_map.keys():
                                pattern = r"\b" + re.escape(old_id) + r"\b"
                                matches = len(re.findall(pattern, content))
                                file_matches += matches

                            if file_matches > 0:
                                print(f"   📝 {file_path}: {file_matches}箇所")
                                total_matches += file_matches

                            total_files += 1

                        except Exception as e:
                            print(f"   ❌ {file_path}: 読み込みエラー")

        print(f"\n📊 ドライラン結果:")
        print(f"   対象ファイル: {total_files}")
        print(f"   予想置換数: {total_matches}")


if __name__ == "__main__":
    replacer = BulkTrackerReplacer()

    # 実行モード選択
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        replacer.dry_run()
    else:
        replacer.run_bulk_replacement()
