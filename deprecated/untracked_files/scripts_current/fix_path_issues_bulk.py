#!/usr/bin/env python3
"""
一括パス問題修正スクリプト

205個のファイルの/mnt/c問題を environment_manager 使用で一括解決
個別修正不要の自動化ツール
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class BulkPathFixer:
    """一括パス修正システム"""

    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.errors = []

        # 修正パターン定義
        self.fix_patterns = [
            # パターン1: 直接的な/mnt/cパス
            {
                "pattern": r'["\']\/mnt\/c\/AItools\/[^"\']*["\']',
                "replacement": self._replace_direct_path,
                "description": "直接的な/mnt/cパスを環境管理関数に置換",
            },
            # パターン2: ハードコードされたテストパス
            {
                "pattern": r'(test_image_path|input_dir|output_dir)\s*=\s*["\']\/mnt\/c[^"\']*["\']',
                "replacement": self._replace_test_path,
                "description": "テスト用パス変数を環境管理関数に置換",
            },
            # パターン3: pathlib.Pathでの/mnt/c使用
            {
                "pattern": r'Path\(["\']\/mnt\/c\/[^"\']*["\']\)',
                "replacement": self._replace_pathlib_usage,
                "description": "PathLib使用での/mnt/cを環境管理に置換",
            },
        ]

    def _replace_direct_path(self, match) -> str:
        """直接パス置換"""
        original_path = match.group(0).strip("\"'")

        # パスを解析してpath_typeとsub_pathsを決定
        if "/lora/train/yado/org/" in original_path:
            # データパス
            sub_path = original_path.split("/lora/train/yado/org/")[1]
            return f'get_path("data", "org", "{sub_path}")'
        elif "/tracker-workspace/" in original_path:
            # 出力パス
            sub_path = original_path.split("/tracker-workspace/")[1]
            return f'get_path("output", "{sub_path}")'
        elif "sam_vit_" in original_path or "yolo" in original_path:
            # モデルパス
            filename = Path(original_path).name
            return f'get_path("models", "{filename}")'
        else:
            # その他は汎用置換
            return f'get_path("data", Path("{original_path}").relative_to(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path("/mnt/c/AItools/").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path("/mnt/c/AItools/").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))))))'

    def _replace_test_path(self, match) -> str:
        """テストパス置換"""
        var_name = match.group(1)
        original_path = match.group(0)

        if "kana05" in original_path:
            return f'{var_name} = get_test_image_path("kana05", "kana05_0001.jpg")'
        elif "kana08" in original_path:
            return f'{var_name} = get_test_image_path("kana08", "kana08_0001.jpg")'
        elif "kaname07" in original_path:
            return f'{var_name} = get_test_image_path("kaname07", "kaname07_0001.jpg")'
        else:
            return f'{var_name} = setup_test_env("{var_name}")["input_dir"]'

    def _replace_pathlib_usage(self, match) -> str:
        """PathLib使用置換"""
        original = match.group(0)
        path_content = original.split("Path(")[1].rstrip(")").strip("\"'")

        if "/lora/train/yado/" in path_content:
            sub_path = path_content.split("/lora/train/yado/")[1]
            return f'Path(get_path("data", "{sub_path}"))'
        else:
            return f'Path(get_path("data", Path("{path_content}").relative_to(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path("/mnt/c/AItools/").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path(get_path("data", Path("/mnt/c/AItools/").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/")))))))'

    def add_import_if_needed(self, content: str) -> str:
        """必要に応じて環境管理インポートを追加"""
        # 既にインポートがあるかチェック
        if "from features.common.environment_manager import" in content:
            return content
        if "import environment_manager" in content:
            return content

        # get_path等の関数が使われているかチェック
        if any(
            func in content for func in ["get_path(", "get_test_image_path(", "setup_test_env("]
        ):
            # インポート追加位置を特定
            lines = content.split("\n")
            import_insert_pos = 0

            # 既存のインポートの最後を探す
            for i, line in enumerate(lines):
                if line.strip().startswith(("import ", "from ")) and not line.strip().startswith(
                    "#"
                ):
                    import_insert_pos = i + 1
                elif line.strip() == "":
                    continue
                elif line.strip().startswith("#"):
                    continue
                else:
                    break

            # インポート文挿入
            import_statement = "from features.common.environment_manager import get_path, get_test_image_path, setup_test_env, is_ci_environment"
            lines.insert(import_insert_pos, import_statement)
            return "\n".join(lines)

        return content

    def process_file(self, file_path: Path) -> bool:
        """単一ファイル処理"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            file_fixes = 0

            # 各パターンで修正適用
            for pattern_info in self.fix_patterns:
                pattern = pattern_info["pattern"]
                replacement_func = pattern_info["replacement"]

                matches = list(re.finditer(pattern, content))
                for match in matches:
                    old_text = match.group(0)
                    new_text = replacement_func(match)
                    content = content.replace(old_text, new_text)
                    file_fixes += 1

            # 必要なインポート追加
            if file_fixes > 0:
                content = self.add_import_if_needed(content)

            # 変更がある場合のみファイル更新
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"✅ 修正完了: {file_path} ({file_fixes}箇所)")
                self.fixes_applied += file_fixes
                return True
            else:
                return False

        except Exception as e:
            error_msg = f"❌ エラー {file_path}: {e}"
            self.errors.append(error_msg)
            print(error_msg)
            return False

    def find_target_files(self) -> List[Path]:
        """修正対象ファイル検索"""
        target_files = []

        # /mnt/cを含むPythonファイルを検索
        for root, dirs, files in os.walk(project_root):
            # 除外ディレクトリ
            dirs[:] = [
                d for d in dirs if d not in [".git", "__pycache__", ".pytest_cache", "sam-env"]
            ]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            if "/mnt/c" in f.read():
                                target_files.append(file_path)
                    except:
                        continue  # 読み取りエラーは無視

        return target_files

    def process_all_files(self, dry_run: bool = False) -> Dict[str, int]:
        """全ファイル一括処理"""
        target_files = self.find_target_files()

        print(f"🔍 /mnt/c問題対象ファイル: {len(target_files)}個")

        if dry_run:
            print("🔄 ドライラン: 実際の修正は行いません")
            for file_path in target_files:
                print(f"  - {file_path}")
            return {"found": len(target_files), "processed": 0, "fixed": 0}

        processed = 0
        fixed = 0

        for file_path in target_files:
            self.files_processed += 1
            processed += 1

            if self.process_file(file_path):
                fixed += 1

        return {
            "found": len(target_files),
            "processed": processed,
            "fixed": fixed,
            "total_fixes": self.fixes_applied,
            "errors": len(self.errors),
        }

    def print_summary(self, results: Dict[str, int]):
        """処理結果サマリー表示"""
        print("\n" + "=" * 60)
        print("📊 一括パス修正結果サマリー")
        print("=" * 60)
        print(f"対象ファイル発見: {results['found']}個")
        print(f"処理完了ファイル: {results['fixed']}個")
        print(f"総修正箇所数: {results.get('total_fixes', 0)}箇所")

        if results.get("errors", 0) > 0:
            print(f"エラー発生: {results['errors']}件")
            print("\n❌ エラー詳細:")
            for error in self.errors:
                print(f"  {error}")

        if results["fixed"] > 0:
            print("\n✅ 修正完了！environment_manager導入で/mnt/c問題根本解決")
            print("📝 次の手順:")
            print("  1. git add .")
            print("  2. git commit -m 'fix: 一括パス修正 - environment_manager導入で205ファイルの/mnt/c問題根本解決'")
            print("  3. git push")
            print("  4. CI実行確認")


def main():
    parser = argparse.ArgumentParser(description="一括パス問題修正ツール")
    parser.add_argument("--dry-run", action="store_true", help="実際の修正を行わず、対象ファイルのみ表示")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")

    args = parser.parse_args()

    print("🚀 一括パス修正スクリプト開始")
    print("📁 プロジェクトルート:", project_root)

    fixer = BulkPathFixer()
    results = fixer.process_all_files(dry_run=args.dry_run)
    fixer.print_summary(results)


if __name__ == "__main__":
    main()
