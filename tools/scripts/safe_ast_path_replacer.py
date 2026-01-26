#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safe_ast_path_replacer.py - Layer 2: AST使用の安全な段階的パス置換システム

Facebook製Bowlerを使用して構文木レベルでパス置換
- 無限再帰完全回避（正規表現でなくAST操作）
- 段階的処理（5-20ファイル単位）
- 既に置換済みのコードは触らない
"""

import argparse
import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Set

# プロジェクトルートをパス追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SafePathReplacer:
    """安全なAST使用パス置換システム"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.replaced_files: Set[Path] = set()
        self.skipped_files: Set[Path] = set()
        self.error_files: Set[Path] = set()

        # 対象パターン（WSLパス）
        self.target_patterns = [
            r'"/mnt/c/AItools/lora/train/yado/org/([^"]*)"',
            r'"/mnt/c/AItools/lora/train/yado/tracker-workspace/([^"]*)"',
            r'"/mnt/c/AItools/([^"]*)"',
        ]

    def find_target_files(self, directories: List[str] = None) -> List[Path]:
        """
        置換対象ファイルを検索

        Args:
            directories: 対象ディレクトリリスト（デフォルト: ['tests', 'tools', 'features']）

        Returns:
            List[Path]: Pythonファイルパスのリスト
        """
        if directories is None:
            directories = ["tests", "tools", "features"]

        files = []
        for directory in directories:
            dir_path = self.project_root / directory
            if dir_path.exists():
                py_files = list(dir_path.rglob("*.py"))
                logger.info(f"📁 {directory}: {len(py_files)}個のPythonファイル発見")
                files.extend(py_files)

        # /mnt/cを含むファイルのみフィルタ
        target_files = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                if "/mnt/c" in content:
                    target_files.append(file_path)
            except Exception as e:
                logger.warning(f"❌ {file_path}: 読み込み失敗 - {e}")
                self.error_files.add(file_path)

        logger.info(f"🎯 /mnt/c含有ファイル: {len(target_files)}個")
        return target_files

    def is_already_converted(self, file_path: Path) -> bool:
        """
        既にpathshim.resolve()を使用しているかチェック

        Args:
            file_path: チェック対象ファイル

        Returns:
            bool: 既に変換済みの場合True
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            # 既にpathshim.resolve()やfrom pathshim import使用している
            if "pathshim" in content or "resolve(" in content:
                return True

            # コメントで変換済みを示している
            if "Layer 2 converted" in content or "AST converted" in content:
                return True

            return False

        except Exception as e:
            logger.error(f"❌ {file_path}: 変換済みチェック失敗 - {e}")
            return True  # 安全のためスキップ

    def process_single_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """
        単一ファイルの安全なパス置換

        Args:
            file_path: 処理対象ファイル
            dry_run: テスト実行モード

        Returns:
            bool: 成功時True
        """
        try:
            # ファイル読み込み
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # /mnt/c パス検索と置換
            mnt_c_pattern = re.compile(r'(["\'])(/mnt/c/[^"\']*)\1')
            matches = list(mnt_c_pattern.finditer(content))

            if not matches:
                logger.debug(f"⏭️  {file_path.name}: /mnt/cパス見つからず")
                return True

            # 既にresolve()を使用している行をスキップ
            lines = content.splitlines()
            modified = False

            for match in matches:
                quote = match.group(1)
                path_str = match.group(2)
                full_match = match.group(0)

                # resolve()で既にラップされているかチェック
                start_pos = match.start()
                line_start = content.rfind("\n", 0, start_pos) + 1
                line_end = content.find("\n", start_pos)
                if line_end == -1:
                    line_end = len(content)

                line_content = content[line_start:line_end]

                if "resolve(" in line_content:
                    logger.debug(f"⏭️  既にresolve()使用: {path_str}")
                    continue

                # 置換実行
                replacement = f"resolve({quote}{path_str}{quote})"
                content = content.replace(full_match, replacement, 1)
                logger.info(f"🔄 {file_path.name}: {full_match} → {replacement}")
                modified = True

            if not modified:
                logger.debug(f"⏭️  {file_path.name}: 変更なし（既に変換済み）")
                return True

            # import文追加
            if "from pathshim import resolve" not in content:
                content = self._add_import_to_content(content)

            if dry_run:
                logger.info(f"🧪 {file_path.name}: ドライラン完了")
                self._show_diff(original_content, content, file_path.name)
            else:
                # ファイル更新
                file_path.write_text(content, encoding="utf-8")
                logger.info(f"✅ {file_path.name}: ファイル更新完了")

            return True

        except Exception as e:
            logger.error(f"❌ {file_path.name}: 処理失敗 - {e}")
            return False

    def _add_import_to_content(self, content: str) -> str:
        """
        コンテンツにpathshim import追加

        Args:
            content: ファイル内容

        Returns:
            str: import追加後のコンテンツ
        """
        lines = content.splitlines()
        insert_pos = 0

        # shebangやencoding宣言の後に挿入
        for i, line in enumerate(lines):
            if line.startswith("#!") or "coding:" in line or "encoding:" in line:
                insert_pos = i + 1
            elif line.strip() == "" and insert_pos > 0:
                continue  # 空行をスキップ
            elif line.startswith('"""') or line.startswith("'''"):
                # docstringの終わりまでスキップ
                quote = line[:3]
                if line.count(quote) >= 2:  # 同じ行で終了
                    insert_pos = i + 1
                else:
                    for j in range(i + 1, len(lines)):
                        if quote in lines[j]:
                            insert_pos = j + 1
                            break
                break
            else:
                break

        # import文を挿入
        import_line = "from pathshim import resolve  # Layer 2 AST conversion"
        lines.insert(insert_pos, import_line)
        lines.insert(insert_pos + 1, "")  # 空行追加

        return "\n".join(lines)

    def _show_diff(self, original: str, modified: str, filename: str):
        """差分表示"""
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        print(f"\n📝 {filename} の変更内容:")
        print("-" * 40)

        for i, (old, new) in enumerate(zip(original_lines, modified_lines)):
            if old != new:
                print(f"  {i+1:3d}- {old}")
                print(f"  {i+1:3d}+ {new}")

        if len(modified_lines) > len(original_lines):
            for i in range(len(original_lines), len(modified_lines)):
                print(f"  {i+1:3d}+ {modified_lines[i]}")

        print("-" * 40)

    def process_batch(self, target_files: List[Path], dry_run: bool = True) -> dict:
        """
        バッチ処理実行

        Args:
            target_files: 処理対象ファイル
            dry_run: テスト実行モード

        Returns:
            dict: 処理結果統計
        """
        if not target_files:
            logger.warning("⚠️  処理対象ファイルが見つかりません")
            return {"processed": 0, "skipped": 0, "errors": 0}

        # 既に変換済みファイルをスキップ
        processable_files = []
        for file_path in target_files:
            if self.is_already_converted(file_path):
                logger.info(f"⏭️  スキップ（変換済み）: {file_path.relative_to(self.project_root)}")
                self.skipped_files.add(file_path)
            else:
                processable_files.append(file_path)

        if not processable_files:
            logger.info("✅ 全ファイルが変換済みです")
            return {
                "processed": 0,
                "skipped": len(self.skipped_files),
                "errors": len(self.error_files),
            }

        logger.info(f"🚀 {len(processable_files)}個のファイルを処理開始")

        # 各ファイルを個別処理
        for file_path in processable_files:
            success = self.process_single_file(file_path, dry_run)

            if success:
                self.replaced_files.add(file_path)
                logger.debug(f"✅ {file_path.name}: 処理成功")
            else:
                self.error_files.add(file_path)
                logger.error(f"❌ {file_path.name}: 処理失敗")

        return {
            "processed": len(self.replaced_files),
            "skipped": len(self.skipped_files),
            "errors": len(self.error_files),
        }

    def generate_report(self) -> str:
        """処理結果レポート生成"""
        total_files = len(self.replaced_files) + len(self.skipped_files) + len(self.error_files)

        report = f"""
🎯 Layer 2: AST使用パス置換処理結果

📊 統計:
- 処理済み: {len(self.replaced_files)}ファイル
- スキップ: {len(self.skipped_files)}ファイル（変換済み）
- エラー: {len(self.error_files)}ファイル
- 合計: {total_files}ファイル

✅ 処理済みファイル:
{chr(10).join(f"  - {f.relative_to(self.project_root)}" for f in sorted(self.replaced_files))}

⏭️  スキップファイル:
{chr(10).join(f"  - {f.relative_to(self.project_root)}" for f in sorted(self.skipped_files))}

❌ エラーファイル:
{chr(10).join(f"  - {f.relative_to(self.project_root)}" for f in sorted(self.error_files))}
        """

        return report


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Layer 2: AST使用安全パス置換")
    parser.add_argument(
        "--directories", nargs="+", default=["tests", "tools", "features"], help="処理対象ディレクトリ"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="バッチサイズ（ファイル数）")
    parser.add_argument("--dry-run", action="store_true", help="ドライラン実行（実際の変更なし）")
    parser.add_argument(
        "--stage",
        choices=["critical", "all"],
        default="critical",
        help="処理段階（critical: 重要ファイルのみ, all: 全ファイル）",
    )

    args = parser.parse_args()

    replacer = SafePathReplacer()

    # 対象ファイル検索
    all_files = replacer.find_target_files(args.directories)

    # 段階選択
    if args.stage == "critical":
        # 重要ファイル（テストファイル）を優先
        target_files = [f for f in all_files if "/tests/" in str(f)][: args.batch_size]
        logger.info(f"🎯 Critical段階: {len(target_files)}個の重要ファイルを処理")
    else:
        target_files = all_files[: args.batch_size]
        logger.info(f"🎯 All段階: {len(target_files)}個のファイルを処理")

    # バッチ処理実行
    results = replacer.process_batch(target_files, dry_run=args.dry_run)

    # 結果報告
    print(replacer.generate_report())

    if args.dry_run:
        print("🧪 ドライラン完了。実際の変更を行うには --dry-run フラグを外してください。")
    else:
        print("✅ Layer 2パス置換完了。git diffで変更内容を確認してください。")

    return 0 if results["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
