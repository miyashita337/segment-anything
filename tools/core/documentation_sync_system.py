#!/usr/bin/env python3
"""
P1-A004: ドキュメント整備システム
実装と仕様の同期、ドキュメント一貫性確保システム

PROGRESS_TRACKER.md準拠のワークフロー実装:
- 実装修正 → 動作確認・ユニットTEST → 抽出パイプライン実行（バックグラウンド）
- 品質評価 → 統合実行スクリプト → ダッシュボード生成
"""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentationItem:
    """ドキュメント項目"""

    file_path: str
    section: str
    content: str
    last_modified: str
    doc_type: str  # README, API, GUIDE, CHANGELOG
    sync_status: str  # SYNCED, OUTDATED, MISSING
    implementation_refs: List[str]
    dependencies: List[str]


@dataclass
class ImplementationItem:
    """実装項目"""

    file_path: str
    function_name: str
    class_name: Optional[str]
    docstring: str
    last_modified: str
    impl_type: str  # CLASS, FUNCTION, MODULE
    test_coverage: bool
    doc_refs: List[str]


@dataclass
class SyncReport:
    """同期レポート"""

    report_id: str
    generated_at: str
    total_docs: int
    total_implementations: int
    synced_items: int
    outdated_items: int
    missing_docs: int
    missing_implementations: int
    sync_rate: float
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]


class DocumentationSyncSystem:
    """ドキュメント整備・同期システム"""

    def __init__(self):
        """初期化"""
        self.project_root = project_root

        # CI環境対応: CI環境では一時ディレクトリを使用
        import os

        if os.getenv("CI_ENVIRONMENT") == "true" or not os.path.exists("/mnt/c"):
            # CI環境では project_root 配下を使用
            self.workspace_root = project_root / "workspace"
        else:
            # PROGRESS_TRACKER.md準拠のワークスペース（ローカル環境）
            self.workspace_root = Path(
                "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace"
            )

        self.workspace_dir = self.workspace_root / "P1-A004"
        self.documentation_dir = self.workspace_dir / "documentation"

        # ディレクトリ作成
        for subdir in ["extraction", "quality", "dashboard", "tests", "documentation"]:
            (self.workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

        # ドキュメント管理パス
        self.sync_report_file = self.workspace_dir / "documentation" / "sync_report.json"
        self.missing_docs_file = self.workspace_dir / "documentation" / "missing_docs.json"
        self.outdated_docs_file = self.workspace_dir / "documentation" / "outdated_docs.json"

        # スキャン対象パス
        self.scan_paths = {
            "docs": self.project_root / "docs",
            "tools": self.project_root / "tools",
            "features": self.project_root / "features",
            "tests": self.project_root / "tests",
            "root_docs": self.project_root,
        }

        print(f"🎯 P1-A004: ドキュメント整備システム初期化完了")
        print(f"ワークスペース: {self.workspace_dir}")
        print(f"スキャン対象: {len(self.scan_paths)}パス")

    def scan_documentation_files(self) -> List[DocumentationItem]:
        """ドキュメントファイルスキャン"""
        logger.info("ドキュメントファイルスキャン開始")

        documentation_items = []

        # ドキュメントファイルパターン
        doc_patterns = {
            "README": r"README.*\.md$",
            "GUIDE": r".*[Gg]uide.*\.md$",
            "API": r".*[Aa]pi.*\.md$",
            "CHANGELOG": r"CHANGELOG.*\.md$",
            "DOC": r".*\.md$",
        }

        for scan_name, scan_path in self.scan_paths.items():
            if not scan_path.exists():
                logger.warning(f"スキャンパスが存在しません: {scan_path}")
                continue

            logger.info(f"スキャン中: {scan_name} ({scan_path})")

            for file_path in scan_path.rglob("*.md"):
                try:
                    # ファイル情報取得
                    stat = file_path.stat()
                    last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

                    # ドキュメントタイプ判定
                    doc_type = "DOC"
                    for dtype, pattern in doc_patterns.items():
                        if re.search(pattern, file_path.name, re.IGNORECASE):
                            doc_type = dtype
                            break

                    # 内容読み込み
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 実装参照解析
                    implementation_refs = self._extract_implementation_refs(content)

                    # 依存関係解析
                    dependencies = self._extract_dependencies(content)

                    doc_item = DocumentationItem(
                        file_path=str(file_path.relative_to(self.project_root)),
                        section=self._extract_main_section(content),
                        content=content[:500] + "..." if len(content) > 500 else content,
                        last_modified=last_modified,
                        doc_type=doc_type,
                        sync_status="UNKNOWN",  # 後で解析
                        implementation_refs=implementation_refs,
                        dependencies=dependencies,
                    )

                    documentation_items.append(doc_item)

                except Exception as e:
                    logger.warning(f"ドキュメント読み込みエラー {file_path}: {e}")

        logger.info(f"ドキュメントスキャン完了: {len(documentation_items)}件")
        return documentation_items

    def scan_implementation_files(self) -> List[ImplementationItem]:
        """実装ファイルスキャン"""
        logger.info("実装ファイルスキャン開始")

        implementation_items = []

        # 実装ファイルスキャン対象
        impl_paths = [
            self.project_root / "tools",
            self.project_root / "features",
            self.project_root / "tests",
        ]

        for impl_path in impl_paths:
            if not impl_path.exists():
                continue

            logger.info(f"実装スキャン: {impl_path}")

            for file_path in impl_path.rglob("*.py"):
                try:
                    # __pycache__等を除外
                    if "__pycache__" in str(file_path) or ".pyc" in str(file_path):
                        continue

                    # ファイル情報取得
                    stat = file_path.stat()
                    last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

                    # Python AST解析（簡易版）
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # クラス・関数抽出
                    classes = self._extract_classes(content)
                    functions = self._extract_functions(content)

                    # クラス項目追加
                    for class_name, docstring in classes:
                        impl_item = ImplementationItem(
                            file_path=str(file_path.relative_to(self.project_root)),
                            function_name="",
                            class_name=class_name,
                            docstring=docstring,
                            last_modified=last_modified,
                            impl_type="CLASS",
                            test_coverage=self._check_test_coverage(file_path, class_name),
                            doc_refs=self._extract_doc_refs(docstring),
                        )
                        implementation_items.append(impl_item)

                    # 関数項目追加
                    for func_name, docstring in functions:
                        impl_item = ImplementationItem(
                            file_path=str(file_path.relative_to(self.project_root)),
                            function_name=func_name,
                            class_name=None,
                            docstring=docstring,
                            last_modified=last_modified,
                            impl_type="FUNCTION",
                            test_coverage=self._check_test_coverage(file_path, func_name),
                            doc_refs=self._extract_doc_refs(docstring),
                        )
                        implementation_items.append(impl_item)

                except Exception as e:
                    logger.warning(f"実装ファイル解析エラー {file_path}: {e}")

        logger.info(f"実装スキャン完了: {len(implementation_items)}件")
        return implementation_items

    def analyze_sync_status(
        self, docs: List[DocumentationItem], impls: List[ImplementationItem]
    ) -> SyncReport:
        """同期状況解析"""
        logger.info("同期状況解析開始")

        # 実装参照のマッピング作成
        impl_map = {}
        for impl in impls:
            key = f"{impl.class_name or impl.function_name}"
            impl_map[key] = impl

        # ドキュメント同期状況判定
        synced_items = 0
        outdated_items = 0
        missing_docs = 0

        for doc in docs:
            has_impl_refs = bool(doc.implementation_refs)

            if has_impl_refs:
                # 実装参照がある場合、実装の存在確認
                refs_exist = any(ref in impl_map for ref in doc.implementation_refs)
                if refs_exist:
                    doc.sync_status = "SYNCED"
                    synced_items += 1
                else:
                    doc.sync_status = "OUTDATED"
                    outdated_items += 1
            else:
                # 実装参照がない場合
                doc.sync_status = "MISSING"
                missing_docs += 1

        # 実装に対応するドキュメントが不足している項目
        documented_impls = set()
        for doc in docs:
            documented_impls.update(doc.implementation_refs)

        missing_implementations = 0
        for impl in impls:
            impl_name = impl.class_name or impl.function_name
            if impl_name not in documented_impls and not impl.docstring:
                missing_implementations += 1

        # 同期率計算
        total_items = len(docs) + len(impls)
        sync_rate = synced_items / total_items if total_items > 0 else 0.0

        # 推奨事項生成
        recommendations = self._generate_sync_recommendations(
            docs, impls, outdated_items, missing_docs, missing_implementations
        )

        # 詳細分析
        detailed_analysis = {
            "doc_types_distribution": self._analyze_doc_types(docs),
            "impl_types_distribution": self._analyze_impl_types(impls),
            "test_coverage_rate": self._analyze_test_coverage(impls),
            "outdated_files": [doc.file_path for doc in docs if doc.sync_status == "OUTDATED"],
            "missing_doc_files": [doc.file_path for doc in docs if doc.sync_status == "MISSING"],
            "undocumented_implementations": [
                f"{impl.file_path}:{impl.class_name or impl.function_name}"
                for impl in impls
                if not impl.docstring and not impl.doc_refs
            ],
        }

        report = SyncReport(
            report_id=f"P1A004_sync_{datetime.now():%Y%m%d_%H%M%S}",
            generated_at=datetime.now().isoformat(),
            total_docs=len(docs),
            total_implementations=len(impls),
            synced_items=synced_items,
            outdated_items=outdated_items,
            missing_docs=missing_docs,
            missing_implementations=missing_implementations,
            sync_rate=sync_rate,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis,
        )

        logger.info(f"同期状況解析完了: 同期率 {sync_rate:.1%}")
        return report

    def generate_documentation_templates(self, missing_items: List[str]) -> List[Path]:
        """不足ドキュメントテンプレート生成"""
        logger.info("ドキュメントテンプレート生成開始")

        generated_files = []

        for item in missing_items[:5]:  # 上位5件のみ生成
            # テンプレート内容生成
            template_content = self._create_doc_template(item)

            # ファイル名生成
            safe_name = re.sub(r"[^\w\-_.]", "_", item)
            template_file = self.documentation_dir / f"{safe_name}_template.md"

            # テンプレート保存
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template_content)

            generated_files.append(template_file)
            logger.info(f"テンプレート生成: {template_file}")

        logger.info(f"ドキュメントテンプレート生成完了: {len(generated_files)}件")
        return generated_files

    def save_sync_report(self, report: SyncReport) -> Path:
        """同期レポート保存"""
        with open(self.sync_report_file, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info(f"同期レポート保存: {self.sync_report_file}")
        return self.sync_report_file

    def execute_full_documentation_sync(self) -> Dict[str, Any]:
        """フルドキュメント同期実行"""
        logger.info("🚀 P1-A004 フルドキュメント同期開始")
        start_time = datetime.now()

        try:
            # 1. ドキュメントスキャン
            docs = self.scan_documentation_files()

            # 2. 実装スキャン
            impls = self.scan_implementation_files()

            # 3. 同期状況解析
            sync_report = self.analyze_sync_status(docs, impls)

            # 4. 同期レポート保存
            report_file = self.save_sync_report(sync_report)

            # 5. 不足ドキュメントテンプレート生成
            missing_items = sync_report.detailed_analysis["undocumented_implementations"]
            template_files = self.generate_documentation_templates(missing_items)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            result = {
                "success": True,
                "processing_time": processing_time,
                "sync_rate": sync_report.sync_rate,
                "total_docs": sync_report.total_docs,
                "total_implementations": sync_report.total_implementations,
                "synced_items": sync_report.synced_items,
                "outdated_items": sync_report.outdated_items,
                "missing_docs": sync_report.missing_docs,
                "missing_implementations": sync_report.missing_implementations,
                "report_file": str(report_file),
                "template_files": [str(f) for f in template_files],
                "recommendations": sync_report.recommendations,
            }

            logger.info(f"✅ P1-A004 フルドキュメント同期完了 (処理時間: {processing_time:.2f}秒)")
            return result

        except Exception as e:
            logger.error(f"ドキュメント同期エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }

    # ヘルパーメソッド
    def _extract_implementation_refs(self, content: str) -> List[str]:
        """実装参照抽出"""
        refs = []
        # クラス名参照
        class_refs = re.findall(r"`([A-Z][a-zA-Z0-9_]*)`", content)
        refs.extend(class_refs)
        # 関数名参照
        func_refs = re.findall(r"`([a-z_][a-zA-Z0-9_]*\(\))`", content)
        refs.extend([ref.replace("()", "") for ref in func_refs])
        return list(set(refs))

    def _extract_dependencies(self, content: str) -> List[str]:
        """依存関係抽出"""
        deps = []
        # ファイルパス参照
        file_refs = re.findall(r"`([a-zA-Z0-9_/\.-]+\.py)`", content)
        deps.extend(file_refs)
        return list(set(deps))

    def _extract_main_section(self, content: str) -> str:
        """メインセクション抽出"""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return "Unknown Section"

    def _extract_classes(self, content: str) -> List[Tuple[str, str]]:
        """クラス抽出"""
        classes = []
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*\n\s*"""([^"]+)"""'
        matches = re.findall(class_pattern, content, re.MULTILINE | re.DOTALL)
        for class_name, docstring in matches:
            classes.append((class_name, docstring.strip()))
        return classes

    def _extract_functions(self, content: str) -> List[Tuple[str, str]]:
        """関数抽出"""
        functions = []
        func_pattern = (
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->[^:]+)?\s*:\s*\n\s*"""([^"]+)"""'
        )
        matches = re.findall(func_pattern, content, re.MULTILINE | re.DOTALL)
        for func_name, docstring in matches:
            functions.append((func_name, docstring.strip()))
        return functions

    def _check_test_coverage(self, file_path: Path, item_name: str) -> bool:
        """テストカバレッジ確認"""
        # 簡易実装: テストファイル存在確認
        test_patterns = [
            self.project_root / "tests" / f"test_{file_path.stem}.py",
            self.project_root / "tests" / "unit" / f"test_{file_path.stem}.py",
            self.project_root / "tests" / "integration" / f"test_{file_path.stem}.py",
        ]

        for test_file in test_patterns:
            if test_file.exists():
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        test_content = f.read()
                        if item_name in test_content:
                            return True
                except:
                    pass
        return False

    def _extract_doc_refs(self, docstring: str) -> List[str]:
        """ドキュメント参照抽出"""
        refs = []
        # ドキュメントファイル参照
        doc_refs = re.findall(r"([A-Z][A-Z_]*\.md)", docstring)
        refs.extend(doc_refs)
        return list(set(refs))

    def _generate_sync_recommendations(
        self, docs, impls, outdated, missing_docs, missing_impls
    ) -> List[str]:
        """同期推奨事項生成"""
        recommendations = []

        if outdated > 0:
            recommendations.append(f"期限切れドキュメント{outdated}件の更新が必要")

        if missing_docs > 0:
            recommendations.append(f"実装参照が不足しているドキュメント{missing_docs}件の修正が必要")

        if missing_impls > 0:
            recommendations.append(f"ドキュメントが不足している実装{missing_impls}件のドキュメント作成が必要")

        # テストカバレッジ低下警告
        untested_impls = len([impl for impl in impls if not impl.test_coverage])
        if untested_impls > len(impls) * 0.3:
            recommendations.append(f"テストカバレッジ改善: {untested_impls}件の実装にテストが不足")

        return recommendations

    def _analyze_doc_types(self, docs) -> Dict[str, int]:
        """ドキュメントタイプ分析"""
        types = {}
        for doc in docs:
            types[doc.doc_type] = types.get(doc.doc_type, 0) + 1
        return types

    def _analyze_impl_types(self, impls) -> Dict[str, int]:
        """実装タイプ分析"""
        types = {}
        for impl in impls:
            types[impl.impl_type] = types.get(impl.impl_type, 0) + 1
        return types

    def _analyze_test_coverage(self, impls) -> float:
        """テストカバレッジ分析"""
        if not impls:
            return 0.0
        tested = len([impl for impl in impls if impl.test_coverage])
        return tested / len(impls)

    def _create_doc_template(self, item: str) -> str:
        """ドキュメントテンプレート作成"""
        return f"""# {item} Documentation

## Overview

This document describes the implementation and usage of `{item}`.

## Description

[Provide detailed description of the component]

## Usage

```python
# Example usage code
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| param1    | str  | Description |

## Returns

| Type | Description |
|------|-------------|
| type | Description |

## Examples

### Basic Usage

```python
# Example code
```

### Advanced Usage

```python
# Advanced example code
```

## Notes

- Important note 1
- Important note 2

## See Also

- Related documentation
- Related implementations

---
*Generated by P1-A004 Documentation Sync System*
*Generated at: {datetime.now().isoformat()}*
"""


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-A004: ドキュメント整備システム")
    parser.add_argument("--full", action="store_true", help="フル同期実行")
    parser.add_argument("--scan-only", action="store_true", help="スキャンのみ実行")

    args = parser.parse_args()

    system = DocumentationSyncSystem()

    if args.full:
        # フル同期実行
        result = system.execute_full_documentation_sync()

        if result["success"]:
            print(f"🎯 P1-A004ドキュメント同期完了")
            print(f"   同期率: {result['sync_rate']:.1%}")
            print(f"   総ドキュメント: {result['total_docs']}件")
            print(f"   総実装: {result['total_implementations']}件")
            print(f"   同期済み: {result['synced_items']}件")
            print(f"   期限切れ: {result['outdated_items']}件")
            print(f"   不足: {result['missing_docs']}件")
            print(f"   処理時間: {result['processing_time']:.2f}秒")
            print(f"   レポート: {result['report_file']}")
            return 0
        else:
            print(f"❌ ドキュメント同期失敗: {result['error']}")
            return 1

    elif args.scan_only:
        # スキャンのみ実行
        docs = system.scan_documentation_files()
        impls = system.scan_implementation_files()

        print(f"✅ スキャン完了")
        print(f"   ドキュメント: {len(docs)}件")
        print(f"   実装: {len(impls)}件")

        return 0

    else:
        print("🎯 P1-A004: ドキュメント整備システム")
        print("使用例:")
        print("  python documentation_sync_system.py --full")
        print("  python documentation_sync_system.py --scan-only")
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
