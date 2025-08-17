#!/usr/bin/env python3
"""
OPT-029: ドキュメント整備システム ユニットテスト

ドキュメント同期システムの動作確認テスト実装
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from tools.core.documentation_sync_system import (
    DocumentationItem,
    ImplementationItem,
    SyncReport,
    DocumentationSyncSystem
)


@pytest.fixture
def temp_workspace():
    """テスト用一時ワークスペース"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documentation_item():
    """サンプルドキュメント項目"""
    return DocumentationItem(
        file_path="docs/sample.md",
        section="Sample Documentation",
        content="# Sample\nThis is sample documentation.",
        last_modified="2025-01-01T00:00:00",
        doc_type="DOC",
        sync_status="SYNCED",
        implementation_refs=["SampleClass", "sample_function"],
        dependencies=["sample.py"]
    )


@pytest.fixture
def sample_implementation_item():
    """サンプル実装項目"""
    return ImplementationItem(
        file_path="tools/sample.py",
        function_name="sample_function",
        class_name=None,
        docstring="Sample function docstring",
        last_modified="2025-01-01T00:00:00",
        impl_type="FUNCTION",
        test_coverage=True,
        doc_refs=["SAMPLE.md"]
    )


class TestDocumentationItem:
    """DocumentationItemクラステスト"""
    
    def test_documentation_item_creation(self, sample_documentation_item):
        """ドキュメント項目作成テスト"""
        doc = sample_documentation_item
        
        assert doc.file_path == "docs/sample.md"
        assert doc.section == "Sample Documentation"
        assert doc.doc_type == "DOC"
        assert doc.sync_status == "SYNCED"
        assert "SampleClass" in doc.implementation_refs
        assert "sample_function" in doc.implementation_refs
        assert "sample.py" in doc.dependencies


class TestImplementationItem:
    """ImplementationItemクラステスト"""
    
    def test_implementation_item_creation(self, sample_implementation_item):
        """実装項目作成テスト"""
        impl = sample_implementation_item
        
        assert impl.file_path == "tools/sample.py"
        assert impl.function_name == "sample_function"
        assert impl.class_name is None
        assert impl.impl_type == "FUNCTION"
        assert impl.test_coverage is True
        assert "SAMPLE.md" in impl.doc_refs


class TestDocumentationSyncSystem:
    """DocumentationSyncSystemクラステスト"""
    
    def test_system_initialization(self):
        """システム初期化テスト"""
        system = DocumentationSyncSystem()
        
        assert system.project_root is not None
        assert system.workspace_dir.exists()
        assert system.documentation_dir.exists()
        assert len(system.scan_paths) == 5
        assert "docs" in system.scan_paths
        assert "tools" in system.scan_paths
    
    def test_extract_implementation_refs(self):
        """実装参照抽出テスト"""
        system = DocumentationSyncSystem()
        
        content = """
        # Documentation
        
        This uses `SampleClass` and `sample_function()` for processing.
        Also see `AnotherClass` implementation.
        """
        
        refs = system._extract_implementation_refs(content)
        
        assert "SampleClass" in refs
        assert "sample_function" in refs
        assert "AnotherClass" in refs
    
    def test_extract_dependencies(self):
        """依存関係抽出テスト"""
        system = DocumentationSyncSystem()
        
        content = """
        # Documentation
        
        See implementation in `tools/sample.py` and `features/test.py`.
        """
        
        deps = system._extract_dependencies(content)
        
        assert "tools/sample.py" in deps
        assert "features/test.py" in deps
    
    def test_extract_main_section(self):
        """メインセクション抽出テスト"""
        system = DocumentationSyncSystem()
        
        # 正しい形式（行頭から開始）
        content = """# Main Title

Some content here.

## Subsection
"""
        
        section = system._extract_main_section(content)
        assert section == "Main Title"
    
    def test_extract_classes(self):
        """クラス抽出テスト"""
        system = DocumentationSyncSystem()
        
        content = '''
class SampleClass:
    """Sample class docstring"""
    
    def method(self):
        pass

class AnotherClass(BaseClass):
    """Another class docstring"""
    pass
'''
        
        classes = system._extract_classes(content)
        
        assert len(classes) == 2
        assert ("SampleClass", "Sample class docstring") in classes
        assert ("AnotherClass", "Another class docstring") in classes
    
    def test_extract_functions(self):
        """関数抽出テスト"""
        system = DocumentationSyncSystem()
        
        content = '''
def sample_function():
    """Sample function docstring"""
    pass

def another_function(param: str) -> int:
    """Another function docstring"""
    return 0
'''
        
        functions = system._extract_functions(content)
        
        assert len(functions) == 2
        assert ("sample_function", "Sample function docstring") in functions
        assert ("another_function", "Another function docstring") in functions
    
    def test_check_test_coverage_existing(self):
        """テストカバレッジ確認（存在）テスト"""
        system = DocumentationSyncSystem()
        
        # モックファイルパスを実際のプロジェクト内に設定
        test_file_path = system.project_root / "tools" / "core" / "documentation_sync_system.py"
        
        # テストファイルが存在する場合をモック
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data="def test_documentation_sync_system():\n    pass")) as mock_file:
            
            mock_exists.return_value = True
            
            coverage = system._check_test_coverage(test_file_path, "documentation_sync_system")
            assert coverage is True
    
    def test_check_test_coverage_missing(self):
        """テストカバレッジ確認（不存在）テスト"""
        system = DocumentationSyncSystem()
        
        test_file_path = Path("nonexistent/file.py")
        
        # テストファイルが存在しない場合
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            coverage = system._check_test_coverage(test_file_path, "nonexistent_function")
            assert coverage is False
    
    def test_extract_doc_refs(self):
        """ドキュメント参照抽出テスト"""
        system = DocumentationSyncSystem()
        
        docstring = """
        Function docstring with references to README.md and API_GUIDE.md.
        See also CHANGELOG.md for version history.
        """
        
        refs = system._extract_doc_refs(docstring)
        
        assert "README.md" in refs
        assert "API_GUIDE.md" in refs
        assert "CHANGELOG.md" in refs
    
    def test_analyze_sync_status(self, sample_documentation_item, sample_implementation_item):
        """同期状況解析テスト"""
        system = DocumentationSyncSystem()
        
        # サンプルデータ作成
        docs = [sample_documentation_item]
        impls = [sample_implementation_item]
        
        # 実装参照を一致させる
        docs[0].implementation_refs = ["sample_function"]
        
        report = system.analyze_sync_status(docs, impls)
        
        assert isinstance(report, SyncReport)
        assert report.total_docs == 1
        assert report.total_implementations == 1
        assert report.synced_items >= 0
        assert 0.0 <= report.sync_rate <= 1.0
        assert len(report.recommendations) >= 0
    
    def test_generate_sync_recommendations(self):
        """同期推奨事項生成テスト"""
        system = DocumentationSyncSystem()
        
        # サンプルデータ
        docs = []
        impls = []
        
        recommendations = system._generate_sync_recommendations(docs, impls, 5, 3, 2)
        
        assert len(recommendations) >= 3
        assert any("期限切れ" in rec for rec in recommendations)
        assert any("実装参照が不足" in rec for rec in recommendations)
        assert any("ドキュメントが不足" in rec for rec in recommendations)
    
    def test_analyze_doc_types(self, sample_documentation_item):
        """ドキュメントタイプ分析テスト"""
        system = DocumentationSyncSystem()
        
        docs = [sample_documentation_item]
        docs[0].doc_type = "README"
        
        doc_types = system._analyze_doc_types(docs)
        
        assert "README" in doc_types
        assert doc_types["README"] == 1
    
    def test_analyze_impl_types(self, sample_implementation_item):
        """実装タイプ分析テスト"""
        system = DocumentationSyncSystem()
        
        impls = [sample_implementation_item]
        
        impl_types = system._analyze_impl_types(impls)
        
        assert "FUNCTION" in impl_types
        assert impl_types["FUNCTION"] == 1
    
    def test_analyze_test_coverage(self, sample_implementation_item):
        """テストカバレッジ分析テスト"""
        system = DocumentationSyncSystem()
        
        # テストカバレッジありの実装
        impl1 = sample_implementation_item
        impl1.test_coverage = True
        
        # テストカバレッジなしの実装
        impl2 = ImplementationItem(
            file_path="tools/another.py",
            function_name="another_function",
            class_name=None,
            docstring="Another function",
            last_modified="2025-01-01T00:00:00",
            impl_type="FUNCTION",
            test_coverage=False,
            doc_refs=[]
        )
        
        impls = [impl1, impl2]
        
        coverage_rate = system._analyze_test_coverage(impls)
        
        assert coverage_rate == 0.5  # 50%
    
    def test_create_doc_template(self):
        """ドキュメントテンプレート作成テスト"""
        system = DocumentationSyncSystem()
        
        template = system._create_doc_template("SampleClass")
        
        assert "# SampleClass Documentation" in template
        assert "## Overview" in template
        assert "## Usage" in template
        assert "## Parameters" in template
        assert "## Returns" in template
        assert "## Examples" in template
        assert "OPT-029 Documentation Sync System" in template
    
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    def test_scan_documentation_files_mock(self, mock_file, mock_glob):
        """ドキュメントファイルスキャン（モック）テスト"""
        system = DocumentationSyncSystem()
        
        # モックファイルパス
        mock_file_path = Mock()
        mock_file_path.name = "README.md"
        mock_file_path.relative_to.return_value = Path("docs/README.md")
        mock_file_path.stat.return_value.st_mtime = datetime.now().timestamp()
        
        mock_glob.return_value = [mock_file_path]
        mock_file.return_value.read.return_value = "# Sample README\n\nThis is a sample."
        
        # パスの存在確認をモック
        with patch('pathlib.Path.exists', return_value=True):
            docs = system.scan_documentation_files()
        
        assert len(docs) >= 0  # スキャン実行を確認
    
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    def test_scan_implementation_files_mock(self, mock_file, mock_glob):
        """実装ファイルスキャン（モック）テスト"""
        system = DocumentationSyncSystem()
        
        # モックファイルパス
        mock_file_path = Mock()
        mock_file_path.name = "sample.py"
        mock_file_path.stem = "sample"
        mock_file_path.relative_to.return_value = Path("tools/sample.py")
        mock_file_path.stat.return_value.st_mtime = datetime.now().timestamp()
        
        mock_glob.return_value = [mock_file_path]
        mock_file.return_value.read.return_value = '''
class SampleClass:
    """Sample class docstring"""
    pass

def sample_function():
    """Sample function docstring"""
    pass
'''
        
        # パスの存在確認をモック
        with patch('pathlib.Path.exists', return_value=True):
            impls = system.scan_implementation_files()
        
        assert len(impls) >= 0  # スキャン実行を確認


class TestSyncReport:
    """SyncReportクラステスト"""
    
    def test_sync_report_creation(self):
        """同期レポート作成テスト"""
        report = SyncReport(
            report_id="test_report_001",
            generated_at="2025-01-01T00:00:00",
            total_docs=100,
            total_implementations=200,
            synced_items=50,
            outdated_items=30,
            missing_docs=20,
            missing_implementations=40,
            sync_rate=0.25,
            recommendations=["推奨事項1", "推奨事項2"],
            detailed_analysis={"test": "data"}
        )
        
        assert report.report_id == "test_report_001"
        assert report.total_docs == 100
        assert report.total_implementations == 200
        assert report.sync_rate == 0.25
        assert len(report.recommendations) == 2
        assert "test" in report.detailed_analysis


class TestIntegration:
    """統合テスト"""
    
    def test_full_documentation_sync_mock(self):
        """フルドキュメント同期（モック）統合テスト"""
        system = DocumentationSyncSystem()
        
        # メソッドをモック
        with patch.object(system, 'scan_documentation_files') as mock_scan_docs, \
             patch.object(system, 'scan_implementation_files') as mock_scan_impls, \
             patch.object(system, 'analyze_sync_status') as mock_analyze, \
             patch.object(system, 'save_sync_report') as mock_save, \
             patch.object(system, 'generate_documentation_templates') as mock_templates:
            
            # モック戻り値設定
            mock_scan_docs.return_value = []
            mock_scan_impls.return_value = []
            mock_analyze.return_value = SyncReport(
                report_id="test_001",
                generated_at="2025-01-01T00:00:00",
                total_docs=0,
                total_implementations=0,
                synced_items=0,
                outdated_items=0,
                missing_docs=0,
                missing_implementations=0,
                sync_rate=0.0,
                recommendations=[],
                detailed_analysis={"undocumented_implementations": []}
            )
            mock_save.return_value = Path("/test/path")
            mock_templates.return_value = []
            
            # フル同期実行
            result = system.execute_full_documentation_sync()
            
            assert result["success"] is True
            assert "processing_time" in result
            assert "sync_rate" in result
            assert "report_file" in result
            
            # メソッド呼び出し確認
            mock_scan_docs.assert_called_once()
            mock_scan_impls.assert_called_once()
            mock_analyze.assert_called_once()
            mock_save.assert_called_once()
            mock_templates.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])