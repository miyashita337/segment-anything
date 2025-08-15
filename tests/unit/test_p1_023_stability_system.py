"""
P1-023: VSCode安定性向上システム - ユニットテスト

メモリ監視・I/O最適化・遅延インポート・統合管理の
包括的テストスイート
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# テスト対象モジュール
from features.common.memory_monitor import MemoryMonitor, get_memory_monitor, monitor_memory
from features.common.io_optimizer import IOOptimizer, get_io_optimizer, optimize_io
from features.common.lazy_import_manager import LazyImportManager, get_lazy_import_manager, with_lazy_imports
from features.common.stability_manager import StabilityManager, get_stability_manager, with_stability_management


class TestMemoryMonitor:
    """メモリ監視システムのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.monitor = MemoryMonitor(
            memory_threshold=80.0,
            vram_threshold=85.0,
            monitor_interval=1.0
        )
    
    def test_memory_monitor_initialization(self):
        """メモリ監視初期化テスト"""
        assert self.monitor.memory_threshold == 80.0
        assert self.monitor.vram_threshold == 85.0
        assert self.monitor.monitor_interval == 1.0
        assert self.monitor.enable_auto_cleanup is True
        assert len(self.monitor.cleanup_callbacks) == 0
    
    def test_get_memory_status(self):
        """メモリ状況取得テスト"""
        status = self.monitor.get_memory_status()
        
        # 必須キー確認
        required_keys = ['ram_used_gb', 'ram_total_gb', 'ram_percent', 'ram_available_gb']
        for key in required_keys:
            assert key in status
            assert isinstance(status[key], (int, float))
        
        # 妥当性確認
        assert 0 <= status['ram_percent'] <= 100
        assert status['ram_total_gb'] > 0
        assert status['ram_used_gb'] >= 0
    
    def test_check_memory_pressure(self):
        """メモリ圧迫チェックテスト"""
        pressure = self.monitor.check_memory_pressure()
        
        assert 'ram_pressure' in pressure
        assert 'vram_pressure' in pressure
        assert isinstance(pressure['ram_pressure'], bool)
        assert isinstance(pressure['vram_pressure'], bool)
    
    def test_register_cleanup_callback(self):
        """クリーンアップコールバック登録テスト"""
        callback_called = False
        
        def test_callback():
            nonlocal callback_called
            callback_called = True
        
        self.monitor.register_cleanup_callback(test_callback)
        assert len(self.monitor.cleanup_callbacks) == 1
        
        # コールバック実行テスト
        self.monitor.perform_memory_cleanup(force=True)
        assert callback_called
    
    def test_perform_memory_cleanup(self):
        """メモリクリーンアップテスト"""
        result = self.monitor.perform_memory_cleanup(force=True)
        
        # 結果形式確認
        expected_keys = ['ram_freed_gb', 'vram_freed_gb', 'cleanup_time_s', 'before_ram_percent', 'after_ram_percent']
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))
        
        # 処理時間は妥当な範囲
        assert 0 <= result['cleanup_time_s'] <= 10.0
        
        # 最終クリーンアップ時刻が更新される
        assert self.monitor.last_cleanup_time is not None
    
    def test_monitor_once(self):
        """一回監視テスト"""
        # 通常状況（圧迫なし）
        with patch.object(self.monitor, 'check_memory_pressure') as mock_pressure:
            mock_pressure.return_value = {'ram_pressure': False, 'vram_pressure': False}
            result = self.monitor.monitor_once()
            assert result is None  # 圧迫なしの場合
        
        # 圧迫状況
        with patch.object(self.monitor, 'check_memory_pressure') as mock_pressure:
            mock_pressure.return_value = {'ram_pressure': True, 'vram_pressure': False}
            result = self.monitor.monitor_once()
            assert result is not None  # クリーンアップ実行
            assert isinstance(result, dict)
    
    def test_get_monitoring_report(self):
        """監視レポート生成テスト"""
        report = self.monitor.get_monitoring_report()
        
        required_keys = ['timestamp', 'memory_status', 'pressure_status', 'monitoring_active', 'thresholds']
        for key in required_keys:
            assert key in report
        
        assert 'ram_threshold' in report['thresholds']
        assert 'vram_threshold' in report['thresholds']
        assert report['thresholds']['ram_threshold'] == 80.0
    
    def test_monitor_memory_decorator(self):
        """メモリ監視デコレータテスト"""
        @monitor_memory(cleanup_after=True)
        def test_function():
            return "テスト成功"
        
        result = test_function()
        assert result == "テスト成功"


class TestIOOptimizer:
    """I/O最適化システムのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.optimizer = IOOptimizer(
            max_concurrent_io=2,
            io_delay_ms=50,
            batch_size=3,
            enable_vscode_sync=True
        )
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """テストクリーンアップ"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_safe_file_write(self):
        """安全ファイル書き込みテスト"""
        test_file = self.test_dir / "test.txt"
        content = "テストコンテンツ"
        
        result = await self.optimizer.safe_file_write(test_file, content)
        assert result is True
        assert test_file.exists()
        assert test_file.read_text(encoding='utf-8') == content
    
    @pytest.mark.asyncio
    async def test_safe_file_read(self):
        """安全ファイル読み込みテスト"""
        test_file = self.test_dir / "read_test.txt"
        content = "読み込みテスト"
        test_file.write_text(content, encoding='utf-8')
        
        result = await self.optimizer.safe_file_read(test_file)
        assert result == content
    
    @pytest.mark.asyncio
    async def test_batch_file_operations(self):
        """バッチファイル操作テスト"""
        operations = []
        for i in range(5):
            operations.append({
                'type': 'write',
                'path': self.test_dir / f"batch_{i}.txt",
                'content': f"バッチ {i}"
            })
        
        progress_calls = []
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        results = await self.optimizer.batch_file_operations(operations, progress_callback)
        
        # 全て成功
        assert len(results) == 5
        assert all(results)
        
        # ファイル確認
        for i in range(5):
            test_file = self.test_dir / f"batch_{i}.txt"
            assert test_file.exists()
            assert test_file.read_text() == f"バッチ {i}"
        
        # 進捗コールバック確認
        assert len(progress_calls) > 0
        assert progress_calls[-1] == (5, 5)
    
    @pytest.mark.asyncio
    async def test_google_sheets_safe_update(self):
        """Google Sheets安全更新テスト"""
        mock_result = {"status": "success"}
        
        def mock_update_func():
            time.sleep(0.1)  # 模擬処理時間
            return mock_result
        
        result = await self.optimizer.google_sheets_safe_update(
            mock_update_func, post_update_delay=0.1
        )
        
        assert result == mock_result
    
    @pytest.mark.asyncio
    async def test_optimized_report_generation(self):
        """最適化レポート生成テスト"""
        reports = {
            'test_report': {
                'title': 'テストレポート',
                'data': {'value': 42},
                'markdown': '# テスト\n\n成功',
                'html': '<h1>テスト</h1><p>成功</p>'
            }
        }
        
        results = await self.optimizer.optimized_report_generation(
            reports, self.test_dir / 'reports'
        )
        
        # 生成結果確認
        assert 'test_report_json' in results
        assert 'test_report_md' in results
        assert 'test_report_html' in results
        assert all(results.values())
        
        # ファイル存在確認
        report_dir = self.test_dir / 'reports'
        assert (report_dir / 'test_report.json').exists()
        assert (report_dir / 'test_report.md').exists()
        assert (report_dir / 'test_report.html').exists()
    
    @pytest.mark.asyncio
    async def test_workspace_sync_workflow(self):
        """ワークスペース同期ワークフローテスト"""
        workspace_dir = self.test_dir / 'workspace'
        
        def mock_sheets_func():
            return "Google Sheets更新成功"
        
        reports_data = {
            'workflow_report': {
                'title': 'ワークフローテスト',
                'markdown': '# ワークフロー\n完了',
                'html': '<h1>ワークフロー</h1><p>完了</p>'
            }
        }
        
        result = await self.optimizer.workspace_sync_workflow(
            workspace_dir=workspace_dir,
            google_sheets_func=mock_sheets_func,
            reports_data=reports_data
        )
        
        # 結果確認
        assert result['workspace_created'] is True
        assert 'reports_generated' in result
        assert result['google_sheets_updated'] is True
        assert result['total_time'] > 0
        
        # ワークスペース構造確認
        assert workspace_dir.exists()
        for subdir in ['extraction', 'quality', 'dashboard', 'tests']:
            assert (workspace_dir / subdir).exists()
    
    def test_get_io_statistics(self):
        """I/O統計情報取得テスト"""
        stats = self.optimizer.get_io_statistics()
        
        required_keys = ['timestamp', 'active_operations', 'active_count', 'statistics', 'configuration']
        for key in required_keys:
            assert key in stats
        
        assert stats['configuration']['max_concurrent_io'] == 2
        assert stats['configuration']['batch_size'] == 3


class TestLazyImportManager:
    """遅延インポート管理システムのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.manager = LazyImportManager()
    
    def test_lazy_import_manager_initialization(self):
        """遅延インポート管理初期化テスト"""
        assert len(self.manager._import_cache) == 0
        assert self.manager._import_stats['total_imports'] == 0
        assert self.manager._import_stats['cache_hits'] == 0
        assert len(self.manager._import_stats['failed_imports']) == 0
    
    def test_register_lazy_import_success(self):
        """遅延インポート登録成功テスト"""
        # 標準ライブラリjsonを使用してテスト
        json_loader = self.manager.register_lazy_import('json')
        json_module = json_loader()
        
        assert json_module is not None
        assert hasattr(json_module, 'loads')
        assert hasattr(json_module, 'dumps')
        
        # キャッシュ確認
        assert 'json' in self.manager._import_cache
        assert self.manager._import_stats['total_imports'] == 1
    
    def test_register_lazy_import_cache_hit(self):
        """遅延インポートキャッシュヒットテスト"""
        json_loader = self.manager.register_lazy_import('json')
        
        # 初回インポート
        json_module1 = json_loader()
        initial_imports = self.manager._import_stats['total_imports']
        
        # 2回目インポート（キャッシュヒット）
        json_module2 = json_loader()
        
        assert json_module1 is json_module2  # 同一インスタンス
        assert self.manager._import_stats['total_imports'] == initial_imports  # インポート数変化なし
        assert self.manager._import_stats['cache_hits'] == 1
    
    def test_register_lazy_import_fallback(self):
        """遅延インポートフォールバックテスト"""
        def fallback_func():
            return "フォールバック結果"
        
        loader = self.manager.register_lazy_import(
            'nonexistent_module',
            fallback_func=fallback_func
        )
        
        result = loader()
        assert result == "フォールバック結果"
        assert 'nonexistent_module' in self.manager._import_stats['failed_imports']
    
    def test_get_torch(self):
        """PyTorch遅延インポートテスト"""
        torch_module = self.manager.get_torch()
        
        # torch利用可能時は実際のモジュール、利用不可時はモック
        assert torch_module is not None
        assert hasattr(torch_module, 'cuda')
        assert hasattr(torch_module.cuda, 'is_available')
    
    def test_preload_essential_modules(self):
        """重要モジュール事前ロードテスト"""
        self.manager.preload_essential_modules(['json', 'os'])
        
        # キャッシュに追加されている
        assert len(self.manager._import_cache) >= 2
    
    def test_clear_cache(self):
        """キャッシュクリアテスト"""
        # キャッシュに何か追加
        json_loader = self.manager.register_lazy_import('json')
        json_loader()
        
        assert len(self.manager._import_cache) > 0
        
        # 特定モジュールクリア
        self.manager.clear_cache('json')
        assert 'json' not in self.manager._import_cache
        
        # 全クリア
        self.manager.preload_essential_modules(['os'])
        self.manager.clear_cache()
        assert len(self.manager._import_cache) == 0
    
    def test_get_import_statistics(self):
        """インポート統計取得テスト"""
        json_loader = self.manager.register_lazy_import('json')
        json_loader()
        
        stats = self.manager.get_import_statistics()
        
        required_keys = ['cached_modules', 'cache_size', 'statistics', 'memory_usage_estimate']
        for key in required_keys:
            assert key in stats
        
        assert 'json' in stats['cached_modules']
        assert stats['cache_size'] >= 1
    
    def test_with_lazy_imports_decorator(self):
        """遅延インポートデコレータテスト"""
        @with_lazy_imports('json', 'os')
        def test_function():
            return "デコレータテスト成功"
        
        result = test_function()
        assert result == "デコレータテスト成功"


class TestStabilityManager:
    """統合安定性管理システムのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.manager = StabilityManager(
            enable_memory_monitoring=True,
            enable_io_optimization=True,
            enable_lazy_imports=True
        )
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """テストクリーンアップ"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_stability_manager_initialization(self):
        """統合安定性管理初期化テスト"""
        assert self.manager.memory_enabled is True
        assert self.manager.io_enabled is True
        assert self.manager.lazy_enabled is True
        assert self.manager.memory_monitor is not None
        assert self.manager.io_optimizer is not None
        assert self.manager.lazy_manager is not None
    
    def test_get_system_status(self):
        """システム状況取得テスト"""
        status = self.manager.get_system_status()
        
        required_keys = ['timestamp', 'components']
        for key in required_keys:
            assert key in status
        
        assert 'memory_monitoring' in status['components']
        assert 'io_optimization' in status['components']
        assert 'lazy_imports' in status['components']
        
        # 有効化されたコンポーネントの情報が含まれる
        if self.manager.memory_enabled:
            assert 'memory' in status
            assert 'memory_pressure' in status
        
        if self.manager.io_enabled:
            assert 'io_stats' in status
        
        if self.manager.lazy_enabled:
            assert 'import_stats' in status
    
    @pytest.mark.asyncio
    async def test_safe_character_extraction(self):
        """安全文字抽出テスト"""
        def mock_extraction_func(input_path, output_path, **kwargs):
            # 模擬抽出処理
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'result.txt').write_text('抽出成功')
            return {'processed_files': 1, 'success': True}
        
        result = await self.manager.safe_character_extraction(
            extraction_func=mock_extraction_func,
            input_path=str(self.test_dir / 'input'),
            output_path=str(self.test_dir / 'output'),
            test_param='テストパラメータ'
        )
        
        assert result['success'] is True
        assert result['extraction_result'] is not None
        assert 'stability_stats' in result
        
        # 出力確認
        output_file = self.test_dir / 'output' / 'result.txt'
        assert output_file.exists()
        assert output_file.read_text() == '抽出成功'
    
    @pytest.mark.asyncio
    async def test_safe_batch_processing(self):
        """安全バッチ処理テスト"""
        def mock_batch_func(items, output_dir, **kwargs):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 各アイテムに対して処理
            for item in items:
                (output_path / f'batch_{item}.txt').write_text(f'処理済み: {item}')
            
            return {'processed': len(items)}
        
        input_items = ['item1', 'item2', 'item3', 'item4', 'item5']
        
        result = await self.manager.safe_batch_processing(
            batch_func=mock_batch_func,
            input_items=input_items,
            output_dir=str(self.test_dir / 'batch_output'),
            batch_size=2
        )
        
        assert result['success'] is True
        assert result['processed_count'] == 5
        assert result['failed_count'] == 0
        assert len(result['batch_results']) == 3  # 5アイテム ÷ 2バッチサイズ = 3バッチ
        
        # 出力ファイル確認
        for item in input_items:
            output_file = self.test_dir / 'batch_output' / f'batch_{item}.txt'
            assert output_file.exists()
    
    @pytest.mark.asyncio
    async def test_integrated_tracker_workflow(self):
        """統合トラッカーワークフローテスト"""
        def mock_extraction_func(input_path, output_path, **kwargs):
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'extraction_result.json').write_text('{"status": "success"}')
            return {'success': True}
        
        def mock_google_sheets_func():
            return {'status': 'updated'}
        
        result = await self.manager.integrated_tracker_workflow(
            tracker_id='TEST-001',
            workspace_base=str(self.test_dir),
            extraction_func=mock_extraction_func,
            extraction_args={
                'input_path': str(self.test_dir / 'input'),
                'param1': 'value1'
            },
            google_sheets_func=mock_google_sheets_func
        )
        
        assert result['success'] is True
        assert result['tracker_id'] == 'TEST-001'
        assert 'phases' in result
        assert 'workspace_setup' in result['phases']
        assert 'extraction' in result['phases']
        assert 'stability_stats' in result
        
        # ワークスペース構造確認
        workspace_dir = self.test_dir / 'TEST-001'
        assert workspace_dir.exists()
        for subdir in ['extraction', 'quality', 'dashboard', 'tests']:
            assert (workspace_dir / subdir).exists()
    
    @pytest.mark.asyncio
    async def test_with_stability_management_decorator(self):
        """統合安定性管理デコレータテスト"""
        @with_stability_management(enable_memory=True, enable_io=True, enable_lazy=True)
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "非同期デコレータテスト成功"
        
        result = await test_async_function()
        assert result == "非同期デコレータテスト成功"
        
        @with_stability_management(enable_memory=True, enable_io=False, enable_lazy=True)
        def test_sync_function():
            return "同期デコレータテスト成功"
        
        result = test_sync_function()
        assert result == "同期デコレータテスト成功"


class TestIntegration:
    """統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """テストクリーンアップ"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """システム全体統合テスト"""
        # P1-023の3つのコンポーネントが正常に連携することを確認
        
        # 1. グローバルインスタンス取得
        memory_monitor = get_memory_monitor()
        io_optimizer = get_io_optimizer()
        lazy_manager = get_lazy_import_manager()
        stability_manager = get_stability_manager()
        
        assert memory_monitor is not None
        assert io_optimizer is not None
        assert lazy_manager is not None
        assert stability_manager is not None
        
        # 2. 遅延インポートでモジュール取得
        json_module = lazy_manager.get_torch()  # PyTorchまたはフォールバック
        assert json_module is not None
        
        # 3. メモリ監視実行
        memory_status = memory_monitor.get_memory_status()
        assert 'ram_percent' in memory_status
        
        # 4. I/O最適化ファイル操作
        test_file = self.test_dir / 'integration_test.txt'
        write_success = await io_optimizer.safe_file_write(test_file, '統合テスト')
        assert write_success is True
        
        read_content = await io_optimizer.safe_file_read(test_file)
        assert read_content == '統合テスト'
        
        # 5. 統合ワークフロー実行
        def mock_extraction(input_path, output_path, **kwargs):
            return {'integration': 'success'}
        
        workflow_result = await stability_manager.integrated_tracker_workflow(
            tracker_id='INTEGRATION-TEST',
            workspace_base=str(self.test_dir),
            extraction_func=mock_extraction,
            extraction_args={'input_path': str(self.test_dir / 'input')}
        )
        
        assert workflow_result['success'] is True
        assert workflow_result['tracker_id'] == 'INTEGRATION-TEST'
        
        # 6. システム状況確認
        system_status = stability_manager.get_system_status()
        assert 'components' in system_status
        assert 'memory' in system_status
        assert 'io_stats' in system_status
        assert 'import_stats' in system_status
        
        print("✅ P1-023統合テスト完全成功")


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "--tb=short"])