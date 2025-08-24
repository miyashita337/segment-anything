"""
P1-023: VSCode安定性向上 - 統合安定性管理システム

メモリ監視・I/O最適化・遅延インポートを統合し、
既存のextract_character.pyシステムに組み込む統一インターフェース
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .io_optimizer import get_io_optimizer, optimize_io
from .lazy_import_manager import get_lazy_import_manager, with_lazy_imports
from .memory_monitor import get_memory_monitor, monitor_memory

logger = logging.getLogger(__name__)


class StabilityManager:
    """
    統合安定性管理クラス
    
    P1-023の3つのコンポーネント（メモリ監視・I/O最適化・遅延インポート）を
    統合して既存システムに組み込む統一管理インターフェース
    """
    
    def __init__(
        self,
        enable_memory_monitoring: bool = True,
        enable_io_optimization: bool = True,
        enable_lazy_imports: bool = True,
        auto_preload_modules: Optional[List[str]] = None
    ):
        """
        Args:
            enable_memory_monitoring: メモリ監視有効化
            enable_io_optimization: I/O最適化有効化  
            enable_lazy_imports: 遅延インポート有効化
            auto_preload_modules: 自動事前ロードモジュール
        """
        self.memory_enabled = enable_memory_monitoring
        self.io_enabled = enable_io_optimization
        self.lazy_enabled = enable_lazy_imports
        
        # コンポーネント初期化
        self.memory_monitor = get_memory_monitor() if self.memory_enabled else None
        self.io_optimizer = get_io_optimizer() if self.io_enabled else None
        self.lazy_manager = get_lazy_import_manager() if self.lazy_enabled else None
        
        # 自動事前ロード
        if auto_preload_modules and self.lazy_manager:
            self.lazy_manager.preload_essential_modules(auto_preload_modules)
        
        logger.info(f"StabilityManager初期化: メモリ={self.memory_enabled}, I/O={self.io_enabled}, 遅延={self.lazy_enabled}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム全体の安定性状況取得"""
        status = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'components': {
                'memory_monitoring': self.memory_enabled,
                'io_optimization': self.io_enabled, 
                'lazy_imports': self.lazy_enabled
            }
        }
        
        # メモリ状況
        if self.memory_monitor:
            status['memory'] = self.memory_monitor.get_memory_status()
            status['memory_pressure'] = self.memory_monitor.check_memory_pressure()
        
        # I/O統計
        if self.io_optimizer:
            status['io_stats'] = self.io_optimizer.get_io_statistics()
        
        # インポート統計
        if self.lazy_manager:
            status['import_stats'] = self.lazy_manager.get_import_statistics()
        
        return status
    
    async def safe_character_extraction(
        self,
        extraction_func: Callable,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        **extraction_kwargs
    ) -> Dict[str, Any]:
        """
        安定性最適化された文字抽出実行
        
        Args:
            extraction_func: 抽出関数
            input_path: 入力パス
            output_path: 出力パス
            **extraction_kwargs: 抽出関数の追加引数
            
        Returns:
            実行結果と安定性統計
        """
        logger.info(f"安定性最適化文字抽出開始: {input_path} -> {output_path}")
        
        result = {
            'success': False,
            'extraction_result': None,
            'stability_stats': {}
        }
        
        try:
            # Phase 1: メモリ事前チェック・クリーンアップ
            if self.memory_monitor:
                initial_memory = self.memory_monitor.get_memory_status()
                logger.debug(f"抽出前メモリ: RAM={initial_memory['ram_percent']:.1f}%")
                
                # 必要に応じて事前クリーンアップ
                pressure = self.memory_monitor.check_memory_pressure()
                if pressure['ram_pressure']:
                    cleanup_result = self.memory_monitor.perform_memory_cleanup()
                    logger.info(f"事前メモリクリーンアップ: RAM解放={cleanup_result.get('ram_freed_gb', 0):.2f}GB")
            
            # Phase 2: 必要モジュールの遅延ロード
            if self.lazy_manager:
                # 抽出に必要なモジュールを事前ロード
                torch = self.lazy_manager.get_torch()
                sam = self.lazy_manager.get_segment_anything()
                yolo = self.lazy_manager.get_ultralytics()
                cv2 = self.lazy_manager.get_opencv()
                
                logger.debug("抽出用モジュール遅延ロード完了")
            
            # Phase 3: I/O最適化されたワークスペース準備
            if self.io_optimizer:
                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                await self.io_optimizer._wait_for_vscode_sync(500)  # VSCode同期待機
            
            # Phase 4: 実際の抽出実行
            logger.debug("最適化された文字抽出実行開始")
            extraction_result = await asyncio.get_event_loop().run_in_executor(
                None,
                extraction_func,
                str(input_path),
                str(output_path),
                **extraction_kwargs
            )
            
            result['extraction_result'] = extraction_result
            result['success'] = True
            
            # Phase 5: 実行後メモリ・I/O最適化
            if self.memory_monitor:
                final_memory = self.memory_monitor.get_memory_status()
                memory_diff = final_memory['ram_percent'] - initial_memory['ram_percent']
                
                result['stability_stats']['memory'] = {
                    'initial_ram_percent': initial_memory['ram_percent'],
                    'final_ram_percent': final_memory['ram_percent'],
                    'ram_usage_diff': memory_diff
                }
                
                # 大幅なメモリ増加時はクリーンアップ
                if memory_diff > 10.0:  # 10%以上増加
                    cleanup_result = self.memory_monitor.perform_memory_cleanup()
                    result['stability_stats']['post_cleanup'] = cleanup_result
            
            # Phase 6: I/O統計・VSCode同期
            if self.io_optimizer:
                await self.io_optimizer._wait_for_vscode_sync(1000)  # 最終同期待機
                result['stability_stats']['io'] = self.io_optimizer.get_io_statistics()
            
            logger.info("安定性最適化文字抽出完了")
            
        except Exception as e:
            logger.error(f"安定性最適化文字抽出エラー: {e}")
            result['error'] = str(e)
            
            # エラー時も緊急クリーンアップ
            if self.memory_monitor:
                emergency_cleanup = self.memory_monitor.perform_memory_cleanup(force=True)
                result['stability_stats']['emergency_cleanup'] = emergency_cleanup
        
        return result
    
    async def safe_batch_processing(
        self,
        batch_func: Callable,
        input_items: List[Any],
        output_dir: Union[str, Path],
        batch_size: int = 5,
        **batch_kwargs
    ) -> Dict[str, Any]:
        """
        安定性最適化されたバッチ処理実行
        
        Args:
            batch_func: バッチ処理関数
            input_items: 入力アイテムリスト
            output_dir: 出力ディレクトリ
            batch_size: バッチサイズ
            **batch_kwargs: バッチ関数の追加引数
            
        Returns:
            バッチ処理結果と安定性統計
        """
        logger.info(f"安定性最適化バッチ処理開始: {len(input_items)}件, バッチサイズ={batch_size}")
        
        result = {
            'success': False,
            'processed_count': 0,
            'failed_count': 0,
            'batch_results': [],
            'stability_stats': {}
        }
        
        try:
            output_dir = Path(output_dir)
            
            # バッチ処理実行
            for i in range(0, len(input_items), batch_size):
                batch = input_items[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(input_items) + batch_size - 1) // batch_size
                
                logger.info(f"バッチ {batch_num}/{total_batches} 処理開始: {len(batch)}件")
                
                # バッチ前メモリチェック
                if self.memory_monitor:
                    batch_memory = self.memory_monitor.monitor_once()
                    if batch_memory:  # クリーンアップが実行された場合
                        logger.info(f"バッチ{batch_num}前クリーンアップ実行")
                
                # バッチ実行
                try:
                    batch_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        batch_func,
                        batch,
                        str(output_dir),
                        **batch_kwargs
                    )
                    
                    result['batch_results'].append({
                        'batch_num': batch_num,
                        'success': True,
                        'result': batch_result
                    })
                    result['processed_count'] += len(batch)
                    
                except Exception as batch_error:
                    logger.error(f"バッチ{batch_num}エラー: {batch_error}")
                    result['batch_results'].append({
                        'batch_num': batch_num,
                        'success': False,
                        'error': str(batch_error)
                    })
                    result['failed_count'] += len(batch)
                
                # バッチ間のI/O・メモリ最適化
                if self.io_optimizer:
                    await self.io_optimizer._wait_for_vscode_sync(200)
                
                if self.memory_monitor and batch_num % 3 == 0:  # 3バッチごとにメモリチェック
                    self.memory_monitor.monitor_once()
            
            result['success'] = result['processed_count'] > 0
            
            # 最終統計
            if self.memory_monitor:
                result['stability_stats']['memory'] = self.memory_monitor.get_monitoring_report()
            
            if self.io_optimizer:
                result['stability_stats']['io'] = self.io_optimizer.get_io_statistics()
            
            if self.lazy_manager:
                result['stability_stats']['imports'] = self.lazy_manager.get_import_statistics()
            
            logger.info(f"安定性最適化バッチ処理完了: 成功={result['processed_count']}, 失敗={result['failed_count']}")
            
        except Exception as e:
            logger.error(f"安定性最適化バッチ処理エラー: {e}")
            result['error'] = str(e)
        
        return result
    
    async def integrated_tracker_workflow(
        self,
        tracker_id: str,
        workspace_base: Union[str, Path],
        extraction_func: Callable,
        extraction_args: Dict[str, Any],
        google_sheets_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        統合トラッカーワークフロー実行
        
        P1-B003で発生したVSCodeハングアップを回避する
        完全最適化されたワークフロー実行
        """
        logger.info(f"統合トラッカーワークフロー開始: {tracker_id}")
        
        workspace_dir = Path(workspace_base) / tracker_id
        
        result = {
            'tracker_id': tracker_id,
            'success': False,
            'phases': {},
            'stability_stats': {}
        }
        
        try:
            # Phase 1: ワークスペース準備（I/O最適化）
            if self.io_optimizer:
                workspace_result = await self.io_optimizer.workspace_sync_workflow(
                    workspace_dir=workspace_dir,
                    google_sheets_func=None,  # 後で実行
                    reports_data=None  # 後で生成
                )
                result['phases']['workspace_setup'] = workspace_result
            else:
                workspace_dir.mkdir(parents=True, exist_ok=True)
                for subdir in ['extraction', 'quality', 'dashboard', 'tests']:
                    (workspace_dir / subdir).mkdir(exist_ok=True)
            
            # Phase 2: 安定性最適化抽出実行
            extraction_result = await self.safe_character_extraction(
                extraction_func=extraction_func,
                input_path=extraction_args['input_path'],
                output_path=workspace_dir / 'extraction',
                **{k: v for k, v in extraction_args.items() if k != 'input_path'}
            )
            result['phases']['extraction'] = extraction_result
            
            # Phase 3: レポート生成（I/O最適化）
            if extraction_result['success'] and self.io_optimizer:
                reports_data = {
                    f'{tracker_id}_report': {
                        'title': f'{tracker_id} 実行レポート',
                        'extraction_result': extraction_result,
                        'markdown': f'# {tracker_id}\n\n実行完了',
                        'html': f'<h1>{tracker_id}</h1><p>実行完了</p>'
                    }
                }
                
                report_result = await self.io_optimizer.optimized_report_generation(
                    reports_data, workspace_dir
                )
                result['phases']['reports'] = report_result
            
            # Phase 4: Google Sheets更新（競合回避）
            if google_sheets_func and self.io_optimizer:
                sheets_result = await self.io_optimizer.google_sheets_safe_update(
                    google_sheets_func, post_update_delay=2.0
                )
                result['phases']['google_sheets'] = {'success': True, 'result': sheets_result}
            
            # Phase 5: 最終安定性統計
            result['stability_stats'] = self.get_system_status()
            result['success'] = True
            
            logger.info(f"統合トラッカーワークフロー完了: {tracker_id}")
            
        except Exception as e:
            logger.error(f"統合トラッカーワークフローエラー: {tracker_id} - {e}")
            result['error'] = str(e)
            
            # エラー時緊急安定化
            if self.memory_monitor:
                emergency_stats = self.memory_monitor.perform_memory_cleanup(force=True)
                result['stability_stats']['emergency'] = emergency_stats
        
        return result


# グローバルインスタンス
_global_stability_manager: Optional[StabilityManager] = None


def get_stability_manager() -> StabilityManager:
    """グローバル安定性管理インスタンス取得"""
    global _global_stability_manager
    if _global_stability_manager is None:
        _global_stability_manager = StabilityManager(
            auto_preload_modules=['torch', 'cv2']  # 基本モジュールは事前ロード
        )
    return _global_stability_manager


# 簡易インターフェース
async def stable_extract_character(input_path: str, output_path: str, **kwargs):
    """安定性最適化文字抽出（簡易インターフェース）"""
    manager = get_stability_manager()
    
    # 実際の抽出関数は動的インポート
    def extract_func(inp, out, **kw):
        # 遅延インポートでextract_character関数を取得
        from features.extraction.commands.extract_character import main as extract_main
        return extract_main(inp, out, **kw)
    
    return await manager.safe_character_extraction(
        extraction_func=extract_func,
        input_path=input_path,
        output_path=output_path,
        **kwargs
    )


# デコレータ: 統合安定性管理
def with_stability_management(
    enable_memory: bool = True,
    enable_io: bool = True,
    enable_lazy: bool = True
):
    """
    関数デコレータ: 統合安定性管理適用
    
    Args:
        enable_memory: メモリ監視有効化
        enable_io: I/O最適化有効化
        enable_lazy: 遅延インポート有効化
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            manager = StabilityManager(
                enable_memory_monitoring=enable_memory,
                enable_io_optimization=enable_io,
                enable_lazy_imports=enable_lazy
            )
            
            logger.debug(f"統合安定性管理実行: {func.__name__}")
            
            try:
                # メモリ監視
                if enable_memory and manager.memory_monitor:
                    manager.memory_monitor.monitor_once()
                
                # 関数実行
                result = await func(*args, **kwargs)
                
                # 実行後最適化
                if enable_memory and manager.memory_monitor:
                    manager.memory_monitor.monitor_once()
                
                if enable_io and manager.io_optimizer:
                    await manager.io_optimizer._wait_for_vscode_sync()
                
                return result
                
            except Exception as e:
                logger.error(f"統合安定性管理エラー: {func.__name__} - {e}")
                
                # エラー時緊急安定化
                if enable_memory and manager.memory_monitor:
                    manager.memory_monitor.perform_memory_cleanup(force=True)
                
                raise
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # 非同期関数か同期関数かを判定
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # テスト実行
    import json
    import tempfile
    
    async def test_stability_manager():
        """統合安定性管理システムのテスト"""
        print("=== P1-023 統合安定性管理システム テスト ===")
        print()
        
        manager = StabilityManager()
        
        # テスト1: システム状況確認
        print("テスト1: システム状況")
        status = manager.get_system_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        print()
        
        # テスト2: 模擬抽出処理
        print("テスト2: 安定性最適化抽出")
        
        def mock_extraction(input_path, output_path, **kwargs):
            import time
            time.sleep(0.1)  # 模擬処理時間
            return {'success': True, 'processed_files': 1}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extraction_result = await manager.safe_character_extraction(
                extraction_func=mock_extraction,
                input_path=temp_dir + '/input',
                output_path=temp_dir + '/output'
            )
            
            print("抽出結果:")
            print(json.dumps(extraction_result, indent=2, ensure_ascii=False))
        print()
        
        # テスト3: デコレータ使用
        print("テスト3: デコレータテスト")
        
        @with_stability_management()
        async def test_function():
            return "安定性管理適用済み関数実行完了"
        
        decorator_result = await test_function()
        print(f"デコレータ結果: {decorator_result}")
    
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_stability_manager())