"""
P1-023: VSCode安定性向上 - I/O競合回避システム

GPT-4分析に基づくファイルI/O最適化:
- 非同期I/O処理によるVSCode競合回避
- バッチ処理によるI/O効率化
- WSL2環境でのファイルシステム最適化
- Google Sheets API競合対策
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class IOOptimizer:
    """
    I/O競合回避・最適化クラス
    
    WSL2-VSCode環境でのファイル操作競合を回避し、
    大量のファイル操作を効率的にバッチ処理する。
    """
    
    def __init__(
        self,
        max_concurrent_io: int = 3,
        io_delay_ms: int = 100,
        batch_size: int = 10,
        enable_vscode_sync: bool = True
    ):
        """
        Args:
            max_concurrent_io: 最大同時I/O操作数
            io_delay_ms: I/O操作間の遅延 (ミリ秒)
            batch_size: バッチ処理サイズ
            enable_vscode_sync: VSCode同期待機有効化
        """
        self.max_concurrent_io = max_concurrent_io
        self.io_delay_ms = io_delay_ms
        self.batch_size = batch_size
        self.enable_vscode_sync = enable_vscode_sync
        
        self.io_semaphore = asyncio.Semaphore(max_concurrent_io)
        self.active_operations = set()
        self.io_stats = {
            'operations_completed': 0,
            'total_wait_time': 0.0,
            'avg_operation_time': 0.0
        }
        
        logger.info(f"IOOptimizer初期化: 同時I/O={max_concurrent_io}, バッチサイズ={batch_size}")
    
    async def _wait_for_vscode_sync(self, delay_ms: Optional[int] = None):
        """VSCode同期待機（ファイル監視競合回避）"""
        if not self.enable_vscode_sync:
            return
        
        wait_time = (delay_ms or self.io_delay_ms) / 1000.0
        await asyncio.sleep(wait_time)
        logger.debug(f"VSCode同期待機完了: {wait_time:.3f}s")
    
    @asynccontextmanager
    async def _io_operation_context(self, operation_name: str):
        """I/O操作のコンテキスト管理"""
        start_time = time.time()
        
        async with self.io_semaphore:
            self.active_operations.add(operation_name)
            logger.debug(f"I/O操作開始: {operation_name}")
            
            try:
                yield
                
            finally:
                self.active_operations.discard(operation_name)
                operation_time = time.time() - start_time
                
                # 統計更新
                self.io_stats['operations_completed'] += 1
                self.io_stats['total_wait_time'] += operation_time
                self.io_stats['avg_operation_time'] = (
                    self.io_stats['total_wait_time'] / self.io_stats['operations_completed']
                )
                
                logger.debug(f"I/O操作完了: {operation_name} ({operation_time:.3f}s)")
                await self._wait_for_vscode_sync()
    
    async def safe_file_write(
        self, 
        file_path: Union[str, Path], 
        content: Union[str, bytes], 
        encoding: str = 'utf-8'
    ) -> bool:
        """競合回避ファイル書き込み"""
        file_path = Path(file_path)
        
        async with self._io_operation_context(f"write:{file_path.name}"):
            try:
                # 親ディレクトリ作成
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 書き込み実行
                if isinstance(content, str):
                    file_path.write_text(content, encoding=encoding)
                else:
                    file_path.write_bytes(content)
                
                logger.debug(f"ファイル書き込み成功: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"ファイル書き込み失敗: {file_path} - {e}")
                return False
    
    async def safe_file_read(
        self, 
        file_path: Union[str, Path], 
        encoding: str = 'utf-8'
    ) -> Optional[Union[str, bytes]]:
        """競合回避ファイル読み込み"""
        file_path = Path(file_path)
        
        async with self._io_operation_context(f"read:{file_path.name}"):
            try:
                if encoding:
                    content = file_path.read_text(encoding=encoding)
                else:
                    content = file_path.read_bytes()
                
                logger.debug(f"ファイル読み込み成功: {file_path}")
                return content
                
            except Exception as e:
                logger.error(f"ファイル読み込み失敗: {file_path} - {e}")
                return None
    
    async def batch_file_operations(
        self, 
        operations: List[Dict[str, Any]], 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[bool]:
        """
        バッチファイル操作実行
        
        Args:
            operations: 操作リスト [{"type": "write", "path": "...", "content": "..."}, ...]
            progress_callback: 進捗コールバック(完了数, 総数)
            
        Returns:
            各操作の成功/失敗結果
        """
        results = []
        total_ops = len(operations)
        
        logger.info(f"バッチファイル操作開始: {total_ops}件")
        
        for i in range(0, total_ops, self.batch_size):
            batch = operations[i:i + self.batch_size]
            batch_tasks = []
            
            for op in batch:
                if op['type'] == 'write':
                    task = self.safe_file_write(op['path'], op['content'], op.get('encoding', 'utf-8'))
                elif op['type'] == 'read':
                    task = self.safe_file_read(op['path'], op.get('encoding', 'utf-8'))
                else:
                    logger.warning(f"不明な操作タイプ: {op['type']}")
                    continue
                
                batch_tasks.append(task)
            
            # バッチ実行
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 結果処理
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"バッチ操作でエラー: {result}")
                    results.append(False)
                else:
                    results.append(bool(result))
            
            # 進捗報告
            completed = len(results)
            if progress_callback:
                progress_callback(completed, total_ops)
            
            logger.debug(f"バッチ進捗: {completed}/{total_ops}")
        
        success_count = sum(results)
        logger.info(f"バッチファイル操作完了: 成功={success_count}/{total_ops}")
        
        return results
    
    async def google_sheets_safe_update(
        self, 
        update_func: Callable[[], Any], 
        post_update_delay: float = 1.0
    ) -> Any:
        """Google Sheets更新の安全実行（ファイル競合回避）"""
        async with self._io_operation_context("google_sheets_update"):
            try:
                # Google Sheets更新実行
                result = await asyncio.get_event_loop().run_in_executor(
                    None, update_func
                )
                
                # 更新後の待機（VSCodeファイル監視との競合回避）
                await asyncio.sleep(post_update_delay)
                logger.debug(f"Google Sheets更新完了 (待機: {post_update_delay}s)")
                
                return result
                
            except Exception as e:
                logger.error(f"Google Sheets更新失敗: {e}")
                raise
    
    async def optimized_report_generation(
        self, 
        reports: Dict[str, Any], 
        output_dir: Union[str, Path]
    ) -> Dict[str, bool]:
        """最適化レポート生成（JSON, Markdown, HTML）"""
        output_dir = Path(output_dir)
        results = {}
        
        # レポート生成操作リスト作成
        operations = []
        
        for report_name, content in reports.items():
            # JSON レポート
            json_path = output_dir / f"{report_name}.json"
            operations.append({
                'type': 'write',
                'path': json_path,
                'content': json.dumps(content, indent=2, ensure_ascii=False)
            })
            
            # Markdown レポート（内容に応じて）
            if isinstance(content, dict) and 'markdown' in content:
                md_path = output_dir / f"{report_name}.md"
                operations.append({
                    'type': 'write',
                    'path': md_path,
                    'content': content['markdown']
                })
            
            # HTML レポート（内容に応じて）
            if isinstance(content, dict) and 'html' in content:
                html_path = output_dir / f"{report_name}.html"
                operations.append({
                    'type': 'write',
                    'path': html_path,
                    'content': content['html']
                })
        
        # バッチ実行
        def progress_callback(completed, total):
            logger.info(f"レポート生成進捗: {completed}/{total}")
        
        operation_results = await self.batch_file_operations(operations, progress_callback)
        
        # 結果整理
        for i, op in enumerate(operations):
            file_path = Path(op['path'])
            results[f"{file_path.stem}_{file_path.suffix[1:]}"] = operation_results[i]
        
        return results
    
    async def workspace_sync_workflow(
        self, 
        workspace_dir: Union[str, Path],
        google_sheets_func: Optional[Callable[[], Any]] = None,
        reports_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        統合ワークスペース同期ワークフロー
        
        P1-B003でのVSCodeハングアップを回避する最適化された処理順序:
        1. ワークスペース準備
        2. レポート生成（バッチ）
        3. Google Sheets更新（遅延付き）
        4. VSCode同期待機
        """
        workspace_dir = Path(workspace_dir)
        workflow_results = {
            'workspace_created': False,
            'reports_generated': {},
            'google_sheets_updated': False,
            'total_time': 0
        }
        
        start_time = time.time()
        logger.info(f"ワークスペース同期ワークフロー開始: {workspace_dir}")
        
        try:
            # Step 1: ワークスペース準備
            workspace_dir.mkdir(parents=True, exist_ok=True)
            for subdir in ['extraction', 'quality', 'dashboard', 'tests']:
                (workspace_dir / subdir).mkdir(exist_ok=True)
            workflow_results['workspace_created'] = True
            logger.info("ワークスペース構造作成完了")
            
            # Step 2: レポート生成（最適化バッチ処理）
            if reports_data:
                report_results = await self.optimized_report_generation(
                    reports_data, workspace_dir
                )
                workflow_results['reports_generated'] = report_results
                logger.info("レポート生成完了")
            
            # Step 3: VSCode同期待機（ファイル監視安定化）
            await self._wait_for_vscode_sync(1500)  # 1.5秒待機
            
            # Step 4: Google Sheets更新（競合回避）
            if google_sheets_func:
                sheets_result = await self.google_sheets_safe_update(
                    google_sheets_func, post_update_delay=2.0
                )
                workflow_results['google_sheets_updated'] = True
                logger.info("Google Sheets更新完了")
            
            # Step 5: 最終同期待機
            await self._wait_for_vscode_sync(1000)  # 1秒最終待機
            
        except Exception as e:
            logger.error(f"ワークスペース同期ワークフローエラー: {e}")
            raise
        
        finally:
            workflow_results['total_time'] = time.time() - start_time
            logger.info(f"ワークスペース同期ワークフロー完了: {workflow_results['total_time']:.2f}s")
        
        return workflow_results
    
    def get_io_statistics(self) -> Dict[str, Any]:
        """I/O統計情報取得"""
        return {
            'timestamp': datetime.now().isoformat(),
            'active_operations': list(self.active_operations),
            'active_count': len(self.active_operations),
            'statistics': self.io_stats.copy(),
            'configuration': {
                'max_concurrent_io': self.max_concurrent_io,
                'io_delay_ms': self.io_delay_ms,
                'batch_size': self.batch_size,
                'vscode_sync_enabled': self.enable_vscode_sync
            }
        }


# グローバルインスタンス
_global_optimizer: Optional[IOOptimizer] = None


def get_io_optimizer() -> IOOptimizer:
    """グローバルI/O最適化インスタンスを取得"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = IOOptimizer()
    return _global_optimizer


# 簡易インターフェース関数群
async def safe_write_file(file_path: Union[str, Path], content: Union[str, bytes], encoding: str = 'utf-8') -> bool:
    """競合回避ファイル書き込み（簡易インターフェース）"""
    optimizer = get_io_optimizer()
    return await optimizer.safe_file_write(file_path, content, encoding)


async def safe_read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[Union[str, bytes]]:
    """競合回避ファイル読み込み（簡易インターフェース）"""
    optimizer = get_io_optimizer()
    return await optimizer.safe_file_read(file_path, encoding)


async def batch_write_files(file_operations: List[Dict[str, Any]]) -> List[bool]:
    """バッチファイル書き込み（簡易インターフェース）"""
    optimizer = get_io_optimizer()
    return await optimizer.batch_file_operations(file_operations)


# デコレータ: I/O最適化
def optimize_io(enable_batch: bool = False, vscode_sync: bool = True):
    """
    関数デコレータ: I/O操作最適化
    
    Args:
        enable_batch: バッチ処理有効化
        vscode_sync: VSCode同期待機有効化
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            optimizer = get_io_optimizer()
            
            # 一時的にVSCode同期設定を変更
            original_sync = optimizer.enable_vscode_sync
            optimizer.enable_vscode_sync = vscode_sync
            
            try:
                logger.debug(f"I/O最適化実行開始: {func.__name__}")
                result = await func(*args, **kwargs)
                
                # VSCode同期待機
                if vscode_sync:
                    await optimizer._wait_for_vscode_sync()
                
                logger.debug(f"I/O最適化実行完了: {func.__name__}")
                return result
                
            finally:
                optimizer.enable_vscode_sync = original_sync
        
        def sync_wrapper(*args, **kwargs):
            # 同期関数の場合は非同期実行
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # 非同期関数か同期関数かを判定
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # テスト実行
    import shutil
    import tempfile
    
    async def test_io_optimizer():
        """I/O最適化システムのテスト"""
        print("=== P1-023 I/O競合回避システム テスト ===")
        print()
        
        optimizer = IOOptimizer()
        
        # テンポラリディレクトリ作成
        test_dir = Path(tempfile.mkdtemp())
        print(f"テストディレクトリ: {test_dir}")
        
        try:
            # テスト1: 単一ファイル操作
            print("テスト1: 単一ファイル操作")
            test_file = test_dir / "test.txt"
            success = await optimizer.safe_file_write(test_file, "テストコンテンツ")
            print(f"書き込み結果: {success}")
            
            content = await optimizer.safe_file_read(test_file)
            print(f"読み込み結果: {content}")
            print()
            
            # テスト2: バッチ操作
            print("テスト2: バッチファイル操作")
            operations = []
            for i in range(5):
                operations.append({
                    'type': 'write',
                    'path': test_dir / f"batch_{i}.txt",
                    'content': f"バッチテスト {i}"
                })
            
            def progress_cb(completed, total):
                print(f"進捗: {completed}/{total}")
            
            batch_results = await optimizer.batch_file_operations(operations, progress_cb)
            print(f"バッチ結果: 成功={sum(batch_results)}/{len(batch_results)}")
            print()
            
            # テスト3: レポート生成
            print("テスト3: レポート生成")
            reports = {
                'test_report': {
                    'title': 'テストレポート',
                    'data': {'test': True, 'value': 42},
                    'markdown': '# テストレポート\n\nテスト成功',
                    'html': '<h1>テストレポート</h1><p>テスト成功</p>'
                }
            }
            
            report_results = await optimizer.optimized_report_generation(
                reports, test_dir / 'reports'
            )
            print(f"レポート生成結果: {report_results}")
            print()
            
            # 統計情報
            stats = optimizer.get_io_statistics()
            print("I/O統計情報:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
        finally:
            # クリーンアップ
            shutil.rmtree(test_dir)
            print(f"\nテストディレクトリ削除: {test_dir}")
    
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_io_optimizer())