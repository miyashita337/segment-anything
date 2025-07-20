#!/usr/bin/env python3
"""
エラー自動修復システム
エラーパターンの検出と自動修復戦略の実装
"""

import os
import sys
import re
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorPattern:
    """エラーパターン定義クラス"""
    
    def __init__(self, pattern: str, error_type: type, recovery_strategy: str):
        """
        初期化
        
        Args:
            pattern: エラーメッセージの正規表現パターン
            error_type: エラータイプ
            recovery_strategy: 修復戦略の名前
        """
        self.pattern = re.compile(pattern)
        self.error_type = error_type
        self.recovery_strategy = recovery_strategy


class ErrorRecovery:
    """エラー自動修復クラス"""
    
    def __init__(self):
        """初期化"""
        self.error_patterns = []
        self.recovery_strategies = {}
        self.recovery_history = []
        self._register_default_patterns()
        self._register_default_strategies()
        
    def _register_default_patterns(self):
        """デフォルトエラーパターン登録"""
        patterns = [
            # インポートエラー
            ErrorPattern(
                r"ModuleNotFoundError.*No module named '(.+)'",
                ModuleNotFoundError,
                "fix_import_error"
            ),
            # ファイル不在エラー
            ErrorPattern(
                r"FileNotFoundError.*No such file or directory.*'(.+)'",
                FileNotFoundError,
                "fix_file_not_found"
            ),
            # CUDA OOMエラー
            ErrorPattern(
                r"CUDA out of memory",
                RuntimeError,
                "fix_cuda_oom"
            ),
            # パス関連エラー
            ErrorPattern(
                r"hooks\.start.*import start",
                ImportError,
                "fix_phase0_import"
            ),
            # タイムアウトエラー
            ErrorPattern(
                r"Command timed out",
                TimeoutError,
                "fix_timeout"
            ),
            # 権限エラー
            ErrorPattern(
                r"Permission denied",
                PermissionError,
                "fix_permission"
            )
        ]
        
        self.error_patterns.extend(patterns)
        
    def _register_default_strategies(self):
        """デフォルト修復戦略登録"""
        self.recovery_strategies = {
            "fix_import_error": self._fix_import_error,
            "fix_file_not_found": self._fix_file_not_found,
            "fix_cuda_oom": self._fix_cuda_oom,
            "fix_phase0_import": self._fix_phase0_import,
            "fix_timeout": self._fix_timeout,
            "fix_permission": self._fix_permission,
            "retry": self._retry_operation,
            "skip": self._skip_operation
        }
        
    def analyze_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """
        エラー分析
        
        Args:
            error: 発生したエラー
            context: エラーコンテキスト情報
            
        Returns:
            推奨される修復戦略名
        """
        error_message = str(error)
        error_type = type(error)
        
        # パターンマッチング
        for pattern in self.error_patterns:
            if pattern.error_type == error_type or pattern.error_type == Exception:
                match = pattern.pattern.search(error_message)
                if match:
                    logger.info(f"Error pattern matched: {pattern.recovery_strategy}")
                    return pattern.recovery_strategy
                    
        # デフォルト戦略
        if context.get('retry_count', 0) < 3:
            return "retry"
        else:
            return "skip"
            
    def apply_recovery(self, strategy: str, error: Exception, 
                      context: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        修復戦略適用
        
        Args:
            strategy: 修復戦略名
            error: 発生したエラー
            context: エラーコンテキスト
            
        Returns:
            (成功フラグ, 修復結果)
        """
        if strategy not in self.recovery_strategies:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return False, None
            
        try:
            recovery_func = self.recovery_strategies[strategy]
            success, result = recovery_func(error, context)
            
            # 履歴記録
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'strategy': strategy,
                'error': str(error),
                'success': success,
                'context': context
            })
            
            return success, result
            
        except Exception as e:
            logger.error(f"Recovery strategy {strategy} failed: {e}")
            return False, None
            
    def _fix_import_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """インポートエラー修復"""
        error_message = str(error)
        
        # モジュール名抽出
        match = re.search(r"No module named '(.+)'", error_message)
        if not match:
            return False, None
            
        module_name = match.group(1)
        logger.info(f"Attempting to fix import error for: {module_name}")
        
        # Phase 0リファクタリング対応
        if module_name.startswith('hooks.'):
            new_module = module_name.replace('hooks.', 'features.common.hooks.')
            context['import_fix'] = f"Replace '{module_name}' with '{new_module}'"
            return True, new_module
            
        if module_name.startswith('commands.'):
            new_module = module_name.replace('commands.', 'features.extraction.commands.')
            context['import_fix'] = f"Replace '{module_name}' with '{new_module}'"
            return True, new_module
            
        return False, None
        
    def _fix_file_not_found(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """ファイル不在エラー修復"""
        error_message = str(error)
        
        # ファイルパス抽出
        match = re.search(r"'(.+)'", error_message)
        if not match:
            return False, None
            
        file_path = match.group(1)
        logger.info(f"File not found: {file_path}")
        
        # 代替パス候補
        alternatives = [
            file_path,
            f"test_small/{Path(file_path).name}",
            f"temp/scripts/testing/{Path(file_path).name}",
            f"temp/scripts/migration/{Path(file_path).name}"
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                logger.info(f"Found alternative: {alt_path}")
                context['file_alternative'] = alt_path
                return True, alt_path
                
        # ディレクトリ作成が必要な場合
        if context.get('create_if_missing', False):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            return True, file_path
            
        return False, None
        
    def _fix_cuda_oom(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """CUDA OOMエラー修復"""
        logger.info("CUDA OOM detected, attempting recovery")
        
        try:
            import torch
            
            # GPUメモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # バッチサイズ削減提案
            current_batch = context.get('batch_size', 1)
            new_batch = max(1, current_batch // 2)
            
            context['batch_size'] = new_batch
            context['cuda_recovery'] = "Reduced batch size and cleared GPU cache"
            
            return True, {'batch_size': new_batch}
            
        except Exception as e:
            logger.error(f"CUDA recovery failed: {e}")
            return False, None
            
    def _fix_phase0_import(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Phase 0インポートエラー修復"""
        file_path = context.get('file_path', '')
        
        if not file_path:
            return False, None
            
        # インポート修正マッピング
        import_fixes = {
            'from hooks.start': 'from features.common.hooks.start',
            'from commands.': 'from features.extraction.commands.',
            'from utils.': 'from features.evaluation.utils.',
            'from models.': 'from features.extraction.models.'
        }
        
        fixes_applied = []
        
        for old_import, new_import in import_fixes.items():
            if old_import in str(error):
                fixes_applied.append((old_import, new_import))
                
        if fixes_applied:
            context['import_fixes'] = fixes_applied
            return True, fixes_applied
            
        return False, None
        
    def _fix_timeout(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """タイムアウトエラー修復"""
        current_timeout = context.get('timeout', 120)
        new_timeout = min(current_timeout * 2, 600)  # 最大10分
        
        context['timeout'] = new_timeout
        context['timeout_recovery'] = f"Increased timeout from {current_timeout}s to {new_timeout}s"
        
        return True, {'timeout': new_timeout}
        
    def _fix_permission(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """権限エラー修復"""
        file_path = context.get('file_path', '')
        
        if file_path and os.path.exists(file_path):
            try:
                # 実行権限付与
                os.chmod(file_path, 0o755)
                context['permission_fix'] = f"Added execute permission to {file_path}"
                return True, None
            except:
                pass
                
        return False, None
        
    def _retry_operation(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """操作リトライ"""
        retry_count = context.get('retry_count', 0)
        max_retries = context.get('max_retries', 3)
        
        if retry_count < max_retries:
            context['retry_count'] = retry_count + 1
            context['retry_recovery'] = f"Retry attempt {retry_count + 1}/{max_retries}"
            return True, {'retry': True}
            
        return False, None
        
    def _skip_operation(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """操作スキップ"""
        context['skip_recovery'] = "Operation skipped due to unrecoverable error"
        return True, {'skip': True}
        

class AutoRecoveryExecutor:
    """自動修復機能付き実行クラス"""
    
    def __init__(self):
        self.error_recovery = ErrorRecovery()
        
    def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """
        自動修復機能付き関数実行
        
        Args:
            func: 実行する関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数
            
        Returns:
            関数の実行結果
        """
        context = {
            'function': func.__name__,
            'retry_count': 0,
            'max_retries': 3
        }
        
        while context['retry_count'] <= context['max_retries']:
            try:
                # 関数実行
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                
                # エラー分析
                strategy = self.error_recovery.analyze_error(e, context)
                
                if strategy:
                    # 修復適用
                    success, recovery_result = self.error_recovery.apply_recovery(
                        strategy, e, context
                    )
                    
                    if success:
                        if recovery_result and recovery_result.get('skip'):
                            logger.info("Skipping operation")
                            return None
                        elif recovery_result and recovery_result.get('retry'):
                            logger.info("Retrying operation")
                            continue
                        else:
                            # 修復結果を適用して再実行
                            if 'import_fix' in context:
                                # インポート修正の場合は特別処理
                                logger.info("Import fix applied, manual intervention required")
                                raise e
                                
                # 修復失敗または戦略なし
                raise e
                
        # 最大リトライ到達
        raise Exception(f"Max retries reached for {func.__name__}")


# シングルトンインスタンス
_auto_recovery_executor = None


def get_auto_recovery_executor() -> AutoRecoveryExecutor:
    """自動修復実行クラスのシングルトンインスタンス取得"""
    global _auto_recovery_executor
    if _auto_recovery_executor is None:
        _auto_recovery_executor = AutoRecoveryExecutor()
    return _auto_recovery_executor