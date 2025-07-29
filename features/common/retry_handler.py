"""
自動リトライ機能
失敗時に最大3回まで自動再実行を行う
"""

import time
import logging
from typing import TypeVar, Callable, Optional, Dict, Any, Tuple
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """リトライ設定"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        exponential_backoff: bool = True,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: Optional[Tuple[type, ...]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_backoff = exponential_backoff
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_on_exceptions = retry_on_exceptions or (Exception,)


class RetryHandler:
    """自動リトライハンドラー"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_stats: Dict[str, int] = {}
    
    def retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """デコレータ: 自動リトライ機能を付与"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            func_name = f"{func.__module__}.{func.__name__}"
            attempt = 0
            last_exception = None
            
            while attempt <= self.config.max_retries:
                try:
                    if attempt > 0:
                        logger.info(f"🔄 リトライ実行 {attempt}/{self.config.max_retries}: {func_name}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"✅ リトライ成功: {func_name} (試行回数: {attempt + 1})")
                        self._record_retry_success(func_name, attempt)
                    
                    return result
                
                except self.config.retry_on_exceptions as e:
                    last_exception = e
                    attempt += 1
                    
                    if attempt > self.config.max_retries:
                        logger.error(f"❌ リトライ上限到達: {func_name}")
                        logger.error(f"   最終エラー: {str(e)}")
                        self._record_retry_failure(func_name)
                        raise
                    
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"⚠️ エラー発生: {str(e)}")
                    logger.info(f"   {delay:.1f}秒後にリトライします...")
                    
                    time.sleep(delay)
            
            # ここには到達しないはずだが、念のため
            if last_exception:
                raise last_exception
        
        return wrapper
    
    def retry_with_fallback(
        self,
        func: Callable[..., T],
        fallback: Callable[..., T]
    ) -> Callable[..., T]:
        """デコレータ: リトライ失敗時にフォールバック関数を実行"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return self.retry(func)(*args, **kwargs)
            except Exception as e:
                logger.warning(f"🔀 フォールバック実行: {func.__name__} -> {fallback.__name__}")
                return fallback(*args, **kwargs)
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """リトライ間隔を計算"""
        if not self.config.exponential_backoff:
            return self.config.initial_delay
        
        delay = self.config.initial_delay * (self.config.backoff_factor ** (attempt - 1))
        return min(delay, self.config.max_delay)
    
    def _record_retry_success(self, func_name: str, attempts: int) -> None:
        """リトライ成功を記録"""
        key = f"{func_name}_success"
        self.retry_stats[key] = self.retry_stats.get(key, 0) + 1
        
        key_attempts = f"{func_name}_total_attempts"
        self.retry_stats[key_attempts] = self.retry_stats.get(key_attempts, 0) + attempts
    
    def _record_retry_failure(self, func_name: str) -> None:
        """リトライ失敗を記録"""
        key = f"{func_name}_failure"
        self.retry_stats[key] = self.retry_stats.get(key, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """リトライ統計を取得"""
        return {
            "retry_stats": self.retry_stats.copy(),
            "config": {
                "max_retries": self.config.max_retries,
                "initial_delay": self.config.initial_delay,
                "exponential_backoff": self.config.exponential_backoff
            }
        }
    
    def reset_statistics(self) -> None:
        """統計をリセット"""
        self.retry_stats.clear()


# グローバルインスタンス
default_retry_handler = RetryHandler()
retry = default_retry_handler.retry
retry_with_fallback = default_retry_handler.retry_with_fallback


# 画像処理特化のリトライ設定
image_processing_config = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    exponential_backoff=True,
    retry_on_exceptions=(RuntimeError, ValueError, IOError)
)

image_retry_handler = RetryHandler(image_processing_config)