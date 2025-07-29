"""
自動リトライ機能のテスト
"""

import pytest
import time
from unittest.mock import Mock, patch
from features.common.retry_handler import RetryHandler, RetryConfig


class TestRetryHandler:
    """RetryHandlerのテスト"""
    
    def test_successful_execution_no_retry(self):
        """正常実行時はリトライしない"""
        handler = RetryHandler()
        mock_func = Mock(return_value="success")
        
        @handler.retry
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_failure(self):
        """失敗時にリトライする"""
        handler = RetryHandler(RetryConfig(max_retries=3, initial_delay=0.1))
        mock_func = Mock(side_effect=[RuntimeError("error"), RuntimeError("error"), "success"])
        
        @handler.retry
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_max_retries_exceeded(self):
        """リトライ上限に達したら例外を再発生"""
        handler = RetryHandler(RetryConfig(max_retries=2, initial_delay=0.1))
        mock_func = Mock(side_effect=RuntimeError("persistent error"))
        
        @handler.retry
        def test_func():
            return mock_func()
        
        with pytest.raises(RuntimeError, match="persistent error"):
            test_func()
        
        # 初回実行 + 2回リトライ = 3回
        assert mock_func.call_count == 3
    
    def test_retry_with_specific_exceptions(self):
        """特定の例外のみリトライ"""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            retry_on_exceptions=(ValueError,)
        )
        handler = RetryHandler(config)
        
        # ValueErrorはリトライ対象
        mock_func1 = Mock(side_effect=[ValueError("retry this"), "success"])
        
        @handler.retry
        def test_func1():
            return mock_func1()
        
        result = test_func1()
        assert result == "success"
        assert mock_func1.call_count == 2
        
        # RuntimeErrorはリトライ対象外
        mock_func2 = Mock(side_effect=RuntimeError("don't retry"))
        
        @handler.retry
        def test_func2():
            return mock_func2()
        
        with pytest.raises(RuntimeError):
            test_func2()
        
        assert mock_func2.call_count == 1
    
    def test_exponential_backoff(self):
        """指数バックオフの動作確認"""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            exponential_backoff=True,
            backoff_factor=2.0
        )
        handler = RetryHandler(config)
        
        # 時間計測用
        delays = []
        original_sleep = time.sleep
        
        def mock_sleep(seconds):
            delays.append(seconds)
            original_sleep(0.001)  # 実際には短時間だけ待機
        
        with patch('time.sleep', side_effect=mock_sleep):
            mock_func = Mock(side_effect=[RuntimeError(), RuntimeError(), "success"])
            
            @handler.retry
            def test_func():
                return mock_func()
            
            result = test_func()
        
        assert result == "success"
        assert len(delays) == 2  # 2回リトライ
        assert delays[0] == pytest.approx(0.1, rel=0.01)  # 初回: 0.1秒
        assert delays[1] == pytest.approx(0.2, rel=0.01)  # 2回目: 0.1 * 2 = 0.2秒
    
    def test_retry_with_fallback(self):
        """フォールバック機能のテスト"""
        handler = RetryHandler(RetryConfig(max_retries=2, initial_delay=0.1))
        
        mock_main = Mock(side_effect=RuntimeError("always fails"))
        mock_fallback = Mock(return_value="fallback result")
        
        # retry_with_fallbackは2つの関数を引数に取る
        decorated = handler.retry_with_fallback(
            lambda: mock_main(),
            lambda: mock_fallback()
        )
        
        result = decorated()
        
        assert result == "fallback result"
        assert mock_main.call_count == 3  # 初回 + 2回リトライ
        assert mock_fallback.call_count == 1
    
    def test_statistics_tracking(self):
        """統計情報の追跡"""
        handler = RetryHandler(RetryConfig(max_retries=3, initial_delay=0.1))
        
        # 成功ケース
        mock_func1 = Mock(side_effect=[RuntimeError(), "success"])
        
        @handler.retry
        def success_func():
            return mock_func1()
        
        success_func()
        
        # 失敗ケース
        mock_func2 = Mock(side_effect=RuntimeError("always fails"))
        
        @handler.retry
        def failure_func():
            return mock_func2()
        
        try:
            failure_func()
        except RuntimeError:
            pass
        
        stats = handler.get_statistics()
        
        # 統計情報の確認
        assert "retry_stats" in stats
        assert "config" in stats
        
        retry_stats = stats["retry_stats"]
        
        # success_funcの統計
        assert f"{success_func.__module__}.success_func_success" in retry_stats
        assert retry_stats[f"{success_func.__module__}.success_func_success"] == 1
        
        # failure_funcの統計
        assert f"{failure_func.__module__}.failure_func_failure" in retry_stats
        assert retry_stats[f"{failure_func.__module__}.failure_func_failure"] == 1
    
    def test_reset_statistics(self):
        """統計情報のリセット"""
        handler = RetryHandler()
        
        # ダミーデータを追加
        handler.retry_stats["test_key"] = 100
        
        # リセット
        handler.reset_statistics()
        
        stats = handler.get_statistics()
        assert len(stats["retry_stats"]) == 0