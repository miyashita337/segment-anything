#!/usr/bin/env python3
"""
エラーハンドリングシステムのテスト
"""

import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from features.common.error_handling import (
        ErrorSeverity, ErrorCategory, BaseCustomError,
        FileNotFoundError, InsufficientMemoryError, GPUNotAvailableError,
        ValidationError, ProcessingError, ErrorHandler, with_error_handling
    )
except ImportError as e:
    pytest.skip(f"Error handling modules not available: {e}", allow_module_level=True)


class TestCustomErrors:
    """カスタムエラークラスのテスト"""
    
    def test_base_custom_error(self):
        """基本エラークラスのテスト"""
        error = BaseCustomError(
            message="テストエラー",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.PROCESSING,
            recoverable=True,
            retry_count=3,
            details={'test': 'value'}
        )
        
        assert error.message == "テストエラー"
        assert error.severity == ErrorSeverity.WARNING
        assert error.category == ErrorCategory.PROCESSING
        assert error.recoverable is True
        assert error.retry_count == 3
        assert error.details['test'] == 'value'
        
        # to_dict テスト
        error_dict = error.to_dict()
        assert error_dict['type'] == 'BaseCustomError'
        assert error_dict['message'] == "テストエラー"
        assert error_dict['severity'] == 'warning'
        assert error_dict['category'] == 'processing_error'
    
    def test_file_not_found_error(self):
        """ファイルエラーのテスト"""
        error = FileNotFoundError("/path/to/missing.txt")
        
        assert "ファイルが見つかりません" in error.message
        assert error.category == ErrorCategory.IO
        assert error.recoverable is False
        assert error.details['filepath'] == "/path/to/missing.txt"
    
    def test_insufficient_memory_error(self):
        """メモリ不足エラーのテスト"""
        error = InsufficientMemoryError(required_mb=1024, available_mb=512)
        
        assert "メモリ不足" in error.message
        assert error.category == ErrorCategory.MEMORY
        assert error.recoverable is True
        assert error.retry_count == 3
        assert error.details['required_mb'] == 1024
        assert error.details['available_mb'] == 512
    
    def test_gpu_not_available_error(self):
        """GPU利用不可エラーのテスト"""
        error = GPUNotAvailableError("CUDA version mismatch")
        
        assert "GPU利用不可" in error.message
        assert error.category == ErrorCategory.GPU
        assert error.recoverable is True
        assert error.details['reason'] == "CUDA version mismatch"
    
    def test_validation_error(self):
        """バリデーションエラーのテスト"""
        error = ValidationError(field="batch_size", value=-1, expected="positive integer")
        
        assert "バリデーションエラー" in error.message
        assert error.category == ErrorCategory.VALIDATION
        assert error.recoverable is False
        assert error.details['field'] == "batch_size"
        assert error.details['value'] == -1
        assert error.details['expected'] == "positive integer"
    
    def test_processing_error(self):
        """処理エラーのテスト"""
        error = ProcessingError(stage="preprocessing", reason="invalid image format")
        
        assert "処理エラー" in error.message
        assert error.category == ErrorCategory.PROCESSING
        assert error.recoverable is True
        assert error.retry_count == 2
        assert error.details['stage'] == "preprocessing"
        assert error.details['reason'] == "invalid image format"


class TestErrorHandler:
    """エラーハンドラーのテスト"""
    
    def test_error_handler_initialization(self, tmp_path):
        """エラーハンドラー初期化のテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        assert handler.log_dir.exists()
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) == 0
    
    def test_handle_error(self, tmp_path):
        """エラー処理のテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        error = ValidationError(field="test", value="invalid", expected="valid")
        result = handler.handle_error(error)
        
        assert len(handler.error_history) == 1
        assert handler.error_history[0]['type'] == 'ValidationError'
        
        # ログファイルが作成されたか確認
        log_files = list(handler.log_dir.glob("*.json"))
        assert len(log_files) == 1
    
    def test_recovery_strategy(self, tmp_path):
        """リカバリー戦略のテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        # リカバリー戦略を登録
        def test_recovery(error):
            return "recovered"
        
        handler.register_recovery_strategy(ProcessingError, test_recovery)
        
        # リカバリー可能なエラーを処理
        error = ProcessingError(stage="test", reason="test error")
        result = handler.handle_error(error)
        
        assert result == "recovered"
    
    def test_error_summary(self, tmp_path):
        """エラーサマリーのテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        # 複数のエラーを追加
        handler.handle_error(ValidationError("field1", "value1", "expected1"))
        handler.handle_error(ProcessingError("stage1", "reason1"))
        handler.handle_error(GPUNotAvailableError("test"))
        
        summary = handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert summary['recoverable_count'] == 2
        assert 'error' in summary['by_severity']
        assert 'warning' in summary['by_severity']
        assert 'validation_error' in summary['by_category']
        assert 'processing_error' in summary['by_category']
        assert 'gpu_error' in summary['by_category']


class TestErrorDecorator:
    """エラーハンドリングデコレーターのテスト"""
    
    def test_decorator_with_custom_error(self, tmp_path):
        """カスタムエラーでのデコレーターテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        @with_error_handling(handler)
        def test_function():
            raise ValidationError("test", "invalid", "valid")
        
        with pytest.raises(ValidationError):
            test_function()
        
        assert len(handler.error_history) == 1
    
    def test_decorator_with_standard_error(self, tmp_path):
        """標準エラーでのデコレーターテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        @with_error_handling(handler)
        def test_function():
            raise ValueError("standard error")
        
        with pytest.raises(BaseCustomError):
            test_function()
        
        assert len(handler.error_history) == 1
        assert handler.error_history[0]['category'] == 'unknown_error'
    
    def test_decorator_with_recovery(self, tmp_path):
        """リカバリー付きデコレーターテスト"""
        handler = ErrorHandler(log_dir=tmp_path / "test_logs")
        
        # リカバリー戦略を登録
        handler.register_recovery_strategy(
            ProcessingError,
            lambda e: "recovered_value"
        )
        
        @with_error_handling(handler)
        def test_function():
            raise ProcessingError("test", "recoverable error")
        
        # リカバリーが成功した場合は値が返される
        result = test_function()
        assert result == "recovered_value"


def test_memory_recovery_strategy():
    """メモリリカバリー戦略のテスト"""
    from features.common.error_handling import memory_recovery_strategy
    
    error = InsufficientMemoryError(required_mb=100, available_mb=50)
    
    # メモリリカバリーを実行（実際のメモリ状況に依存）
    result = memory_recovery_strategy(error)
    
    # 結果はbool値
    assert isinstance(result, bool)


def test_gpu_fallback_strategy():
    """GPUフォールバック戦略のテスト"""
    from features.common.error_handling import gpu_fallback_strategy
    
    error = GPUNotAvailableError("test")
    result = gpu_fallback_strategy(error)
    
    assert result == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])