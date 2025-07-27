#!/usr/bin/env python3
"""
階層的エラーハンドリングシステム
PH2-002: より具体的で回復可能なエラー処理の実装
"""

import sys
import traceback
import logging
from enum import Enum
from typing import Optional, Callable, Any, Dict, Type
from functools import wraps
from datetime import datetime
import json
from pathlib import Path


class ErrorSeverity(Enum):
    """エラーの重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""

    IO = "io_error"
    MEMORY = "memory_error"
    GPU = "gpu_error"
    VALIDATION = "validation_error"
    NETWORK = "network_error"
    CONFIGURATION = "configuration_error"
    PROCESSING = "processing_error"
    RESOURCE = "resource_error"
    UNKNOWN = "unknown_error"


class BaseCustomError(Exception):
    """カスタムエラーの基底クラス"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        recoverable: bool = False,
        retry_count: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.retry_count = retry_count
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """エラー情報を辞書形式で返す"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# 具体的なエラークラス
class FileNotFoundError(BaseCustomError):
    """ファイルが見つからないエラー"""

    def __init__(self, filepath: str, **kwargs):
        super().__init__(
            f"ファイルが見つかりません: {filepath}",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.IO,
            recoverable=False,
            details={"filepath": filepath},
            **kwargs,
        )


class InsufficientMemoryError(BaseCustomError):
    """メモリ不足エラー"""

    def __init__(self, required_mb: float, available_mb: float, **kwargs):
        super().__init__(
            f"メモリ不足: 必要 {required_mb:.1f}MB, 利用可能 {available_mb:.1f}MB",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.MEMORY,
            recoverable=True,
            retry_count=3,
            details={"required_mb": required_mb, "available_mb": available_mb},
            **kwargs,
        )


class GPUNotAvailableError(BaseCustomError):
    """GPU利用不可エラー"""

    def __init__(self, reason: str = "CUDA not available", **kwargs):
        super().__init__(
            f"GPU利用不可: {reason}",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.GPU,
            recoverable=True,
            details={"reason": reason},
            **kwargs,
        )


class ValidationError(BaseCustomError):
    """バリデーションエラー"""

    def __init__(self, field: str, value: Any, expected: str, **kwargs):
        super().__init__(
            f"バリデーションエラー: {field}={value}, 期待値: {expected}",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.VALIDATION,
            recoverable=False,
            details={"field": field, "value": value, "expected": expected},
            **kwargs,
        )


class ProcessingError(BaseCustomError):
    """処理エラー"""

    def __init__(self, stage: str, reason: str, **kwargs):
        super().__init__(
            f"処理エラー [{stage}]: {reason}",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PROCESSING,
            recoverable=True,
            retry_count=2,
            details={"stage": stage, "reason": reason},
            **kwargs,
        )


class ResourceError(BaseCustomError):
    """リソースエラー"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            category=kwargs.get("category", ErrorCategory.RESOURCE),
            recoverable=True,
            retry_count=1,
            **kwargs,
        )


class ErrorHandler:
    """エラーハンドリングマネージャー"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/errors")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_history = []
        self.recovery_strategies = {}
        self._setup_logging()

    def _setup_logging(self):
        """ロギング設定"""
        self.logger = logging.getLogger("ErrorHandler")
        self.logger.setLevel(logging.DEBUG)

        # ファイルハンドラー
        fh = logging.FileHandler(self.log_dir / f"errors_{datetime.now():%Y%m%d}.log")
        fh.setLevel(logging.WARNING)

        # コンソールハンドラー
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # フォーマッター
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def register_recovery_strategy(
        self, error_type: Type[BaseCustomError], strategy: Callable[[BaseCustomError], Any]
    ):
        """リカバリー戦略を登録"""
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: BaseCustomError) -> Optional[Any]:
        """エラーを処理"""
        # エラー履歴に追加
        self.error_history.append(error.to_dict())

        # ログ出力
        log_method = getattr(self.logger, error.severity.value)
        log_method(f"{error.category.value}: {error.message}")

        # 詳細情報をJSONで保存
        error_file = (
            self.log_dir / f"error_{datetime.now():%Y%m%d_%H%M%S}_{error.category.value}.json"
        )
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error.to_dict(), f, indent=2, ensure_ascii=False)

        # リカバリー可能な場合は戦略を実行
        if error.recoverable and type(error) in self.recovery_strategies:
            try:
                return self.recovery_strategies[type(error)](error)
            except Exception as recovery_error:
                self.logger.error(f"リカバリー失敗: {recovery_error}")

        return None

    def get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリーを取得"""
        summary = {
            "total_errors": len(self.error_history),
            "by_severity": {},
            "by_category": {},
            "recoverable_count": 0,
            "recent_errors": self.error_history[-10:],
        }

        for error in self.error_history:
            # 重要度別集計
            severity = error["severity"]
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

            # カテゴリ別集計
            category = error["category"]
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

            # リカバリー可能数
            if error["recoverable"]:
                summary["recoverable_count"] += 1

        return summary


def with_error_handling(handler: ErrorHandler):
    """エラーハンドリングデコレーター"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseCustomError as e:
                # カスタムエラーの場合はハンドラーで処理
                result = handler.handle_error(e)
                if result is not None:
                    return result
                raise
            except MemoryError as e:
                # メモリエラーをカスタムエラーに変換
                custom_error = InsufficientMemoryError(
                    required_mb=0, available_mb=0, details={"original_error": str(e)}  # 不明  # 不明
                )
                handler.handle_error(custom_error)
                raise custom_error
            except FileNotFoundError as e:
                # ファイルエラーをカスタムエラーに変換
                custom_error = FileNotFoundError(
                    filepath=str(e.filename) if hasattr(e, "filename") else "unknown",
                    details={"original_error": str(e)},
                )
                handler.handle_error(custom_error)
                raise custom_error
            except Exception as e:
                # その他のエラー
                custom_error = BaseCustomError(
                    message=str(e),
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.UNKNOWN,
                    details={"type": type(e).__name__, "traceback": traceback.format_exc()},
                )
                handler.handle_error(custom_error)
                raise custom_error

        return wrapper

    return decorator


# グローバルエラーハンドラーインスタンス
global_error_handler = ErrorHandler()


# リカバリー戦略の例
def memory_recovery_strategy(error: InsufficientMemoryError) -> bool:
    """メモリ不足時のリカバリー戦略"""
    import gc
    import torch

    print("🔧 メモリクリーンアップ実行中...")

    # ガベージコレクション
    gc.collect()

    # GPU メモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 再度メモリチェック
    import psutil

    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024 * 1024)

    print(f"✅ クリーンアップ完了: {available_mb:.1f}MB 利用可能")

    return available_mb >= error.details.get("required_mb", 0)


def gpu_fallback_strategy(error: GPUNotAvailableError) -> str:
    """GPU利用不可時のフォールバック戦略"""
    print("⚠️ GPU利用不可のためCPUモードに切り替えます")
    return "cpu"


# デフォルトのリカバリー戦略を登録
global_error_handler.register_recovery_strategy(InsufficientMemoryError, memory_recovery_strategy)
global_error_handler.register_recovery_strategy(GPUNotAvailableError, gpu_fallback_strategy)
