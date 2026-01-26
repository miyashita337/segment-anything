#!/usr/bin/env python3
"""
CI環境検出・軽量化設定管理モジュール
CI-INTEGRATION-001: GitHub Actions CI専用軽量化システム

環境変数によるCI検出、CPU専用モード、軽量モデル自動選択を提供
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CIProvider(Enum):
    """CI環境種別"""

    GITHUB_ACTIONS = "github_actions"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class CIConfiguration:
    """CI環境設定"""

    is_ci: bool
    provider: CIProvider
    cpu_only: bool
    memory_limit_disabled: bool
    lightweight_models: bool
    max_processing_time: int  # seconds
    yolo_model: str
    sam_model: str


class CIEnvironmentDetector:
    """CI環境検出・設定管理"""

    @staticmethod
    def detect_ci_environment() -> CIConfiguration:
        """
        CI環境を検出し、適切な設定を返す

        Returns:
            CIConfiguration: CI環境設定
        """
        # CI環境検出
        is_ci = (
            os.getenv("CI_ENVIRONMENT", "false").lower() == "true"
            or os.getenv("CI", "false").lower() == "true"
            or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
        )

        # CI プロバイダー検出
        if os.getenv("GITHUB_ACTIONS"):
            provider = CIProvider.GITHUB_ACTIONS
        elif is_ci:
            provider = CIProvider.UNKNOWN
        else:
            provider = CIProvider.LOCAL

        # CI専用設定
        cpu_only = is_ci or os.getenv("CPU_ONLY", "false").lower() == "true"

        memory_limit_disabled = (
            is_ci or os.getenv("MEMORY_LIMIT_DISABLED", "false").lower() == "true"
        )

        lightweight_models = is_ci

        # 軽量化モデル選択（CI環境時）
        if is_ci:
            yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt")  # nano版
            sam_model = os.getenv("SAM_MODEL", "sam_vit_b_01ec64.pth")  # base版
            max_processing_time = int(os.getenv("MAX_PROCESSING_TIME", "300"))
        else:
            yolo_model = os.getenv("YOLO_MODEL", "yolov8x.pt")  # extra-large版
            sam_model = os.getenv("SAM_MODEL", "sam_vit_h_4b8939.pth")  # huge版
            max_processing_time = int(os.getenv("MAX_PROCESSING_TIME", "1800"))

        config = CIConfiguration(
            is_ci=is_ci,
            provider=provider,
            cpu_only=cpu_only,
            memory_limit_disabled=memory_limit_disabled,
            lightweight_models=lightweight_models,
            max_processing_time=max_processing_time,
            yolo_model=yolo_model,
            sam_model=sam_model,
        )

        logger.info(f"🔧 Environment detected: {provider.value} (CI={is_ci}, CPU_ONLY={cpu_only})")
        if is_ci:
            logger.info(f"   YOLO: {yolo_model}, SAM: {sam_model}")
            logger.info(f"   Max processing: {max_processing_time}s")

        return config

    @staticmethod
    def get_optimized_extraction_params() -> Dict[str, Any]:
        """
        CI環境に最適化された抽出パラメータを取得

        Returns:
            最適化パラメータ辞書
        """
        config = CIEnvironmentDetector.detect_ci_environment()

        if config.is_ci:
            # CI環境: 軽量化・高速処理優先
            return {
                "yolo_model": config.yolo_model,
                "sam_model": config.sam_model,
                "yolo_confidence": 0.1,  # 少し高めで高速化
                "max_masks": 5,  # 制限を厳しく
                "score_threshold": 0.1,
                "cpu_only": config.cpu_only,
                "timeout": config.max_processing_time,
                "quality_method": "balanced",  # シンプルな評価
                "memory_optimization": True,
                "ci_mode": True,
                "processing_notes": ["CI環境での軽量化処理", "CPU専用モード", "90%処理時間短縮設定"],
            }
        else:
            # ローカル環境: 品質優先
            return {
                "yolo_model": config.yolo_model,
                "sam_model": config.sam_model,
                "yolo_confidence": 0.07,
                "max_masks": 10,
                "score_threshold": 0.07,
                "cpu_only": False,
                "timeout": config.max_processing_time,
                "quality_method": "balanced",
                "memory_optimization": False,
                "ci_mode": False,
                "processing_notes": ["ローカル環境での高品質処理", "GPU使用可能時は自動切替", "標準品質・速度バランス"],
            }

    @staticmethod
    def log_environment_info():
        """環境情報をログ出力"""
        config = CIEnvironmentDetector.detect_ci_environment()

        logger.info("🌍 Environment Information:")
        logger.info(f"   Provider: {config.provider.value}")
        logger.info(f"   CI Mode: {config.is_ci}")
        logger.info(f"   CPU Only: {config.cpu_only}")
        logger.info(f"   Memory Optimization: {config.memory_limit_disabled}")
        logger.info(f"   Lightweight Models: {config.lightweight_models}")
        logger.info(f"   YOLO Model: {config.yolo_model}")
        logger.info(f"   SAM Model: {config.sam_model}")
        logger.info(f"   Processing Timeout: {config.max_processing_time}s")

        # 環境変数の詳細
        env_vars = [
            "CI",
            "CI_ENVIRONMENT",
            "GITHUB_ACTIONS",
            "CPU_ONLY",
            "MEMORY_LIMIT_DISABLED",
            "YOLO_MODEL",
            "SAM_MODEL",
            "MAX_PROCESSING_TIME",
        ]

        logger.debug("🔍 Environment Variables:")
        for var in env_vars:
            value = os.getenv(var, "<not set>")
            logger.debug(f"   {var}: {value}")


def get_ci_config() -> CIConfiguration:
    """CI設定取得（簡易アクセス関数）"""
    return CIEnvironmentDetector.detect_ci_environment()


def is_ci_environment() -> bool:
    """CI環境判定（簡易アクセス関数）"""
    return CIEnvironmentDetector.detect_ci_environment().is_ci


def get_extraction_params() -> Dict[str, Any]:
    """抽出パラメータ取得（簡易アクセス関数）"""
    return CIEnvironmentDetector.get_optimized_extraction_params()


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    print("🧪 CI Environment Detection Test")
    print("=" * 40)

    CIEnvironmentDetector.log_environment_info()

    print("\n📋 Extraction Parameters:")
    params = get_extraction_params()
    for key, value in params.items():
        print(f"   {key}: {value}")

    print(f"\n🔍 Is CI Environment: {is_ci_environment()}")
