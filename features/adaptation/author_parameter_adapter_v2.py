#!/usr/bin/env python3
"""
QCA-001: 作者別パラメータ適応システム実装（設定ファイルベース版）
ディレクトリ構造から作者を自動識別し、各作者の絵柄特性に最適化された
SAM/YOLOパラメータを自動適用するシステム

設定ファイル: config/author_config.yaml
Created for: QCA-001 - 作者別パラメータ適応システム・ディレクトリ構造ベース自動最適化
Author: Claude Code Integration System
"""

import logging
import os
import re
import yaml
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuthorCharacteristics(Enum):
    """作者の描画特性分類"""

    BALANCED = "balanced"  # バランス型
    DETAIL_ORIENTED = "detail"  # 細密描写特化
    SIMPLE_STYLE = "simple"  # シンプルスタイル


@dataclass
class AuthorProfile:
    """作者プロファイル定義"""

    author_id: str
    characteristics: AuthorCharacteristics
    sam_profile: str
    yolo_confidence: float
    max_masks: int
    score_threshold: float
    description: str
    processing_notes: List[str]


class AuthorParameterAdapterV2:
    """
    作者別パラメータ適応システム（設定ファイルベース）

    ディレクトリ構造から作者を自動検出し、
    各作者の絵柄特性に最適化されたパラメータを提供する

    設定ファイル: config/author_config.yaml
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス（未指定時はデフォルト設定使用）
        """
        self.config = self._load_config(config_path)
        self.author_profiles = self._build_author_profiles()
        self.default_profile = self._build_default_profile()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if config_path is None:
            # デフォルト設定ファイルパス
            config_path = Path(__file__).parent.parent.parent / "config" / "author_config.yaml"

        if Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✅ 設定ファイル読み込み成功: {config_path}")
                    return config
            except Exception as e:
                logger.warning(f"⚠️ 設定ファイル読み込みエラー: {e}")
        else:
            logger.warning(f"⚠️ 設定ファイル未発見: {config_path}")

        # ハードコーディングされたデフォルト設定（フォールバック）
        logger.info("📋 デフォルト設定を使用")
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得（設定ファイルが見つからない場合のフォールバック）"""
        return {
            "author_profiles": {
                "yado": {
                    "characteristics": "BALANCED",
                    "sam_profile": "character_focused",
                    "yolo_confidence": 0.07,
                    "max_masks": 10,
                    "score_threshold": 0.07,
                    "description": "yado作者: バランス型キャラクター重視・多作品対応",
                    "processing_notes": ["kana03-09全作品に共通の絵柄", "キャラクター抽出に特化した設定"],
                },
                "kiri": {
                    "characteristics": "DETAIL_ORIENTED",
                    "sam_profile": "precision_focused",
                    "yolo_confidence": 0.05,
                    "max_masks": 8,
                    "score_threshold": 0.05,
                    "description": "kiri作者: 細密描写特化・高品質重視（元aichi）",
                    "processing_notes": ["細かい線画・詳細な描写", "高品質抽出を優先"],
                },
                "zundamon": {
                    "characteristics": "SIMPLE_STYLE",
                    "sam_profile": "speed_optimized",
                    "yolo_confidence": 0.08,
                    "max_masks": 6,
                    "score_threshold": 0.08,
                    "description": "zundamon作者: シンプルスタイル・効率重視",
                    "processing_notes": ["シンプルで明確な線画", "高速処理可能"],
                },
            },
            "default_profile": {
                "characteristics": "BALANCED",
                "sam_profile": "character_focused",
                "yolo_confidence": 0.07,
                "max_masks": 8,
                "score_threshold": 0.07,
                "description": "default: バランス型汎用設定",
                "processing_notes": ["汎用バランス型設定"],
            },
            "detection": {"known_authors": ["yado", "kiri", "zundamon"], "fallback_search": True},
        }

    def _build_author_profiles(self) -> Dict[str, AuthorProfile]:
        """設定からAuthorProfileオブジェクトを構築"""
        profiles = {}

        for author_id, config in self.config.get("author_profiles", {}).items():
            profiles[author_id] = self._config_to_profile(author_id, config)

        return profiles

    def _build_default_profile(self) -> AuthorProfile:
        """デフォルトプロファイル構築"""
        config = self.config.get("default_profile", {})
        return self._config_to_profile("default", config)

    def _config_to_profile(self, author_id: str, config: Dict[str, Any]) -> AuthorProfile:
        """設定辞書からAuthorProfileオブジェクトを生成"""
        # AuthorCharacteristicsの変換
        characteristics_map = {
            "BALANCED": AuthorCharacteristics.BALANCED,
            "DETAIL_ORIENTED": AuthorCharacteristics.DETAIL_ORIENTED,
            "SIMPLE_STYLE": AuthorCharacteristics.SIMPLE_STYLE,
        }

        characteristics = characteristics_map.get(
            config.get("characteristics", "BALANCED"), AuthorCharacteristics.BALANCED
        )

        return AuthorProfile(
            author_id=author_id,
            characteristics=characteristics,
            sam_profile=config.get("sam_profile", "character_focused"),
            yolo_confidence=config.get("yolo_confidence", 0.07),
            max_masks=config.get("max_masks", 8),
            score_threshold=config.get("score_threshold", 0.07),
            description=config.get("description", f"{author_id}作者: 設定なし"),
            processing_notes=config.get("processing_notes", []),
        )

    def detect_author_from_path(self, image_path: str) -> Optional[str]:
        """
        パスから作者を自動検出（設定ベース）

        Args:
            image_path: 画像ファイルパス

        Returns:
            作者名または None
        """
        try:
            path_obj = Path(image_path)
            path_parts = path_obj.parts

            known_authors = self.config.get("detection", {}).get("known_authors", [])

            # /train/{作者名}/ パターンを検索
            train_index = -1
            for i, part in enumerate(path_parts):
                if part.lower() == "train":
                    train_index = i
                    break

            if train_index >= 0 and train_index + 1 < len(path_parts):
                # train の次の部分が作者名
                potential_author = path_parts[train_index + 1]

                if potential_author.lower() in [a.lower() for a in known_authors]:
                    logger.debug(
                        f"🔍 Author detected from path: {potential_author} (from train/{potential_author})"
                    )
                    return potential_author.lower()

            # フォールバック: パス内の既知作者名を検索
            if self.config.get("detection", {}).get("fallback_search", True):
                for part in path_parts:
                    part_lower = part.lower()
                    if part_lower in [a.lower() for a in known_authors]:
                        logger.debug(f"🔍 Author detected from path part: {part_lower}")
                        return part_lower

            logger.debug(f"⚠️ No known author found in path: {image_path}")
            return None

        except Exception as e:
            logger.error(f"❌ Author detection error: {e}")
            return None

    def get_author_profile(self, author: str) -> AuthorProfile:
        """
        作者プロファイル取得

        Args:
            author: 作者名

        Returns:
            AuthorProfile オブジェクト
        """
        return self.author_profiles.get(author, self.default_profile)

    def apply_author_optimization(
        self, image_path: str, sam_config: Optional[Dict] = None, force_author: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        作者別最適化適用

        Args:
            image_path: 画像パス
            sam_config: SAM設定（オプション）
            force_author: 強制作者指定（オプション）

        Returns:
            最適化設定
        """
        # 作者検出
        detected_author = force_author or self.detect_author_from_path(image_path)

        if detected_author:
            profile = self.get_author_profile(detected_author)
            logger.info(f"🎯 作者別最適化適用: {detected_author} → {profile.description}")
        else:
            profile = self.default_profile
            logger.info(f"⚡ デフォルト設定適用: {profile.description}")

        # 最適化設定生成
        optimization_config = {
            "detected_author": detected_author,
            "profile": profile,
            "sam_profile": profile.sam_profile,
            "yolo_confidence": profile.yolo_confidence,
            "max_masks": profile.max_masks,
            "score_threshold": profile.score_threshold,
        }

        return optimization_config

    def get_sam_config(self, author: Optional[str] = None) -> str:
        """
        作者用SAMプロファイル取得

        Args:
            author: 作者名

        Returns:
            SAMプロファイル名
        """
        profile = self.get_author_profile(author) if author else self.default_profile
        return profile.sam_profile

    @property
    def AUTHOR_PROFILES(self) -> Dict[str, AuthorProfile]:
        """後方互換性のためのプロパティ"""
        return self.author_profiles

    def list_authors(self) -> List[str]:
        """登録作者一覧取得"""
        return list(self.author_profiles.keys())

    def reload_config(self, config_path: Optional[str] = None):
        """設定ファイル再読み込み"""
        self.config = self._load_config(config_path)
        self.author_profiles = self._build_author_profiles()
        self.default_profile = self._build_default_profile()
        logger.info("🔄 設定ファイル再読み込み完了")


# 後方互換性のためのエイリアス
AuthorParameterAdapter = AuthorParameterAdapterV2


# 静的メソッドの後方互換性サポート
def detect_author_from_path(image_path: str) -> Optional[str]:
    """
    静的メソッド互換性ラッパー
    （既存コードとの互換性のため）
    """
    adapter = AuthorParameterAdapterV2()
    return adapter.detect_author_from_path(image_path)
