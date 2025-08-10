#!/usr/bin/env python3
"""
QCA-001: 作者別パラメータ適応システム実装
ディレクトリ構造から作者を自動識別し、各作者の絵柄特性に最適化された
SAM/YOLOパラメータを自動適用するシステム

Created for: QCA-001 - 作者別パラメータ適応システム・ディレクトリ構造ベース自動最適化
Author: Claude Code Integration System
"""

import os
import re
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuthorCharacteristics(Enum):
    """作者の特性分類"""
    DETAIL_ORIENTED = "detail_oriented"      # 細かい描写重視
    BALANCED = "balanced"                    # バランス型
    SPEED_FOCUSED = "speed_focused"          # 高速処理重視
    COMPLEX_SCENES = "complex_scenes"        # 複雑なシーン
    SIMPLE_STYLE = "simple_style"            # シンプルスタイル


@dataclass
class AuthorProfile:
    """作者プロファイル"""
    author_id: str
    characteristics: AuthorCharacteristics
    sam_profile: str
    yolo_confidence: float
    max_masks: int
    score_threshold: float
    description: str
    processing_notes: List[str]


class AuthorParameterAdapter:
    """
    作者別パラメータ適応システム（設定ファイルベース）
    
    ディレクトリ構造から作者を自動検出し、
    各作者の絵柄特性に最適化されたパラメータを提供する
    
    パス構造: /train/{作者名}/org/{作品名}/
    例: /mnt/c/AItools/lora/train/yado/org/kana05/ → 作者="yado"
    
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
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if config_path is None:
            # デフォルト設定ファイルパス
            config_path = Path(__file__).parent.parent.parent / "config" / "author_config.yaml"
            
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # ハードコーディングされたデフォルト設定（フォールバック）
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得（設定ファイルが見つからない場合のフォールバック）"""
        return {
            'author_profiles': {
                'yado': {
                    'characteristics': 'BALANCED',
                    'sam_profile': 'character_focused',
                    'yolo_confidence': 0.07,
                    'max_masks': 10,
                    'score_threshold': 0.07,
                    'description': 'yado作者: バランス型キャラクター重視・多作品対応',
                    'processing_notes': ['kana03-09全作品に共通の絵柄', 'キャラクター抽出に特化した設定']
                },
                'aichi': {
                    'characteristics': 'DETAIL_ORIENTED',
                    'sam_profile': 'precision_focused',
                    'yolo_confidence': 0.05,
                    'max_masks': 8,
                    'score_threshold': 0.05,
                    'description': 'aichi作者: 細密描写特化・高品質重視',
                    'processing_notes': ['細かい線画・詳細な描写', '高品質抽出を優先']
                },
                'kiri': {
                    'characteristics': 'DETAIL_ORIENTED',
                    'sam_profile': 'precision_focused',
                    'yolo_confidence': 0.05,
                    'max_masks': 8,
                    'score_threshold': 0.05,
                    'description': 'kiri作者: 細密描写特化・高品質重視（元aichi）',
                    'processing_notes': ['細かい線画・詳細な描写', '高品質抽出を優先']
                },
                'zundamon': {
                    'characteristics': 'SIMPLE_STYLE',
                    'sam_profile': 'speed_optimized',
                    'yolo_confidence': 0.08,
                    'max_masks': 6,
                    'score_threshold': 0.08,
                    'description': 'zundamon作者: シンプルスタイル・効率重視',
                    'processing_notes': ['シンプルで明確な線画', '高速処理可能']
                }
            },
            'detection': {
                'known_authors': ['yado', 'aichi', 'kiri', 'zundamon']
            }
        }
    
    def _build_author_profiles(self) -> Dict[str, AuthorProfile]:
        """設定からAuthorProfileオブジェクトを構築"""
        profiles = {}
        
        for author_id, config in self.config.get('author_profiles', {}).items():
            # AuthorCharacteristicsの変換
            characteristics_map = {
                'BALANCED': AuthorCharacteristics.BALANCED,
                'DETAIL_ORIENTED': AuthorCharacteristics.DETAIL_ORIENTED,
                'SIMPLE_STYLE': AuthorCharacteristics.SIMPLE_STYLE
            }
            
            characteristics = characteristics_map.get(config.get('characteristics', 'BALANCED'), 
                                                    AuthorCharacteristics.BALANCED)
            
            profiles[author_id] = AuthorProfile(
                author_id=author_id,
                characteristics=characteristics,
                sam_profile=config.get('sam_profile', 'character_focused'),
                yolo_confidence=config.get('yolo_confidence', 0.07),
                max_masks=config.get('max_masks', 8),
                score_threshold=config.get('score_threshold', 0.07),
                description=config.get('description', f'{author_id}作者: 設定なし'),
                processing_notes=config.get('processing_notes', [])
            )
        
        return profiles
    
    @property
    def AUTHOR_PROFILES(self) -> Dict[str, AuthorProfile]:
        """後方互換性のためのプロパティ"""
        return self.author_profiles
    
    # 旧ハードコーディング定義（削除予定）
    _LEGACY_AUTHOR_PROFILES = {
        "yado": AuthorProfile(
            author_id="yado",
            characteristics=AuthorCharacteristics.BALANCED,
            sam_profile="character_focused",
            yolo_confidence=0.07,  # バランス型・キャラクター重視
            max_masks=10,
            score_threshold=0.07,
            description="yado作者: バランス型キャラクター重視・多作品対応",
            processing_notes=[
                "kana03-09全作品に共通の絵柄",
                "キャラクター抽出に特化した設定",
                "多様なシーン・ポーズに対応",
                "安定した品質・速度バランス"
            ]
        ),
        
        "aichi": AuthorProfile(
            author_id="aichi",
            characteristics=AuthorCharacteristics.DETAIL_ORIENTED,
            sam_profile="precision_focused",
            yolo_confidence=0.05,  # 細密描写のため低信頼度
            max_masks=8,
            score_threshold=0.05,
            description="aichi作者: 細密描写特化・高品質重視",
            processing_notes=[
                "細かい線画・詳細な描写",
                "高品質抽出を優先",
                "処理時間よりも品質重視",
                "背景との分離に注意"
            ]
        ),
        
        "kiri": AuthorProfile(
            author_id="kiri",
            characteristics=AuthorCharacteristics.DETAIL_ORIENTED,
            sam_profile="precision_focused",
            yolo_confidence=0.05,  # 細密描写のため低信頼度（元aichi設定）
            max_masks=8,
            score_threshold=0.05,
            description="kiri作者: 細密描写特化・高品質重視（元aichi）",
            processing_notes=[
                "細かい線画・詳細な描写",
                "高品質抽出を優先",
                "処理時間よりも品質重視",
                "背景との分離に注意"
            ]
        ),
        
        "zundamon": AuthorProfile(
            author_id="zundamon",
            characteristics=AuthorCharacteristics.SIMPLE_STYLE,
            sam_profile="speed_optimized",
            yolo_confidence=0.08,  # シンプルスタイルのため標準
            max_masks=6,
            score_threshold=0.08,
            description="zundamon作者: シンプルスタイル・効率重視",
            processing_notes=[
                "シンプルで明確な線画",
                "高速処理可能",
                "効率的なバッチ処理",
                "安定した基本品質"
            ]
        )
    }
    
    # デフォルト設定（作者不明時）
    DEFAULT_PROFILE = AuthorProfile(
        author_id="default",
        characteristics=AuthorCharacteristics.BALANCED,
        sam_profile="balanced",
        yolo_confidence=0.07,
        max_masks=10,
        score_threshold=0.07,
        description="デフォルト設定・汎用バランス型",
        processing_notes=[
            "汎用設定",
            "未知の作者・データセットに対応",
            "標準的な品質・速度バランス"
        ]
    )
    
    @staticmethod
    def detect_author_from_path(image_path: str) -> Optional[str]:
        """
        パスから作者を自動検出
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            作者名（yado, aichi, zundamon, kiri）または None
            
        Examples:
            /mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg -> "yado"
            /path/to/aichi/10_aichi/image.jpg -> "aichi"
            /some/zundamon/work/image.png -> "zundamon"
        """
        try:
            path_obj = Path(image_path)
            path_parts = path_obj.parts
            
            # /train/{作者名}/ パターンを検索
            train_index = -1
            for i, part in enumerate(path_parts):
                if part.lower() == 'train':
                    train_index = i
                    break
            
            if train_index >= 0 and train_index + 1 < len(path_parts):
                # train の次の部分が作者名
                potential_author = path_parts[train_index + 1]
                
                # 既知の作者名かチェック
                if potential_author.lower() in ['yado', 'aichi', 'zundamon', 'kiri']:
                    logger.debug(f"🔍 Author detected from path: {potential_author} (from train/{potential_author})")
                    return potential_author.lower()
            
            # フォールバック: パス内の既知作者名を検索
            for part in path_parts:
                part_lower = part.lower()
                if part_lower in ['yado', 'aichi', 'zundamon', 'kiri']:
                    logger.debug(f"🔍 Author detected from path part: {part_lower}")
                    return part_lower
            
            logger.debug(f"⚠️ No known author found in path: {image_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting author from path {image_path}: {e}")
            return None
    
    @classmethod
    def get_author_profile(cls, author_id: Optional[str]) -> AuthorProfile:
        """
        作者プロファイルを取得
        
        Args:
            author_id: 作者ID
            
        Returns:
            AuthorProfile: 作者プロファイル（不明時はデフォルト）
        """
        if author_id and author_id in cls.AUTHOR_PROFILES:
            profile = cls.AUTHOR_PROFILES[author_id]
            logger.info(f"📋 Author profile loaded: {author_id} - {profile.description}")
            return profile
        else:
            if author_id:
                logger.warning(f"⚠️ Unknown author: {author_id}, using default profile")
            else:
                logger.info("📋 No author detected, using default profile")
            return cls.DEFAULT_PROFILE
    
    @classmethod
    def get_optimized_parameters(cls, author_id: Optional[str]) -> Dict[str, Any]:
        """
        作者に最適化されたパラメータを取得
        
        Args:
            author_id: 作者ID
            
        Returns:
            最適化パラメータの辞書
        """
        profile = cls.get_author_profile(author_id)
        
        parameters = {
            "sam_profile": profile.sam_profile,
            "yolo_confidence": profile.yolo_confidence,
            "max_masks": profile.max_masks,
            "score_threshold": profile.score_threshold,
            "author_id": profile.author_id,
            "characteristics": profile.characteristics.value,
            "description": profile.description,
            "processing_notes": profile.processing_notes
        }
        
        logger.info(f"⚙️ Optimized parameters for {profile.author_id}: "
                   f"SAM={profile.sam_profile}, YOLO={profile.yolo_confidence}, "
                   f"Masks={profile.max_masks}")
        
        return parameters
    
    @classmethod
    def apply_author_optimization(cls, image_path: str) -> Dict[str, Any]:
        """
        パスから作者を検出し、最適パラメータを返す
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            最適化パラメータの辞書
        """
        author_id = cls.detect_author_from_path(image_path)
        parameters = cls.get_optimized_parameters(author_id)
        
        logger.info(f"🎯 Author optimization applied for {image_path}: "
                   f"Author={author_id or 'default'}")
        
        return parameters
    
    @classmethod
    def get_all_authors(cls) -> List[str]:
        """
        対応している全作者IDのリストを取得
        
        Returns:
            作者IDのリスト
        """
        return list(cls.AUTHOR_PROFILES.keys())
    
    @classmethod
    def get_author_statistics(cls) -> Dict[str, Any]:
        """
        作者別統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            "total_authors": len(cls.AUTHOR_PROFILES),
            "characteristics_distribution": {},
            "sam_profiles": {},
            "confidence_range": {
                "min": min(p.yolo_confidence for p in cls.AUTHOR_PROFILES.values()),
                "max": max(p.yolo_confidence for p in cls.AUTHOR_PROFILES.values()),
                "avg": sum(p.yolo_confidence for p in cls.AUTHOR_PROFILES.values()) / len(cls.AUTHOR_PROFILES)
            }
        }
        
        # 特性分布
        for profile in cls.AUTHOR_PROFILES.values():
            char = profile.characteristics.value
            stats["characteristics_distribution"][char] = stats["characteristics_distribution"].get(char, 0) + 1
            
            sam_prof = profile.sam_profile
            stats["sam_profiles"][sam_prof] = stats["sam_profiles"].get(sam_prof, 0) + 1
        
        return stats


def test_author_detection():
    """テスト用関数"""
    test_paths = [
        "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg",
        "/mnt/c/AItools/lora/train/aichi/org/work01/image.jpg", 
        "/path/to/train/zundamon/org/test/test.png",
        "/invalid/path/image.jpg"
    ]
    
    print("🧪 Author Detection Test")
    print("=" * 40)
    
    for path in test_paths:
        author = AuthorParameterAdapter.detect_author_from_path(path)
        params = AuthorParameterAdapter.apply_author_optimization(path)
        print(f"Path: {path}")
        print(f"Author: {author}")
        print(f"Profile: {params['sam_profile']}")
        print(f"Confidence: {params['yolo_confidence']}")
        print()


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    test_author_detection()
    
    # 統計表示
    stats = AuthorParameterAdapter.get_author_statistics()
    print("📊 Author Statistics:")
    print(f"Total Authors: {stats['total_authors']}")
    print(f"Characteristics: {stats['characteristics_distribution']}")
    print(f"Confidence Range: {stats['confidence_range']['min']:.3f} - {stats['confidence_range']['max']:.3f}")