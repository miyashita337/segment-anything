#!/usr/bin/env python3
"""
環境依存パス管理システム

大規模コードベースでの/mnt/c問題を根本解決するための統一管理モジュール
CI環境・ローカル環境の差異を吸収し、205個のファイル個別修正を不要にする
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """環境依存設定統一管理クラス"""
    
    def __init__(self):
        """環境検出と初期設定"""
        self.is_ci = self._detect_ci_environment()
        self.base_paths = self._initialize_base_paths()
        self.temp_dirs = {}  # 一時ディレクトリキャッシュ
        
        logger.info(f"Environment: {'CI' if self.is_ci else 'Local'}")
    
    def _detect_ci_environment(self) -> bool:
        """CI環境検出"""
        return (
            os.getenv('CI_ENVIRONMENT') == 'true' or
            os.getenv('GITHUB_ACTIONS') == 'true' or
            os.getenv('CI') == 'true' or
            not os.path.exists('/mnt/c')
        )
    
    def _initialize_base_paths(self) -> Dict[str, str]:
        """基本パス初期化"""
        if self.is_ci:
            # CI環境用パス
            temp_base = tempfile.mkdtemp(prefix="segment_anything_ci_")
            return {
                'project_root': os.getcwd(),
                'data_base': temp_base,
                'models_base': temp_base + '/models',
                'output_base': temp_base + '/output',
                'test_base': temp_base + '/test_data'
            }
        else:
            # ローカル環境用パス
            return {
                'project_root': '/mnt/c/AItools/segment-anything',
                'data_base': '/mnt/c/AItools/lora/train/yado',
                'models_base': '/mnt/c/AItools/segment-anything',
                'output_base': '/mnt/c/AItools/lora/train/yado/tracker-workspace',
                'test_base': '/mnt/c/AItools/segment-anything/test_small'
            }
    
    def get_path(self, path_type: str, *sub_paths: str) -> str:
        """統一パス取得インターフェース
        
        Args:
            path_type: 'data', 'models', 'output', 'test' など
            *sub_paths: サブディレクトリパス
        
        Returns:
            環境に適したフルパス
        """
        base_key = f"{path_type}_base"
        
        if base_key not in self.base_paths:
            raise ValueError(f"Unknown path type: {path_type}")
        
        base_path = Path(self.base_paths[base_key])
        
        # サブパスを結合
        full_path = base_path
        for sub_path in sub_paths:
            full_path = full_path / sub_path
        
        # CI環境では必要に応じてディレクトリ作成
        if self.is_ci and path_type in ['output', 'test']:
            full_path.mkdir(parents=True, exist_ok=True)
        
        return str(full_path)
    
    def get_test_image_path(self, dataset: str, filename: str) -> str:
        """テスト用画像パス取得"""
        if self.is_ci:
            # CI環境ではダミー画像作成
            return self._create_dummy_image(dataset, filename)
        else:
            # ローカル環境では実際のパス
            return self.get_path('data', 'org', dataset, filename)
    
    def _create_dummy_image(self, dataset: str, filename: str) -> str:
        """CI用ダミー画像作成"""
        test_dir = Path(self.get_path('test', dataset))
        test_dir.mkdir(parents=True, exist_ok=True)
        
        dummy_path = test_dir / filename
        
        if not dummy_path.exists():
            # 512x512のダミー画像作成
            import numpy as np
            import cv2
            
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(dummy_path), dummy_image)
            logger.debug(f"Created dummy image: {dummy_path}")
        
        return str(dummy_path)
    
    def get_output_workspace(self, tracker_id: str) -> str:
        """トラッカーワークスペースパス取得"""
        return self.get_path('output', tracker_id)
    
    def setup_test_environment(self, test_name: str) -> Dict[str, str]:
        """テスト環境セットアップ
        
        Returns:
            テスト用パス辞書
        """
        if test_name in self.temp_dirs:
            return self.temp_dirs[test_name]
        
        if self.is_ci:
            temp_dir = tempfile.mkdtemp(prefix=f"{test_name}_")
            paths = {
                'input_dir': temp_dir + '/input',
                'output_dir': temp_dir + '/output',
                'test_images': temp_dir + '/test_images'
            }
            
            # 必要ディレクトリ作成
            for path in paths.values():
                Path(path).mkdir(parents=True, exist_ok=True)
            
            self.temp_dirs[test_name] = paths
            return paths
        else:
            # ローカル環境用パス
            return {
                'input_dir': self.get_path('data', 'org', 'kana05'),
                'output_dir': self.get_path('output', test_name),
                'test_images': self.get_path('test')
            }
    
    def cleanup_test_environment(self, test_name: str):
        """テスト環境クリーンアップ"""
        if test_name in self.temp_dirs and self.is_ci:
            import shutil
            try:
                # CI環境では一時ディレクトリを削除
                temp_paths = self.temp_dirs[test_name]
                for path in temp_paths.values():
                    if Path(path).exists():
                        shutil.rmtree(path, ignore_errors=True)
                del self.temp_dirs[test_name]
                logger.debug(f"Cleaned up test environment: {test_name}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {test_name}: {e}")


# グローバルインスタンス（シングルトンパターン）
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """環境マネージャーシングルトン取得"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


# 便利関数群
def get_path(path_type: str, *sub_paths: str) -> str:
    """統一パス取得（便利関数）"""
    return get_environment_manager().get_path(path_type, *sub_paths)


def is_ci_environment() -> bool:
    """CI環境判定（便利関数）"""
    return get_environment_manager().is_ci


def get_test_image_path(dataset: str, filename: str) -> str:
    """テスト画像パス取得（便利関数）"""
    return get_environment_manager().get_test_image_path(dataset, filename)


def setup_test_env(test_name: str) -> Dict[str, str]:
    """テスト環境セットアップ（便利関数）"""
    return get_environment_manager().setup_test_environment(test_name)


def cleanup_test_env(test_name: str):
    """テスト環境クリーンアップ（便利関数）"""
    return get_environment_manager().cleanup_test_environment(test_name)