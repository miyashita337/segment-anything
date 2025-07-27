#!/usr/bin/env python3
"""
標準出力パス管理システム
仕様書準拠の出力パス生成と検証機能を提供
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OutputCategory(Enum):
    """出力カテゴリ定義"""
    EXTRACTION = "extractions"
    QUALITY_REPORT = "quality_reports"
    TEST_RESULT = "test_results"
    DASHBOARD = "dashboard"
    TEMP = "temp"


@dataclass
class WorkspaceStructure:
    """ワークスペース構造定義"""
    base_path: Path
    tracker_id: str
    
    def __post_init__(self):
        """初期化後処理"""
        if not self.tracker_id:
            raise ValueError("tracker_id is required")
        
        # トラッカーIDの形式チェック
        if not self._validate_tracker_id(self.tracker_id):
            raise ValueError(f"Invalid tracker_id format: {self.tracker_id}")
    
    def _validate_tracker_id(self, tracker_id: str) -> bool:
        """トラッカーID形式検証"""
        # 基本パターン: PH1-001, PH2-002, baseline, 等
        valid_patterns = [
            tracker_id.startswith("PH"),  # PHで始まる
            tracker_id == "baseline",
            tracker_id == "backup",
            tracker_id == "AUDIT",  # 監査用特別ID
            tracker_id == "TEST",   # テスト用特別ID
            "-" in tracker_id or tracker_id in ["baseline", "backup", "AUDIT", "TEST"]
        ]
        return any(valid_patterns)


class OutputPathManager:
    """標準出力パス管理クラス"""
    
    # 仕様書準拠のベースパス
    WORKSPACE_BASE = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
    
    def __init__(self, tracker_id: str, base_path: Optional[Path] = None):
        """
        初期化
        
        Args:
            tracker_id: トラッカーID (例: PH2-002, baseline)
            base_path: カスタムベースパス（テスト用）
        """
        self.tracker_id = tracker_id
        self.base_path = base_path or self.WORKSPACE_BASE
        
        # ワークスペース構造初期化
        self.workspace = WorkspaceStructure(
            base_path=self.base_path,
            tracker_id=tracker_id
        )
        
        # 環境変数からの設定上書き
        env_base = os.getenv('WORKSPACE_BASE')
        if env_base:
            self.base_path = Path(env_base)
            logger.info(f"Using WORKSPACE_BASE from environment: {env_base}")
    
    def get_tracker_root(self) -> Path:
        """トラッカー専用ルートディレクトリ取得"""
        return self.base_path / self.tracker_id
    
    def get_output_path(
        self, 
        category: OutputCategory, 
        subcategory: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        標準出力パス生成
        
        Args:
            category: 出力カテゴリ
            subcategory: サブカテゴリ（オプション）
            filename: ファイル名（オプション）
            
        Returns:
            完全な出力パス
            
        Example:
            get_output_path(OutputCategory.DASHBOARD, filename="report.html")
            → /workspace/PH2-002/dashboard/report.html
        """
        # ベースパス構築
        path = self.get_tracker_root()
        
        # カテゴリ別パス設定
        if category == OutputCategory.DASHBOARD:
            path = path / "dashboard"
        elif category == OutputCategory.EXTRACTION:
            path = path / "extraction"
        elif category == OutputCategory.QUALITY_REPORT:
            path = path / "quality"
        elif category == OutputCategory.TEST_RESULT:
            path = path / "tests"
        elif category == OutputCategory.TEMP:
            path = path / "temp"
        else:
            # フォールバック
            path = path / category.value
        
        # サブカテゴリ追加
        if subcategory:
            path = path / subcategory
        
        # ファイル名追加
        if filename:
            path = path / filename
        
        return path
    
    def ensure_output_dir(
        self, 
        category: OutputCategory, 
        subcategory: Optional[str] = None
    ) -> Path:
        """
        出力ディレクトリ確保（作成）
        
        Args:
            category: 出力カテゴリ
            subcategory: サブカテゴリ（オプション）
            
        Returns:
            作成されたディレクトリパス
        """
        output_dir = self.get_output_path(category, subcategory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ensured output directory: {output_dir}")
        return output_dir
    
    def validate_compliance(self) -> Dict[str, Any]:
        """
        仕様書準拠チェック
        
        Returns:
            検証結果の詳細
        """
        results = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # ベースパスチェック
        if not self.base_path.exists():
            results["issues"].append(f"Base workspace path does not exist: {self.base_path}")
            results["compliant"] = False
        
        # トラッカーIDチェック
        if not self.workspace._validate_tracker_id(self.tracker_id):
            results["issues"].append(f"Invalid tracker_id format: {self.tracker_id}")
            results["compliant"] = False
        
        # 権限チェック
        try:
            test_dir = self.get_output_path(OutputCategory.TEMP, "test")
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / "permission_test.tmp"
            test_file.write_text("test")
            test_file.unlink()
            test_dir.rmdir()
        except Exception as e:
            results["issues"].append(f"No write permission: {e}")
            results["compliant"] = False
        
        # 推奨事項
        if self.tracker_id.startswith("test"):
            results["recommendations"].append("Consider using TEMP category for test outputs")
        
        return results
    
    def get_standard_structure(self) -> Dict[str, Path]:
        """
        標準ディレクトリ構造取得
        
        Returns:
            標準ディレクトリパス一覧
        """
        return {
            "root": self.get_tracker_root(),
            "dashboard": self.get_output_path(OutputCategory.DASHBOARD),
            "extraction": self.get_output_path(OutputCategory.EXTRACTION),
            "quality": self.get_output_path(OutputCategory.QUALITY_REPORT),
            "tests": self.get_output_path(OutputCategory.TEST_RESULT),
            "temp": self.get_output_path(OutputCategory.TEMP)
        }
    
    def create_standard_structure(self) -> List[Path]:
        """
        標準ディレクトリ構造作成
        
        Returns:
            作成されたディレクトリ一覧
        """
        created_dirs = []
        structure = self.get_standard_structure()
        
        for name, path in structure.items():
            if name != "root":  # ルートは自動作成される
                self.ensure_output_dir(OutputCategory(name) if name in [e.value for e in OutputCategory] else OutputCategory.TEMP)
                created_dirs.append(path)
        
        logger.info(f"Created standard directory structure for {self.tracker_id}")
        return created_dirs
    
    @classmethod
    def detect_existing_trackers(cls, base_path: Optional[Path] = None) -> List[str]:
        """
        既存トラッカーID検出
        
        Args:
            base_path: 検索ベースパス
            
        Returns:
            発見されたトラッカーID一覧
        """
        search_path = base_path or cls.WORKSPACE_BASE
        trackers = []
        
        if search_path.exists():
            for item in search_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # トラッカーIDとして有効か検証
                    try:
                        workspace = WorkspaceStructure(search_path, item.name)
                        trackers.append(item.name)
                    except ValueError:
                        # 無効なトラッカーIDはスキップ
                        pass
        
        return sorted(trackers)
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"OutputPathManager(tracker_id='{self.tracker_id}', base='{self.base_path}')"
    
    def __repr__(self) -> str:
        """デバッグ表現"""
        return self.__str__()


# ユーティリティ関数
def get_workspace_manager(tracker_id: str) -> OutputPathManager:
    """ワークスペースマネージャー取得（シングルトン風）"""
    return OutputPathManager(tracker_id)


def ensure_compliant_output(tracker_id: str, category: OutputCategory, filename: str) -> Path:
    """
    仕様書準拠出力パス取得・ディレクトリ作成
    
    Args:
        tracker_id: トラッカーID
        category: 出力カテゴリ
        filename: ファイル名
        
    Returns:
        完全な出力ファイルパス
    """
    manager = OutputPathManager(tracker_id)
    
    # 仕様書準拠チェック
    compliance = manager.validate_compliance()
    if not compliance["compliant"]:
        logger.warning(f"Compliance issues detected: {compliance['issues']}")
    
    # ディレクトリ確保
    manager.ensure_output_dir(category)
    
    # 完全パス生成
    return manager.get_output_path(category, filename=filename)


if __name__ == "__main__":
    # 使用例とテスト
    print("=== Output Path Manager Test ===")
    
    # 基本使用例
    manager = OutputPathManager("PH2-002")
    print(f"Manager: {manager}")
    
    # パス生成例
    dashboard_path = manager.get_output_path(OutputCategory.DASHBOARD, filename="report.html")
    print(f"Dashboard path: {dashboard_path}")
    
    # 仕様書準拠チェック
    compliance = manager.validate_compliance()
    print(f"Compliance: {compliance}")
    
    # 既存トラッカー検出
    trackers = OutputPathManager.detect_existing_trackers()
    print(f"Existing trackers: {trackers}")