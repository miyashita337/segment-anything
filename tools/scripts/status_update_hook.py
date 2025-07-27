#!/usr/bin/env python3
"""
ステータス変更フック統一インターフェース
全プロセスでGoogle Sheets自動更新を実現

技術仕様: ../../spec/GOOGLE_SHEETS_INTEGRATION.md
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# progress_tracker統合
sys.path.append(str(Path(__file__).parent))
try:
    from progress_tracker.data_models import TaskStatus, ComponentStatus, TaskRecord, PriorityLevel
    from google_sheets_updater import GoogleSheetsUpdater, update_progress_tracker_record
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False
    logging.warning("progress_tracker統合機能が利用できません")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatusUpdateHook:
    """ステータス変更フック統一インターフェース"""
    
    def __init__(self, tracker_id: str = "PH2-002", priority: Optional[PriorityLevel] = None):
        """
        初期化
        
        Args:
            tracker_id: トラッカーID（デフォルト: PH2-002）
            priority: 優先度レベル（デフォルト: 優先度中）
        """
        self.tracker_id = tracker_id
        self.priority = priority or PriorityLevel.MEDIUM
        self.is_available = PROGRESS_TRACKER_AVAILABLE
        
        if not self.is_available:
            logger.warning(f"Google Sheets更新機能無効: {tracker_id}")
    
    def update_status(self, status: str, description: str = "") -> bool:
        """
        メインステータス更新
        
        Args:
            status: ステータス文字列（TaskStatus対応）
            description: 説明文
            
        Returns:
            bool: 更新成功フラグ
        """
        if not self.is_available:
            return False
            
        try:
            # TaskStatus変換
            if status == "開始":
                task_status = TaskStatus.IN_PROGRESS
            elif status == "実装完了":
                task_status = TaskStatus.IMPLEMENTATION_DONE
            elif status == "動作確認":
                task_status = TaskStatus.OPERATION_CHECK
            elif status == "品質チェック":
                task_status = TaskStatus.QUALITY_CHECK
            elif status == "抽出パイプライン":
                task_status = TaskStatus.EXTRACTION_PIPELINE
            elif status == "完了":
                task_status = TaskStatus.RELEASE
            else:
                task_status = TaskStatus.IN_PROGRESS
            
            # 仮の品質データ作成（実際のデータがない場合）
            quality_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": description or "処理中",
                "total_images": 0,
                "evaluation_metrics": [],
                "mask_metrics": [],
                "objective_metrics": [],
                "overall_score": 0.0,
                "passed_metrics": 0,
                "total_metrics": 0,
                "status": "IN_PROGRESS",
                "priority_improvements": []
            }
            
            # Google Sheets更新
            success = update_progress_tracker_record(self.tracker_id, quality_data, self.priority)
            
            if success:
                logger.info(f"ステータス更新成功: {self.tracker_id} -> {status}")
            else:
                logger.warning(f"ステータス更新失敗: {self.tracker_id} -> {status}")
                
            return success
            
        except Exception as e:
            logger.error(f"ステータス更新エラー: {e}")
            return False
    
    def update_component_status(self, component: str, status: str) -> bool:
        """
        コンポーネント別ステータス更新
        
        Args:
            component: コンポーネント名（operation_check, quality_evaluation等）
            status: ステータス（完了/失敗/実行中等）
            
        Returns:
            bool: 更新成功フラグ
        """
        if not self.is_available:
            return False
            
        try:
            # ComponentStatus変換
            if status == "完了":
                comp_status = ComponentStatus.COMPLETED
            elif status == "失敗":
                comp_status = ComponentStatus.FAILED
            elif status == "実行中":
                comp_status = ComponentStatus.IN_PROGRESS
            elif status == "スキップ":
                comp_status = ComponentStatus.SKIPPED
            else:
                comp_status = ComponentStatus.EMPTY
            
            logger.info(f"コンポーネントステータス更新: {self.tracker_id}.{component} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"コンポーネントステータス更新エラー: {e}")
            return False
    
    def update_extraction_start(self, dataset_name: str, total_images: int = 0) -> bool:
        """
        抽出開始時の更新
        
        Args:
            dataset_name: データセット名
            total_images: 総画像数
        """
        success = self.update_status("開始", f"抽出開始: {dataset_name}")
        if success:
            self.update_component_status("extraction_pipeline", "実行中")
        return success
    
    def update_extraction_complete(self, dataset_name: str, total_images: int, 
                                 success_count: int) -> bool:
        """
        抽出完了時の更新
        
        Args:
            dataset_name: データセット名
            total_images: 総画像数
            success_count: 成功数
        """
        success = self.update_status("抽出パイプライン", 
                                   f"抽出完了: {success_count}/{total_images}")
        if success:
            self.update_component_status("extraction_pipeline", "完了")
        return success
    
    def update_quality_check_start(self) -> bool:
        """品質チェック開始時の更新"""
        success = self.update_status("品質チェック", "品質評価実行中")
        if success:
            self.update_component_status("quality_evaluation", "実行中")
        return success
    
    def update_quality_check_complete(self, quality_data: Dict[str, Any]) -> bool:
        """
        品質チェック完了時の更新（実際の品質データ使用）
        
        Args:
            quality_data: 統合品質レポートデータ
        """
        try:
            success = update_progress_tracker_record(self.tracker_id, quality_data, self.priority)
            if success:
                self.update_component_status("quality_evaluation", "完了")
                self.update_component_status("dashboard_generation", "完了")
                logger.info(f"品質チェック完了更新: {self.tracker_id}")
            return success
        except Exception as e:
            logger.error(f"品質チェック完了更新エラー: {e}")
            return False
    
    def update_error_status(self, error_type: str, error_message: str) -> bool:
        """
        エラー発生時の更新
        
        Args:
            error_type: エラータイプ
            error_message: エラーメッセージ
        """
        return self.update_status("エラー", f"{error_type}: {error_message}")


# 便利関数
def create_hook(tracker_id: str = "PH2-002", priority: Optional[PriorityLevel] = None) -> StatusUpdateHook:
    """ステータス更新フック作成"""
    return StatusUpdateHook(tracker_id, priority)


def update_extraction_status(tracker_id: str, status: str, **kwargs) -> bool:
    """抽出ステータス更新"""
    hook = create_hook(tracker_id)
    
    if status == "start":
        return hook.update_extraction_start(
            kwargs.get("dataset_name", ""),
            kwargs.get("total_images", 0)
        )
    elif status == "complete":
        return hook.update_extraction_complete(
            kwargs.get("dataset_name", ""),
            kwargs.get("total_images", 0),
            kwargs.get("success_count", 0)
        )
    elif status == "error":
        return hook.update_error_status(
            kwargs.get("error_type", "UnknownError"),
            kwargs.get("error_message", "")
        )
    else:
        return hook.update_status(status, kwargs.get("description", ""))


def update_quality_status(tracker_id: str, quality_data: Dict[str, Any]) -> bool:
    """品質チェックステータス更新"""
    hook = create_hook(tracker_id)
    return hook.update_quality_check_complete(quality_data)


if __name__ == "__main__":
    # テスト実行
    hook = create_hook()
    
    # テスト1: 抽出開始
    print("テスト1: 抽出開始")
    hook.update_extraction_start("test_dataset", 5)
    
    # テスト2: 抽出完了
    print("テスト2: 抽出完了")
    hook.update_extraction_complete("test_dataset", 5, 4)
    
    # テスト3: 品質チェック開始
    print("テスト3: 品質チェック開始")
    hook.update_quality_check_start()