#!/usr/bin/env python3
"""
PlanCommandHandler テストスイート
KIRO-006 Phase 2: ワークフロー計画・起票システムのテスト

PlanCommandHandlerの入力検証、Google Sheets統合、
エラーハンドリングの包括的テストを提供します。
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools.workflow.plan_command_handler import PlanCommandHandler
from tools.progress_tracker.data_models import TaskStatus, PriorityLevel, TaskRecord


class TestPlanCommandHandler(unittest.TestCase):
    """PlanCommandHandlerテストクラス"""
    
    def setUp(self):
        """テスト前準備"""
        self.handler = PlanCommandHandler()
        # ProgressManagerをモック化
        self.handler.progress_manager = Mock()
    
    def test_validate_inputs_success(self):
        """入力検証成功テスト"""
        # 正常な入力
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_inputs_empty_tracker_id(self):
        """トラッカーID空文字テスト"""
        is_valid, error = self.handler.validate_inputs(
            "", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("トラッカーIDが指定されていません", error)
    
    def test_validate_inputs_empty_summary(self):
        """概要空文字テスト"""
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001", 
            "", 
            "テスト詳細説明"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("概要が指定されていません", error)
    
    def test_validate_inputs_empty_details(self):
        """詳細空文字テスト"""
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001", 
            "テスト概要", 
            ""
        )
        
        self.assertFalse(is_valid)
        self.assertIn("詳細が指定されていません", error)
    
    def test_validate_inputs_whitespace_only(self):
        """空白文字のみテスト"""
        is_valid, error = self.handler.validate_inputs(
            "   ", 
            "   ", 
            "   "
        )
        
        self.assertFalse(is_valid)
        self.assertIn("トラッカーIDが指定されていません", error)
    
    def test_validate_inputs_details_too_long(self):
        """詳細文字数制限テスト"""
        long_details = "a" * (PlanCommandHandler.MAX_DETAILS_LENGTH + 1)
        
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001", 
            "テスト概要", 
            long_details
        )
        
        self.assertFalse(is_valid)
        self.assertIn("詳細が文字数制限を超えています", error)
        self.assertIn(f"{PlanCommandHandler.MAX_DETAILS_LENGTH + 1:,}文字", error)
    
    def test_validate_inputs_details_max_length(self):
        """詳細最大文字数境界値テスト"""
        max_details = "a" * PlanCommandHandler.MAX_DETAILS_LENGTH
        
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001", 
            "テスト概要", 
            max_details
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_tracker_id_format_valid(self):
        """トラッカーID形式検証（正常）"""
        valid_ids = [
            "TRACKER-001",
            "KIRO-006", 
            "QUAL-044",
            "A-1",
            "TEST123-999"
        ]
        
        for tracker_id in valid_ids:
            with self.subTest(tracker_id=tracker_id):
                result = self.handler._validate_tracker_id_format(tracker_id)
                self.assertTrue(result, f"Valid ID should pass: {tracker_id}")
    
    def test_validate_tracker_id_format_invalid(self):
        """トラッカーID形式検証（異常）"""
        invalid_ids = [
            "tracker-001",  # 小文字
            "TRACKER_001",  # アンダースコア
            "TRACKER-",     # 番号なし
            "-001",         # プレフィックスなし
            "TRACKER001",   # ハイフンなし
            "TRACKER-ABC",  # 番号が文字
            "123-TRACKER",  # 数字から開始
            "",             # 空文字
            "TRACKER-01-02" # 複数ハイフン
        ]
        
        for tracker_id in invalid_ids:
            with self.subTest(tracker_id=tracker_id):
                result = self.handler._validate_tracker_id_format(tracker_id)
                self.assertFalse(result, f"Invalid ID should fail: {tracker_id}")
    
    def test_check_existing_tracker_not_exists(self):
        """既存トラッカー確認（存在しない）"""
        # get_taskがNoneを返すようにモック設定
        self.handler.progress_manager.get_task.return_value = None
        
        exists, info = self.handler.check_existing_tracker("TRACKER-001")
        
        self.assertFalse(exists)
        self.assertIsNone(info)
        self.handler.progress_manager.get_task.assert_called_once_with("TRACKER-001")
    
    def test_check_existing_tracker_exists(self):
        """既存トラッカー確認（存在する）"""
        # モックタスクレコード作成
        mock_task = Mock()
        mock_task.tracker_id = "TRACKER-001"
        mock_task.status = TaskStatus.NOT_STARTED
        mock_task.created_date = "2025-01-06"
        mock_task.description = "既存のテストタスク"
        
        self.handler.progress_manager.get_task.return_value = mock_task
        
        exists, info = self.handler.check_existing_tracker("TRACKER-001")
        
        self.assertTrue(exists)
        self.assertIsNotNone(info)
        self.assertEqual(info['tracker_id'], "TRACKER-001")
        self.assertEqual(info['status'], TaskStatus.NOT_STARTED.value)
        self.assertEqual(info['created_date'], "2025-01-06")
        self.assertEqual(info['description'], "既存のテストタスク")
    
    def test_check_existing_tracker_error(self):
        """既存トラッカー確認（エラー）"""
        # get_taskでエラーが発生するようにモック設定
        self.handler.progress_manager.get_task.side_effect = Exception("API Error")
        
        exists, info = self.handler.check_existing_tracker("TRACKER-001")
        
        self.assertFalse(exists)
        self.assertIsNone(info)
    
    def test_create_google_sheets_tracker_success(self):
        """Google Sheetsトラッカー作成成功テスト"""
        # モックタスクレコード作成
        mock_task = Mock()
        mock_task.tracker_id = "TRACKER-001"
        mock_task.status = TaskStatus.NOT_STARTED
        mock_task.created_date = "2025-01-06 10:00:00"
        
        self.handler.progress_manager.create_task.return_value = mock_task
        
        success, message = self.handler.create_google_sheets_tracker(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertTrue(success)
        self.assertIn("Google Sheetsにトラッカーを起票しました", message)
        self.assertIn("TRACKER-001", message)
        self.assertIn("テスト概要", message)
        
        # create_taskが正しい引数で呼ばれたか確認
        self.handler.progress_manager.create_task.assert_called_once()
        call_args = self.handler.progress_manager.create_task.call_args
        self.assertEqual(call_args[1]['tracker_id'], "TRACKER-001")
        self.assertIn("テスト概要", call_args[1]['description'])
        self.assertIn("テスト詳細説明", call_args[1]['description'])
    
    def test_create_google_sheets_tracker_no_manager(self):
        """Google Sheetsトラッカー作成（マネージャーなし）"""
        self.handler.progress_manager = None
        
        success, message = self.handler.create_google_sheets_tracker(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertFalse(success)
        self.assertIn("Google Sheets連携が利用できません", message)
    
    def test_create_google_sheets_tracker_api_error(self):
        """Google Sheetsトラッカー作成（APIエラー）"""
        # create_taskでエラーが発生するようにモック設定
        self.handler.progress_manager.create_task.side_effect = Exception("API Connection Error")
        
        success, message = self.handler.create_google_sheets_tracker(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertFalse(success)
        self.assertIn("Google Sheets起票に失敗しました", message)
        self.assertIn("API Connection Error", message)
        self.assertIn("トラブルシューティング", message)
    
    def test_execute_plan_command_success(self):
        """planコマンド実行成功テスト"""
        # 作成成功
        mock_task = Mock()
        mock_task.tracker_id = "TRACKER-001"
        mock_task.status = TaskStatus.NOT_STARTED
        mock_task.created_date = "2025-01-06 10:00:00"
        self.handler.progress_manager.create_task.return_value = mock_task
        
        # get_taskの戻り値を設定（最初はNone、作成後は作成されたタスク）
        self.handler.progress_manager.get_task.side_effect = [None, mock_task]
        
        success, message = self.handler.execute_plan_command(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertTrue(success)
        self.assertIn("Google Sheetsにトラッカーを起票しました", message)
    
    def test_execute_plan_command_validation_error(self):
        """planコマンド実行（入力検証エラー）"""
        success, message = self.handler.execute_plan_command(
            "", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertFalse(success)
        self.assertIn("トラッカーIDが指定されていません", message)
    
    def test_execute_plan_command_existing_tracker(self):
        """planコマンド実行（既存トラッカー）"""
        # 既存トラッカーあり
        mock_task = Mock()
        mock_task.tracker_id = "TRACKER-001"
        mock_task.status = TaskStatus.IN_PROGRESS
        mock_task.created_date = "2025-01-05"
        mock_task.description = "既存のタスク"
        self.handler.progress_manager.get_task.return_value = mock_task
        
        success, message = self.handler.execute_plan_command(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明"
        )
        
        self.assertFalse(success)
        self.assertIn("トラッカーが既に存在します", message)
        self.assertIn("TRACKER-001", message)
    
    def test_execute_plan_command_priority_mapping(self):
        """planコマンド実行（優先度マッピング）"""
        # 作成成功
        mock_task = Mock()
        mock_task.tracker_id = "TRACKER-001"
        mock_task.status = TaskStatus.NOT_STARTED
        mock_task.created_date = "2025-01-06 10:00:00"
        self.handler.progress_manager.create_task.return_value = mock_task
        
        # get_taskの戻り値を設定（最初はNone、作成後は作成されたタスク）
        self.handler.progress_manager.get_task.side_effect = [None, mock_task]
        
        # 高優先度で実行
        success, message = self.handler.execute_plan_command(
            "TRACKER-001", 
            "テスト概要", 
            "テスト詳細説明",
            "high"
        )
        
        self.assertTrue(success)
        self.assertIn("優先度高", message)
    
    def test_get_usage_help(self):
        """使用方法ヘルプテスト"""
        help_text = self.handler.get_usage_help()
        
        self.assertIn("planコマンド使用方法", help_text)
        self.assertIn("TRACKER_ID", help_text)
        self.assertIn("概要", help_text)
        self.assertIn("詳細", help_text)
        self.assertIn(f"{PlanCommandHandler.MAX_DETAILS_LENGTH:,}文字", help_text)
        self.assertIn("使用例", help_text)
    
    def test_japanese_characters(self):
        """日本語文字テスト"""
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001",
            "日本語の概要テスト",
            "これは日本語の詳細説明です。\n改行も含まれています。\n特殊文字：！？（）「」"
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_unicode_characters(self):
        """Unicode文字テスト"""
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001",
            "Unicode test: 🚀 ✅ ❌",
            "Emoji and symbols: 📋 🔧 ⚠️ 🎯"
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_boundary_values(self):
        """境界値テスト"""
        # 詳細が制限ちょうど
        max_details = "a" * PlanCommandHandler.MAX_DETAILS_LENGTH
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001",
            "境界値テスト",
            max_details
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # 詳細が制限+1
        over_details = "a" * (PlanCommandHandler.MAX_DETAILS_LENGTH + 1)
        is_valid, error = self.handler.validate_inputs(
            "TRACKER-001",
            "境界値テスト",
            over_details
        )
        
        self.assertFalse(is_valid)
        self.assertIn("文字数制限を超えています", error)


class TestPlanCommandHandlerIntegration(unittest.TestCase):
    """PlanCommandHandler統合テスト"""
    
    @patch('tools.workflow.plan_command_handler.ProgressManager')
    def test_initialization_with_config(self, mock_progress_manager_class):
        """設定ファイルでの初期化テスト"""
        # モック設定
        mock_config = Mock()
        mock_progress_manager = Mock()
        mock_progress_manager_class.return_value = mock_progress_manager
        
        with patch('tools.progress_tracker.config.get_default_config', return_value=mock_config):
            handler = PlanCommandHandler()
            
            self.assertIsNotNone(handler.progress_manager)
            mock_progress_manager_class.assert_called_once_with(mock_config)
    
    @patch('tools.workflow.plan_command_handler.ProgressManager')
    def test_initialization_error_handling(self, mock_progress_manager_class):
        """初期化エラーハンドリングテスト"""
        # 初期化でエラーが発生するようにモック設定
        mock_progress_manager_class.side_effect = Exception("Config Error")
        
        with patch('tools.progress_tracker.config.get_default_config', side_effect=Exception("Config Error")):
            handler = PlanCommandHandler()
            
            self.assertIsNone(handler.progress_manager)


if __name__ == '__main__':
    # ログレベル設定
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # テスト実行
    unittest.main(verbosity=2)