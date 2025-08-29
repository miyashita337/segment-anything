#!/usr/bin/env python3
"""
Level 4: 承認ワークフローテストシステム

承認プロセス、通知、進捗管理の統合テスト
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

# Mock systems import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.mocks.mock_pushover import (
    get_mock_pushover_client, 
    reset_mock_pushover_client, 
    MockPushoverClient
)
from tests.mocks.mock_approval_system import (
    get_mock_approval_system,
    reset_mock_approval_system,
    MockApprovalSystem,
    ApprovalStatus,
    ApprovalStage
)
from tests.mocks.mock_progress_manager import (
    get_mock_progress_manager,
    reset_mock_progress_manager,
    MockProgressManager,
    TaskStatus,
    TaskPriority
)
from tests.mocks.mock_google_sheets import get_mock_sheets_client, reset_mock_sheets_client


class TestApprovalWorkflow:
    """Level 4: 承認ワークフロー統合テストクラス"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        # 全モックシステムをリセット
        reset_mock_pushover_client()
        reset_mock_approval_system()
        reset_mock_progress_manager()
        reset_mock_sheets_client()
        
        # モックインスタンス取得
        self.pushover_client = get_mock_pushover_client()
        self.approval_system = get_mock_approval_system()
        self.progress_manager = get_mock_progress_manager()
        self.sheets_client = get_mock_sheets_client()
        
    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        reset_mock_pushover_client()
        reset_mock_approval_system()
        reset_mock_progress_manager()
        reset_mock_sheets_client()

    # ==================== Pushover通知システムテスト ====================
    
    def test_pushover_basic_notification(self):
        """基本的な通知送信テスト"""
        result = self.pushover_client.send_notification(
            message="テスト通知メッセージ",
            title="テストタイトル",
            priority=1
        )
        
        assert result["status"] == 1
        assert "request" in result
        assert "message_id" in result
        
        # 送信済みメッセージ確認
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 1
        assert messages[0].message == "テスト通知メッセージ"
        assert messages[0].title == "テストタイトル"
        assert messages[0].priority == 1
    
    def test_pushover_extraction_complete_notification(self):
        """抽出完了通知テスト"""
        result = self.pushover_client.send_extraction_complete_notification(
            tracker_id="TEST-001",
            total_images=50,
            successful_extractions=45,
            success_rate=90.0,
            workspace_path="/workspace/TEST-001",
            attachment_images=["image1.jpg", "image2.jpg"]
        )
        
        assert result["status"] == 1
        
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 1
        
        message = messages[0]
        assert "TEST-001 抽出完了" in message.message
        assert "総画像数: 50枚" in message.message
        assert "成功抽出: 45枚" in message.message
        assert "成功率: 90.0%" in message.message
        assert message.title == "Claude Code - TEST-001 Complete"
        assert message.priority == 1
    
    def test_pushover_approval_request_notification(self):
        """承認依頼通知テスト"""
        result = self.pushover_client.send_approval_request(
            tracker_id="TEST-002",
            stage="quality_check",
            details="品質チェック完了。承認をお願いします。",
            approval_url="http://example.com/approve/123"
        )
        
        assert result["status"] == 1
        
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 1
        
        message = messages[0]
        assert "TEST-002 承認依頼" in message.message
        assert "ステージ: quality_check" in message.message
        assert "品質チェック完了。承認をお願いします。" in message.message
        assert "http://example.com/approve/123" in message.message
        assert message.title == "承認依頼 - TEST-002"
        assert message.priority == 1
    
    def test_pushover_quality_alert_notification(self):
        """品質アラート通知テスト"""
        failed_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        
        result = self.pushover_client.send_quality_alert(
            tracker_id="TEST-003",
            quality_score=0.45,
            threshold=0.60,
            failed_images=failed_images
        )
        
        assert result["status"] == 1
        
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 1
        
        message = messages[0]
        assert "TEST-003 品質アラート" in message.message
        assert "品質スコア: 0.450" in message.message
        assert "閾値: 0.600" in message.message
        assert "失敗画像: 3件" in message.message
        assert "image1.jpg" in message.message
        assert message.title == "品質アラート - TEST-003"
        assert message.priority == 2
    
    def test_pushover_failure_mode(self):
        """Pushover失敗モードテスト"""
        self.pushover_client.set_failure_mode(True)
        
        result = self.pushover_client.send_notification(
            message="失敗テストメッセージ",
            title="失敗テスト"
        )
        
        assert result["status"] == 0
        assert "error" in result
        assert result["error"] == "Mock API failure"
        
        # 失敗時はメッセージが保存されない
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 0
        
        # 統計確認
        stats = self.pushover_client.get_api_statistics()
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 1
        assert stats["messages_sent"] == 0
    
    def test_pushover_message_search(self):
        """メッセージ検索機能テスト"""
        # 複数の通知を送信
        self.pushover_client.send_notification("テスト1", "承認依頼 - TEST-001")
        self.pushover_client.send_notification("テスト2", "Claude Code - TEST-002")
        self.pushover_client.send_notification("テスト3", "承認依頼 - TEST-003")
        
        # パターン検索テスト
        approval_messages = self.pushover_client.get_messages_by_title_pattern("承認依頼")
        assert len(approval_messages) == 2
        assert approval_messages[0].title == "承認依頼 - TEST-001"
        assert approval_messages[1].title == "承認依頼 - TEST-003"
        
        claude_messages = self.pushover_client.get_messages_by_title_pattern("Claude Code")
        assert len(claude_messages) == 1
        assert claude_messages[0].title == "Claude Code - TEST-002"

    # ==================== 承認システムテスト ====================
    
    def test_approval_request_creation(self):
        """承認依頼作成テスト"""
        request = self.approval_system.create_approval_request(
            tracker_id="TEST-004",
            stage=ApprovalStage.QUALITY_CHECK,
            title="品質チェック承認依頼",
            details="抽出品質が閾値を下回りました。手動確認をお願いします。",
            requestor="claude_code",
            expires_in_hours=24
        )
        
        assert request.request_id.startswith("APPR-")
        assert request.tracker_id == "TEST-004"
        assert request.stage == ApprovalStage.QUALITY_CHECK
        assert request.status == ApprovalStatus.PENDING
        assert request.approval_url is not None
        assert "http://100.123.241.106:8088/approval/" in request.approval_url
        
        # 有効期限確認（24時間後）
        expected_expiry = datetime.now() + timedelta(hours=24)
        time_diff = abs((request.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # 1分以内の誤差は許容
    
    def test_approval_request_approval(self):
        """承認依頼の承認テスト"""
        # 承認依頼作成
        request = self.approval_system.create_approval_request(
            tracker_id="TEST-005",
            stage=ApprovalStage.FINAL_APPROVAL,
            title="最終承認依頼",
            details="全プロセス完了。最終承認をお願いします。"
        )
        
        # 承認実行
        success = self.approval_system.approve_request(
            request.request_id,
            approver="test_user@example.com",
            comment="品質基準を満たしています。承認します。"
        )
        
        assert success is True
        
        # 承認結果確認
        updated_request = self.approval_system.get_request(request.request_id)
        assert updated_request.status == ApprovalStatus.APPROVED
        assert updated_request.approved_by == "test_user@example.com"
        assert updated_request.approved_at is not None
        assert updated_request.metadata["approval_comment"] == "品質基準を満たしています。承認します。"
    
    def test_approval_request_rejection(self):
        """承認依頼の拒否テスト"""
        # 承認依頼作成
        request = self.approval_system.create_approval_request(
            tracker_id="TEST-006",
            stage=ApprovalStage.QUALITY_CHECK,
            title="品質チェック承認依頼",
            details="品質チェック完了"
        )
        
        # 拒否実行
        success = self.approval_system.reject_request(
            request.request_id,
            rejector="test_manager@example.com",
            reason="品質基準を満たしていません。再処理が必要です。"
        )
        
        assert success is True
        
        # 拒否結果確認
        updated_request = self.approval_system.get_request(request.request_id)
        assert updated_request.status == ApprovalStatus.REJECTED
        assert updated_request.approved_by == "test_manager@example.com"
        assert updated_request.approved_at is not None
        assert updated_request.rejection_reason == "品質基準を満たしていません。再処理が必要です。"
    
    def test_approval_request_expiration(self):
        """承認依頼の期限切れテスト"""
        # 期限切れの承認依頼作成（過去時刻設定）
        request = self.approval_system.create_approval_request(
            tracker_id="TEST-007",
            stage=ApprovalStage.DEPLOYMENT,
            title="デプロイ承認依頼",
            details="デプロイ準備完了"
        )
        
        # 手動で有効期限を過去に設定
        request.expires_at = datetime.now() - timedelta(hours=1)
        
        # 期限切れ後の承認試行
        success = self.approval_system.approve_request(request.request_id)
        assert success is False
        
        # ステータス確認（期限切れに自動変更）
        updated_request = self.approval_system.get_request(request.request_id)
        assert updated_request.status == ApprovalStatus.EXPIRED
    
    def test_approval_auto_modes(self):
        """自動承認・拒否モードテスト"""
        # 自動承認モード
        self.approval_system.set_auto_approve_mode(True)
        
        request1 = self.approval_system.create_approval_request(
            tracker_id="TEST-008",
            stage=ApprovalStage.EXTRACTION_START,
            title="抽出開始承認",
            details="自動承認テスト"
        )
        
        assert request1.status == ApprovalStatus.APPROVED
        assert request1.approved_by == "auto_system"
        
        # 自動拒否モード
        self.approval_system.set_auto_reject_mode(True)
        
        request2 = self.approval_system.create_approval_request(
            tracker_id="TEST-009",
            stage=ApprovalStage.QUALITY_CHECK,
            title="品質チェック承認",
            details="自動拒否テスト"
        )
        
        assert request2.status == ApprovalStatus.REJECTED
        assert request2.approved_by == "auto_system"
        assert request2.rejection_reason == "Automatically rejected by mock system"
    
    def test_approval_stage_requirement_check(self):
        """承認要否判定テスト"""
        # 各ステージの承認要否確認
        assert self.approval_system.check_stage_approval_required(
            "TEST-010", ApprovalStage.EXTRACTION_START) is False
        
        assert self.approval_system.check_stage_approval_required(
            "TEST-010", ApprovalStage.QUALITY_CHECK) is True
        
        assert self.approval_system.check_stage_approval_required(
            "TEST-010", ApprovalStage.STATISTICAL_ANALYSIS) is True
        
        assert self.approval_system.check_stage_approval_required(
            "TEST-010", ApprovalStage.FINAL_APPROVAL) is True
        
        assert self.approval_system.check_stage_approval_required(
            "TEST-010", ApprovalStage.DEPLOYMENT) is True
    
    def test_approval_statistics(self):
        """承認統計情報テスト"""
        # 複数の承認依頼作成（異なるステータス）
        self.approval_system.create_approval_request(
            "TEST-011", ApprovalStage.QUALITY_CHECK, "承認1", "詳細1"
        )
        
        request2 = self.approval_system.create_approval_request(
            "TEST-012", ApprovalStage.FINAL_APPROVAL, "承認2", "詳細2"
        )
        self.approval_system.approve_request(request2.request_id)
        
        request3 = self.approval_system.create_approval_request(
            "TEST-013", ApprovalStage.DEPLOYMENT, "承認3", "詳細3"
        )
        self.approval_system.reject_request(request3.request_id, reason="テスト拒否")
        
        # 統計情報取得
        stats = self.approval_system.get_approval_statistics()
        
        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["approved"] == 1
        assert stats["by_status"]["rejected"] == 1
        assert stats["by_stage"]["quality_check"] == 1
        assert stats["by_stage"]["final_approval"] == 1
        assert stats["by_stage"]["deployment"] == 1
        
        # 処理時間統計の存在確認
        assert "average_processing_time_seconds" in stats
        assert "max_processing_time_seconds" in stats
        assert "min_processing_time_seconds" in stats

    # ==================== 進捗管理システムテスト ====================
    
    def test_progress_task_creation(self):
        """進捗タスク作成テスト"""
        task = self.progress_manager.create_task(
            tracker_id="TEST-014",
            title="画像抽出タスク",
            description="SAM+YOLO による画像抽出処理",
            priority=TaskPriority.HIGH,
            estimated_hours=2.5,
            assignee="claude_code"
        )
        
        assert task.task_id.startswith("TASK-")
        assert task.tracker_id == "TEST-014"
        assert task.title == "画像抽出タスク"
        assert task.status == TaskStatus.NOT_STARTED
        assert task.priority == TaskPriority.HIGH
        assert task.estimated_hours == 2.5
        assert task.assignee == "claude_code"
        assert task.progress_percentage == 0
    
    def test_progress_task_lifecycle(self):
        """進捗タスクライフサイクルテスト"""
        # タスク作成
        task = self.progress_manager.create_task(
            tracker_id="TEST-015",
            title="品質評価タスク",
            description="抽出結果の品質評価"
        )
        
        assert task.status == TaskStatus.NOT_STARTED
        
        # タスク開始
        success = self.progress_manager.start_task(task.task_id, assignee="evaluator")
        assert success is True
        
        updated_task = self.progress_manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.assignee == "evaluator"
        assert updated_task.started_at is not None
        
        # 進捗更新
        success = self.progress_manager.update_progress(task.task_id, 50)
        assert success is True
        
        updated_task = self.progress_manager.get_task(task.task_id)
        assert updated_task.progress_percentage == 50
        
        # タスク完了
        success = self.progress_manager.complete_task(task.task_id, actual_hours=1.8)
        assert success is True
        
        updated_task = self.progress_manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.progress_percentage == 100
        assert updated_task.actual_hours == 1.8
        assert updated_task.completed_at is not None
    
    def test_progress_task_dependencies(self):
        """進捗タスク依存関係テスト"""
        # 依存元タスク作成
        base_task = self.progress_manager.create_task(
            tracker_id="TEST-016",
            title="基礎タスク",
            description="依存元タスク"
        )
        
        # 依存先タスク作成
        dependent_task = self.progress_manager.create_task(
            tracker_id="TEST-016", 
            title="依存タスク",
            description="基礎タスクに依存するタスク",
            dependencies=[base_task.task_id]
        )
        
        # 依存元が未完了の状態で依存先を開始（ブロックされるはず）
        success = self.progress_manager.start_task(dependent_task.task_id)
        assert success is False
        
        updated_dependent_task = self.progress_manager.get_task(dependent_task.task_id)
        assert updated_dependent_task.status == TaskStatus.BLOCKED
        
        # 依存元タスクを完了
        self.progress_manager.start_task(base_task.task_id)
        self.progress_manager.complete_task(base_task.task_id)
        
        # 依存先タスクを開始（今度は成功するはず）
        success = self.progress_manager.start_task(dependent_task.task_id)
        assert success is True
        
        updated_dependent_task = self.progress_manager.get_task(dependent_task.task_id)
        assert updated_dependent_task.status == TaskStatus.IN_PROGRESS
    
    def test_progress_task_failure_and_cancellation(self):
        """進捗タスク失敗・キャンセルテスト"""
        # 失敗テスト
        task1 = self.progress_manager.create_task(
            tracker_id="TEST-017",
            title="失敗テストタスク",
            description="失敗をテストするタスク"
        )
        
        self.progress_manager.start_task(task1.task_id)
        success = self.progress_manager.fail_task(task1.task_id, "テスト失敗理由")
        assert success is True
        
        updated_task1 = self.progress_manager.get_task(task1.task_id)
        assert updated_task1.status == TaskStatus.FAILED
        assert updated_task1.metadata["failure_reason"] == "テスト失敗理由"
        assert "failed_at" in updated_task1.metadata
        
        # キャンセルテスト
        task2 = self.progress_manager.create_task(
            tracker_id="TEST-018",
            title="キャンセルテストタスク",
            description="キャンセルをテストするタスク"
        )
        
        success = self.progress_manager.cancel_task(task2.task_id, "テストキャンセル理由")
        assert success is True
        
        updated_task2 = self.progress_manager.get_task(task2.task_id)
        assert updated_task2.status == TaskStatus.CANCELLED
        assert updated_task2.metadata["cancellation_reason"] == "テストキャンセル理由"
        assert "cancelled_at" in updated_task2.metadata
    
    def test_progress_tracker_calculation(self):
        """トラッカー進捗計算テスト"""
        tracker_id = "TEST-019"
        
        # 複数タスク作成
        task1 = self.progress_manager.create_task(tracker_id, "タスク1", "説明1")
        task2 = self.progress_manager.create_task(tracker_id, "タスク2", "説明2")
        task3 = self.progress_manager.create_task(tracker_id, "タスク3", "説明3")
        task4 = self.progress_manager.create_task(tracker_id, "タスク4", "説明4")
        
        # 各タスクを異なる状態に設定
        self.progress_manager.start_task(task1.task_id)
        self.progress_manager.complete_task(task1.task_id)  # 100%完了
        
        self.progress_manager.start_task(task2.task_id)
        self.progress_manager.update_progress(task2.task_id, 75)  # 75%進捗
        
        self.progress_manager.start_task(task3.task_id)
        self.progress_manager.fail_task(task3.task_id, "失敗")  # 失敗
        
        # task4は未開始のまま
        
        # 進捗計算
        progress_info = self.progress_manager.calculate_tracker_progress(tracker_id)
        
        assert progress_info["tracker_id"] == tracker_id
        assert progress_info["total_tasks"] == 4
        assert progress_info["completed_tasks"] == 1
        assert progress_info["failed_tasks"] == 1
        assert progress_info["in_progress_tasks"] == 1
        assert progress_info["not_started_tasks"] == 1
        
        # 全体進捗率 = (100 + 75 + 0 + 0) / (4 * 100) = 175/400 = 43.75%
        expected_progress = 43.8  # 四捨五入
        assert abs(progress_info["overall_progress"] - expected_progress) < 0.1
    
    def test_progress_statistics(self):
        """進捗統計情報テスト"""
        # 複数のタスクを作成（異なる状態・優先度）
        task1 = self.progress_manager.create_task(
            "TEST-020", "高優先度タスク", "説明", priority=TaskPriority.HIGH
        )
        self.progress_manager.start_task(task1.task_id)
        self.progress_manager.complete_task(task1.task_id, actual_hours=2.0)
        
        task2 = self.progress_manager.create_task(
            "TEST-021", "中優先度タスク", "説明", priority=TaskPriority.MEDIUM
        )
        self.progress_manager.start_task(task2.task_id)
        self.progress_manager.update_progress(task2.task_id, 60)
        
        task3 = self.progress_manager.create_task(
            "TEST-022", "低優先度タスク", "説明", priority=TaskPriority.LOW
        )
        self.progress_manager.start_task(task3.task_id)
        self.progress_manager.fail_task(task3.task_id, "テスト失敗")
        
        # 統計情報取得
        stats = self.progress_manager.get_progress_statistics()
        
        assert stats["total_tasks"] == 3
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["in_progress"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_priority"]["low"] == 1
        
        # 作業時間統計
        assert "average_completion_hours" in stats
        assert stats["average_completion_hours"] == 2.0
        
        # 進捗率統計
        assert "average_progress_percentage" in stats
        assert stats["average_progress_percentage"] == 60.0

    # ==================== 統合ワークフローテスト ====================
    
    def test_integrated_approval_workflow(self):
        """統合承認ワークフローテスト"""
        tracker_id = "TEST-023"
        
        # 1. 進捗タスク作成
        extraction_task = self.progress_manager.create_task(
            tracker_id=tracker_id,
            title="画像抽出処理",
            description="SAM+YOLO抽出処理",
            priority=TaskPriority.HIGH
        )
        
        quality_task = self.progress_manager.create_task(
            tracker_id=tracker_id,
            title="品質評価処理",
            description="抽出結果品質評価",
            dependencies=[extraction_task.task_id]
        )
        
        # 2. 抽出開始（承認不要ステージ）
        assert not self.approval_system.check_stage_approval_required(
            tracker_id, ApprovalStage.EXTRACTION_START
        )
        
        self.progress_manager.start_task(extraction_task.task_id)
        self.progress_manager.update_progress(extraction_task.task_id, 100)
        
        # 抽出完了通知
        self.pushover_client.send_extraction_complete_notification(
            tracker_id=tracker_id,
            total_images=30,
            successful_extractions=28,
            success_rate=93.3,
            workspace_path=f"/workspace/{tracker_id}",
            attachment_images=["sample1.jpg", "sample2.jpg"]
        )
        
        # 3. 品質チェック承認依頼作成（承認必要ステージ）
        assert self.approval_system.check_stage_approval_required(
            tracker_id, ApprovalStage.QUALITY_CHECK
        )
        
        approval_request = self.approval_system.create_approval_request(
            tracker_id=tracker_id,
            stage=ApprovalStage.QUALITY_CHECK,
            title="品質チェック承認依頼",
            details="抽出処理完了。品質チェックの承認をお願いします。"
        )
        
        # 承認依頼通知
        self.pushover_client.send_approval_request(
            tracker_id=tracker_id,
            stage="quality_check",
            details="品質チェック完了。承認をお願いします。",
            approval_url=approval_request.approval_url
        )
        
        # 4. 承認実行
        self.approval_system.approve_request(
            approval_request.request_id,
            approver="quality_manager@example.com",
            comment="品質基準を満たしています。"
        )
        
        # 5. 品質タスク開始（依存関係解除済み）
        self.progress_manager.start_task(quality_task.task_id)
        self.progress_manager.complete_task(quality_task.task_id)
        
        # 6. 結果検証
        # 通知確認
        messages = self.pushover_client.get_sent_messages()
        assert len(messages) == 2  # 抽出完了 + 承認依頼
        
        extraction_msg = [m for m in messages if "抽出完了" in m.message][0]
        assert tracker_id in extraction_msg.message
        assert "成功率: 93.3%" in extraction_msg.message
        
        approval_msg = [m for m in messages if "承認依頼" in m.message][0]
        assert tracker_id in approval_msg.message
        assert "quality_check" in approval_msg.message
        
        # 承認状況確認
        final_request = self.approval_system.get_request(approval_request.request_id)
        assert final_request.status == ApprovalStatus.APPROVED
        assert final_request.approved_by == "quality_manager@example.com"
        
        # 進捗状況確認
        progress_info = self.progress_manager.calculate_tracker_progress(tracker_id)
        assert progress_info["completed_tasks"] == 2
        assert progress_info["overall_progress"] == 100.0
    
    def test_integrated_quality_alert_workflow(self):
        """統合品質アラートワークフローテスト"""
        tracker_id = "TEST-024"
        
        # 1. 品質問題のあるタスク作成
        task = self.progress_manager.create_task(
            tracker_id=tracker_id,
            title="低品質抽出処理",
            description="品質問題のある抽出処理"
        )
        
        self.progress_manager.start_task(task.task_id)
        
        # 2. 品質アラート送信
        failed_images = [
            "low_quality1.jpg", "low_quality2.jpg", 
            "low_quality3.jpg", "low_quality4.jpg", "low_quality5.jpg"
        ]
        
        self.pushover_client.send_quality_alert(
            tracker_id=tracker_id,
            quality_score=0.35,
            threshold=0.60,
            failed_images=failed_images
        )
        
        # 3. 緊急承認依頼作成
        approval_request = self.approval_system.create_approval_request(
            tracker_id=tracker_id,
            stage=ApprovalStage.QUALITY_CHECK,
            title="緊急品質問題対応",
            details=f"品質スコア 0.35 (閾値: 0.60)\n失敗画像: {len(failed_images)}件",
            expires_in_hours=2  # 2時間以内の対応要求
        )
        
        # 4. タスク失敗処理
        self.progress_manager.fail_task(
            task.task_id, 
            f"品質基準未達成: スコア 0.35 < 閾値 0.60"
        )
        
        # 5. 結果検証
        # 品質アラート通知確認
        alert_messages = self.pushover_client.get_messages_by_title_pattern("品質アラート")
        assert len(alert_messages) == 1
        
        alert_msg = alert_messages[0]
        assert alert_msg.priority == 2  # 緊急
        assert "品質スコア: 0.350" in alert_msg.message
        assert "閾値: 0.600" in alert_msg.message
        assert "失敗画像: 5件" in alert_msg.message
        
        # 承認依頼確認
        assert approval_request.expires_at is not None
        expected_expiry = datetime.now() + timedelta(hours=2)
        time_diff = abs((approval_request.expires_at - expected_expiry).total_seconds())
        assert time_diff < 300  # 5分以内の誤差は許容
        
        # タスク失敗確認
        failed_task = self.progress_manager.get_task(task.task_id)
        assert failed_task.status == TaskStatus.FAILED
        assert "品質基準未達成" in failed_task.metadata["failure_reason"]
        
        # 統計確認
        progress_info = self.progress_manager.calculate_tracker_progress(tracker_id)
        assert progress_info["failed_tasks"] == 1
        assert progress_info["overall_progress"] == 0.0
    
    def test_notification_callback_integration(self):
        """通知コールバック統合テスト"""
        callback_events = []
        
        def test_callback(event_type, task):
            callback_events.append({
                "event": event_type,
                "task_id": task.task_id,
                "tracker_id": task.tracker_id,
                "status": task.status.value
            })
        
        # コールバック登録
        self.progress_manager.add_notification_callback(test_callback)
        
        # タスクライフサイクル実行
        task = self.progress_manager.create_task("TEST-025", "コールバックテスト", "説明")
        self.progress_manager.start_task(task.task_id)
        self.progress_manager.update_progress(task.task_id, 50)
        self.progress_manager.complete_task(task.task_id)
        
        # コールバックイベント確認
        assert len(callback_events) == 4  # created, started, progress_updated, completed
        
        assert callback_events[0]["event"] == "task_created"
        assert callback_events[0]["status"] == "not_started"
        
        assert callback_events[1]["event"] == "task_started"
        assert callback_events[1]["status"] == "in_progress"
        
        assert callback_events[2]["event"] == "progress_updated"
        assert callback_events[2]["status"] == "in_progress"
        
        assert callback_events[3]["event"] == "task_completed"
        assert callback_events[3]["status"] == "completed"
        
        # 全イベントが同じタスクを参照
        for event in callback_events:
            assert event["task_id"] == task.task_id
            assert event["tracker_id"] == "TEST-025"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])