#!/usr/bin/env python3
"""
Mock Approval System for workflow testing

承認プロセスをモックしてテスト可能にするシステム
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ApprovalStatus(Enum):
    """承認ステータス"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalStage(Enum):
    """承認ステージ"""

    EXTRACTION_START = "extraction_start"
    QUALITY_CHECK = "quality_check"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    FINAL_APPROVAL = "final_approval"
    DEPLOYMENT = "deployment"


@dataclass
class ApprovalRequest:
    """承認依頼データクラス"""

    request_id: str
    tracker_id: str
    stage: ApprovalStage
    title: str
    details: str
    requestor: str
    created_at: datetime
    status: ApprovalStatus
    expires_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    approval_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返却"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        if self.approved_at:
            data["approved_at"] = self.approved_at.isoformat()
        data["stage"] = self.stage.value
        data["status"] = self.status.value
        return data


class MockApprovalSystem:
    """Mock承認システム"""

    def __init__(self):
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.approval_log_file = Path("tests/fixtures/mock_approval_log.json")
        self.request_counter = 0
        self.auto_approve_mode = False
        self.auto_reject_mode = False
        self.approval_delay_seconds = 0

        # 設定可能な承認者リスト
        self.available_approvers = ["user@example.com", "manager@example.com", "admin@example.com"]
        self.default_approver = "user@example.com"

    def create_approval_request(
        self,
        tracker_id: str,
        stage: ApprovalStage,
        title: str,
        details: str,
        requestor: str = "claude_code",
        expires_in_hours: int = 24,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        承認依頼作成

        Args:
            tracker_id: トラッカーID
            stage: 承認ステージ
            title: 承認タイトル
            details: 詳細情報
            requestor: 依頼者
            expires_in_hours: 有効期限（時間）
            metadata: メタデータ

        Returns:
            作成された承認依頼
        """
        self.request_counter += 1
        request_id = f"APPR-{self.request_counter:04d}"

        now = datetime.now()
        expires_at = now + timedelta(hours=expires_in_hours) if expires_in_hours > 0 else None

        approval_url = f"http://100.123.241.106:8088/approval/{request_id}"

        request = ApprovalRequest(
            request_id=request_id,
            tracker_id=tracker_id,
            stage=stage,
            title=title,
            details=details,
            requestor=requestor,
            created_at=now,
            status=ApprovalStatus.PENDING,
            expires_at=expires_at,
            approval_url=approval_url,
            metadata=metadata or {},
        )

        self.approval_requests[request_id] = request

        # 自動承認・自動拒否モードの処理
        if self.auto_approve_mode:
            self._auto_process_approval(request_id, True)
        elif self.auto_reject_mode:
            self._auto_process_approval(request_id, False)

        self._save_approval_log()
        return request

    def approve_request(
        self, request_id: str, approver: str = None, comment: Optional[str] = None
    ) -> bool:
        """
        承認依頼を承認

        Args:
            request_id: 承認依頼ID
            approver: 承認者
            comment: コメント

        Returns:
            承認成功フラグ
        """
        if request_id not in self.approval_requests:
            return False

        request = self.approval_requests[request_id]

        # ステータスチェック
        if request.status != ApprovalStatus.PENDING:
            return False

        # 有効期限チェック
        if request.expires_at and datetime.now() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            self._save_approval_log()
            return False

        # 承認処理
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver or self.default_approver
        request.approved_at = datetime.now()

        if comment:
            if not request.metadata:
                request.metadata = {}
            request.metadata["approval_comment"] = comment

        self._save_approval_log()
        return True

    def reject_request(
        self, request_id: str, rejector: str = None, reason: Optional[str] = None
    ) -> bool:
        """
        承認依頼を拒否

        Args:
            request_id: 承認依頼ID
            rejector: 拒否者
            reason: 拒否理由

        Returns:
            拒否成功フラグ
        """
        if request_id not in self.approval_requests:
            return False

        request = self.approval_requests[request_id]

        # ステータスチェック
        if request.status != ApprovalStatus.PENDING:
            return False

        # 拒否処理
        request.status = ApprovalStatus.REJECTED
        request.approved_by = rejector or self.default_approver
        request.approved_at = datetime.now()
        request.rejection_reason = reason or "No reason provided"

        self._save_approval_log()
        return True

    def cancel_request(self, request_id: str, reason: Optional[str] = None) -> bool:
        """
        承認依頼をキャンセル

        Args:
            request_id: 承認依頼ID
            reason: キャンセル理由

        Returns:
            キャンセル成功フラグ
        """
        if request_id not in self.approval_requests:
            return False

        request = self.approval_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.CANCELLED
        if reason:
            if not request.metadata:
                request.metadata = {}
            request.metadata["cancellation_reason"] = reason

        self._save_approval_log()
        return True

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """承認依頼取得"""
        return self.approval_requests.get(request_id)

    def get_requests_by_tracker(self, tracker_id: str) -> List[ApprovalRequest]:
        """トラッカーID別承認依頼取得"""
        return [req for req in self.approval_requests.values() if req.tracker_id == tracker_id]

    def get_requests_by_status(self, status: ApprovalStatus) -> List[ApprovalRequest]:
        """ステータス別承認依頼取得"""
        return [req for req in self.approval_requests.values() if req.status == status]

    def get_pending_requests(self, include_expired: bool = False) -> List[ApprovalRequest]:
        """保留中の承認依頼取得"""
        pending_requests = []
        now = datetime.now()

        for req in self.approval_requests.values():
            if req.status != ApprovalStatus.PENDING:
                continue

            # 期限切れチェック
            if req.expires_at and now > req.expires_at:
                req.status = ApprovalStatus.EXPIRED
                continue

            if include_expired or req.status == ApprovalStatus.PENDING:
                pending_requests.append(req)

        if pending_requests:
            self._save_approval_log()

        return pending_requests

    def check_stage_approval_required(self, tracker_id: str, stage: ApprovalStage) -> bool:
        """
        指定ステージで承認が必要かチェック

        Args:
            tracker_id: トラッカーID
            stage: チェック対象ステージ

        Returns:
            承認必要フラグ
        """
        # ステージ別承認要否設定（実際のシステムでは設定ファイルから読み込み）
        approval_required_stages = {
            ApprovalStage.EXTRACTION_START: False,  # 抽出開始は自動承認
            ApprovalStage.QUALITY_CHECK: True,  # 品質チェックは承認必要
            ApprovalStage.STATISTICAL_ANALYSIS: True,  # 統計分析は承認必要
            ApprovalStage.FINAL_APPROVAL: True,  # 最終承認は必要
            ApprovalStage.DEPLOYMENT: True,  # デプロイは承認必要
        }

        return approval_required_stages.get(stage, True)

    def wait_for_approval(
        self, request_id: str, timeout_seconds: int = 300, check_interval: int = 5
    ) -> ApprovalStatus:
        """
        承認待機（モックでは即座に結果を返す）

        Args:
            request_id: 承認依頼ID
            timeout_seconds: タイムアウト秒数
            check_interval: チェック間隔秒数

        Returns:
            最終的な承認ステータス
        """
        request = self.get_request(request_id)
        if not request:
            return ApprovalStatus.CANCELLED

        # モック環境では即座に状態を返す
        if self.approval_delay_seconds > 0:
            # 実際の環境では time.sleep(self.approval_delay_seconds) するが、
            # テスト環境では承認者設定とスタータス変更のみシミュレーション
            pass

        return request.status

    def set_auto_approve_mode(self, enabled: bool):
        """自動承認モード設定"""
        self.auto_approve_mode = enabled
        if enabled:
            self.auto_reject_mode = False

    def set_auto_reject_mode(self, enabled: bool):
        """自動拒否モード設定"""
        self.auto_reject_mode = enabled
        if enabled:
            self.auto_approve_mode = False

    def set_approval_delay(self, seconds: int):
        """承認遅延秒数設定"""
        self.approval_delay_seconds = max(0, seconds)

    def get_approval_statistics(self) -> Dict[str, Any]:
        """承認統計情報取得"""
        total = len(self.approval_requests)
        if total == 0:
            return {"total": 0}

        stats = {"total": total}

        # ステータス別集計
        status_counts = {}
        for status in ApprovalStatus:
            count = sum(1 for req in self.approval_requests.values() if req.status == status)
            status_counts[status.value] = count

        stats["by_status"] = status_counts

        # ステージ別集計
        stage_counts = {}
        for stage in ApprovalStage:
            count = sum(1 for req in self.approval_requests.values() if req.stage == stage)
            stage_counts[stage.value] = count

        stats["by_stage"] = stage_counts

        # 処理時間統計
        processed_requests = [req for req in self.approval_requests.values() if req.approved_at]
        if processed_requests:
            processing_times = []
            for req in processed_requests:
                delta = req.approved_at - req.created_at
                processing_times.append(delta.total_seconds())

            stats["average_processing_time_seconds"] = sum(processing_times) / len(processing_times)
            stats["max_processing_time_seconds"] = max(processing_times)
            stats["min_processing_time_seconds"] = min(processing_times)

        return stats

    def clear_all_requests(self):
        """全承認依頼クリア"""
        self.approval_requests.clear()
        self.request_counter = 0
        if self.approval_log_file.exists():
            self.approval_log_file.unlink()

    def _auto_process_approval(self, request_id: str, approve: bool):
        """自動承認・拒否処理"""
        if approve:
            self.approve_request(
                request_id, approver="auto_system", comment="Automatically approved by mock system"
            )
        else:
            self.reject_request(
                request_id, rejector="auto_system", reason="Automatically rejected by mock system"
            )

    def _save_approval_log(self):
        """承認ログファイル保存"""
        self.approval_log_file.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            "requests": {req_id: req.to_dict() for req_id, req in self.approval_requests.items()},
            "statistics": self.get_approval_statistics(),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.approval_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


# グローバルモックインスタンス（シングルトン）
_mock_approval_instance = None


def get_mock_approval_system() -> MockApprovalSystem:
    """Mock承認システムのシングルトン取得"""
    global _mock_approval_instance
    if _mock_approval_instance is None:
        _mock_approval_instance = MockApprovalSystem()
    return _mock_approval_instance


def reset_mock_approval_system():
    """Mock承認システムのリセット"""
    global _mock_approval_instance
    if _mock_approval_instance:
        _mock_approval_instance.clear_all_requests()
    _mock_approval_instance = None
