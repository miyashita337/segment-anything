#!/usr/bin/env python3
"""
KIRO-006 Phase 2 - Pytest Configuration and Fixtures
ワークフロー計画・起票システムのテスト共通設定
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """プロジェクトルートパスを提供"""
    return project_root


@pytest.fixture(scope="function")
def temp_workspace():
    """テスト用の一時ワークスペースを作成"""
    temp_dir = tempfile.mkdtemp(prefix="kiro006_test_")
    
    # 必要なディレクトリ構造を作成
    workspace_dirs = [
        "config",
        ".workflow_state", 
        ".workflow_approvals",
        "logs",
        "workspace"
    ]
    
    for dir_name in workspace_dirs:
        os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)
    
    # 元のディレクトリを保存
    original_cwd = os.getcwd()
    
    # テスト用ディレクトリに移動
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # 元のディレクトリに戻る
    os.chdir(original_cwd)
    
    # 一時ディレクトリを削除
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_progress_manager():
    """ProgressManagerのモックを提供"""
    mock_manager = Mock()
    
    # 基本的なメソッドのモック設定
    mock_manager.get_task.return_value = None
    mock_manager.create_task.return_value = Mock()
    
    # 設定のモック
    mock_config = Mock()
    mock_config.sheet_url = "https://docs.google.com/spreadsheets/d/test/edit"
    mock_manager.config = mock_config
    
    return mock_manager


@pytest.fixture(scope="function")
def mock_workflow_controller():
    """WorkflowControllerのモックを提供"""
    mock_controller = Mock()
    
    # 基本的なメソッドのモック設定
    mock_controller.create_tracker_workflow.return_value = True
    mock_controller.get_workflow_status.return_value = {
        "current_phase": "Phase 0.5",
        "current_step": "branch_verification", 
        "can_proceed": True,
        "completed_steps": [],
        "pending_approvals": []
    }
    
    return mock_controller


@pytest.fixture(scope="function")
def sample_tracker_data():
    """テスト用のサンプルトラッカーデータを提供"""
    return {
        "tracker_id": "TEST-001",
        "summary": "テスト用概要",
        "details": "テスト用詳細説明です。" * 10,  # 適度な長さ
        "priority": "medium"
    }


@pytest.fixture(scope="function")
def long_details_data():
    """文字数制限テスト用の長い詳細データを提供"""
    from tools.workflow.plan_command_handler import PlanCommandHandler
    
    # 制限を超える長さのデータ
    long_text = "a" * (PlanCommandHandler.MAX_DETAILS_LENGTH + 100)
    
    return {
        "tracker_id": "LONG-001",
        "summary": "文字数制限テスト",
        "details": long_text,
        "priority": "high"
    }


@pytest.fixture(scope="function")
def invalid_tracker_ids():
    """無効なトラッカーIDのリストを提供"""
    return [
        "",                 # 空文字
        "   ",             # 空白のみ
        "invalid",         # ハイフンなし
        "tracker-001",     # 小文字
        "TRACKER_001",     # アンダースコア
        "TRACKER-",        # 番号なし
        "-001",            # プレフィックスなし
        "TRACKER-ABC",     # 番号が文字
        "123-TRACKER",     # 数字から開始
        "TRACKER@001",     # 特殊文字
    ]


@pytest.fixture(scope="function")
def valid_tracker_ids():
    """有効なトラッカーIDのリストを提供"""
    return [
        "TRACKER-001",
        "KIRO-006",
        "QUAL-044", 
        "A-1",
        "TEST123-999",
        "FEATURE-001",
        "BUG-123",
        "URGENT-001"
    ]


@pytest.fixture(autouse=True)
def setup_logging():
    """テスト用ログ設定"""
    import logging
    
    # テスト中はログレベルをWARNING以上に設定
    logging.getLogger().setLevel(logging.WARNING)
    
    # 特定のモジュールのログを抑制
    logging.getLogger('httplib2').setLevel(logging.ERROR)
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)
    logging.getLogger('google').setLevel(logging.ERROR)


@pytest.fixture(scope="function")
def mock_google_sheets_config():
    """Google Sheets設定のモックを提供"""
    config_data = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nTEST_KEY\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "test-client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://token.googleapis.com/token"
    }
    
    return config_data


@pytest.fixture(scope="function") 
def mock_sqlite_database(temp_workspace):
    """テスト用SQLiteデータベースのモックを提供"""
    db_path = os.path.join(temp_workspace, ".workflow_state", "workflow.db")
    
    # データベースディレクトリを作成
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    return db_path


# マーカー定義
def pytest_configure(config):
    """Pytestマーカーの設定"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line(
        "markers", "cli: CLI interface tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "security: Security-related tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "workflow: Workflow system tests (KIRO-006)"
    )
    config.addinivalue_line(
        "markers", "plan_command: PlanCommandHandler tests"
    )
    config.addinivalue_line(
        "markers", "create_command: CreateCommandHandler tests"
    )
    config.addinivalue_line(
        "markers", "backward_compatibility: Tests ensuring existing functionality works"
    )


# テスト実行前後のフック
def pytest_sessionstart(session):
    """テストセッション開始時の処理"""
    print("\n🚀 KIRO-006 Phase 2 - Workflow Plan Command System Tests Starting...")


def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時の処理"""
    if exitstatus == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests failed with exit status: {exitstatus}")


# テスト結果のカスタマイズ
def pytest_runtest_makereport(item, call):
    """テスト結果レポートのカスタマイズ"""
    if call.when == "call":
        # テスト名にマーカー情報を追加
        markers = [marker.name for marker in item.iter_markers()]
        if markers:
            item.user_properties.append(("markers", ", ".join(markers)))


# 並列実行時の設定（pytest-xdistが利用可能な場合）
def pytest_configure_node(node):
    """並列実行ノードの設定"""
    # 各ワーカーに固有の設定を追加
    node.workerinput["worker_id"] = node.gateway.id