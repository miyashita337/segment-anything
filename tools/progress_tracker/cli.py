#!/usr/bin/env python3
"""
進捗管理システム CLI
コマンドラインからの進捗管理操作
"""

import argparse
import sys
import logging
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.config import get_default_config, check_configuration, print_setup_instructions
from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskStatus, ComponentStatus, ProgressTrackerError
from tools.progress_tracker.connection_monitor import show_user_friendly_status
from tools.progress_tracker.execution_permission import (
    ExecutionPermissionManager, PermissionLevel, ActionType, 
    require_permission, get_permission_manager
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cmd_check_config(args):
    """設定状況確認"""
    print("🔧 進捗管理システム設定確認")
    print("=" * 50)
    
    status = check_configuration()
    
    print(f"設定状態: {'✅ 正常' if status['config_valid'] else '❌ 問題あり'}")
    print(f"認証ファイル: {'✅ 存在' if status['auth_file_exists'] else '❌ 不在'}")
    print(f"スプレッドシートID: {'✅ 設定済み' if status['spreadsheet_id_set'] else '❌ 未設定'}")
    
    if status['issues']:
        print("\n⚠️ 問題:")
        for issue in status['issues']:
            print(f"  - {issue}")
        
        print("\n💡 セットアップ手順:")
        print("python tools/progress_tracker/config.py")
        return 1
    
    # 実際の接続テスト
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        tasks = manager.get_all_tasks()
        print(f"\n✅ Google Sheets接続成功")
        print(f"📊 現在のタスク数: {len(tasks)}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Google Sheets接続失敗: {e}")
        return 1


def cmd_show_status(args):
    """ステータス表示"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        manager.print_status_summary()
        return 0
        
    except Exception as e:
        print(f"❌ ステータス表示エラー: {e}")
        return 1


@require_permission(ActionType.WRITE)
def cmd_create_task(args):
    """新規タスク作成（必須パラメータ強制版）"""
    
    # 必須パラメータチェック
    missing_params = []
    if not args.tracker_id:
        missing_params.append("トラッカーID")
    if not hasattr(args, 'description') or not args.description:
        missing_params.append("概要 (--description)")
    if not hasattr(args, 'details') or not args.details:
        missing_params.append("詳細 (--details)")
    
    if missing_params:
        print(f"❌ エラー: 以下の必須パラメータが不足しています:")
        for param in missing_params:
            print(f"   - {param}")
        print("\n💡 必須パラメータ:")
        print("  - トラッカーID: 例) QUAL-036")
        print("  - 概要 (--description): タスクの概要説明")
        print("  - 詳細 (--details): 実装詳細・技術仕様")
        print("  - 登録日付: 自動設定（現在時刻）")
        print("\n✅ 正しい使用例:")
        print('python tools/progress_tracker/cli.py create QUAL-036 \\')
        print('  --description "Quality Workflowリファクタリング" \\')
        print('  --details "create_phase1_extraction_report.py固定値→UnifiedQualityChecker実測値統合"')
        return 1
    
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # 基本タスク作成
        task = manager.create_task(args.tracker_id, args.description)
        
        # 詳細情報の追加（Google Sheetsクライアント直接操作）
        try:
            client = manager.client
            
            # タスク行を特定
            all_values = client.get_sheet_values('A:Z')
            task_row = None
            
            for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップ
                if row and len(row) > 0 and row[0] == args.tracker_id:
                    task_row = i
                    break
            
            if task_row:
                # 自動日付設定
                import datetime
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 優先度設定（デフォルト：高）
                priority = getattr(args, 'priority', '高')
                
                # Google Sheetsの列構造に基づいて更新
                # C列: 優先度, D列: 登録日付, E列: 更新日付, F列: 概要, G列: 詳細
                
                client.update_sheet_values(f'C{task_row}', [[priority]])        # 優先度
                client.update_sheet_values(f'D{task_row}', [[current_time]])    # 登録日付
                client.update_sheet_values(f'E{task_row}', [[current_time]])    # 更新日付
                client.update_sheet_values(f'F{task_row}', [[args.description]])  # 概要
                client.update_sheet_values(f'G{task_row}', [[args.details]])    # 詳細
                
                print(f"✅ タスク作成成功: {task.tracker_id}")
                print(f"📝 概要: {args.description}")
                print(f"📋 詳細: {len(args.details)}文字")
                print(f"🎯 優先度: {priority}")
                print(f"📅 登録日付: {current_time}")
                
            else:
                print(f"⚠️ 基本タスクは作成されましたが、詳細情報の更新に失敗しました")
                
        except Exception as e:
            print(f"⚠️ 詳細情報更新エラー: {e}")
            print("基本タスクは作成済みです")
        
        return 0
        
    except Exception as e:
        print(f"❌ タスク作成エラー: {e}")
        return 1

@require_permission(ActionType.WRITE)
def cmd_create_task_basic(args):
    """基本タスク作成（旧create コマンド・後方互換性用）"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        task = manager.create_task(args.tracker_id, args.description or "")
        print(f"✅ 基本タスク作成成功: {task.tracker_id}")
        print(f"📝 説明: {task.description}")
        print("⚠️ 注意: 詳細情報は含まれていません。必要に応じて update-details コマンドを使用してください")
        return 0
        
    except Exception as e:
        print(f"❌ 基本タスク作成エラー: {e}")
        return 1


@require_permission(ActionType.WRITE)
def cmd_create_task_enhanced(args):
    """拡張タスク作成（詳細・優先度・日付対応）"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # 基本タスク作成
        task = manager.create_task(args.tracker_id, args.description or "")
        
        # 詳細情報がある場合は更新
        if args.details:
            try:
                # Google Sheets クライアント直接操作で詳細情報を更新
                client = manager.client
                
                # タスク行を特定
                all_values = client.get_sheet_values('A:Z')
                task_row = None
                
                for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップ
                    if row and len(row) > 0 and row[0] == args.tracker_id:
                        task_row = i
                        break
                
                if task_row:
                    # 登録日付・更新日付の設定（指定された日付または現在日時）
                    import datetime
                    if args.registration_date:
                        reg_date = args.registration_date
                        update_date = args.registration_date
                    else:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        reg_date = now
                        update_date = now
                    
                    # 優先度設定
                    priority = args.priority or "高"
                    
                    # 概要設定（説明と同じにするか、別途指定）
                    summary = args.summary or args.description or ""
                    
                    # Google Sheetsの列構造に基づいて更新
                    # C列: 優先度, D列: 登録日付, E列: 更新日付, F列: 概要, G列: 詳細
                    
                    # 優先度更新
                    client.update_sheet_values(f'C{task_row}', [[priority]])
                    
                    # 登録日付更新  
                    client.update_sheet_values(f'D{task_row}', [[reg_date]])
                    
                    # 更新日付更新
                    client.update_sheet_values(f'E{task_row}', [[update_date]])
                    
                    # 概要更新
                    if summary:
                        client.update_sheet_values(f'F{task_row}', [[summary]])
                    
                    # 詳細更新
                    client.update_sheet_values(f'G{task_row}', [[args.details]])
                    
                    print(f"📋 拡張情報更新完了:")
                    print(f"   優先度: {priority}")
                    print(f"   登録日付: {reg_date}")
                    print(f"   詳細: {len(args.details)}文字")
                    
            except Exception as e:
                print(f"⚠️ 拡張情報更新エラー: {e}")
                print("基本タスクは作成済みです")
        
        return 0
        
    except Exception as e:
        print(f"❌ 拡張タスク作成エラー: {e}")
        return 1


@require_permission(ActionType.WRITE)
def cmd_update_status(args):
    """ステータス更新"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # ステータス文字列をEnumに変換
        status = None
        for s in TaskStatus:
            if s.value == args.status:
                status = s
                break
        
        if not status:
            print(f"❌ 無効なステータス: {args.status}")
            print("有効なステータス:")
            for s in TaskStatus:
                print(f"  - {s.value}")
            return 1
        
        task = manager.update_task_status(args.tracker_id, status)
        print(f"✅ ステータス更新成功: {task.tracker_id} -> {status.value}")
        return 0
        
    except Exception as e:
        print(f"❌ ステータス更新エラー: {e}")
        return 1


def cmd_update_component(args):
    """コンポーネントステータス更新"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # コンポーネントステータス文字列をEnumに変換
        comp_status = None
        for s in ComponentStatus:
            if s.value == args.component_status:
                comp_status = s
                break
        
        if not comp_status:
            print(f"❌ 無効なコンポーネントステータス: {args.component_status}")
            print("有効なステータス:")
            for s in ComponentStatus:
                print(f"  - {s.value}")
            return 1
        
        task = manager.update_component_status(args.tracker_id, args.component, comp_status)
        print(f"✅ コンポーネント更新成功: {task.tracker_id}.{args.component} -> {comp_status.value}")
        return 0
        
    except Exception as e:
        print(f"❌ コンポーネント更新エラー: {e}")
        return 1


def cmd_workflow_update(args):
    """ワークフロー結果更新"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # ワークフロー結果をパース
        workflow_results = {}
        if hasattr(args, 'extraction_success'):
            workflow_results['extraction_pipeline'] = args.extraction_success
        if hasattr(args, 'quality_success'):
            workflow_results['quality_evaluation'] = args.quality_success
        if hasattr(args, 'dashboard_success'):
            workflow_results['dashboard_generation'] = args.dashboard_success
        if hasattr(args, 'test_success'):
            workflow_results['unit_test'] = args.test_success
        
        task = manager.bulk_update_from_workflow(args.tracker_id, workflow_results)
        print(f"✅ ワークフロー更新成功: {task.tracker_id}")
        print(f"📊 ステータス: {task.status.value}")
        return 0
        
    except Exception as e:
        print(f"❌ ワークフロー更新エラー: {e}")
        return 1


def cmd_list_tasks(args):
    """タスク一覧表示"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        if args.status:
            # 特定ステータスのタスク
            status = None
            for s in TaskStatus:
                if s.value == args.status:
                    status = s
                    break
            
            if not status:
                print(f"❌ 無効なステータス: {args.status}")
                return 1
            
            tasks = manager.get_tasks_by_status(status)
            print(f"📋 {status.value}のタスク: {len(tasks)}件")
        else:
            # 全タスク
            tasks = manager.get_all_tasks()
            print(f"📋 全タスク: {len(tasks)}件")
        
        if tasks:
            print("=" * 60)
            for task in tasks:
                updated = task.updated_date.strftime('%Y-%m-%d') if task.updated_date else "未更新"
                print(f"{task.tracker_id}: {task.status.value} (更新: {updated})")
                if task.description:
                    print(f"  📝 {task.description}")
        
        return 0
        
    except Exception as e:
        print(f"❌ タスク一覧エラー: {e}")
        return 1


def cmd_init_sheet(args):
    """シート初期化"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # 強制初期化
        manager.client.initialize_sheet()
        print("✅ シート初期化完了")
        
        # 初期化確認
        tasks = manager.get_all_tasks()
        print(f"📊 現在のタスク数: {len(tasks)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ シート初期化エラー: {e}")
        return 1


def cmd_update_metrics(args):
    """10指標更新"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # 指標辞書作成
        metrics_dict = {}
        if hasattr(args, 'lca') and args.lca is not None:
            metrics_dict['LCA'] = args.lca
        if hasattr(args, 'ab_rate') and args.ab_rate is not None:
            metrics_dict['A/B評価率'] = args.ab_rate
        if hasattr(args, 'fps') and args.fps is not None:
            metrics_dict['FPS'] = args.fps
        if hasattr(args, 'c_plus_rate') and args.c_plus_rate is not None:
            metrics_dict['C以上評価率'] = args.c_plus_rate
        if hasattr(args, 'avg_coverage') and args.avg_coverage is not None:
            metrics_dict['平均カバレッジ率'] = args.avg_coverage
        if hasattr(args, 'avg_compactness') and args.avg_compactness is not None:
            metrics_dict['平均コンパクトネス'] = args.avg_compactness
        if hasattr(args, 'avg_fill_rate') and args.avg_fill_rate is not None:
            metrics_dict['平均フィル率'] = args.avg_fill_rate
        if hasattr(args, 'sci') and args.sci is not None:
            metrics_dict['SCI (Semantic Completeness Index)'] = args.sci
        if hasattr(args, 'pla') and args.pla is not None:
            metrics_dict['PLA (Pixel-Level Accuracy)'] = args.pla
        if hasattr(args, 'ple') and args.ple is not None:
            metrics_dict['PLE (Progressive Learning Efficiency)'] = args.ple
        
        task = manager.update_task_metrics(args.tracker_id, metrics_dict)
        print(f"✅ メトリクス更新成功: {task.tracker_id}")
        print(f"📊 更新指標数: {len(metrics_dict)}")
        return 0
        
    except Exception as e:
        print(f"❌ メトリクス更新エラー: {e}")
        return 1


def cmd_import_quality_results(args):
    """統合品質チェッカー結果インポート"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # JSONファイルから品質結果を読み込み
        import json
        with open(args.results_file, 'r', encoding='utf-8') as f:
            quality_results = json.load(f)
        
        task = manager.update_from_quality_checker_results(args.tracker_id, quality_results)
        print(f"✅ 品質結果インポート成功: {task.tracker_id}")
        return 0
        
    except Exception as e:
        print(f"❌ 品質結果インポートエラー: {e}")
        return 1


def cmd_update_task_details(args):
    """既存タスクの詳細情報更新"""
    try:
        config = get_default_config()
        manager = ProgressManager(config)
        
        # Google Sheets クライアント取得
        client = manager.client
        
        # タスク行を特定
        all_values = client.get_sheet_values('A:Z')
        task_row = None
        
        for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップ
            if row and len(row) > 0 and row[0] == args.tracker_id:
                task_row = i
                break
        
        if not task_row:
            print(f"❌ タスクが見つかりません: {args.tracker_id}")
            return 1
        
        # 更新項目の準備
        import datetime
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        updates_made = []
        
        # 詳細情報の更新
        if args.details:
            client.update_sheet_values(f'G{task_row}', [[args.details]])
            updates_made.append(f"詳細情報: {len(args.details)}文字")
        
        # 優先度の更新
        if args.priority:
            client.update_sheet_values(f'C{task_row}', [[args.priority]])
            updates_made.append(f"優先度: {args.priority}")
        
        # 概要の更新
        if args.summary:
            client.update_sheet_values(f'F{task_row}', [[args.summary]])
            updates_made.append(f"概要更新")
        
        # 登録日付の更新（指定がある場合のみ）
        if args.registration_date:
            client.update_sheet_values(f'D{task_row}', [[args.registration_date]])
            updates_made.append(f"登録日付: {args.registration_date}")
        
        # 更新日付は必ず現在時刻に更新
        client.update_sheet_values(f'E{task_row}', [[current_time]])
        updates_made.append(f"更新日付: {current_time}")
        
        print(f"✅ タスク詳細更新成功: {args.tracker_id}")
        for update in updates_made:
            print(f"   📋 {update}")
        
        return 0
        
    except Exception as e:
        print(f"❌ タスク詳細更新エラー: {e}")
        return 1


def cmd_permission_status(args):
    """権限状態表示"""
    manager = get_permission_manager()
    
    print("🔒 Claude実行権限管理システム")
    print("=" * 50)
    
    # 有効状態
    enabled = manager.enabled
    print(f"システム状態: {'✅ 有効' if enabled else '⚠️ 無効'}")
    
    if not enabled:
        print("\n💡 有効化方法:")
        print("  export CLAUDE_PERMISSION_ENABLED=true")
        return 0
    
    # 現在の権限レベル
    current_level = manager.get_current_level()
    print(f"現在の権限レベル: {current_level.name}")
    print(f"  説明: {get_level_description(current_level)}")
    
    # セッション情報
    print(f"\nセッションID: {manager.session_id}")
    print(f"開始時刻: {manager.state.get('started_at', '不明')}")
    
    # 最近の監査ログ
    audit_log = manager.get_audit_log(limit=5)
    if audit_log:
        print("\n📋 最近のアクティビティ:")
        for entry in audit_log:
            event_type = entry.get('event_type', '')
            timestamp = entry.get('timestamp', '')
            data = entry.get('data', {})
            
            if event_type == 'permission_check':
                action = data.get('action', '')
                allowed = data.get('allowed', False)
                status = '✅' if allowed else '❌'
                print(f"  {status} {timestamp[:19]} - {action}")
            elif event_type == 'permission_change':
                old_level = data.get('old_level', '')
                new_level = data.get('new_level', '')
                print(f"  🔄 {timestamp[:19]} - 権限変更: {old_level} → {new_level}")
    
    return 0


def cmd_set_permission(args):
    """権限レベル設定"""
    manager = get_permission_manager()
    
    if not manager.enabled:
        print("⚠️ 権限管理システムが無効です")
        print("有効化: export CLAUDE_PERMISSION_ENABLED=true")
        return 1
    
    try:
        new_level = PermissionLevel[args.level.upper()]
        old_level = manager.get_current_level()
        
        if old_level == new_level:
            print(f"ℹ️ 既に {new_level.name} に設定されています")
            return 0
        
        # 確認
        print(f"🔄 権限レベル変更:")
        print(f"  現在: {old_level.name} - {get_level_description(old_level)}")
        print(f"  変更後: {new_level.name} - {get_level_description(new_level)}")
        
        response = input("\n変更しますか？ (y/N): ").strip().lower()
        if response != 'y':
            print("キャンセルしました")
            return 0
        
        manager.set_permission_level(new_level)
        print(f"✅ 権限レベルを {new_level.name} に変更しました")
        return 0
        
    except KeyError:
        print(f"❌ 無効な権限レベル: {args.level}")
        print("有効なレベル:")
        for level in PermissionLevel:
            print(f"  - {level.name}: {get_level_description(level)}")
        return 1
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1


def cmd_permission_audit(args):
    """監査ログ表示"""
    manager = get_permission_manager()
    
    if not manager.enabled:
        print("⚠️ 権限管理システムが無効です")
        return 1
    
    print("📋 監査ログ")
    print("=" * 70)
    
    audit_log = manager.get_audit_log(limit=args.limit)
    if not audit_log:
        print("ログエントリがありません")
        return 0
    
    for entry in audit_log:
        timestamp = entry.get('timestamp', '')[:19]
        event_type = entry.get('event_type', '')
        data = entry.get('data', {})
        
        if event_type == 'permission_check':
            action = data.get('action', '')
            target = data.get('target', '')
            allowed = data.get('allowed', False)
            level = data.get('level', '')
            
            status = '✅ 許可' if allowed else '❌ 拒否'
            print(f"{timestamp} [{level}] {status}: {action}")
            if target:
                print(f"  対象: {target}")
        
        elif event_type == 'permission_change':
            old_level = data.get('old_level', '')
            new_level = data.get('new_level', '')
            print(f"{timestamp} 🔄 権限変更: {old_level} → {new_level}")
        
        elif event_type == 'user_confirmation':
            action = data.get('action', '')
            approved = data.get('approved', False)
            status = '✅ 承認' if approved else '❌ 拒否'
            print(f"{timestamp} 👤 ユーザー確認 {status}: {action}")
    
    return 0


def get_level_description(level: PermissionLevel) -> str:
    """権限レベルの説明取得"""
    descriptions = {
        PermissionLevel.READ_ONLY: "読み取り専用（ファイル変更不可）",
        PermissionLevel.PLAN_ONLY: "計画モード（実装実行不可）",
        PermissionLevel.EXECUTE_STEP_BY_STEP: "段階実行（各操作に確認必要）",
        PermissionLevel.EXECUTE_FULL: "完全実行権限（制限なし）"
    }
    return descriptions.get(level, "不明")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="進捗管理システム CLI")
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # 設定確認コマンド
    check_parser = subparsers.add_parser('check-config', help='設定状況確認')
    check_parser.set_defaults(func=cmd_check_config)
    
    # ステータス表示コマンド
    status_parser = subparsers.add_parser('status', help='全体ステータス表示')
    status_parser.set_defaults(func=cmd_show_status)
    
    # タスク作成コマンド（必須パラメータ強制版）
    create_parser = subparsers.add_parser('create', help='新規タスク作成（必須パラメータ：概要・詳細）')
    create_parser.add_argument('tracker_id', help='トラッカーID (例: QUAL-036)')
    create_parser.add_argument('--description', '-d', required=True, help='タスク概要説明（必須）')
    create_parser.add_argument('--details', '--detail', required=True, help='詳細情報（必須）')
    create_parser.add_argument('--priority', '-p', choices=['高', '中', '低'], default='高', help='優先度 (デフォルト: 高)')
    create_parser.set_defaults(func=cmd_create_task)
    
    # 基本タスク作成コマンド（後方互換性用）
    create_basic_parser = subparsers.add_parser('create-basic', help='基本タスク作成（詳細なし・後方互換性用）')
    create_basic_parser.add_argument('tracker_id', help='トラッカーID (例: PHS-005)')
    create_basic_parser.add_argument('--description', '-d', help='タスク説明')
    create_basic_parser.set_defaults(func=cmd_create_task_basic)
    
    # 拡張タスク作成コマンド
    create_enhanced_parser = subparsers.add_parser('create-enhanced', help='拡張タスク作成（詳細・優先度・日付対応）')
    create_enhanced_parser.add_argument('tracker_id', help='トラッカーID (例: PHS-005)')
    create_enhanced_parser.add_argument('--description', '-d', help='タスク説明')
    create_enhanced_parser.add_argument('--details', '--detail', help='詳細情報（長文対応）')
    create_enhanced_parser.add_argument('--priority', '-p', choices=['高', '中', '低'], default='高', help='優先度 (デフォルト: 高)')
    create_enhanced_parser.add_argument('--summary', '-s', help='概要（省略時は説明を使用）')
    create_enhanced_parser.add_argument('--registration-date', '--reg-date', help='登録日付 (yyyy-mm-dd hh:mm:ss形式、省略時は現在時刻)')
    create_enhanced_parser.set_defaults(func=cmd_create_task_enhanced)
    
    # ステータス更新コマンド
    update_parser = subparsers.add_parser('update', help='タスクステータス更新')
    update_parser.add_argument('tracker_id', help='トラッカーID')
    update_parser.add_argument('status', help='新しいステータス')
    update_parser.set_defaults(func=cmd_update_status)
    
    # コンポーネント更新コマンド
    comp_parser = subparsers.add_parser('update-component', help='コンポーネントステータス更新')
    comp_parser.add_argument('tracker_id', help='トラッカーID')
    comp_parser.add_argument('component', help='コンポーネント名')
    comp_parser.add_argument('component_status', help='コンポーネントステータス')
    comp_parser.set_defaults(func=cmd_update_component)
    
    # ワークフロー更新コマンド
    workflow_parser = subparsers.add_parser('workflow', help='ワークフロー結果更新')
    workflow_parser.add_argument('tracker_id', help='トラッカーID')
    workflow_parser.add_argument('--extraction-success', type=bool, default=True, help='抽出パイプライン成功')
    workflow_parser.add_argument('--quality-success', type=bool, default=True, help='品質評価成功')
    workflow_parser.add_argument('--dashboard-success', type=bool, default=True, help='ダッシュボード生成成功')
    workflow_parser.add_argument('--test-success', type=bool, default=True, help='テスト成功')
    workflow_parser.set_defaults(func=cmd_workflow_update)
    
    # タスク一覧コマンド
    list_parser = subparsers.add_parser('list', help='タスク一覧表示')
    list_parser.add_argument('--status', '-s', help='特定ステータスのタスクのみ表示')
    list_parser.set_defaults(func=cmd_list_tasks)
    
    # シート初期化コマンド
    init_parser = subparsers.add_parser('init', help='シート初期化')
    init_parser.set_defaults(func=cmd_init_sheet)
    
    # セットアップ手順表示
    setup_parser = subparsers.add_parser('setup', help='セットアップ手順表示')
    setup_parser.set_defaults(func=lambda args: print_setup_instructions() or 0)
    
    # 10指標更新コマンド
    metrics_parser = subparsers.add_parser('update-metrics', help='10指標更新')
    metrics_parser.add_argument('tracker_id', help='トラッカーID')
    metrics_parser.add_argument('--lca', type=float, help='LCA (バウンディングボックス精度)')
    metrics_parser.add_argument('--ab-rate', type=float, help='A/B評価率')
    metrics_parser.add_argument('--fps', type=float, help='FPS (処理速度)')
    metrics_parser.add_argument('--c-plus-rate', type=float, help='C以上評価率')
    metrics_parser.add_argument('--avg-coverage', type=float, help='平均カバレッジ率')
    metrics_parser.add_argument('--avg-compactness', type=float, help='平均コンパクトネス')
    metrics_parser.add_argument('--avg-fill-rate', type=float, help='平均フィル率')
    metrics_parser.add_argument('--sci', type=float, help='SCI (Semantic Completeness Index)')
    metrics_parser.add_argument('--pla', type=float, help='PLA (Pixel-Level Accuracy)')
    metrics_parser.add_argument('--ple', type=float, help='PLE (Progressive Learning Efficiency)')
    metrics_parser.set_defaults(func=cmd_update_metrics)
    
    # 品質結果インポートコマンド
    import_parser = subparsers.add_parser('import-quality', help='統合品質チェッカー結果インポート')
    import_parser.add_argument('tracker_id', help='トラッカーID')
    import_parser.add_argument('results_file', help='品質結果JSONファイルパス')
    import_parser.set_defaults(func=cmd_import_quality_results)
    
    # タスク詳細更新コマンド
    update_details_parser = subparsers.add_parser('update-details', help='既存タスクの詳細情報更新')
    update_details_parser.add_argument('tracker_id', help='トラッカーID')
    update_details_parser.add_argument('--details', '--detail', help='詳細情報（長文対応）')
    update_details_parser.add_argument('--priority', '-p', choices=['高', '中', '低'], help='優先度')
    update_details_parser.add_argument('--summary', '-s', help='概要')
    update_details_parser.add_argument('--registration-date', '--reg-date', help='登録日付 (yyyy-mm-dd hh:mm:ss形式)')
    update_details_parser.set_defaults(func=cmd_update_task_details)
    
    # 接続監視コマンド
    status_parser = subparsers.add_parser('connection-status', help='API接続状況確認')
    status_parser.set_defaults(func=lambda args: 0 if show_user_friendly_status() else 1)
    
    # ===== 権限管理コマンド =====
    # 権限状態表示
    perm_status_parser = subparsers.add_parser('permission-status', help='権限管理システム状態表示')
    perm_status_parser.set_defaults(func=cmd_permission_status)
    
    # 権限レベル設定
    perm_set_parser = subparsers.add_parser('set-permission', help='権限レベル設定')
    perm_set_parser.add_argument('level', 
                                  choices=['READ_ONLY', 'PLAN_ONLY', 'EXECUTE_STEP_BY_STEP', 'EXECUTE_FULL'],
                                  help='権限レベル')
    perm_set_parser.set_defaults(func=cmd_set_permission)
    
    # 監査ログ表示
    perm_audit_parser = subparsers.add_parser('permission-audit', help='権限システム監査ログ表示')
    perm_audit_parser.add_argument('--limit', type=int, default=20, help='表示件数（デフォルト: 20）')
    perm_audit_parser.set_defaults(func=cmd_permission_audit)
    
    # 引数パース
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"コマンド実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())