# 設計書

## 概要

ワークフロー計画・起票システムの設計書です。既存の`create`コマンドからGoogle Sheets機能を分離し、新しい`plan`コマンドに移行します。同時に、混乱を避けるためにPlanModeエスカレーション関連の用語をTaskFailureEスカレーションにリファクタリングします。

## アーキテクチャ

### 現在のシステム分析

```
現在のcreateコマンド:
┌─────────────────────────────────────┐
│ create {TRACKER_ID}                 │
│ ├── SQLiteワークフロー状態作成       │
│ ├── Google Sheets起票               │
│ └── ステータス自動更新               │
└─────────────────────────────────────┘

問題:
- 責任が混在（計画 + 実行）
- ワークフロー開始前の調査段階がない
- 用語の混乱（PlanMode vs Claude Plan mode）
```

### 新しいアーキテクチャ

```
新しい分離アーキテクチャ:
┌─────────────────────────────────────┐
│ 1. plan {TRACKER_ID} {概要} {詳細}  │
│    ├── 入力検証（3引数必須）         │
│    ├── 詳細文字数制限（20,000文字）  │
│    ├── Google Sheets起票            │
│    └── 起票確認メッセージ            │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 2. create {TRACKER_ID}              │
│    ├── SQLiteワークフロー状態作成    │
│    ├── 初期ステップ設定              │
│    └── ワークフロー開始確認          │
└─────────────────────────────────────┘

利点:
- 明確な責任分離
- 段階的なワークフロー管理
- 適切な用語使用
```

## コンポーネント設計

### 1. PlanCommandHandler

**目的**: planコマンドの処理とGoogle Sheets起票

```python
class PlanCommandHandler:
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.max_detail_length = 20000
    
    def execute_plan(self, tracker_id: str, summary: str, details: str) -> bool:
        """
        planコマンド実行
        
        Args:
            tracker_id: トラッカーID
            summary: 概要
            details: 詳細（最大20,000文字）
            
        Returns:
            実行成功フラグ
        """
        # 入力検証
        if not self._validate_inputs(tracker_id, summary, details):
            return False
        
        # Google Sheets起票
        try:
            task = self.progress_manager.create_task(
                tracker_id=tracker_id,
                description=summary,
                details=details
            )
            
            print(f"✅ Google Sheetsにトラッカーを起票: {tracker_id}")
            print(f"   概要: {summary}")
            print(f"   詳細: {len(details)}文字")
            print(f"   ステータス: {task.status.value}")
            print(f"   作成日時: {task.created_date}")
            
            return True
            
        except Exception as e:
            print(f"❌ Google Sheets起票に失敗: {e}")
            return False
    
    def _validate_inputs(self, tracker_id: str, summary: str, details: str) -> bool:
        """入力検証"""
        if not tracker_id or not tracker_id.strip():
            print("❌ エラー: トラッカーIDが必要です")
            return False
        
        if not summary or not summary.strip():
            print("❌ エラー: 概要が必要です")
            return False
        
        if not details or not details.strip():
            print("❌ エラー: 詳細が必要です")
            return False
        
        if len(details) > self.max_detail_length:
            print(f"❌ エラー: 詳細は{self.max_detail_length}文字以内で入力してください")
            print(f"   現在の文字数: {len(details)}文字")
            return False
        
        return True
```

### 2. CreateCommandHandler（リファクタリング）

**目的**: SQLiteワークフロー状態管理のみに特化

```python
class CreateCommandHandler:
    def __init__(self, workflow_controller):
        self.workflow_controller = workflow_controller
    
    def execute_create(self, tracker_id: str) -> bool:
        """
        createコマンド実行（SQLiteのみ）
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            実行成功フラグ
        """
        # SQLiteワークフロー状態作成のみ
        success = self.workflow_controller.create_tracker_workflow(tracker_id)
        
        if not success:
            print(f"❌ トラッカーのワークフロー作成に失敗: {tracker_id}")
            return False
        
        print(f"✅ SQLiteワークフロー状態を作成: {tracker_id}")
        current_step = self.workflow_controller.state_manager.get_current_step(tracker_id)
        print(f"   現在のステップ: {current_step}")
        print(f"   ワークフロー開始準備完了")
        
        # Google Sheets機能は削除済み
        print(f"ℹ️  Google Sheets起票は `plan` コマンドで事前に実行してください")
        
        return True
```

### 3. TaskFailureEscalator（リファクタリング）

**目的**: PlanModeEscalatorをTaskFailureEscalatorにリネーム

```python
class TaskFailureEscalator:
    """タスク失敗エスカレーター（旧PlanModeEscalator）"""
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.escalation_file = self.workspace / "task_failure_escalation.json"
        logger.info("TaskFailureEscalator initialized")
    
    def create_escalation(self,
                         task_id: str,
                         task_type: str,
                         error: str,
                         retry_count: int,
                         command: str) -> Dict[str, Any]:
        """
        タスク失敗エスカレーション作成
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            error: エラー内容
            retry_count: リトライ回数
            command: 実行コマンド
            
        Returns:
            エスカレーション情報
        """
        escalation = {
            'task_id': task_id,
            'task_type': task_type,
            'error': error,
            'retry_count': retry_count,
            'command': command,
            'created_at': datetime.now().isoformat(),
            'status': 'pending_review',
            'suggested_actions': self._suggest_actions(task_type, error)
        }
        
        # ファイル保存
        try:
            with open(self.escalation_file, 'w') as f:
                json.dump(escalation, f, indent=2)
            logger.info(f"Task failure escalation created for {task_id}")
        except Exception as e:
            logger.error(f"Failed to create escalation file: {e}")
        
        return escalation
    
    def get_escalation_prompt(self, escalation: Dict[str, Any]) -> str:
        """
        タスク失敗エスカレーション用プロンプト生成
        
        Args:
            escalation: エスカレーション情報
            
        Returns:
            エスカレーションプロンプト
        """
        prompt = f"""🚨 タスク失敗のためレビューが必要です

## エラー情報
- **タスクID**: {escalation['task_id']}
- **タスクタイプ**: {escalation['task_type']}
- **リトライ回数**: {escalation['retry_count']}
- **作成時刻**: {escalation['created_at']}

## エラー詳細
```
{escalation['error']}
```

## 実行コマンド
```bash
{escalation['command']}
```

## 推奨アクション
"""
        
        for i, action in enumerate(escalation['suggested_actions'], 1):
            prompt += f"{i}. {action}\n"
        
        prompt += """
## 次のステップ
1. エラーの根本原因を分析
2. 推奨アクションから適切な対応を選択
3. 必要に応じてコード修正やパラメータ調整
4. タスクを再実行

このエラーをどのように解決すべきか検討してください。
"""
        
        return prompt
```

### 4. NotificationBridge（更新）

**目的**: TaskFailureEscalatorとの統合

```python
class NotificationBridge:
    """通知ブリッジ統合クラス"""
    
    def __init__(self, workspace_path: str, tracker_id: str = "QUAL-044"):
        self.workspace_path = workspace_path
        self.tracker_id = tracker_id
        
        # コンポーネント初期化
        self.pushover = PushoverNotifier()
        self.escalator = TaskFailureEscalator(workspace_path)  # 更新
        
        # 既存の機能は維持
        self.notification_history: Set[str] = set()
        self.last_notification_times: Dict[str, float] = {}
        # ... 他の初期化処理
    
    def handle_task_failure(self,
                          task_id: str,
                          task_type: str,
                          error: str,
                          retry_count: int,
                          command: str) -> Dict[str, Any]:
        """
        タスク失敗ハンドリング
        """
        logger.info(f"Handling task failure: {task_id}")
        
        # Pushover通知送信
        self.pushover.send_task_failed(
            task_id=task_id,
            task_type=task_type,
            error=error,
            retry_count=retry_count
        )
        
        # タスク失敗エスカレーション作成
        escalation = self.escalator.create_escalation(
            task_id=task_id,
            task_type=task_type,
            error=error,
            retry_count=retry_count,
            command=command
        )
        
        # 失敗ログ記録
        self._log_event({
            'event': 'task_failed',
            'task_id': task_id,
            'task_type': task_type,
            'error': error,
            'retry_count': retry_count,
            'escalation_created': True,
            'timestamp': datetime.now().isoformat()
        })
        
        return escalation
```

## CLI統合設計

### コマンド構造

```python
def main():
    """メインCLIエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="ワークフロー管理システム CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s plan TRACKER-001 "概要" "詳細"  # Google Sheets起票
  %(prog)s create TRACKER-001             # SQLiteワークフロー開始
  %(prog)s status TRACKER-001             # ワークフロー状態確認
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # planコマンド（新規）
    plan_parser = subparsers.add_parser('plan', help='Google Sheetsにトラッカーを起票')
    plan_parser.add_argument('tracker_id', help='トラッカーID (例: TRACKER-001)')
    plan_parser.add_argument('summary', help='概要')
    plan_parser.add_argument('details', help='詳細（最大20,000文字）')
    
    # createコマンド（更新）
    create_parser = subparsers.add_parser('create', help='SQLiteワークフローを開始')
    create_parser.add_argument('tracker_id', help='トラッカーID')
    
    # 既存コマンドは維持
    # ...
```

## マイグレーション戦略

### 1. 段階的リファクタリング

**Phase 1: 新機能追加**
- PlanCommandHandlerの実装
- planコマンドの追加
- 既存機能は維持

**Phase 2: 用語変更**
- PlanModeEscalator → TaskFailureEscalator
- planmode_escalation.json → task_failure_escalation.json
- 関連する全ての参照を更新

**Phase 3: 機能分離**
- createコマンドからGoogle Sheets機能を削除
- CreateCommandHandlerの更新
- 後方互換性の確保

### 2. ファイル移行

```python
def migrate_escalation_files(workspace_path: str):
    """エスカレーションファイルの移行"""
    old_file = Path(workspace_path) / "planmode_escalation.json"
    new_file = Path(workspace_path) / "task_failure_escalation.json"
    
    if old_file.exists() and not new_file.exists():
        shutil.move(str(old_file), str(new_file))
        logger.info(f"Migrated escalation file: {old_file} -> {new_file}")
```

## エラーハンドリング

### 入力検証エラー

```python
class ValidationError(Exception):
    """入力検証エラー"""
    pass

def validate_plan_inputs(tracker_id: str, summary: str, details: str):
    """planコマンド入力検証"""
    errors = []
    
    if not tracker_id.strip():
        errors.append("トラッカーIDが必要です")
    
    if not summary.strip():
        errors.append("概要が必要です")
    
    if not details.strip():
        errors.append("詳細が必要です")
    
    if len(details) > 20000:
        errors.append(f"詳細は20,000文字以内で入力してください（現在: {len(details)}文字）")
    
    if errors:
        raise ValidationError("\n".join(f"❌ {error}" for error in errors))
```

### Google Sheets API エラー

```python
def handle_sheets_error(error: Exception) -> str:
    """Google Sheets APIエラーハンドリング"""
    if "quota" in str(error).lower():
        return "Google Sheets APIの制限に達しました。しばらく待ってから再試行してください。"
    elif "permission" in str(error).lower():
        return "Google Sheetsへのアクセス権限がありません。設定を確認してください。"
    elif "network" in str(error).lower():
        return "ネットワークエラーが発生しました。接続を確認してください。"
    else:
        return f"Google Sheets APIエラー: {error}"
```

## テスト戦略

### 単体テスト

```python
class TestPlanCommandHandler(unittest.TestCase):
    def test_valid_plan_execution(self):
        """正常なplan実行テスト"""
        handler = PlanCommandHandler(mock_progress_manager)
        result = handler.execute_plan("TEST-001", "テスト概要", "テスト詳細")
        self.assertTrue(result)
    
    def test_detail_length_validation(self):
        """詳細文字数制限テスト"""
        handler = PlanCommandHandler(mock_progress_manager)
        long_details = "a" * 20001
        result = handler.execute_plan("TEST-001", "概要", long_details)
        self.assertFalse(result)

class TestTaskFailureEscalator(unittest.TestCase):
    def test_escalation_creation(self):
        """エスカレーション作成テスト"""
        escalator = TaskFailureEscalator("/tmp/test")
        escalation = escalator.create_escalation(
            "test_001", "pytest", "ImportError", 2, "python -m pytest"
        )
        self.assertEqual(escalation['task_id'], "test_001")
        self.assertIn('suggested_actions', escalation)
```

### 統合テスト

```python
class TestWorkflowIntegration(unittest.TestCase):
    def test_plan_to_create_workflow(self):
        """plan→createワークフローテスト"""
        # 1. planコマンド実行
        plan_result = execute_plan_command("TEST-001", "概要", "詳細")
        self.assertTrue(plan_result)
        
        # 2. createコマンド実行
        create_result = execute_create_command("TEST-001")
        self.assertTrue(create_result)
        
        # 3. 状態確認
        status = get_workflow_status("TEST-001")
        self.assertEqual(status['current_step'], 'branch_verification')
```

## パフォーマンス考慮事項

### Google Sheets API最適化

- バッチ処理の活用
- キャッシュ機能の実装
- リトライ機能の追加
- タイムアウト設定の最適化

### SQLite最適化

- インデックスの適切な設定
- トランザクション管理
- 接続プールの活用
- 定期的なVACUUM実行

この設計により、明確な責任分離と適切な用語使用を実現し、既存機能を維持しながら新しいワークフローを導入できます。