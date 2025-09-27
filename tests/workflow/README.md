# ワークフローCLIユニットテストガイド

## 概要

ワークフローシステムの全21コマンドに対する包括的なユニットテストスイートです。モック機能により実際のDB・Google Sheetsを更新せずにテストを実行できます。

## テストファイル構成

```
tests/workflow/
├── fixtures/
│   ├── __init__.py
│   ├── mock_data.py              # テストデータ定義
│   └── workflow_fixtures.py     # テスト基盤クラス
├── test_plan_command_handler.py   # planコマンドテスト
├── test_create_command_handler.py # createコマンドテスト
├── test_basic_workflow_commands.py # status, instructions, stepテスト
├── test_management_commands.py    # approvals, process, sheets, template, guideテスト
├── test_subagent_command_handler.py # SubAgentコマンドテスト
├── test_workflow_cli.py           # CLI統合テスト
└── README.md                      # このファイル
```

## テスト対象コマンド

### 基本コマンド (10個)
- `plan`: Google Sheets起票
- `create`: SQLiteワークフロー作成
- `status`: 状態取得
- `instructions`: 指示取得
- `step`: ステップ実行
- `approvals`: 承認リスト表示
- `process`: プロセス状態確認
- `sheets`: Google Sheets状態確認
- `template`: テンプレート生成
- `guide`: ガイド表示

### SubAgentコマンド (11個)
- `subagent-extraction`: 抽出処理開始
- `subagent-status`: 状態確認
- `subagent-wait`: 完了待機
- `subagent-retry`: 再実行
- `subagent-terminate`: 終了
- `subagent-cleanup`: ロッククリーンアップ
- `subagent-locks-status`: ロック状況確認
- `subagent-cleanup-all`: 全ロッククリーンアップ
- `subagent-auto-retry-check`: 自動再実行条件確認
- `subagent-auto-retry`: 自動再実行
- `subagent-auto-retry-all`: 全自動再実行

## モック戦略

### 主要なモック対象
- **ProgressManager**: Google Sheets API呼び出し
- **WorkflowController**: SQLite状態管理
- **SubAgentMonitor**: プロセス監視システム
- **SubAgentLockManager**: ロック管理
- **WorkspaceConfig**: 設定管理
- **ファイルシステム操作**: os.path, Path operations

### モック設定パターン
```python
# 基本的なモック設定
self.add_mock('module.function', return_value=mock_value)

# 複数回呼び出し対応
self.mock_manager.get_task.side_effect = [None, mock_task]

# エラーケーステスト
self.mock_manager.create_task.side_effect = Exception("Test error")
```

## テスト実行方法

### 個別テスト実行
```bash
# planコマンドテスト
python -m pytest tests/workflow/test_plan_command_handler.py -v

# 特定のテストケース
python -m pytest tests/workflow/test_plan_command_handler.py::TestPlanCommandHandler::test_plan_command_success -v
```

### 全体テスト実行
```bash
# 新規作成したテストのみ
python -m pytest tests/workflow/test_plan_command_handler.py tests/workflow/test_create_command_handler.py -v

# タイムアウト設定付き
python -m pytest tests/workflow/ --timeout=300 -v
```

### テストカバレッジ確認
```bash
# カバレッジ計測
python -m pytest tests/workflow/ --cov=tools.workflow --cov-report=html

# カバレッジレポート閲覧
open htmlcov/index.html
```

## テストデータ

### サンプルデータ
```python
SAMPLE_TRACKER_ID = "TEST-001"
SAMPLE_SUMMARY = "テスト用概要"
SAMPLE_DETAILS = "これはテスト用の詳細説明です。" * 10
SAMPLE_AUTHOR_NAME = "yado"
```

### モックタスクデータ
```python
MOCK_TASKS = {
    "TRACKER-001": MockTask(
        tracker_id="TRACKER-001",
        description="テスト用タスク1",
        status="planning",
        created_date="2025-09-27 10:00:00",
        updated_date="2025-09-27 10:00:00"
    )
}
```

## トラブルシューティング

### よくある問題

#### 1. ImportError: モジュールが見つからない
```bash
# 解決方法：Pythonパスを設定
export PYTHONPATH=/mnt/c/AItools/segment-anything:$PYTHONPATH
```

#### 2. AttributeError: モックが見つからない
```python
# 解決方法：正確なインポートパスを指定
self.add_mock('tools.workflow.plan_command_handler.ProgressManager')
```

#### 3. 複数回呼び出しのモック失敗
```python
# 解決方法：side_effectを使用
mock_manager.get_task.side_effect = [None, mock_task]
```

#### 4. テストタイムアウト
```bash
# 解決方法：タイムアウト値を調整
python -m pytest --timeout=600 tests/workflow/
```

### デバッグ方法

#### 1. モックの呼び出し確認
```python
# モックが呼ばれたか確認
self.mock_manager.create_task.assert_called_once()

# 呼び出し引数を確認
args, kwargs = self.mock_manager.create_task.call_args
print(f"Called with: {args}, {kwargs}")
```

#### 2. 実際の戻り値確認
```python
# 実際の戻り値をデバッグ出力
success, message = handler.execute_plan_command(...)
print(f"Success: {success}, Message: {message}")
```

#### 3. 例外詳細確認
```python
# 例外の詳細をキャッチ
try:
    result = handler.some_method()
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

## テスト拡張ガイド

### 新しいコマンドのテスト追加
1. `test_new_command.py` ファイルを作成
2. `WorkflowTestBase` を継承したテストクラスを作成
3. 必要なモックを `setUp` で設定
4. 正常系・異常系・境界値テストを実装

### 新しいモックデータ追加
1. `fixtures/mock_data.py` に新しいデータを追加
2. `MockData` クラスに取得メソッドを追加
3. テストで新しいデータを使用

### カスタムフィクスチャ作成
1. `fixtures/workflow_fixtures.py` に新しいヘルパーメソッドを追加
2. `WorkflowTestBase` クラスを拡張
3. 再利用可能なモック設定を提供

## ベストプラクティス

### テスト設計
- **単一責任**: 1つのテストで1つの機能をテスト
- **独立性**: テスト間で状態を共有しない
- **再現性**: 同じ条件で同じ結果が得られる

### モック設計
- **最小限**: 必要な部分のみをモック
- **現実的**: 実際の動作に近いモック
- **明示的**: モックの目的と動作を明確に

### エラーハンドリング
- **網羅的**: 正常系・異常系・境界値をテスト
- **具体的**: 期待されるエラーメッセージも検証
- **現実的**: 実際に発生しうるエラーをテスト

## 注意事項

### 実行環境
- sam-env仮想環境での実行を推奨
- CI環境では仮想環境チェックが自動スキップ
- Pythonパスの設定が必要

### モックの制限
- 実際のAPIやデータベースの動作とは異なる場合がある
- 複雑な状態遷移は完全にモックできない場合がある
- 実装変更時にモックの更新が必要

### パフォーマンス
- 大量のモックはテスト実行時間を増加させる
- 不要なモック設定は削除する
- 並列実行時の競合状態に注意

## 今後の改善点

### テストカバレッジ向上
- エッジケースの追加テスト
- 統合テストの拡充
- パフォーマンステストの追加

### モック精度向上
- 実際のAPIレスポンスに基づくモック
- 動的なモックデータ生成
- エラーケースの詳細化

### 自動化改善
- CI/CDパイプラインでの自動実行
- テスト結果の自動レポート生成
- 回帰テストの自動化

---

## 参考資料

- [unittest.mock ドキュメント](https://docs.python.org/3/library/unittest.mock.html)
- [pytest ドキュメント](https://docs.pytest.org/)
- [Python テストガイド](https://docs.python.org/3/library/unittest.html)