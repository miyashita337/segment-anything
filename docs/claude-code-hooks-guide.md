# Claude Code Hook統合システム使用ガイド

**INTG-087: Hook統合・SubAgent統合・品質保証システム 完全版ガイド**

## 🎯 概要

このドキュメントはClaude Code Hook統合システムの使用方法と、INTG-087で実装された各種統合機能の完全ガイドです。

## 📋 システム構成

### 1. Hook統合システム
- **PreToolUse Hook**: コマンド実行前の検証・制御
- **PostToolUse Hook**: コマンド実行後の進捗記録
- **UserPromptSubmit Hook**: ユーザープロンプト分析・コンテキスト提供

### 2. SubAgent長時間タスクキューシステム
- **タスクキュー管理**: 長時間タスクの効率的な実行制御
- **リソース管理**: メモリ・CPU使用量の監視
- **直接実行防止**: extract_character.py等の必須SubAgent経由実行

### 3. 品質保証冪等性システム
- **冪等性テスト**: 複数回実行での結果一貫性検証
- **品質ダッシュボード**: 統合品質チェックシステム
- **統計分析**: 抽出結果の詳細分析・検証

### 4. ワークフロー進捗自動追跡システム
- **13ステップ追跡**: 4フェーズ・13ステップの詳細進捗管理
- **Google Sheets連携**: 外部システムとの進捗同期
- **チェックリスト状態管理**: 永続的な進捗状態保存

## 🔧 使用方法

### Hook システム基本操作

#### 1. Hook設定確認
```bash
# Hook設定ファイル確認
cat .claude/hooks.json

# Hook実行テスト
bash tools/hooks/test_hook_execution.sh
```

#### 2. ワークフローコンプライアンス検証 (PreToolUse)
自動実行：以下のコマンド実行時に自動的にチェック
- `python features/extraction/commands/extract_character.py`
- `python tools/progress_tracker/cli.py`
- `bash tools/scripts/run_quality_workflow.sh`

**検証内容:**
- トラッカーID抽出・検証
- ブランチ検証 (feature/TRACKER_ID形式)
- 入力パス存在チェック
- ワークフロー順序検証

#### 3. ワークフロー進捗更新 (PostToolUse)
自動実行：コマンド実行後に進捗を自動記録

**記録内容:**
- sam-env仮想環境確認完了
- Google Sheets連携完了
- 抽出処理実行・完了
- 品質ワークフロー実行・完了
- ファイル編集記録

#### 4. トラッカーコンテキスト分析 (UserPromptSubmit)
自動実行：ユーザープロンプト送信時に実行

**分析内容:**
- トラッカーID自動検出
- 現在フェーズ判定
- 5つのドキュメント準拠チェック
- 次ステップ推奨事項生成

### SubAgent システム使用方法

#### 1. 基本操作

```bash
# 環境変数設定（推奨）
export TRACKER_ID="INTG-087"
export TRACKER_WORKSPACE="/mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-087"

# タスクをキューに追加
python tools/queue/subagent_wrapper.py enqueue extract_character \
  "python features/extraction/commands/extract_character.py /input/path -o /output/path --batch"

# キューからタスク実行
python tools/queue/subagent_wrapper.py execute

# キュー状況確認
python tools/queue/subagent_wrapper.py status

# 完了タスククリーンアップ
python tools/queue/subagent_wrapper.py cleanup 7

# 実行中タスク停止
python tools/queue/subagent_wrapper.py kill <task_id>

# 全実行中タスク停止
python tools/queue/subagent_wrapper.py kill-all

# 実行中タスク一覧表示
python tools/queue/subagent_wrapper.py list-running
```

#### 2. extract_character.py 直接実行防止

**問題**: 直接実行を試みると以下のエラーが表示されます
```
❌ extract_character.py の直接実行は禁止されています

🚨 INTG-087 SubAgent統合システムにより、長時間タスクは必ず
SubAgentキューシステム経由で実行する必要があります。
```

**解決方法**: SubAgent経由での実行
```bash
# 正しい方法
python tools/queue/subagent_wrapper.py enqueue extraction_task \
  "python features/extraction/commands/extract_character.py /path/to/input -o /path/to/output --batch"

python tools/queue/subagent_wrapper.py execute
```

**緊急時の直接実行** (非推奨):
```bash
# 緊急時のみ使用
SUBAGENT_EXECUTION=true python features/extraction/commands/extract_character.py [args...]
```

### タスク停止機能詳細

#### 1. 実行中タスクの確認
```bash
python tools/queue/subagent_wrapper.py list-running
```

**出力例:**
```json
{
  "running_count": 1,
  "tasks": {
    "INTG-088_extract_1757923937": {
      "task_id": "INTG-088_extract_1757923937",
      "pid": 12345,
      "command": "python features/extraction/commands/extract_character.py ...",
      "started_at": "2025-09-15T08:12:23.749359+00:00",
      "status": "running",
      "cpu_percent": 15.2,
      "memory_mb": 1024,
      "status_detail": "running"
    }
  }
}
```

#### 2. 特定タスクの停止
```bash
# タスクIDを指定して停止
python tools/queue/subagent_wrapper.py kill INTG-088_extract_1757923937
```

**動作:**
- SIGTERM（正常終了要求）を送信
- 5秒待機後、応答がなければSIGKILL（強制終了）
- 実行中タスクリストから自動削除
- ログに停止理由を記録

#### 3. 全タスクの一括停止
```bash
# 実行中の全タスクを停止
python tools/queue/subagent_wrapper.py kill-all
```

#### 4. 安全な停止メカニズム
- **段階的停止**: SIGTERM → 5秒待機 → SIGKILL
- **状態管理**: タスクレジストリーに停止記録
- **プロセス確認**: psutilによる実際のプロセス存在確認
- **エラー処理**: 既に終了済みプロセスの適切な処理

### 品質保証システム使用方法

#### 1. 基本品質チェック
```bash
# 通常の品質チェック実行
python tools/testing/dashboard_quality_validator.py INTG-087

# サーバーURL指定
python tools/testing/dashboard_quality_validator.py INTG-087 \
  --server-url http://100.123.241.106:8088

# レポート保存
python tools/testing/dashboard_quality_validator.py INTG-087 \
  --save-report /tmp/quality_report.json
```

#### 2. 冪等性テスト (INTG-087新機能)
```bash
# 冪等性テスト実行（3回繰り返し）
python tools/testing/dashboard_quality_validator.py INTG-087 \
  --idempotency-test 3

# カスタム繰り返し回数
python tools/testing/dashboard_quality_validator.py INTG-087 \
  --idempotency-test 5 --save-report /tmp/idempotency_report.json
```

**冪等性テスト結果例:**
```
📊 冪等性テスト結果
==========================================
実行回数: 3
一貫性スコア: 98.50%
冪等性判定: ✅ 合格
```

### ワークフロー進捗追跡システム

#### 1. 13ステップワークフロー構成

**Phase 0.5: ブランチ検証**
1. feature/TRACKER_ID ブランチ確認

**Phase 1: 計画・準備フェーズ**
2. sam-env 仮想環境確認
3. Google Sheets 連携・進捗同期
4. SOW（作業範囲確定書）作成

**Phase 2: 実装フェーズ**
5. 実装作業開始
6. コード開発・修正
7. テスト実行・検証
8. 実装完了・承認

**Phase 3: 品質フェーズ**
9. 品質ワークフロー実行
10. ダッシュボード生成・検証
11. 最終品質確認
12. 完了報告・レビュー
13. Git操作・マージ

#### 2. 進捗状態ファイル構造

```json
{
  "tracker_id": "INTG-087",
  "created_at": "2025-09-09T12:00:00Z",
  "phase_0_5_branch": true,
  "phase_1_planning": {
    "sam_env_check": true,
    "google_sheets_sync": true,
    "sow_creation": true
  },
  "phase_2_implementation": {
    "started": true,
    "approval": false
  },
  "phase_3_quality": {
    "workflow_executed": false,
    "dashboard_created": false
  }
}
```

#### 3. 手動進捗更新

```bash
# Google Sheets進捗更新
python tools/progress_tracker/cli.py update INTG-087 "Phase 1完了"

# 進捗状況確認
python tools/progress_tracker/cli.py status INTG-087
```

## 📊 統合テストスイート

### 1. Hook統合テスト
```bash
# Hook統合システムテスト
python -m pytest tests/unit/test_hook_integration.py -v

# ワークフロー状態管理テスト  
python -m pytest tests/unit/test_workflow_state.py -v

# ワークスペース設定テスト
python -m pytest tests/unit/test_workspace_config.py -v
```

### 2. Hook実行テストスイート
```bash
# Hook実行テスト完全版
bash tools/hooks/test_hook_execution.sh

# テスト結果サマリー例:
# 📈 総テスト数: 15
# ✅ 成功: 13
# ❌ 失敗: 1  
# ⚠️ 部分成功: 1
# 🎯 成功率: 93.3%
```

### 3. SubAgent統合テスト
```bash
# SubAgentキューシステムテスト
python tools/queue/subagent_wrapper.py enqueue test_task "echo 'SubAgent test'"
python tools/queue/subagent_wrapper.py execute
python tools/queue/subagent_wrapper.py status
```

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. Hook実行エラー

**問題**: `bash: tools/hooks/validate_workflow_compliance.sh: Permission denied`
```bash
# 解決: 実行権限付与
chmod +x tools/hooks/*.sh
```

**問題**: `workspace_config.py エラー`
```bash
# 解決: Pythonパス確認
export PYTHONPATH="/mnt/c/AItools/segment-anything:$PYTHONPATH"
```

#### 2. SubAgent実行エラー

**問題**: `❌ extract_character.py の直接実行は禁止されています`
```bash
# 解決: SubAgent経由実行
python tools/queue/subagent_wrapper.py enqueue extract_task \
  "python features/extraction/commands/extract_character.py [args...]"
python tools/queue/subagent_wrapper.py execute
```

#### 3. 品質チェックエラー

**問題**: `extraction_result.json が存在しません`
```bash
# 解決: 品質ワークフロー実行
bash tools/scripts/run_quality_workflow.sh INTG-087
```

#### 4. ワークフロー進捗エラー

**問題**: `チェックリスト状態ファイルが存在しません`
```bash
# 解決: ワークスペースディレクトリ作成
mkdir -p /mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-087/.workflow
```

### デバッグモード

```bash
# 詳細ログ出力
export LOG_LEVEL=DEBUG
export VERBOSE=1

# Hook実行詳細ログ
bash -x tools/hooks/validate_workflow_compliance.sh

# SubAgent詳細ログ
python tools/queue/subagent_wrapper.py --debug status
```

## 📚 関連ドキュメント

- **統合テンプレート**: `docs/workflows/templates/unified_tracker_template.md`
- **ワークフローチェックリスト**: `docs/workflows/checklists/tracker_workflow_checklist.md`
- **入力パス検証**: `docs/checklists/input_path_validation_checklist.md`
- **ダッシュボード品質**: `docs/checklists/dashboard_quality_checklist.md`
- **技術仕様**: `docs/technical_specifications.md`

## 🎉 INTG-087 統合システム完成

INTG-087により、以下の統合システムが完成しました：

✅ **Hook統合システム** - 自動化されたワークフロー制御
✅ **SubAgent統合システム** - 長時間タスクの効率的管理
✅ **品質保証システム** - 冪等性・一貫性保証
✅ **進捗追跡システム** - 13ステップ詳細管理

これらのシステムにより、トラッカーベースのワークフローが大幅に改善され、品質・効率・追跡性が向上しました。

---

**作成日**: 2025-09-09  
**バージョン**: INTG-087 完全版  
**メンテナンス**: Claude Code Hook統合システム