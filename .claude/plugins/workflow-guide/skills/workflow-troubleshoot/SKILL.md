---
description: |
  ワークフロー強制実行システム（workflow_cli.py）の使い方とトラブルシューティング。

  トリガーフレーズ:
  - 「workflow_cli.pyの使い方」
  - 「ワークフローでエラー」
  - 「ステップが進まない」
  - 「承認待ちで止まっている」
  - 「Google Sheets連携エラー」
---

# Workflow CLI Guide

ワークフロー強制実行システムの基本コマンド。

## ⚠️ 重要な前提条件

1. **1ワークフロー1画像抽出がマスト**: `create`を実行したら、画像抽出（SubAgent）は必ず通るステップ。タスクの種類に関係なく全ステップを完了する必要がある。

2. **stepでエラー → status/instructionsで解決策を探す**: エラーが出たら「このワークフローは不要」と判断せず、コマンドで状態を確認して解決する。

3. **ブランチ一致が必須**: `feature/{TRACKER_ID}`ブランチでないとワークフローは進行しない。

## Quick Reference

```bash
# 1. Google Sheets起票
python tools/workflow/workflow_cli.py plan {TRACKER_ID} "概要" "詳細" "作者名"

# 2. ローカルワークフロー開始
python tools/workflow/workflow_cli.py create {TRACKER_ID}

# 3. 現在のステップ指示確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}

# 4. ステップ実行
python tools/workflow/workflow_cli.py step {TRACKER_ID}

# 5. 状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 6. 承認待ち一覧
python tools/workflow/workflow_cli.py approvals
```

## Common Issues

### 「stepでエラーが発生した」

```bash
# 1. まず状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 2. 現在のステップで何をすべきか確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}

# 3. エラー内容に応じて対処
# - ブランチ不一致 → git checkout feature/{TRACKER_ID}
# - SubAgent未起動 → subagent-extractionコマンドを実行
# - 承認待ち → approvalsで確認して承認ファイル作成
```

### 「Google Sheets接続エラー」

```bash
# 設定確認
python tools/progress_tracker/cli.py check-config

# 接続テスト
python tools/progress_tracker/test_connection.py
```

### 「承認待ちでブロックされている」

```bash
# 承認待ち確認
python tools/workflow/workflow_cli.py approvals

# 承認ファイル作成（手動）
echo '{"approved": true, "approved_by": "ユーザー名"}' > .workflow_approvals/{APPROVAL_ID}_approved.json
```

### 「ステップが進まない」

```bash
# 状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 指示確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}
```

### 「quality_workflowでDashboard/Quality report not found」

**原因**: ステップごとの生成物と検証対象の対応

| ステップ | 生成物 | 検証対象 |
|---------|--------|---------|
| quality_workflow | unified_quality_report.json | unified_quality_report.json |
| dashboard_generation | dashboard.html, index.html | dashboard.html, index.html |

**対処**:
```bash
# quality_workflowの場合: 品質レポートを確認
ls -la {workspace}/{TRACKER_ID}/quality/unified_quality_report.json

# 存在しない場合: run_quality_workflow.shを手動実行
./tools/scripts/run_quality_workflow.sh {TRACKER_ID}

# dashboard_generationの場合: ダッシュボードを確認
ls -la {workspace}/{TRACKER_ID}/dashboard/dashboard.html
```

### 「間違ったブランチでcreateしてしまった」

ワークフロー状態をリセットする:
```bash
# SQLiteから該当トラッカーの状態を削除
sqlite3 workflow_state.db "DELETE FROM workflow_states WHERE tracker_id='{TRACKER_ID}';"

# 正しいブランチに切り替えてから再度create
git checkout -b feature/{TRACKER_ID}
python tools/workflow/workflow_cli.py create {TRACKER_ID}
```

---

詳細は `./references/` を参照:
- `./references/workflow_cli_guide.md` - CLIコマンド詳細
- `./references/cli_integration_guide.md` - 統合ワークフロー手順
