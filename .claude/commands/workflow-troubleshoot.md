---
name: "workflow-troubleshoot"
description: "workflow_cli.pyの使い方とトラブルシューティング"
---

# Workflow CLI Guide

ワークフロー強制実行システムの基本コマンドとトラブルシューティング。

## ⚠️ 重要な前提条件

1. **1ワークフロー1画像抽出がマスト**: `create`を実行したら、画像抽出は必ず通るステップ
2. **stepでエラー → status/instructionsで解決策を探す**: エラーが出たらコマンドで状態を確認
3. **ブランチ一致が必須**: `feature/{TRACKER_ID}`ブランチでないと進行しない

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

## Step-Specific Error Patterns

### quality_workflow ステップ
| エラー | 対処法 |
|--------|--------|
| `Quality report not found` | `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}` |

### dashboard_generation ステップ
| エラー | 対処法 |
|--------|--------|
| `必須ファイル不在: ダッシュボードHTML` | `python features/evaluation/dashboard_generator.py --tracker-id {TRACKER_ID}` |
| `Dashboard HTML not found` | `cp {workspace}/{TRACKER_ID}/dashboard/dashboard.html {workspace}/{TRACKER_ID}/index.html` |

### final_approval ステップ
| エラー | 対処法 |
|--------|--------|
| `承認待ち` | `python tools/workflow/workflow_cli.py approvals` で確認し、承認ファイルを作成 |

### extraction ステップ
| エラー | 対処法 |
|--------|--------|
| `入力ディレクトリが存在しません` | `{workspace}/{TRACKER_ID}/source_images/` に入力画像を配置 |

## Common Issues

### stepでエラーが発生した
```bash
python tools/workflow/workflow_cli.py status {TRACKER_ID}
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}
```

### 間違ったブランチでcreateしてしまった
```bash
sqlite3 workflow_state.db "DELETE FROM workflow_states WHERE tracker_id='{TRACKER_ID}';"
git checkout -b feature/{TRACKER_ID}
python tools/workflow/workflow_cli.py create {TRACKER_ID}
```

### Google Sheets接続エラー
```bash
python tools/progress_tracker/cli.py check-config
python tools/progress_tracker/test_connection.py
```
