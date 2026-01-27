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

---

詳細は `./references/` を参照:
- `./references/workflow_cli_guide.md` - CLIコマンド詳細
- `./references/cli_integration_guide.md` - 統合ワークフロー手順
