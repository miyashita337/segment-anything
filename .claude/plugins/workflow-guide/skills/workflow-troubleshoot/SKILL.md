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

## Step-Specific Error Patterns

ステップ別のエラーパターンと対処法。stepコマンドでエラーが発生したらこのセクションを参照。

### quality_workflow ステップ

| エラーパターン | 原因 | 対処法 |
|---------------|------|--------|
| `Quality report not found` | unified_quality_report.jsonが未生成 | `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}` |
| `Dashboard/Quality report not found` | 品質レポートまたはダッシュボードが未生成 | 下記の詳細対処を参照 |

**詳細対処**:
```bash
# 品質レポートを確認
ls -la {workspace}/{TRACKER_ID}/quality/unified_quality_report.json

# 存在しない場合: run_quality_workflow.shを手動実行
./tools/scripts/run_quality_workflow.sh {TRACKER_ID}
```

### dashboard_generation ステップ

| エラーパターン | 原因 | 対処法 |
|---------------|------|--------|
| `必須ファイル不在: ダッシュボードHTMLファイル` | dashboard.htmlが未生成 | `python features/evaluation/dashboard_generator.py --tracker-id {TRACKER_ID}` |
| `統計分析結果でBaseLineが有効な値ではありません` | 検証ロジックのバグ（`: 0`パターン） | workflow_controller.pyの検証ロジック修正が必要 |
| `Dashboard HTML not found` | index.htmlへのコピーが未完了 | 下記の詳細対処を参照 |

**詳細対処**:
```bash
# ダッシュボードを確認
ls -la {workspace}/{TRACKER_ID}/dashboard/dashboard.html

# 存在しない場合: ダッシュボード生成
python features/evaluation/dashboard_generator.py --tracker-id {TRACKER_ID}

# index.htmlへのコピー
cp {workspace}/{TRACKER_ID}/dashboard/dashboard.html {workspace}/{TRACKER_ID}/index.html

# 統合サーバー更新
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/refresh
```

### final_approval ステップ

| エラーパターン | 原因 | 対処法 |
|---------------|------|--------|
| `承認待ち` | ユーザーの承認が必要 | 承認ファイル作成 |
| `Approval file not found` | 承認ファイルが未作成 | 下記の詳細対処を参照 |

**詳細対処**:
```bash
# 承認待ち確認
python tools/workflow/workflow_cli.py approvals

# 承認ファイル作成
echo '{"approved": true, "approved_by": "ユーザー名"}' > .workflow_approvals/{APPROVAL_ID}_approved.json
```

### extraction ステップ

| エラーパターン | 原因 | 対処法 |
|---------------|------|--------|
| `入力ディレクトリが存在しません` | source_imagesが未設定 | 入力画像を配置 |
| `SubAgent未起動` | 抽出処理が未実行 | `subagent-extraction`コマンドを実行 |

**詳細対処**:
```bash
# 入力ディレクトリ確認
ls -la {workspace}/{TRACKER_ID}/source_images/

# 入力画像を配置してから抽出実行
python features/extraction/commands/extract_character.py --batch --input {workspace}/{TRACKER_ID}/source_images/
```

---

## Common Issues

汎用的なトラブルシューティング。ステップ固有でないエラーに対応。

### 「stepでエラーが発生した」

```bash
# 1. まず状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 2. 現在のステップで何をすべきか確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}

# 3. エラー内容に応じて対処（上記Step-Specific Error Patternsを参照）
```

### 「Google Sheets接続エラー」

```bash
# 設定確認
python tools/progress_tracker/cli.py check-config

# 接続テスト
python tools/progress_tracker/test_connection.py
```

### 「間違ったブランチでcreateしてしまった」

```bash
# SQLiteから該当トラッカーの状態を削除
sqlite3 workflow_state.db "DELETE FROM workflow_states WHERE tracker_id='{TRACKER_ID}';"

# 正しいブランチに切り替えてから再度create
git checkout -b feature/{TRACKER_ID}
python tools/workflow/workflow_cli.py create {TRACKER_ID}
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
