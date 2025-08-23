# Google Sheets進捗管理システム - 統一リファレンス

**📋 このドキュメントについて**
このドキュメントは、Google Sheets進捗管理システムの**唯一の正式な参照元**です。
すべての関連ドキュメント・コード・設定は、この情報を基準として作成・更新されています。

**⚠️ 重要**: Google Sheets関連の情報は、必ずこのドキュメントを参照してください。

---

## 🔗 Google Sheetsアクセス情報

### メインスプレッドシート
**URL**: https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit?gid=0#gid=0

**Spreadsheet ID**: `10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA`

**シート名**: `シート1` (実際の運用名、設定上は`Progress Tracker`)

**更新日時**: 2025-07-28 (最新更新確認日)

---

## 📊 列構成仕様（23列: A-W）

**最終更新**: 2025-07-28  
**構成**: 23列拡張システム（A-W列）

### 基本情報列（A-G列）
| 列 | フィールド名 | 説明 | 例 |
|---|---|---|---|
| A | tracker_id | トラッカーID | P1-010, PHS-005 |
| B | priority | 優先度 | 優先度最高, 優先度高, 優先度中, 優先度低 |
| C | status | ステータス | 着手前, 着手中, /release |
| D | created_date | 登録日付 | 2025-07-28 |
| E | updated_date | 更新日付 | 2025-07-28 |
| F | description | 概要 | 自動マスク修正機能実装 |
| G | details | 詳細 | 66.57%改善率達成 |

### コンポーネント状況列（H-M列）
| 列 | フィールド名 | 説明 | 値 |
|---|---|---|---|
| H | operation_check | 動作確認 | 完了, 失敗, 未実行, 実行中 |
| I | unit_test | テストUNIT | 完了, 失敗, 未実行, 実行中 |
| J | quality_evaluation | 品質評価 | 完了, 失敗, 未実行, 実行中 |
| K | integration_script | 統合実行スクリプト | 完了, 失敗, 未実行, 実行中 |
| L | dashboard_generation | ダッシュボード生成 | 完了, 失敗, 未実行, 実行中 |
| M | extraction_pipeline | 抽出パイプライン | 完了, 失敗, 未実行, 実行中 |

### 統計分析列（X-AC列）
| 列 | 指標名 | 説明 | 範囲 |
|---|---|---|---|
| X | current_score | Current品質スコア | 0.0-1.0 |
| Y | baseline_score | BaseLine品質スコア | 0.0-1.0 |
| Z | p_value | p値（統計的有意性） | 0.0-1.0 |
| AA | effect_size | 効果サイズ（Cohen's d） | -∞ to +∞ |
| AB | improvement_rate | 改善率（%） | -100% to +∞ |
| AC | statistical_significance | 統計的有意性 | 有意/非有意 |

---

## 🔧 API設定情報

### 認証設定
**認証ファイル**: `config/google_sheets_auth.json`  
**認証方式**: Google Service Account  
**必要権限**: Google Sheets API (読み取り・書き込み)  

### API制限
**読み取り**: 100 requests/100秒/ユーザー  
**書き込み**: 100 requests/100秒/ユーザー  
**セル更新**: 500 requests/100秒/ユーザー  

---

## 📋 ステータス定義

### タスクステータス
- `着手前` - 未着手状態
- `着手中` - 実装開始
- `実装完了` - コード実装完了
- `動作確認` - 動作確認済み  
- `テストUNIT` - 単体テスト完了
- `品質チェック` - 品質評価完了
- `抽出パイプライン` - パイプライン動作確認済み
- `/release` - 完全完了・リリース済み
- `終了` - プロジェクト終了

### コンポーネントステータス
- `` (空) - 未実行
- `完了` - 正常完了
- `失敗` - エラーまたは失敗
- `スキップ` - 意図的スキップ
- `実行中` - 処理中

---

## 🚀 操作方法

### CLI コマンド
```bash
# ステータス確認
python3 tools/progress_tracker/cli.py status

# タスク更新
python3 tools/progress_tracker/cli.py update P1-010 "/release"

# 指標更新
python3 tools/progress_tracker/cli.py update-metrics P1-010 --current 0.85 --baseline 0.67

# 接続確認
python3 tools/progress_tracker/cli.py connection-status

# 権限管理（INCI-001新機能）
python3 tools/progress_tracker/cli.py permission-status    # 現在の実行権限確認
python3 tools/progress_tracker/cli.py set-permission LEVEL # 権限レベル設定
python3 tools/progress_tracker/cli.py permission-audit     # 権限変更履歴確認
```

### 自動更新トリガー
1. **品質ワークフロー実行時**: `./tools/scripts/run_quality_workflow.sh`
2. **ダッシュボード生成時**: `tools/core/quality_dashboard.py`
3. **抽出パイプライン完了時**: `features/extraction/commands/extract_character.py`

---

## 🔄 更新履歴

### 2025-07-28
- 23列システム（A-W）への拡張完了
- 統計指標追加（X-AC列: Current/BaseLine/p値/効果サイズ/改善率/統計的有意性）
- P1-010での動作確認完了

### 2025-07-27  
- 初期21列システム（A-U）運用開始
- Google Sheets API連携実装

---

## ⚠️ 注意事項

### セキュリティ
- 認証ファイル（`config/google_sheets_auth.json`）はGit管理対象外
- サービスアカウントキーの漏洩に注意
- スプレッドシート共有権限は必要最小限に設定

### 運用ルール
- **このドキュメントが情報の正**です
- Google Sheets構造変更時は、必ずこのドキュメントを最初に更新
- 他のドキュメントはこのドキュメントを参照する形で記載

### トラブルシューティング
- 列数エラー: このドキュメントの列構成を確認
- 認証エラー: `tools/progress_tracker/cli.py connection-status`で診断
- API制限エラー: 1分待機後に再試行

---

**📞 サポート**: このドキュメントの情報に疑問がある場合は、Claude Codeに相談してください。