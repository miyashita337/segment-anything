# Google Sheets進捗管理システム セットアップガイド（2025年版）

**⚠️ 重要**: 最新情報は [`docs/google_sheets_reference.md`](../google_sheets_reference.md) を参照してください。

**最終更新**: 2025-07-28  
**実装状況**: ✅ 完全実装済み・運用中  
**構成**: 23列拡張システム（A-W列）対応

## 概要
PROGRESS_TRACKER.mdからGoogle Sheetsへの移行により、リアルタイム進捗管理とClaude Codeによる自動更新を実現します。現在、PH2-002で実装されたSpreadSheet API修正により、安定した運用が可能になっています。

## 🔧 セットアップ手順（実装済み）

**注意**: 以下の手順は既に完了しており、現在運用中です。新規環境構築時の参考として残しています。

### 1. Google Cloud Console設定 ✅

#### a) プロジェクト作成 ✅
1. https://console.cloud.google.com/ にアクセス
2. 新しいプロジェクトを作成 (例: "progress-tracker")
3. プロジェクトを選択

#### b) Google Sheets API有効化 ✅
1. ナビゲーションメニュー → ライブラリ
2. "Google Sheets API" を検索
3. 「有効にする」をクリック

### 2. サービスアカウント作成 ✅

#### a) サービスアカウント設定 ✅
1. ナビゲーションメニュー → IAM & Admin → サービスアカウント
2. 「サービスアカウントを作成」をクリック
3. サービスアカウント名: `progress-tracker-service`
4. 説明: `進捗管理システム用サービスアカウント`
5. 「作成して続行」をクリック

#### b) 権限設定 ✅
- 役割は設定しない（スプレッドシート個別権限で管理）

## 🔧 現在の運用状況

### 実装されている機能
- ✅ **23列管理**: A-W列での包括的進捗管理（10指標完全対応）
- ✅ **自動シート名検出**: 「シート1」等の実際の名前を自動認識
- ✅ **Unicode範囲指定**: 日本語シート名への対応
- ✅ **エラーハンドリング**: API障害時の詳細案内と手動対応
- ✅ **接続監視**: リアルタイム接続状況の監視

### Claude Code コマンド
```bash
# 接続状況確認
python3 tools/progress_tracker/cli.py connection-status

# タスクステータス更新
python3 tools/progress_tracker/cli.py update [タスクID] [新ステータス]

# 品質指標更新
python3 tools/progress_tracker/cli.py update-metrics [タスクID] --pla 0.85 --sci 0.75

# 設定確認
python3 tools/progress_tracker/cli.py check-config
```
- 「完了」をクリック

#### c) 認証キー作成
1. 作成したサービスアカウントをクリック
2. 「キー」タブ → 「キーを追加」 → 「新しいキーを作成」
3. 形式: JSON を選択
4. 「作成」をクリック
5. JSONファイルが自動ダウンロードされる

### 3. スプレッドシート権限設定

#### a) スプレッドシートアクセス
1. https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit
2. 「共有」ボタンをクリック

#### b) サービスアカウント権限付与
1. ダウンロードしたJSONファイルを開く
2. `client_email` の値をコピー (例: `progress-tracker-service@your-project.iam.gserviceaccount.com`)
3. スプレッドシートの共有画面でこのメールアドレスを追加
4. 権限: 「編集者」に設定
5. 「送信」をクリック

### 4. 認証ファイル配置

#### a) ファイル配置
```bash
# プロジェクトルートから実行
mkdir -p config
cp ~/Downloads/your-service-account-file.json config/google_sheets_auth.json
```

#### b) ファイル権限設定
```bash
chmod 600 config/google_sheets_auth.json
```

### 5. 動作確認

#### a) 設定確認
```bash
python tools/progress_tracker/cli.py check-config
```

期待される出力:
```
🔧 進捗管理システム設定確認
==================================================
設定状態: ✅ 正常
認証ファイル: ✅ 存在
スプレッドシートID: ✅ 設定済み

✅ Google Sheets接続成功
📊 現在のタスク数: 0
```

#### b) シート初期化
```bash
python tools/progress_tracker/cli.py init
```

#### c) 初期データ移行
```bash
# ドライラン（確認のみ）
python tools/progress_tracker/migration_tool.py --markdown-path docs/workflows/PROGRESS_TRACKER.md

# 実際の移行実行
python tools/progress_tracker/migration_tool.py --markdown-path docs/workflows/PROGRESS_TRACKER.md --execute
```

## 🚀 使用方法

### CLI コマンド

#### ステータス確認
```bash
python tools/progress_tracker/cli.py status
```

#### 新規タスク作成
```bash
python tools/progress_tracker/cli.py create PH2-004 --description "新機能実装"
```

#### ステータス更新
```bash
python tools/progress_tracker/cli.py update PH2-004 "実装完了"
```

#### タスク一覧表示
```bash
python tools/progress_tracker/cli.py list
python tools/progress_tracker/cli.py list --status "着手中"
```

#### 10指標更新
```bash
# 個別指標更新
python tools/progress_tracker/cli.py update-metrics PH2-001 --lca 0.85 --ab-rate 0.67 --fps 12.5

# 品質チェッカー結果インポート
python tools/progress_tracker/cli.py import-quality PH2-001 results/quality_report.json
```

### ワークフロー統合

#### ワークフロー統合実行
```bash
# 品質ワークフロー実行（10指標自動取得）
python tools/progress_tracker/workflow_integration.py PH2-001 --mode quality --quality-method balanced

# 抽出パイプライン実行
python tools/progress_tracker/workflow_integration.py PH2-001 --mode extraction

# ワークフロー完了マーク
python tools/progress_tracker/workflow_integration.py PH2-001 --mode complete
```

#### 従来スクリプトとの統合
```bash
# ワークフロー実行時に自動更新
./bin/shell/run_quality_workflow.sh

# リリース時に自動更新  
./bin/shell/release.sh
```

## 📊 スプレッドシート構成

**⚠️ 重要**: 最新の列構成・指標定義は [`docs/google_sheets_reference.md`](../google_sheets_reference.md) を参照してください。

**現在の構成**: 23列拡張システム（A-W列）
- **基本情報**: A-G列（トラッカーID、優先度、ステータス等）  
- **コンポーネント状況**: H-M列（動作確認、テスト、品質評価等）
- **10指標**: N-W列（LCA、A/B評価率、FPS、SCI、PLA、PLE等）

### ステータス種類
- `着手前` - 未着手状態
- `着手中` - 実装開始
- `実装完了` - コード実装完了
- `動作確認` - 動作確認済み
- `テストUNIT` - 単体テスト完了
- `品質チェック` - 品質評価完了
- `抽出パイプライン` - パイプライン動作確認済み
- `リリース` - 完全完了

## 🔧 トラブルシューティング

### 認証エラー
```
❌ Google Sheets API認証失敗
```
**対処法:**
1. 認証ファイルパスを確認: `config/google_sheets_auth.json`
2. ファイル形式がJSONかチェック
3. サービスアカウントキーが正しいかチェック

### 権限エラー
```
❌ API実行失敗: The caller does not have permission
```
**対処法:**
1. スプレッドシートの共有設定を確認
2. サービスアカウントメールが「編集者」権限を持っているかチェック
3. スプレッドシートIDが正しいかチェック

### 接続エラー
```
❌ シートデータ取得失敗
```
**対処法:**
1. インターネット接続を確認
2. Google Sheets APIが有効化されているかチェック
3. スプレッドシートが存在するかチェック

## ⚠️ セキュリティ注意事項

1. **認証ファイルの保護**
   - `config/google_sheets_auth.json` はGitにコミットしない
   - `.gitignore` に追加済み
   - ファイル権限を600に設定

2. **スプレッドシートアクセス**
   - サービスアカウントにのみ必要最小限の権限を付与
   - 個人アカウントでの共有は避ける
   - 定期的にアクセスログを確認

## 📈 運用フロー

### 開発時
1. タスク開始時: `python tools/progress_tracker/cli.py update TASK-ID "着手中"`
2. 実装完了時: `python tools/progress_tracker/cli.py update TASK-ID "実装完了"`
3. ワークフロー実行: `./bin/shell/run_quality_workflow.sh` (自動更新)
4. リリース: `./bin/shell/release.sh` (自動更新)

### 進捗確認
- **リアルタイム**: Google Sheetsで直接確認
- **CLI**: `python tools/progress_tracker/cli.py status`
- **レポート**: Google Sheetsの集計機能を活用

これで、従来のMarkdownベースの進捗管理から、リアルタイムで共有可能なGoogle Sheetsベースのシステムに移行できます。