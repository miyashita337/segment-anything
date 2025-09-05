# Google Sheets データ取得ツール

## 概要

Google Sheetsの進捗管理スプレッドシートからデータを取得・表示するツールです。

**スプレッドシートURL**: https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit  
**詳細**: `docs/integrations/external/google_sheets_reference.md` を参照

## 利用可能なスクリプト

### 1. `read_google_sheets.py` - 実際のAPI版
実際のGoogle Sheets APIを使用してリアルタイムでデータを取得します。

**前提条件**: Google Sheets API認証設定が必要
- `config/google_sheets_auth.json` - サービスアカウント認証ファイル

### 2. `read_google_sheets_demo.py` - デモ版（推奨）
API設定なしで即座に使用できるサンプルデータ版です。

## 基本的な使用方法（デモ版）

```bash
# 全タスク表示（最初の5件のみ）
python3 tools/read_google_sheets_demo.py --all --limit 5

# 統計情報表示
python3 tools/read_google_sheets_demo.py --stats

# 優先度別フィルタリング
python3 tools/read_google_sheets_demo.py --priority 優先度最高

# ステータス別フィルタリング
python3 tools/read_google_sheets_demo.py --status 実装完了

# 特定タスクの詳細表示
python3 tools/read_google_sheets_demo.py --tracker-id PH2-002 --detail

# JSON形式で出力
python3 tools/read_google_sheets_demo.py --all --json tasks.json
```

## コマンドオプション

### 基本表示オプション
- `--all` : 全タスク表示
- `--tracker-id ID` : 特定トラッカーIDで検索
- `--priority LEVEL` : 優先度でフィルタ（優先度最高/高/中/低）
- `--status STATUS` : ステータスでフィルタ
- `--stats` : 統計情報表示

### 出力制御オプション
- `--limit N` : 表示件数を制限
- `--detail` : 詳細表示モード
- `--json PATH` : JSON形式でファイル出力
- `--csv PATH` : CSV形式でファイル出力（実際のAPI版のみ）

### ヘルプ・設定
- `--setup-help` : Google Sheets API設定方法を表示
- `--help` : 使用方法を表示

## 表示例

### テーブル表示
```
========================================================================================================================
ID           優先度      ステータス        登録日          概要                                                          
========================================================================================================================
P1-005       優先度高     着手前          2025-07-27   自動マスク修正機能: マスクエッジ自動スムージング・ノイズ除去                             
P1-A001      優先度最高    実装完了         2025-07-27   改善コード復旧: deprecatedから本番環境への復帰                               
PH2-002      優先度中     リリース         2025-07-27   スケーラビリティ改善: 大規模データセット対応                                     
========================================================================================================================
```

### 統計情報
```
📊 サンプルタスク統計情報
============================================================
総タスク数: 20

🎯 優先度別統計:
  優先度最高     :   3件 ( 15.0%)
  優先度高      :   7件 ( 35.0%)
  優先度中      :   8件 ( 40.0%)
  優先度低      :   2件 ( 10.0%)

📈 ステータス別統計:
  着手前            :  12件 ( 60.0%)
  実装完了           :   2件 ( 10.0%)
  品質チェック         :   2件 ( 10.0%)
  着手中            :   2件 ( 10.0%)
  リリース           :   2件 ( 10.0%)
```

## Google Sheets API 実際の設定方法

実際のスプレッドシートデータを取得するには以下の設定が必要です：

### 1. Google Cloud Console設定
1. https://console.cloud.google.com/ にアクセス
2. 新しいプロジェクト作成 (例: "progress-tracker")
3. Google Sheets API を有効化

### 2. サービスアカウント作成
1. IAM & Admin → サービスアカウント
2. "サービスアカウントを作成" をクリック
3. 名前: "progress-tracker-service" (任意)
4. キーを作成 → JSON形式でダウンロード

### 3. スプレッドシート権限設定
1. スプレッドシートを開く
2. 共有ボタンをクリック
3. サービスアカウントのメールアドレスを追加
4. 権限: "編集者" に設定

### 4. 認証ファイル配置
ダウンロードしたJSONファイルを以下の場所に配置:
```
config/google_sheets_auth.json
```

### 5. 動作確認
```bash
python3 tools/read_google_sheets.py --all --limit 5
```

## 取得可能なデータ項目

- **基本情報**: トラッカーID、優先度、ステータス、登録日、更新日、概要
- **コンポーネント状況**: 動作確認、テストUNIT、品質評価、統合実行スクリプト、ダッシュボード生成、抽出パイプライン
- **メトリクス**: LCA、A/B評価率、FPS、C以上評価率、各種品質指標（SCI、PLA、PLE等）

## トラブルシューティング

### よくあるエラー

1. **認証ファイルが見つかりません**
   ```
   ❌ Google Sheets設定に問題があります:
      - 認証ファイルが見つかりません: config/google_sheets_auth.json
   ```
   → `--setup-help` で設定方法を確認してください

2. **API認証失敗**
   ```
   ❌ Google Sheets API接続失敗: Service account info was not in the expected format
   ```
   → サービスアカウントJSONファイルが正しく配置されているか確認

3. **権限エラー**
   → スプレッドシートにサービスアカウントの編集権限が付与されているか確認

### デモ版の利用
API設定が困難な場合は、デモ版をご利用ください：
```bash
python3 tools/read_google_sheets_demo.py --all
```

## 関連ファイル

- `tools/progress_tracker/` - Google Sheets API関連の設定・データモデル
- `tools/google_sheets_updater.py` - 書き込み用ツール
- `config/google_sheets_auth.json` - API認証ファイル（要設定）

## 注意事項

- API版は実際のスプレッドシートをリアルタイムで読み取ります
- デモ版はサンプルデータを使用しています
- 書き込みについては `google_sheets_updater.py` を使用してください
- API利用には Google Cloud Platform のアカウントが必要です