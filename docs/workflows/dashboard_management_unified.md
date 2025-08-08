# ダッシュボード統合管理ガイド

**最終更新**: 2025-08-08  
**バージョン**: 3.0（外部アクセス対応版）  
**目的**: 全ダッシュボード情報の一元管理と統一アクセス

---

## 🎯 概要

このドキュメントは、Segment Anythingプロジェクトにおける全ダッシュボード機能の統合管理情報を提供します。すべてのダッシュボード関連情報はこのファイルを参照してください。

---

## 🌐 ダッシュボードアクセス

### 統合ダッシュボードサーバー（ポート8088）

**内部アクセスURL**:
```
http://localhost:8088/tracker/{トラッカーID}
```

**外部アクセスURL（Tailscale経由）**:
```
http://100.123.241.106:8088/tracker/{トラッカーID}
```

**Basic認証情報**:
- ユーザー名: `admin`
- パスワード: `dashboard2025!`

**アクセス例**:
- INTEGRATE-3-6-01: http://100.123.241.106:8088/tracker/INTEGRATE-3-6-01
- INTEGRATE-3-6-02: http://100.123.241.106:8088/tracker/INTEGRATE-3-6-02
- INTEGRATE-3-6-03: http://100.123.241.106:8088/tracker/INTEGRATE-3-6-03
- INTEGRATE-3-6-04: http://100.123.241.106:8088/tracker/INTEGRATE-3-6-04
- P1-021: http://100.123.241.106:8088/tracker/P1-021
- PH2-002: http://100.123.241.106:8088/tracker/PH2-002

### ✅ ダッシュボード完成条件

**必須要件**: `http://localhost:8088/tracker/{トラッカーID}` が正常に表示されることがダッシュボード完成の絶対条件です。

### サーバー管理

```bash
# サーバー起動確認
ps aux | grep integrated_dashboard_server

# サーバー起動（停止している場合）
python3 integrated_dashboard_server.py &

# ダッシュボードリフレッシュ（新しいダッシュボード追加後）
curl http://localhost:8088/refresh
```

---

## 📁 ダッシュボードファイル構造

### 標準ディレクトリ構造

```
/mnt/c/AItools/lora/train/yado/tracker-workspace/
├── {TRACKER_ID}/
│   └── dashboard/
│       ├── dashboard.html          # メインダッシュボード（必須）
│       ├── quality_dashboard.html/ # 品質ダッシュボード（オプション）
│       └── *.png                  # チャート画像
└── main_dashboard.html            # プロジェクト全体ダッシュボード
```

### ファイル命名規則

- **メインダッシュボード**: `dashboard.html` （必須）
- **品質ダッシュボード**: `quality_dashboard.html/` 配下に配置
- **チャート画像**: 各ダッシュボードと同階層に配置

---

## 🔧 ダッシュボード生成

### 標準ワークフロー

1. **品質レポート生成**
   ```bash
   python3 create_phase1_extraction_report.py \
     /path/to/extraction/ \
     /path/to/tracker-workspace/{TRACKER_ID}/quality/extraction_report.json
   ```

2. **統合品質チェック**
   ```bash
   python3 tools/core/unified_quality_checker.py \
     --results /path/to/quality/extraction_report.json \
     --output /path/to/quality/unified_quality_report.json
   ```

3. **ダッシュボード生成**
   ```bash
   python3 tools/core/quality_dashboard.py \
     --report /path/to/quality/unified_quality_report.json \
     --output /path/to/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html
   ```

4. **サーバーリフレッシュ**
   ```bash
   curl http://localhost:8088/refresh
   ```

### 自動化スクリプト

```bash
# 品質ワークフロー実行（推奨）
./tools/scripts/run_quality_workflow.sh {TRACKER_ID}
```

---

## 📊 ダッシュボード種類

### 1. トラッカー別ダッシュボード

**内部URL**: `http://localhost:8088/tracker/{TRACKER_ID}`  
**外部URL**: `http://100.123.241.106:8088/tracker/{TRACKER_ID}`

**内容**:
- 抽出成功率
- 品質指標（LCA, A/B評価率, FPS, SCI）
- 比較チャート
- 改善提案

**生成コマンド**: `tools/core/quality_dashboard.py`

**最新追加トラッカー（INTEGRATE-3-6シリーズ）**:
- **INTEGRATE-3-6-01**: Phase 3-6統合初期版（PNG形式）
- **INTEGRATE-3-6-02**: Phase 3-6改良版（JPG形式、100%成功率）
- **INTEGRATE-3-6-03**: YOLO汎用版検証（79.2%成功率）
- **INTEGRATE-3-6-04**: アニメ特化版検証（100%成功率）

### 2. メインダッシュボード

**URL**: `http://localhost:8088/`

**内容**:
- プロジェクト全体統計
- トラッカー比較
- 進捗一覧

**生成コマンド**: `generate_main_dashboard.py`

### 3. リアルタイムダッシュボード（実験的）

**URL**: 各トラッカーのサブページ

**内容**:
- リアルタイム品質監視
- 処理進捗
- パフォーマンス監視

---

## 🛠 トラブルシューティング

### よくある問題

#### 1. ダッシュボードが表示されない

**症状**: `❌ ダッシュボードが見つかりません`

**解決方法**:
```bash
# 1. ファイル存在確認
ls /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/

# 2. dashboard.htmlが存在するか確認
ls /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html

# 3. サーバーリフレッシュ
curl http://localhost:8088/refresh

# 4. サーバー再起動
pkill -f integrated_dashboard_server
python3 integrated_dashboard_server.py &
```

#### 2. ダッシュボードが古い

**解決方法**:
```bash
# サーバーリフレッシュ
curl http://localhost:8088/refresh

# ブラウザキャッシュクリア（Ctrl+F5）
```

#### 3. ポート8088が使用できない

**解決方法**:
```bash
# ポート使用状況確認
netstat -tulpn | grep 8088

# プロセス終了
pkill -f integrated_dashboard_server

# 再起動
python3 integrated_dashboard_server.py &
```

---

## 🔍 API エンドポイント

### ダッシュボード一覧取得

```bash
curl http://localhost:8088/api/dashboards
```

**レスポンス例**:
```json
{
  "main": "/path/to/main_dashboard.html",
  "P1-021/dashboard": "/path/to/P1-021/dashboard/dashboard.html",
  "P1-A001/quality_dashboard": "/path/to/P1-A001/dashboard/quality_dashboard.html"
}
```

### サーバー情報

```bash
curl http://localhost:8088/
```

---

## 📚 関連ドキュメント

### 必須参照

- **設定**: `spec/OUTPUT_PATH_STANDARDS.md` - 出力パス標準
- **ワークフロー**: `docs/workflows/output_directory_config.md` - ディレクトリ構成
- **品質**: `docs/workflows/integrated_quality_check_guide.md` - 品質チェック

### 実装参照

- **サーバー**: `integrated_dashboard_server.py` - 統合サーバー実装
- **生成**: `tools/core/quality_dashboard.py` - ダッシュボード生成
- **設定**: `config/workspace_config.py` - ワークスペース設定

---

## 🚀 パフォーマンス最適化

### 推奨設定

```bash
# 環境変数設定
export TRACKER_WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/tracker-workspace"

# サーバー自動起動（オプション）
# ~/.bashrc に追加
alias dashboard-server='python3 /mnt/c/AItools/segment-anything/integrated_dashboard_server.py &'
```

### 監視

```bash
# サーバー状態確認
curl -s -o /dev/null -w "%{http_code}" http://localhost:8088/

# ログ確認
tail -f integrated_dashboard_server.log
```

---

## 📝 変更履歴

- **2025-08-02 v2.0**: 統合管理版作成、URL条件明確化
- **2025-07-xx v1.x**: 個別ダッシュボード管理（旧版）

**重要**: ダッシュボード関連の全情報はこのファイルに統合されました。他のダッシュボード関連ドキュメントは参照のみとし、メイン情報はこのファイルを参照してください。