# Tools Manager 使用ガイド

統合管理CLI (`tools/manager.py`) は、Tools Directory内の各種ツールを統一インターフェースで管理するためのツールです。

## インストール

追加インストールは不要です。既存の環境で動作します。

```bash
python tools/manager.py --help
```

## 主要機能

### 1. Google Sheets タスク管理

#### タスク一覧表示
```bash
# 全タスク表示
python tools/manager.py sheets list

# 優先度でフィルタ
python tools/manager.py sheets list --priority 優先度最高

# ステータスでフィルタ
python tools/manager.py sheets list --status 着手中

# 表示件数制限
python tools/manager.py sheets list --limit 10
```

#### タスク詳細表示
```bash
python tools/manager.py sheets read TDR-002
python tools/manager.py sheets read P1-A001
```

#### タスク更新
```bash
# ステータス更新
python tools/manager.py sheets update TDR-002 --status 実装完了

# 優先度更新（将来実装）
python tools/manager.py sheets update TDR-002 --priority 優先度中
```

### 2. バッチ処理管理

#### バッチスクリプト一覧
```bash
python tools/manager.py batch list
```

出力例：
```
============================================================
バッチ処理スクリプト一覧
============================================================
 1. batch_task_ticketing.py                      13.7 KB  2025-07-27 20:23
 2. kana08_enhanced_stable_batch.py              15.3 KB  2025-07-27 22:44
 3. kana08_stable_batch_restored.py              13.1 KB  2025-07-27 21:45

合計: 3ファイル
```

#### バッチ実行
```bash
# スクリプト名で実行
python tools/manager.py batch run kana08_enhanced_stable_batch

# 拡張子付きでも可
python tools/manager.py batch run batch_task_ticketing.py

# 引数を渡す
python tools/manager.py batch run kana08_enhanced --input_dir ./test_small
```

### 3. ツール整理・メンテナンス

#### 古いスクリプトの検索・アーカイブ
```bash
# 30日以上前のスクリプト検索（デフォルト）
python tools/manager.py cleanup

# 60日以上前のスクリプト検索
python tools/manager.py cleanup --days 60

# 自動アーカイブモード（確認なし）
TOOLS_MANAGER_AUTO_MODE=1 python tools/manager.py cleanup --days 30
```

#### 個別ファイルのアーカイブ
```bash
# 特定ファイルをdeprecated/に移動
python tools/manager.py archive tools/scripts/old_script.py
```

### 4. 統計情報

```bash
python tools/manager.py stats
```

出力例：
```
============================================================
Tools Directory 統計情報
============================================================
core           :   6 ファイル
batch          :   3 ファイル
testing        :   6 ファイル
scripts        :   6 ファイル
utils          :   4 ファイル
legacy         :   4 ファイル
progress_tracker:  10 ファイル
------------------------------
合計           :  39 ファイル
総サイズ       :   498.9 KB
```

## 環境変数

- `TOOLS_MANAGER_AUTO_MODE`: 1に設定すると対話的確認をスキップ（自動化用）

## ユースケース

### 日次メンテナンス
```bash
# 統計確認
python tools/manager.py stats

# 古いスクリプト整理
python tools/manager.py cleanup --days 7
```

### タスク管理ワークフロー
```bash
# 優先度最高のタスク確認
python tools/manager.py sheets list --priority 優先度最高

# タスク詳細確認
python tools/manager.py sheets read TDR-002

# ステータス更新
python tools/manager.py sheets update TDR-002 --status 着手中
```

### バッチ処理実行
```bash
# 利用可能なバッチ確認
python tools/manager.py batch list

# バッチ実行
python tools/manager.py batch run kana08_enhanced_stable_batch
```

## トラブルシューティング

### Google Sheets API認証エラー
- `config/google_sheets_auth.json`が存在することを確認
- 適切な権限が設定されていることを確認

### バッチ実行エラー
- スクリプト名が正しいか確認
- 必要な引数が渡されているか確認
- スクリプトが実行可能か確認

### アーカイブエラー
- ファイルパスが正しいか確認
- deprecated/tools_archive/ディレクトリへの書き込み権限を確認

---
*TDR-002実装 - 2025-07-28*