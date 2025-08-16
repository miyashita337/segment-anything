# P1-021: 週次ソースコード肥大化解決システム使用方法

## 概要

P1-021は、プロジェクト内の不要なファイル・肥大化要因を週1回自動検出・整理する統合システムです。

## 主要機能

### 1. ログファイル整理
- 7日以上古いログファイルの自動削除
- ログディレクトリの圧縮・整理
- 対象: `logs/`, `workspace/`, `*.log`

### 2. 一時ファイル・キャッシュ除去
- Python キャッシュ (`__pycache__`, `*.pyc`)
- 一時ファイル (`*.tmp`, `*.cache`)
- ビルド成果物 (`build/`, `dist/`, `.pytest_cache/`)

### 3. 画像ファイル整理
- プロジェクトルート直下の画像を `deprecated/misplaced_images/` に移動
- Git管理外の画像ファイルの適切な配置

### 4. Git最適化
- `git gc --prune=now` による最適化
- 到達不能オブジェクトの除去
- パックファイル最適化

## 使用方法

### 手動実行

```bash
# ドライラン（削除せずに確認のみ）
python3 tools/scripts/weekly_cleanup.py --dry-run --verbose

# 実際の実行（インタラクティブ確認あり）
python3 tools/scripts/weekly_cleanup.py --verbose

# 自動実行（確認なし、cron用）
python3 tools/scripts/weekly_cleanup.py --force --verbose
```

### 自動実行設定（cron）

```bash
# crontab設定（毎週日曜日午前2時）
crontab -e

# 以下を追加
0 2 * * 0 /mnt/c/AItools/segment-anything/bin/shell/weekly_cleanup_cron.sh
```

### テスト実行

```bash
# cron用スクリプトのテスト
./bin/shell/weekly_cleanup_cron.sh --test
```

## 安全性・除外設定

### 保護されるファイル・ディレクトリ

- **重要ファイル**: 
  - SAMモデル (`sam_vit_*.pth`)
  - YOLOモデル (`yolov8*.pt`)
  - LoRAモデル (`*.safetensors`)
  - 設定ファイル (`requirements.txt`, `setup.py`)

- **重要ディレクトリ**:
  - `.git/`
  - `sam-env/`, `venv/`
  - `core/segment_anything/` (Meta原本実装)
  - `tests/fixtures/` (テストデータ)

### カスタマイズ

`tools/scripts/weekly_cleanup.py` の `exclude_patterns` と `cleanup_targets` を編集することで、保護対象・削除対象をカスタマイズ可能。

## ログ・レポート

### 実行ログ
- 場所: `logs/weekly_cleanup/`
- 形式: `cron_YYYYMMDD_HHMMSS.log`
- 内容: 実行詳細、削除ファイル一覧、エラー情報

### クリーンアップレポート
- 場所: `logs/weekly_cleanup/cleanup_report_YYYYMMDD_HHMMSS.json`
- 内容: 削除ファイル数、解放容量、統計情報

### 通知
- Pushover設定がある場合、完了通知を自動送信
- 成功/エラー別に通知レベルを調整

## 期待効果

- **ディスク容量**: 不要ファイル除去により10-50%の容量削減
- **Git性能**: リポジトリ最適化により操作速度向上
- **開発効率**: 整理されたプロジェクト構造による可視性向上
- **メンテナンス負荷**: 手動管理からの解放

## トラブルシューティング

### よくある問題

1. **権限エラー**
   ```bash
   chmod +x tools/scripts/weekly_cleanup.py
   chmod +x bin/shell/weekly_cleanup_cron.sh
   ```

2. **Python環境エラー**
   ```bash
   # 仮想環境の確認
   which python3
   python3 --version
   ```

3. **ディスク容量不足**
   ```bash
   # 事前確認
   df -h /mnt/c/AItools/segment-anything
   ```

### ログ確認

```bash
# 最新のログ確認
ls -la logs/weekly_cleanup/
tail -50 logs/weekly_cleanup/cron_*.log
```

### 緊急停止

実行中のクリーンアップを停止する場合:
```bash
pkill -f weekly_cleanup.py
```

## 開発者向け情報

### アーキテクチャ

- `WeeklyCleanupManager`: 統合管理クラス
- モジュラー設計: 各清掃タスクは独立実行可能
- エラーハンドリング: 一部失敗でも他タスクは継続
- レポート機能: JSON形式での詳細統計

### 拡張方法

新しいクリーンアップタスクの追加:

1. `WeeklyCleanupManager` にメソッド追加
2. `run_full_cleanup()` から呼び出し
3. `cleanup_targets` に対象パターン追加

### テスト

```bash
# 単体テスト
python3 -m pytest tests/unit/test_weekly_cleanup.py

# 統合テスト
python3 -m pytest tests/integration/test_weekly_cleanup_integration.py
```