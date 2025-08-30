# 包括的ワークフローテストシステム

## 📋 概要
QUAL-036での入力パス検証不全を受けて構築した4レベルテストシステム。
実装済みテストコード: 1,944行（tests/workflow/配下）

## 🎯 設計意図
入力パス検証の仕様違反を防ぐため、ワークフロー全体を4レベルに分割してテスト。
各レベルが独立して機能し、問題の早期発見を実現。

## 🏗️ テスト構成

### Level 1: 基本ワークフロー (252行)
**ファイル**: `tests/workflow/test_basic_workflow.py`
- 入力パス検証（26テストケース）
- CLAUDE.md準拠の厳格チェック
- トラッカーID形式検証

### Level 2: 品質ワークフロー (423行)  
**ファイル**: `tests/workflow/test_quality_workflow.py`
- SAM+YOLO抽出処理（23テストケース）
- A-F評価の統計的分析
- ダッシュボード生成検証

### Level 3: 統計分析 (486行)
**ファイル**: `tests/workflow/test_statistical_workflow.py`
- Cohen's d効果サイズ計算（36テストケース）
- Welch's t検定実装
- Google Sheets API統合

### Level 4: 承認プロセス (783行)
**ファイル**: `tests/workflow/test_approval_workflow.py`
- Pushover通知システム（42テストケース）
- 承認フロー管理
- 進捗追跡機能

## 🚀 実行方法

```bash
# 全レベル実行
./bin/shell/run_workflow_tests.sh

# 特定レベル実行
./bin/shell/run_workflow_tests.sh --level level_1

# Python直接実行
pytest tests/workflow/test_basic_workflow.py -v
```

## 🔍 トラブルシューティング

### インポートエラー
```bash
export PYTHONPATH="/mnt/c/AItools/segment-anything:${PYTHONPATH:-}"
```

### パーミッションエラー
```bash
chmod +x bin/shell/run_workflow_tests.sh
```

### パッケージ不足
```bash
pip install pytest pytest-mock
```

## 📊 品質指標
- **テストケース総数**: 127件
- **コードカバレッジ目標**: 90%以上
- **実行時間**: 300秒以下
- **成功率基準**: 95%以上

## 🔧 モックシステム
- `tests/mocks/mock_sam_yolo.py`: SAM+YOLO抽出器モック
- `tests/mocks/mock_google_sheets.py`: Google Sheets APIモック
- `tests/mocks/mock_pushover.py`: Pushover通知モック
- `tests/mocks/mock_approval_system.py`: 承認システムモック

詳細はソースコードを参照。

---

**最終更新**: 2025-08-30  
**対応バージョン**: Level 1-4 統合テストシステム