# Google Sheets 自動更新統合仕様

## 概要
品質チェック、抽出パイプライン、評価システムの結果をGoogle Spreadsheetに自動反映するシステム。
progress_tracker統合により重複防止と統一データ管理を実現。

## 更新トリガー（リアルタイム自動更新）
1. **抽出開始時**: ステータス「実行中」に更新
2. **抽出完了時**: ステータス「抽出パイプライン」完了、品質指標更新
3. **品質チェック完了時**: unified_quality_report.json生成後、ダッシュボード生成
4. **評価システム実行時**: dashboard.htmlとPNG生成後、全指標最終更新
5. **エラー発生時**: 各段階でエラーステータス自動更新

## 更新内容

### 1. ステータス更新
- トラッカーID
- 処理状態（処理中/完了/エラー）
- タイムスタンプ
- データセット名

### 2. 品質評価更新
- A/B評価率
- C以上評価率  
- FPS
- SCI (Semantic Completeness Index)
- PLA (Pixel-Level Accuracy)
- PLE (Progressive Learning Efficiency)

### 3. 処理結果更新
- 総画像数
- 成功数/失敗数
- 総合スコア
- 改善提案リスト

## 実装アーキテクチャ（progress_tracker統合版）

### progress_tracker統合設定
```python
# 既存progress_tracker設定活用
from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.data_models import MetricsRecord, TaskRecord

config = get_default_config()
# Spreadsheet ID: 10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA
# 認証ファイル: config/google_sheets_auth.json
```

### 重複防止機能
- **既存レコード検索**: `find_existing_record(tracker_id)`でA列検索
- **条件分岐更新**: 
  - 既存あり → `values().update()`で行指定上書き
  - 新規 → `values().append()`で末尾追加
- **パフォーマンス最適化**: A列のみ検索で高速化

### シート構造（A-V列）
- **A列**: トラッカーID（tracker_id）
- **B列**: 優先度（priority） - 優先度最高/高/中/低
- **C-F列**: 基本情報（status, created_date, updated_date, description）
- **G-L列**: コンポーネント別ステータス（動作確認, テストUNIT, 品質評価等）
- **M-V列**: 10指標（LCA, A/B評価率, FPS, SCI, PLA, PLE等）

### 更新フック統合箇所
1. `tools/quality_dashboard.py` - ダッシュボード生成後（実装済み）
2. `features/extraction/commands/extract_character.py` - 抽出開始・完了時
3. `features/evaluation/` - 評価システム実行時
4. `tools/status_update_hook.py` - 統一ステータス更新インターフェース

## セキュリティ考慮
- サービスアカウントキーの安全な管理
- 書き込み権限の制限
- API制限とレート制限対応