# 仕様書参照チェックリスト

**最終更新**: 2025-07-27  
**目的**: 新機能実装時の仕様書準拠確保とPH2-002問題の再発防止

---

## 🎯 使用タイミング

### 必須適用場面
- [ ] 新機能・ツール実装時
- [ ] 出力ファイル・ディレクトリ設定時
- [ ] ダッシュボード・レポート生成時
- [ ] バッチ処理・ワークフロー作成時
- [ ] 既存システムの拡張・修正時

---

## 📋 事前確認チェックリスト

### 1. 出力パス設定（🚨 最重要）
- [ ] `docs/workflows/output_directory_config.md` を確認済み
- [ ] ベースディレクトリ仕様 `/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/` を使用
- [ ] トラッカーIDパターン `{tracker_id}` を適用
- [ ] 相対パス（`./`, `../`, カレントディレクトリ基準）を回避
- [ ] `OutputPathManager` クラスの使用を検討

### 2. トラッカーID管理
- [ ] `docs/workflows/PROGRESS_TRACKER.md` でトラッカー管理方法を確認
- [ ] Google Sheets進捗管理システムとの連携確認
- [ ] 命名規則（PH1-001, PH2-002, baseline等）の準拠
- [ ] 既存トラッカーとの重複回避

### 3. ワークフロー統合
- [ ] 既存のワークフロー（抽出→品質チェック→ダッシュボード）との整合性確認
- [ ] `tools/unified_quality_checker.py` との連携可能性確認
- [ ] バッチ処理スクリプトとの整合性確認

### 4. 品質基準
- [ ] A/B評価率など品質指標の定義確認
- [ ] PLA、SCI、PLE指標の適用可能性確認
- [ ] 評価フォーマット（JSON、CSV等）の標準化確認

---

## 🔍 実装中チェックポイント

### コード実装時
- [ ] **ハードコードパス禁止**: `"dashboard_output"`, `"results/"` 等の固定パス使用なし
- [ ] **環境変数対応**: `WORKSPACE_BASE` 等の環境設定対応
- [ ] **パス検証**: 書き込み権限・ディレクトリ存在確認
- [ ] **エラーハンドリング**: パス生成失敗時の適切な処理

### ファイル出力時
```python
# ❌ 悪い例
output_dir = Path("dashboard_output")
output_dir.mkdir(exist_ok=True)

# ✅ 良い例
from features.common.output_path_manager import OutputPathManager, OutputCategory
manager = OutputPathManager("PH2-002")
output_path = manager.ensure_output_dir(OutputCategory.DASHBOARD)
```

### 設定ファイル・スクリプト
- [ ] 絶対パスの使用
- [ ] トラッカーID変数化（ハードコーディング回避）
- [ ] 設定外部化（config.json、環境変数等）

---

## 📖 必須参照ドキュメント

### 主要仕様書（実装前必読）
1. **`docs/workflows/output_directory_config.md`** - 出力ディレクトリ標準仕様
2. **`docs/workflows/PROGRESS_TRACKER.md`** - 進捗管理・トラッカーID仕様
3. **`docs/workflows/README.md`** - 全体ワークフロー概要

### 参考テンプレート
4. **`docs/templates/implementation_report_template.md`** - 実装レポート標準形式
5. **`features/common/output_path_manager.py`** - 標準パス管理クラス

### 既存実装例
6. **`workspace/PH2-001/`, `workspace/baseline/`** - 正しい構造例
7. **`tools/unified_quality_checker.py`** - 統合品質チェック連携例

---

## 🔧 実装パターン

### 推奨: OutputPathManager使用
```python
from features.common.output_path_manager import (
    OutputPathManager, 
    OutputCategory,
    ensure_compliant_output
)

# 基本使用
manager = OutputPathManager("PH2-002")
dashboard_dir = manager.ensure_output_dir(OutputCategory.DASHBOARD)
report_path = manager.get_output_path(
    OutputCategory.DASHBOARD, 
    filename="comprehensive_report.html"
)

# 簡易使用
output_file = ensure_compliant_output(
    tracker_id="PH2-002",
    category=OutputCategory.DASHBOARD,
    filename="dashboard.html"
)
```

### 従来方式（非推奨だが許容）
```python
# 最低限の仕様準拠
WORKSPACE_BASE = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
tracker_id = "PH2-002"  # 変数化必須
output_dir = WORKSPACE_BASE / tracker_id / "dashboard"
output_dir.mkdir(parents=True, exist_ok=True)
```

---

## ✅ レビュー・テストチェックリスト

### 機能テスト
- [ ] 正しいパスに出力されることをテスト実行で確認
- [ ] 権限エラー・ディスク容量エラーの適切な処理確認
- [ ] 既存ファイルの上書き・バックアップ動作確認

### 仕様準拠テスト
- [ ] `OutputPathManager.validate_compliance()` でチェック実行
- [ ] 複数トラッカーIDでの動作確認
- [ ] 環境変数設定での動作確認

### 統合テスト
- [ ] 既存ワークフローとの連携確認
- [ ] 他ツールからの出力ファイル参照確認
- [ ] CI/CDパイプラインでの動作確認

---

## 🚨 禁止事項

### 絶対に使用してはいけないパターン
```python
# ❌ 相対パス
output_dir = Path("results")
output_dir = Path("./dashboard_output")

# ❌ ハードコードパス  
output_dir = Path("/tmp/results")
output_dir = Path("C:\\Users\\output")

# ❌ トラッカーID無視
output_dir = Path("/workspace/fixed_name")

# ❌ プロジェクトルートへの直接出力
output_file = Path("report.html")
```

### 警告すべきパターン
```python
# ⚠️ 環境依存（改善推奨）
output_dir = Path.home() / "results"

# ⚠️ 検証不足（リスクあり）
output_dir = Path(user_input_path)  # バリデーション必須
```

---

## 📊 問題発生時の対応手順

### 1. 即座の対応
1. 問題のあるパス設定を特定
2. 正しいパスへの出力確認
3. 影響範囲の調査（他の類似コード）

### 2. 根本対策
1. 本チェックリストでの検証実行
2. `OutputPathManager` への移行検討
3. 自動テストへの準拠チェック追加

### 3. 再発防止
1. チームへの周知・教育
2. CI/CDへのチェック組み込み
3. ドキュメント改善・更新

---

## 📈 改善提案プロセス

### 仕様書に不明点がある場合
1. **現状調査**: 既存実装での対応方法確認
2. **問題提起**: 具体的な課題と提案をドキュメント化
3. **合意形成**: チーム内での議論・仕様策定
4. **文書更新**: 仕様書・チェックリストの更新

### 新しいパターンが必要な場合
1. **要件分析**: なぜ既存パターンで対応できないか明確化
2. **設計検討**: 後方互換性・拡張性を考慮した設計
3. **プロトタイプ**: 小規模実装での検証
4. **標準化**: `OutputPathManager` 等への機能追加

---

## 🎯 成功の定義

### 短期目標
- [ ] PH2-002類似問題の根絶
- [ ] 新機能でのパス設定ミス防止
- [ ] チェックリスト100%適用

### 長期目標  
- [ ] 全既存コードの仕様準拠
- [ ] 自動チェック機能の完全統合
- [ ] 開発効率の向上（迷い・修正コスト削減）

---

**注意**: このチェックリストは生きた文書です。新しい問題パターンや改善案があれば随時更新してください。