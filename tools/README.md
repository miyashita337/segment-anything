# Tools Directory Structure

Tools directoryはTDR-001により機能別に整理されました。

## 📁 ディレクトリ構造

### core/ - 中核ツール
継続的に使用される重要なツール群
- `google_sheets_updater.py` - Google Sheets統合管理
- `quality_dashboard.py` - 品質ダッシュボード生成
- `run_auto_pipeline.py` - 自動パイプライン実行
- `run_objective_evaluation.py` - 客観的評価実行
- `sam_yolo_character_segment.py` - キャラクター抽出パイプライン
- `unified_quality_checker.py` - 統合品質チェッカー

### batch/ - バッチ処理系
データセットのバッチ処理用スクリプト
- `batch_task_ticketing.py` - バッチタスクチケット管理
- `kana08_enhanced_stable_batch.py` - kana08改良版バッチ処理
- `kana08_stable_batch_restored.py` - kana08安定版バッチ処理

### testing/ - テスト・評価系
テストスクリプトと評価ツール
- `test_*.py` - 各種テストスクリプト
- `validate_evaluation_data.py` - 評価データ検証
- `evaluation/` - 評価レポート生成ツール群

### scripts/ - 一時的・特定目的スクリプト
特定タスク用の実行スクリプト（将来的にdeprecated/へ移動候補）
- タスク固有のリリーススクリプト
- 一回限りのデータ処理スクリプト
- シェルスクリプト実行ファイル

### utils/ - ユーティリティ
共通的なユーティリティツール
- `init_models.py` - モデル初期化
- `cleanup_repository.py` - リポジトリ整理
- `audit_path_compliance.py` - パス準拠監査
- `file_protection_checklist.py` - ファイル保護チェック

### legacy/ - レガシー・重複機能
将来的に削除・統合対象（TDR-002で対処予定）
- 重複したGoogle Sheets読み取りツール
- 旧バージョンの品質チェッカー

### progress_tracker/ - Progress Trackerモジュール
既存のProgress Tracker統合システム（そのまま維持）

## 🚀 使用方法

### Import例
```python
# 中核ツールのインポート
from tools.core.google_sheets_updater import GoogleSheetsUpdater
from tools.core.quality_dashboard import QualityDashboard

# テストツールのインポート
from tools.testing.test_phase3_cli import test_phase3

# Progress Trackerのインポート（変更なし）
from tools.progress_tracker.data_models import TaskRecord
```

### 新規ファイル作成ルール
1. **継続使用ツール** → `core/`
2. **バッチ処理** → `batch/`
3. **テスト** → `testing/`
4. **一時作業** → `scripts/`
5. **共通機能** → `utils/`

## 📋 メンテナンス

### 定期整理（月次）
- `scripts/`内の実行済みファイル → `deprecated/tools_archive/`
- `legacy/`内の未使用ファイル → 削除検討

### 次期計画
- **TDR-002**: 統合管理CLI作成（tools/manager.py）
- **TDR-003**: ガバナンスルール確立

---
*TDR-001実装 - 2025-07-28*