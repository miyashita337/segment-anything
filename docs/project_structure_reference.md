# プロジェクト構造統一リファレンス

**作成日**: 2025-07-28  
**重要度**: 高  
**目的**: プロジェクトの構造情報を一元化した統一リファレンス

---

## 📋 このドキュメントについて

このドキュメントは、プロジェクトの構造に関する**唯一の正式な参照元**です。  
すべてのプロジェクト構造関連の情報は、この統一リファレンスを参照してください。

**⚠️ 重要**: 他のドキュメントでプロジェクト構造を扱う場合は、「詳細: `docs/project_structure_reference.md` を参照」と記載してください。

---

## 🏗️ 新しい階層化アーキテクチャ（Phase 0 リファクタリング後）

### 基本設計原則

```yaml
architectural_principles:
  core_separation: "Meta Facebook実装（core/）とカスタム実装（features/）の完全分離"
  functional_organization: "機能別ディレクトリ構成"
  test_integration: "統合テストスイート（tests/）"
  tool_centralization: "実行可能スクリプトの集約（tools/）"
  documentation_clarity: "ドキュメントの階層化（docs/）"
```

### ルートレベル構造

```
segment-anything/
├── 📁 core/                    # 元Facebook実装（未改変）
│   ├── segment_anything/       # SAM コアライブラリ
│   ├── scripts/               # バッチ処理・ONNX変換
│   └── demo/                  # React デモアプリ
│
├── 📁 features/               # 自作機能実装
│   ├── extraction/            # キャラクター抽出
│   ├── evaluation/            # 品質評価システム
│   ├── processing/            # 前処理・後処理
│   └── common/                # 共通ユーティリティ
│
├── 📁 tools/                  # 実行可能スクリプト
│   ├── batch/                 # バッチ処理スクリプト
│   ├── core/                  # コア機能スクリプト
│   ├── progress_tracker/      # Google Sheets連携
│   ├── scripts/               # ユーティリティスクリプト
│   ├── testing/               # テストスクリプト
│   └── utils/                 # 汎用ユーティリティ
│
├── 📁 tests/                  # テストスイート
│   ├── unit/                  # 単体テスト
│   ├── integration/           # 統合テスト
│   └── fixtures/              # テストデータ
│
├── 📁 docs/                   # ドキュメント
│   ├── workflows/             # ワークフローガイド
│   ├── issues/                # 問題追跡文書
│   ├── migration/             # 移行関連文書
│   └── checklists/            # チェックリスト
│
├── 📁 config/                 # 設定ファイル
├── 📁 bin/shell/              # シェルスクリプト
├── 📁 test_small/             # テスト用小規模画像セット
├── 📁 deprecated/             # 廃止予定ファイル
├── 📁 logs/                   # ログファイル
└── 📄 各種設定・ドキュメントファイル
```

---

## 📂 詳細ディレクトリ解説

### core/ - Meta Facebook実装（未改変）

```
core/
├── segment_anything/          # SAMコアモジュール
│   ├── modeling/              # モデル定義
│   │   ├── image_encoder.py   # ViT画像エンコーダー
│   │   ├── mask_decoder.py    # マスクデコーダー
│   │   ├── prompt_encoder.py  # プロンプトエンコーダー
│   │   └── sam.py             # SAMメインクラス
│   ├── utils/                 # ユーティリティ
│   │   ├── onnx.py            # ONNX変換
│   │   └── transforms.py      # 画像変換
│   ├── build_sam.py           # SAM構築関数
│   └── predictor.py           # 予測インターフェース
├── scripts/                   # 元のスクリプト群
│   ├── amg.py                 # 自動マスク生成
│   └── export_onnx_model.py   # ONNX出力
└── demo/                      # WebUIデモ
    ├── src/                   # React前端
    └── configs/               # デモ設定
```

**原則**: この階層は Meta の公式実装を保持。変更は最小限に留める。

### features/ - カスタム機能実装

```
features/
├── extraction/                # キャラクター抽出
│   ├── commands/              # CLI コマンドモジュール
│   │   ├── extract_character.py    # メイン抽出コマンド
│   │   └── quick_interactive.py    # インタラクティブ抽出
│   ├── models/                # SAM/YOLO ラッパークラス
│   │   ├── sam_wrapper.py     # SAMラッパー
│   │   └── yolo_wrapper.py    # YOLOラッパー
│   └── pipeline/              # 抽出パイプライン
├── evaluation/                # 品質評価システム
│   ├── metrics/               # 評価指標実装
│   │   ├── pla_calculator.py  # PLA計算
│   │   ├── sci_calculator.py  # SCI計算
│   │   └── ple_tracker.py     # PLE追跡
│   └── utils/                 # 評価・学習ユーティリティ
├── processing/                # 前処理・後処理
│   ├── preprocessing/         # 画像前処理
│   └── postprocessing/        # 画像後処理
│       └── auto_mask_correction.py  # 自動マスク修正
└── common/                    # 共通ユーティリティ
    ├── hooks/                 # 初期化・設定
    ├── notification/          # 通知システム
    └── performance/           # パフォーマンス監視
```

**原則**: プロジェクト固有の機能実装。機能別に明確に分離。

### tools/ - 実行可能スクリプト

```
tools/
├── batch/                     # バッチ処理スクリプト
│   └── kana08_enhanced_stable_batch.py  # 安定版バッチ処理
├── core/                      # コア機能スクリプト
│   ├── unified_quality_checker.py      # 統合品質チェッカー
│   ├── quality_dashboard.py            # 品質ダッシュボード
│   └── run_auto_pipeline.py            # 自動パイプライン
├── progress_tracker/          # Google Sheets連携
│   ├── cli.py                 # CLI インターフェース
│   ├── sheets_client.py       # API クライアント
│   └── data_models.py         # データモデル
├── scripts/                   # ユーティリティスクリプト
│   └── run_quality_workflow.sh         # 品質ワークフロー
├── testing/                   # テストスクリプト
│   ├── test_phase2_simple.py  # Phase 2テスト
│   └── test_phase3_cli.py     # Phase 3テスト
└── utils/                     # 汎用ユーティリティ
    └── init_models.py         # モデル初期化
```

**原則**: 直接実行可能なスクリプトを集約。機能別に整理。

### tests/ - テストスイート

```
tests/
├── unit/                      # 単体テスト
│   ├── test_extract.py        # 抽出機能テスト
│   ├── test_auto_mask_correction.py  # マスク修正テスト
│   └── test_unified_quality_standard.py  # 品質基準テスト
├── integration/               # 統合テスト
│   └── test_extraction_pipeline.py  # パイプラインテスト
└── fixtures/                  # テストデータ
    ├── sample_images/         # サンプル画像
    └── expected_outputs/      # 期待出力
```

**原則**: 全機能の自動テスト。段階的テスト（unit → integration）。

### docs/ - ドキュメント

```
docs/
├── workflows/                 # ワークフローガイド
│   ├── README.md              # メインワークフロー
│   ├── quality_evaluation_guide.md   # 品質評価ガイド
│   └── output_directory_config.md    # 出力ディレクトリ設定
├── issues/                    # 問題追跡文書
├── migration/                 # 移行関連文書
├── checklists/                # チェックリスト
│   └── tracker_workflow_checklist.md  # ワークフローチェック
├── google_sheets_reference.md  # Google Sheets統一リファレンス
├── technical_specifications.md # 技術仕様統一リファレンス
├── dependency_reference.md     # 依存関係統一リファレンス
└── project_structure_reference.md  # 本ドキュメント
```

**原則**: 情報の階層化。統一リファレンス方式の採用。

---

## 🔧 必須ファイル（ワークフローテンプレート用）

### プロジェクト設定・文書
- `README.md` - プロジェクト概要（ワークフロー必須）
- `CLAUDE.md` - Claude Code 設定（ワークフロー必須）  
- `CHANGELOG.md` - 変更履歴（リリース管理必須）
- `setup.py` - パッケージ設定（環境構築必須）

### コア機能
- `core/` フォルダ一式 - Meta SAM 本体実装（システム中核）
- `features/` フォルダ一式 - カスタム機能実装（主要機能）
- `tests/` フォルダ一式 - テストスイート（品質保証必須）

### 設定・ツール  
- `config/` フォルダ - 設定ファイル（システム設定必須）
- `bin/shell/linter.sh` - コード品質チェック（ワークフロー必須）
- `.gitignore` - Git 設定（バージョン管理必須）

### ワークフロー文書
- `docs/workflows/` フォルダ一式 - ワークフローテンプレート（プロセス中核）

---

## 🗂️ ファイル管理方針

### 削除推奨ファイル
```yaml
delete_candidates:
  temporary_files: "*.pid, *.tmp, auto_execution_log.json"
  old_logs: "phase2_batch*.log, kana*_batch.log, run_batch_*.log"
  debug_scripts: "debug_evaluation.py, benchmark_*.py"
  test_scripts: "create_all_26_files.py"
```

### deprecated/ 移行対象
```yaml
deprecated_candidates:
  old_analyzers: "true_content_evaluator.py, visual_intent_analyzer.py"
  old_reports: "backup_analysis_report.md, improvement_report.html"
  legacy_extractors: "extract_kana08_batch.py, kana08_*.py"
  legacy_runners: "run_phase2_*.py"
```

### logs/ 移行対象
```yaml
log_candidates:
  log_files: "*.log"
  progress_files: "*_progress.json, batch_evaluation_results*.json"
  completion_messages: "phase2_completion_message.txt"
```

### セキュリティ考慮ファイル
```yaml
security_exclusions:
  image_files: "すべての画像ファイル（.jpg, .png, .webp等）"
  model_files: "*.pth, *.pt（GitLFS管理）"
  auth_files: "config/*_auth.json"
  private_data: "test_small/ の実際の画像データ"
```

---

## 📋 ディレクトリ管理チェックリスト

### ✅ 新規ファイル作成時
- [ ] 適切なディレクトリに配置されているか？
- [ ] 機能別分類に従っているか？
- [ ] セキュリティ考慮（画像ファイル等）は適切か？
- [ ] テストファイルが対応するディレクトリにあるか？

### ✅ 月次整理項目
```yaml
monthly_cleanup:
  - [ ] logs/ ディレクトリの古いログファイル整理
  - [ ] deprecated/ ディレクトリの定期的な確認・削除
  - [ ] test_small/ の実際の画像データの除外確認
  - [ ] 一時ファイル（*.tmp, *.pid）の削除

quarterly_review:
  - [ ] ディレクトリ構造の最適化検討
  - [ ] 使用頻度の低いスクリプトの整理
  - [ ] ドキュメント構造の見直し
  - [ ] テストディレクトリの整理
```

### 🚨 禁止事項
- ❌ **core/ ディレクトリの構造変更**: Meta実装の保持のため
- ❌ **画像ファイルのcommit**: セキュリティ原則違反
- ❌ **プロジェクトルート直下への無秩序なファイル配置**
- ❌ **機能横断的なファイルの features/ 配置**

---

## 🔄 移行・整理の標準手順

### 新規ディレクトリ作成
```bash
# 必要なディレクトリを作成
mkdir -p deprecated logs config docs/migration docs/issues
mkdir -p tests/unit tests/integration tests/fixtures
```

### ファイル移行
```bash
# 廃止予定ファイルを deprecated/ に移動
mv old_analyzer.py deprecated/

# ログファイルを logs/ に移動  
mv *.log logs/

# 設定ファイルを config/ に整理
mv *_config.json config/
```

### .gitignore 更新
```bash
# セキュリティ重要項目を .gitignore に追加
echo "*.jpg" >> .gitignore
echo "*.png" >> .gitignore  
echo "*.pth" >> .gitignore
echo "config/*_auth.json" >> .gitignore
```

---

## 🎯 現在の構造統計

### ファイル分布（推定）
```yaml
current_distribution:
  total_files: "~380+ ファイル"
  core_files: "~50 ファイル（Meta実装）"
  features_files: "~80 ファイル（カスタム実装）"
  tools_files: "~40 ファイル（実行スクリプト）"
  tests_files: "~30 ファイル（テストスイート）"
  docs_files: "~60 ファイル（ドキュメント）"
  config_files: "~20 ファイル（設定）"
  temporary_files: "~100 ファイル（要整理）"
```

### 整理効果予測
```yaml
cleanup_targets:
  deletion_candidates: "~50 ファイル（一時・重複ファイル）"
  deprecated_migration: "~30 ファイル（古い実装）"
  logs_migration: "~20 ファイル（ログファイル）"
  after_cleanup_total: "~280 ファイル（26%削減）"
```

---

**重要**: この構造リファレンスは、プロジェクトのファイル管理の基盤となる重要な文書です。  
構造変更は慎重に検討し、必ず全体影響を評価してから実行してください。

**更新履歴**:
- 2025-07-28: 統一リファレンス作成（`folder_structure.md` + `docs/file-structure.md` + 構造整理方針 統合）