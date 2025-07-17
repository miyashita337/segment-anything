# Segment Anything プロジェクト ファイル構造ドキュメント

**作成日**: 2025-07-17  
**最終更新**: 2025-07-17  
**Phase 0 タスク**: [P0-001] 現在のファイル構造の完全把握・ドキュメント化

---

## 📊 プロジェクト概要

このプロジェクトは、MetaのSegment Anything Model (SAM)とYOLOを組み合わせたキャラクター抽出パイプラインです。
Phase 0のリファクタリングにより、フラットな構造から階層化された構造に移行しました。

## 🗂 ディレクトリ構造

### ルートレベル
```
segment-anything/
├── 📁 .claude/          # Claude Code設定
├── 📁 .git/            # Gitリポジトリ
├── 📁 assets/          # 静的アセット
├── 📁 backup-*/        # バックアップディレクトリ (Phase 0)
├── 📁 core/            # Facebook元実装 (Phase 0移行)
├── 📁 features/        # 自作機能実装 (Phase 0移行)
├── 📁 tests/           # テストコード (Phase 0統合)
├── 📁 tools/           # 実行可能スクリプト
├── 📁 config/          # 設定ファイル
├── 📁 docs/            # ドキュメント
├── 📁 image_evaluator/ # Node.js評価システム
├── 📁 logs/            # ログファイル
├── 📁 notebooks/       # Jupyter Notebooks
├── 📁 results_*/       # 処理結果出力
├── 📁 sam-env/         # Python仮想環境
└── 📄 各種設定・ドキュメントファイル
```

### core/ - Facebook元実装 (Phase 0移行済み)
```
core/
├── 📁 segment_anything/    # SAMコアモジュール
│   ├── 📁 modeling/        # モデル定義
│   └── 📁 utils/           # ユーティリティ
├── 📁 scripts/             # 元のスクリプト群
└── 📁 demo/                # WebUIデモ
    ├── 📁 src/
    └── 📁 configs/
```

### features/ - 自作機能実装 (Phase 0移行済み)
```
features/
├── 📁 extraction/          # キャラクター抽出機能
│   ├── 📁 commands/        # CLI コマンド
│   └── 📁 models/          # SAM/YOLOラッパー
├── 📁 evaluation/          # 品質評価システム
│   └── 📁 utils/           # 評価ユーティリティ
├── 📁 processing/          # 前処理・後処理
│   ├── 📁 preprocessing/   # 画像前処理
│   └── 📁 postprocessing/  # 画像後処理
└── 📁 common/              # 共通機能
    ├── 📁 hooks/           # 初期化システム
    ├── 📁 notification/    # 通知システム
    └── 📁 performance/     # パフォーマンス監視
```

### tests/ - テストコード (Phase 0統合済み)
```
tests/
├── 📁 unit/               # 単体テスト
├── 📁 integration/        # 統合テスト
│   └── test_option_a_fixes.py  # Option A修正テスト
└── 📁 fixtures/           # テストデータ
```

## 📄 重要ファイル

### 設定・ドキュメント
- `PROGRESS_TRACKER.md` - 進捗管理メインドキュメント
- `CLAUDE.md` - Claude Code設定
- `README.md` - プロジェクト概要
- `setup.py` - Python パッケージ設定

### 実行ファイル
- `init_models.py` - 統合モデル初期化スクリプト
- `run_batch_extraction.py` - バッチ実行スクリプト
- `test_basic_functionality.py` - 基本機能テスト

### 設定ファイル
- `config/pushover.json` - 通知設定
- `.flake8` - コード品質設定
- `.gitignore` - Git無視設定

## 🔄 Phase 0 移行の変更点

### 移行済み
1. **Facebook実装分離**: `segment_anything/` → `core/segment_anything/`
2. **自作実装移行**: 散らばった実装 → `features/` 階層化
3. **テスト統合**: 各所のテスト → `tests/` 統合
4. **バックアップ作成**: `backup-20250716-2236/` 完全バックアップ

### 自動初期化システム
- `features/common/hooks/start.py` - Option A修正
- `initialize_models()` 関数によるモデル自動ロード
- エラー時の自動復旧機能

## 📊 ファイル統計

### ディレクトリ数
- メインディレクトリ: 15個
- サブディレクトリ: 300個以上 (sam-env含む)

### 重要なPythonファイル
- コアモジュール: `core/segment_anything/` (7ファイル)
- 抽出機能: `features/extraction/` (8ファイル)
- テストコード: `tests/` (5ファイル)

### 結果ディレクトリ
- `results_batch/` - 通常バッチ結果
- `results_batch_final/` - 最終バッチ結果
- `results_test/` - テスト結果

## 🚀 次のステップ

Phase 0完了後は以下の構造改善を予定:
1. **Phase 1**: 品質評価システム改善
2. **Phase 2**: Singletonパターン実装
3. **Phase 3**: 設定ベース管理
4. **Phase 4**: 自立型開発ループ導入

---

*このドキュメントは Phase 0 完了時点での状況を記録しています。*