# 📦 バックアップ計画・現状分析

**作成日**: 2025-07-16  
**対象**: Phase 0 プロジェクト構造リファクタリング前のバックアップ

---

## 📊 現在のファイル構造分析

### 🗂️ **フォルダ構造（重要度順）**

#### 🔴 **最重要**: バックアップ必須
- **`commands/`** - 自作のキャラクター抽出コマンド
- **`models/`** - SAM/YOLOラッパークラス
- **`utils/`** - 前処理・後処理・通知ユーティリティ
- **`hooks/`** - 初期化・設定モジュール
- **`config/`** - 設定ファイル（pushover.json等）
- **`tests/`** - テストファイル

#### 🟡 **中重要**: 選択的バックアップ
- **`segment_anything/`** - 元Facebook実装（改造版）
- **`scripts/`** - バッチ処理・ONNX変換
- **`docs/`** - ドキュメント
- **`notebooks/`** - Jupyter notebook

#### 🔵 **低重要**: 必要に応じて
- **`demo/`** - デモアプリケーション
- **`image_evaluator/`** - Node.js評価システム
- **`assets/`** - 画像素材

#### ⚪ **除外**: バックアップ不要
- **`sam-env/`** - 仮想環境（再作成可能）
- **`results_*`** - 処理結果（大容量）
- **`character_boudingbox_preview/`** - プレビュー画像
- **`test_*`** - テストデータ（再作成可能）
- **`__pycache__/`** - キャッシュファイル

### 📋 **ルート直下の重要ファイル**

#### 🔴 **最重要ファイル**
- **`CLAUDE.md`** - プロジェクト指示書
- **`PROGRESS_TRACKER.md`** - 進捗追跡（本ドキュメント）
- **`requirements.txt`** - Python依存関係
- **`setup.py`** - パッケージ設定
- **`bin/shell/linter.sh`** - コード品質チェック

#### 🟡 **重要ファイル**
- **`*.py`** - 個別Pythonスクリプト
  - `sam_yolo_character_segment.py`
  - `test_*.py`
  - `extract_*.py`
- **`*.md`** - ドキュメント
  - `evaluation_report_20250705.md`
  - `exec_log01.md`
  - `README.md`
- **`*.sh`** - 実行スクリプト
  - `run_v042_*.sh`
  - `bin/shell/monitor_v042_tests.sh`

#### 🔵 **設定ファイル**
- **`*.json`** - 設定・進捗ファイル
- **`*.cfg`** - 設定ファイル

---

## 📦 バックアップ実行計画

### Phase 1: 重要ファイル・フォルダのバックアップ
```bash
# バックアップディレクトリ作成
mkdir -p /mnt/c/AItools/segment-anything-backup-$(date +%Y%m%d)

# 最重要フォルダのバックアップ
cp -r commands/ models/ utils/ hooks/ config/ tests/ backup_dir/

# 重要ファイルのバックアップ
cp CLAUDE.md PROGRESS_TRACKER.md requirements.txt setup.py bin/shell/linter.sh backup_dir/
cp *.py *.md *.sh *.json backup_dir/
```

### Phase 2: 設定・環境情報のバックアップ
```bash
# 仮想環境の依存関係
pip freeze > backup_dir/pip_freeze.txt

# Git設定のバックアップ
cp -r .git/ backup_dir/ 2>/dev/null || echo "Git backup skipped"

# 設定ファイル群
cp -r .claude/ backup_dir/
```

### Phase 3: 選択的バックアップ
```bash
# 改造版segment_anything（元Facebook実装）
cp -r segment_anything/ backup_dir/

# スクリプト・ドキュメント
cp -r scripts/ docs/ notebooks/ backup_dir/
```

---

## 🗂️ 提案する新フォルダ構造

### 新構造案
```
segment-anything/
├── core/                    # 元のFacebook実装
│   ├── segment_anything/    # 元のモジュール（変更なし）
│   ├── scripts/            # 元のスクリプト
│   └── demo/               # 元のデモ
├── features/               # 自作機能実装
│   ├── extraction/         # キャラクター抽出
│   │   ├── commands/       # 現在のcommands/
│   │   ├── models/         # 現在のmodels/
│   │   └── __init__.py
│   ├── evaluation/         # 品質評価
│   │   ├── utils/          # 現在のutils/の一部
│   │   └── __init__.py
│   ├── processing/         # 前処理・後処理
│   │   ├── preprocessing/
│   │   ├── postprocessing/
│   │   └── __init__.py
│   └── common/             # 共通ユーティリティ
│       ├── notification/
│       ├── performance/
│       └── __init__.py
├── tests/                  # 統合テスト
│   ├── unit/              # 単体テスト
│   ├── integration/       # 統合テスト
│   └── fixtures/          # テストデータ
├── tools/                  # 実行可能スクリプト
│   ├── extract_*.py       # 現在のルート直下スクリプト
│   ├── sam_*.py
│   └── test_*.py
├── docs/                   # ドキュメント統合
├── config/                 # 設定ファイル
├── logs/                   # ログファイル
└── backup/                 # バックアップディレクトリ
```

### 移行マッピング
```
現在                        →  新構造
commands/                   →  features/extraction/commands/
models/                     →  features/extraction/models/
utils/                      →  features/evaluation/utils/ + features/common/
hooks/                      →  features/common/hooks/
tests/                      →  tests/unit/
segment_anything/           →  core/segment_anything/
scripts/                    →  core/scripts/
demo/                       →  core/demo/
*.py (root)                 →  tools/
docs/                       →  docs/
config/                     →  config/
```

---

## ⚠️ 移行時の注意事項

### 1. **インポート文の修正**
- 既存のインポート文が大幅に変更される
- `from commands.extract_character import ...` → `from features.extraction.commands.extract_character import ...`

### 2. **パス設定の修正**
- 設定ファイルのパス参照
- ログファイルのパス参照
- テストファイルのパス参照

### 3. **実行スクリプトの修正**
- シェルスクリプトのパス修正
- Python実行パスの修正

---

## 📋 バックアップ完了チェックリスト

### Phase 1: 重要ファイル
- [ ] commands/ フォルダ
- [ ] models/ フォルダ
- [ ] utils/ フォルダ
- [ ] hooks/ フォルダ
- [ ] config/ フォルダ
- [ ] tests/ フォルダ
- [ ] CLAUDE.md
- [ ] PROGRESS_TRACKER.md
- [ ] requirements.txt
- [ ] setup.py
- [ ] bin/shell/linter.sh

### Phase 2: 設定・環境
- [ ] pip freeze 出力
- [ ] .git/ フォルダ
- [ ] .claude/ フォルダ

### Phase 3: 選択的バックアップ
- [ ] segment_anything/ フォルダ
- [ ] scripts/ フォルダ
- [ ] docs/ フォルダ
- [ ] notebooks/ フォルダ

### 完了確認
- [ ] バックアップディレクトリの作成確認
- [ ] 重要ファイルの整合性確認
- [ ] バックアップサイズの確認
- [ ] 復元テストの実行

---

*このドキュメントは PROGRESS_TRACKER.md のタスク [P0-001] [P0-002] [P0-003] に対応しています。*