# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

日本語で応答してください。

## ⚠️ 重要: トラッカーワークフロー必須要件（4回目仕様違反の恒久対策）

### 🚨 絶対厳守事項
**全トラッカータスク（P1-005, PH2-001等）は以下が必須完了要件です：**

1. **ワークスペース出力必須**: `/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/`
2. **品質ワークフロー実行必須**: `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}`
3. **ダッシュボード生成必須**: `workspace/{TRACKER_ID}/dashboard/dashboard.html`
4. **実装だけでは未完了**: 機能実装 ≠ タスク完了

### 📋 実行前必須確認（プリフライト・チェック）
- [ ] `spec/OUTPUT_PATH_STANDARDS.md` の確認
- [ ] `docs/workflows/output_directory_config.md` の確認  
- [ ] 既存完了トラッカー（P1-A001, P1-A002等）のパターン確認
- [ ] `features/common/output_path_manager.py` の使用検討
- [ ] ワークスペース構造の事前設計

### ❌ 完了NGパターン（過去4回の失敗例 - 絶対回避）
- 機能実装のみで「完了」報告
- ワークスペースディレクトリ未作成
- 品質ワークフロー未実行
- ダッシュボード未生成
- 仕様書確認の省略

### ✅ 正しい完了判定基準
```bash
# 以下がすべて存在することを確認
ls /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/
├── extraction/          # 抽出結果
├── quality/            # 品質レポート
├── dashboard/          # HTMLダッシュボード
└── tests/              # テスト結果
```

### 🚨 **実装完了の絶対条件（チェックボックス式）**

**以下がすべて ✅ になるまで「完了」と言ってはいけない:**

```bash
# 環境変数設定
export TRACKER_WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/tracker-workspace"
```

#### 📋 **完了チェックリスト**
- [ ] 機能実装完了
- [ ] ワークスペース作成: `${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/`
- [ ] 抽出パイプライン実行: `./tools/scripts/run_quality_workflow.sh ${TRACKER_ID}`
- [ ] ダッシュボード確認: `${TRACKER_ID}/dashboard/dashboard.html`
- [ ] Google Sheets更新: "/release"

#### 📁 **ワークスペース必須ディレクトリ**
- [ ] `${TRACKER_ID}/extraction/` (抽出結果)
- [ ] `${TRACKER_ID}/quality/` (品質レポート)  
- [ ] `${TRACKER_ID}/dashboard/` (HTMLダッシュボード)
- [ ] `${TRACKER_ID}/tests/` (テスト結果)

#### ❌ **禁止表現（ワークスペース出力完了まで使用禁止）**
- "完了しました"
- "実装が完了"  
- "タスク完了"
- "次のタスクへ"

#### ✅ **推奨表現**
- "機能実装段階完了。ワークスペース出力を開始します"
- "実装済み。品質チェック実行中"
- "品質ワークフロー実行完了。ダッシュボード確認待ち"

## 🔄 **CRITICAL: シリアル処理必須要件**

### 🚨 **トラッカー実装の絶対原則**

**全トラッカーは必ずシリアル（順次）処理で対応すること。パラレル（並行）実装は厳禁。**

#### ❌ **パラレル実装禁止の理由**

1. **品質保証の破綻**: 品質チェックや抽出パイプラインは、そのトラッカーで修正した内容を元に品質数値や画像出力を行う
2. **トレーサビリティ不可**: 同時進行では「どの時点での品質チェック」「どの時点での抽出パイプライン」か特定不可能
3. **データ整合性の喪失**: 複数の変更が同時に進むと、品質レポートとコード変更の対応関係が不明

#### ✅ **正しいシリアル処理フロー**

```
P1-XXX (実装) → 品質チェック → 抽出パイプライン → ダッシュボード → /release → 次のトラッカー開始
```

#### 🔒 **完了(/release)判定の厳格化**

**トラッカー完了の必須条件:**
1. ✅ 機能実装完了
2. ✅ 単体・統合テスト合格
3. ✅ 品質ワークフロー実行完了
4. ✅ 抽出パイプライン実行完了（該当する場合）
5. ✅ ダッシュボード生成・確認完了
6. ✅ Google Sheetsのステータス更新（"/release"）
7. ✅ **次トラッカー開始前に、現トラッカーの全出力物確認完了**

#### ⚠️ **違反防止策**

- **Claude は複数のトラッカーを同時に扱わない**
- **現在のトラッカーが/releaseに到達するまで、次のトラッカーは開始しない**
- **品質チェック結果は、そのトラッカーの変更のみを反映していることを保証する**
- **「次のタスクへ進む」前に、ユーザーに現在のトラッカーの完了確認を求める**

#### 🎯 **実装時の確認フロー**

```bash
# トラッカー開始前の確認
1. 前のトラッカーは/releaseになっているか？
2. 現在のワークスペースは空の状態か？
3. 入力ディレクトリは存在するか？

# トラッカー完了前の確認  
4. 全必須ディレクトリが存在するか？
5. ダッシュボードにデータが正しく表示されているか？
6. Google Sheetsが正しく更新されているか？
```

## 🛡️ **入力ディレクトリ存在チェック必須要件**

### 🚨 **入力検証の絶対原則**

**存在しない入力ディレクトリでの処理実行は厳禁。強引な処理継続は禁止。**

#### ❌ **禁止される動作**

- 存在しないディレクトリでの処理継続
- 「とりあえず作成」「空で進める」等の回避策
- エラーを無視した強制実行
- 「後で修正する」前提での進行

#### ✅ **必須実装事項**

1. **事前検証**: 全ての入力パス存在確認
2. **即座終了**: 不正パス検出時の処理停止
3. **明確エラー**: 原因とパスを含む統一エラーメッセージ
4. **ユーザー通知**: 修正すべき内容の明示

#### 📋 **統一エラーメッセージ形式**

```bash
❌ エラー: 入力ディレクトリが存在しません
   パス: {指定されたパス}
   
🔧 対処方法:
   1. パスの確認: ls {親ディレクトリ}
   2. 正しいパスの指定
   3. 必要に応じてディレクトリ作成
   
⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です
```

#### 🔍 **チェック対象ファイル**

- `tools/core/sam_yolo_character_segment.py`
- `tools/scripts/run_quality_workflow.sh`
- `create_phase1_extraction_report.py`
- 全ての抽出・処理関連スクリプト

## プロジェクト概要

このプロジェクトは、Meta の Segment Anything Model (SAM) と YOLOv8 を組み合わせたアニメキャラクター抽出システムです。漫画・アニメ画像からキャラクターを自動検出・抽出し、LoRA学習用データセットを生成することが主目的です。

## 📚 ドキュメント参照ポリシー

**基本原則**: 「ドキュメントを作る努力よりも、削っても真の意味や仕様は通せるようにまとめる」

プロジェクト内の情報は以下の統一リファレンスを参照してください：

### 統一リファレンス一覧

- **🔗 Google Sheets進捗管理**: [`docs/google_sheets_reference.md`](docs/google_sheets_reference.md) を参照
- **⚙️ GitHub Actions統合**: [`docs/github_actions_reference.md`](docs/github_actions_reference.md) を参照
- **📋 技術仕様**: [`docs/technical_specifications.md`](docs/technical_specifications.md) を参照
- **📦 依存関係管理**: [`docs/dependency_reference.md`](docs/dependency_reference.md) を参照
- **🏗️ プロジェクト構造**: [`docs/project_structure_reference.md`](docs/project_structure_reference.md) を参照

### 参照形式

他のドキュメントでは以下の形式で統一リファレンスを参照：

```markdown
**GitHub Actions設定については**: `docs/github_actions_reference.md` を参照
**技術仕様の詳細については**: `docs/technical_specifications.md` を参照
**依存関係管理については**: `docs/dependency_reference.md` を参照
```

### メンテナンス原則

1. **情報の一元化**: 各分野の情報は1つの統一リファレンスに集約
2. **重複の排除**: 同じ内容を複数のファイルに記載しない
3. **参照の徹底**: 詳細情報は統一リファレンスへの参照で済ませる
4. **定期的監査**: 重複ドキュメントの定期的な確認と統合

## 重要なセキュリティ原則

**画像ファイルは秘匿情報として扱うこと**

- ❌ 画像ファイルの commit 禁止
- ❌ プロジェクトルート直下への画像出力禁止  
- ✅ `/mnt/c/AItools/segment-anything/` 直下以外への出力必須
- ✅ `.gitignore` での画像関連パス完全除外

## 重要ファイル保護原則

### 絶対保護対象ファイル

以下のファイルは移動・削除前に必ずユーザー確認が必要：

- **Week N成果物**: `*week*`パターンのファイル（Week 4成果等）
- **stable版ファイル**: `*stable*`パターンのファイル
- **品質改善機能**: `*quality*`, `*evaluation*`, `*sci*`パターン
- **core/features実装**: システム中核・カスタム機能

### 重要度分類システム

```yaml
CRITICAL: Week N成果、stable版、core機能 → 絶対保護
HIGH: features実装、品質改善機能 → ユーザー確認必須  
MEDIUM: tools、設定ファイル → 慎重確認
LOW: ログ、一時ファイル → 安全確認後移行
```

### 判定禁止事項

- ❌ **ファイル名のみでの価値判断**
- ❌ **表面的なパターンマッチング分類**
- ❌ **機能価値の軽視**
- ❌ **ユーザー確認の省略**

### 必須確認プロセス

1. **ファイル内容の詳細精査** - 機能・クラス・依存関係の完全把握
2. **重要ファイルの個別確認** - ユーザーへの事前リスト提示と許可取得
3. **動作確認テスト** - 移行前の重要機能実行テスト
4. **段階的実行** - 一括移行禁止、重要度別段階実行

## 主要コマンド

### 環境セットアップ
```bash
# 仮想環境作成・有効化
python -m venv sam-env
source sam-env/bin/activate  # Linux
sam-env\Scripts\activate     # Windows

# 開発依存関係込みインストール（推奨）
pip install -e .[dev]

# または基本インストール
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx ultralytics easyocr
```

### キャラクター抽出実行
```bash
# メイン抽出コマンド
python features/extraction/commands/extract_character.py input_image.jpg -o output_dir/

# バッチ処理
python features/extraction/commands/extract_character.py input_dir/ -o output_dir/ --batch

# インタラクティブ抽出（100%成功率）
python features/extraction/commands/quick_interactive.py image.jpg --points 750,1000,pos 800,1200,pos

# 自動パイプライン実行
python tools/core/run_auto_pipeline.py

# レガシー互換ツール
python tools/core/sam_yolo_character_segment.py --mode reproduce-auto --input_dir ./test_small/ --output_dir ./results/
```

### テスト・品質チェック
```bash
# 統合品質チェック（flake8, black, mypy, isort）
./bin/shell/linter.sh

# 個別テスト実行
python -m pytest tests/unit/test_extract.py -v
python -m pytest tests/integration/test_extraction_pipeline.py -v

# 段階的テスト
python tools/testing/test_phase2_simple.py
python tools/testing/test_phase3_cli.py
python tools/testing/test_difficult_pose.py

# 品質評価
python tools/core/unified_quality_checker.py
python tools/core/quality_dashboard.py
```

### モデル管理
```bash
# モデル初期化確認
python tools/utils/init_models.py

# CUDA利用可能性確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

## Google Sheets連携（進捗管理）
```bash
# 進捗管理CLIツール
python tools/progress_tracker/cli.py status  # 現在の進捗確認
python tools/progress_tracker/cli.py update  # 進捗更新

# 自動更新フック設定
export GOOGLE_SHEETS_HOOK_ENABLED=true
# 実行時に自動的にGoogle Sheetsに進捗が反映される
```

## バッチ処理とレジューム機能
```bash
# バッチ処理実行（レジューム対応）
python tools/batch/kana08_enhanced_stable_batch.py \
  --input_dir /path/to/images \
  --output_dir /path/to/output \
  --resume  # 中断箇所から再開

# 処理状況モニタリング
./bin/shell/monitor_v042_tests.sh
```

## 単体テスト実行
```bash
# 単体テストの個別実行
python -m pytest tests/unit/test_batch_extraction.py::test_extract_character -v
python -m pytest tests/unit/test_documentation_sync_system.py -v
python -m pytest tests/unit/test_unified_quality_standard.py -v

# 統合テスト実行
python -m pytest tests/integration/ -v
```

## プロジェクト構造

### 新しい階層化アーキテクチャ（Phase 0 リファクタリング後）

```
├── core/                    # 元Facebook実装（未改変）
│   ├── segment_anything/    # SAM コアライブラリ
│   ├── scripts/            # バッチ処理・ONNX変換
│   └── demo/               # React デモアプリ
│
├── features/               # 自作機能実装
│   ├── extraction/         # キャラクター抽出
│   │   ├── commands/       # CLI コマンドモジュール
│   │   └── models/         # SAM/YOLO ラッパークラス
│   ├── evaluation/         # 品質評価システム
│   │   └── utils/          # 評価・学習ユーティリティ
│   ├── processing/         # 前処理・後処理
│   │   ├── preprocessing/  # 画像前処理
│   │   └── postprocessing/ # 画像後処理
│   └── common/             # 共通ユーティリティ
│       ├── hooks/          # 初期化・設定
│       ├── notification/   # 通知システム
│       └── performance/    # パフォーマンス監視
│
├── tools/                  # 実行可能スクリプト
│   ├── batch/             # バッチ処理スクリプト
│   ├── core/              # コア機能スクリプト
│   ├── progress_tracker/  # Google Sheets連携
│   ├── scripts/           # ユーティリティスクリプト
│   ├── testing/           # テストスクリプト
│   └── utils/             # 汎用ユーティリティ
│
├── tests/                  # テストスイート
│   ├── unit/              # 単体テスト
│   ├── integration/       # 統合テスト
│   └── fixtures/          # テストデータ
│
├── bin/shell/             # シェルスクリプト
└── test_small/            # テスト用小規模画像セット
```

### 処理フロー
1. **YOLO検出**: キャラクター候補の境界ボックス検出（閾値0.07、アニメ特化調整済み）
2. **SAM精密分割**: YOLOの結果をプロンプトとした高精度セグメンテーション
3. **品質評価**: 5つの評価手法（balanced, confidence, size, fullbody, central）から最適選択
4. **改善処理**: A評価以外に対して適応的マスク拡張・手足切断防止処理
5. **後処理**: マスク適用、背景除去、リサイズ等

## 開発ガイドライン

### 必須モデルファイル
- `sam_vit_h_4b8939.pth` - SAM ViT-H モデル（2.6GB）
- `yolov8n.pt`, `yolov8x.pt` - YOLO v8 モデル
- これらのファイルは `.gitignore` で除外されている

### コード品質基準
- **black**: バージョン23.* 必須、100文字行制限
- **isort**: バージョン5.12.0 必須
- **flake8**: `.flake8` 設定に従う
- **mypy**: 型チェック（setup.py, notebooks除く）

統合チェックは `./bin/shell/linter.sh` で実行。

### テスト戦略
- **段階的テスト**: phase2 → phase3 の順で実行
- **統合テスト**: `tests/integration/` に配置
- **フィクスチャ**: `tests/fixtures/` でテストデータ管理
- **実画像テスト**: `test_small/` の小規模データセットを使用

### 通知システム
```bash
# Pushover設定（オプション）
cp config/pushover.json.example config/pushover.json
# user_key, api_token を設定
```

## アーキテクチャ設計判断

### YOLO + SAM 2段階アプローチ
- **YOLO**: 高速なキャラクター候補検出
- **SAM**: YOLOボックスをプロンプトとした精密セグメンテーション
- この組み合わせで速度と精度を両立

### 品質評価システム（5段階）
1. **balanced** - バランス重視（推奨）
2. **confidence_priority** - 信頼度優先
3. **size_priority** - サイズ優先
4. **fullbody_priority** - 全身検出優先
5. **central_priority** - 中心位置優先

### インタラクティブ抽出システム
- **GUI版**: `features/evaluation/utils/interactive_assistant.py`（X11環境必要）
- **CLI版**: `features/extraction/commands/quick_interactive.py`
- 自動処理失敗時の手動介入により100%成功率を達成

## トラブルシューティング

### よくある問題
```bash
# CUDA利用不可 → PyTorch CUDA版再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# メモリ不足 → 軽量モデル使用
export YOLO_MODEL=yolov8n.pt  # デフォルトはyolov8x.pt

# モデルファイル不在 → 手動ダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 権限エラー → 実行権限付与
chmod +x bin/shell/*.sh
```

### デバッグ方法
```bash
# ログレベル調整
export LOG_LEVEL=DEBUG

# 小規模テスト実行
python tools/testing/test_phase2_simple.py

# パフォーマンス監視
python -c "
import torch
import psutil
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
print(f'RAM: {psutil.virtual_memory().total // 1024**3} GB')
"
```

## 実績データ（最新版）
- **処理成功率**: 96.7% (148/153画像)
- **品質評価**: balanced手法で30%、size_priority手法で40%の成功率
- **Phase 3インタラクティブ**: 100%成功率（従来自動処理0%から大幅改善）
- **平均品質スコア**: 0.742（範囲: 0.482-0.938）

## 重要な制約事項
- **GPU推奨**: CPU処理は極めて遅い（8GB VRAM以上推奨）
- **メモリ制限**: RAM 2GB、VRAM 8GB、処理時間5分/画像で安定性管理
- **バッチ処理**: 5-8秒/画像、大規模データセットでは数時間要する
- **レジューム機能**: 処理中断時の再開をサポート
- **アニメ特化**: YOLO閾値0.07にアニメキャラクター向け調整済み