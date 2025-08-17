# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

日本語で応答してください。

## 📌 重要: バージョニングルール（2025-08-08制定）

**ユーザーから明示的な指示があるまで、以下のルールを厳守すること：**

### バージョン管理方針
- ✅ **マイナーバージョンのみ更新**: v0.9.1 → v0.9.2 → v0.9.3...
- ❌ **ミドルバージョン更新禁止**: v0.9.x → v0.10.0（ユーザー指示があるまで禁止）
- ❌ **メジャーバージョン更新禁止**: v0.x.x → v1.0.0（ユーザー指示があるまで禁止）

### 適用対象
- git commit時のバージョン番号
- CHANGELOG.md更新時
- リリースタグ作成時
- package.json等の設定ファイル更新時

### 例外条件
ユーザーから以下のような明示的指示があった場合のみミドル/メジャーバージョンを更新：
- 「次はv0.10.0でリリースして」
- 「メジャーアップデートとして扱って」
- 「ミドルバージョンを上げて」

## ⚠️ 重要: トラッカーワークフロー必須要件（4 回目仕様違反の恒久対策）

### 🚨 絶対厳守事項

**全トラッカータスク（OPTETETETETETETET-010, PHS-005 等）は以下が必須完了要件です：**

1. **ワークスペース出力必須**: `/mnt/c/AItools/lora/train/{作者名}/tracker-workspace/{TRACKER_ID}/`
2. **品質ワークフロー実行必須**: `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}`
3. **ダッシュボード生成必須**: `tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html`
4. **実装だけでは未完了**: 機能実装 ≠ タスク完了

### 📋 実行前必須確認（プリフライト・チェック）

- [ ] `spec/OUTPUT_PATH_STANDARDS.md` の確認
- [ ] `docs/workflows/output_directory_config.md` の確認
- [ ] 既存完了トラッカー（OPT-023, OPT-024 等）のパターン確認
- [ ] `features/common/output_path_manager.py` の使用検討
- [ ] ワークスペース構造の事前設計

### ❌ 完了 NG パターン（過去 4 回の失敗例 - 絶対回避）

- 機能実装のみで「完了」報告
- ワークスペースディレクトリ未作成
- 品質ワークフロー未実行
- ダッシュボード未生成
- 仕様書確認の省略

### ✅ 正しい完了判定基準

```bash
# 以下がすべて存在することを確認
ls /mnt/c/AItools/lora/train/{作者名}/tracker-workspace/{TRACKER_ID}/
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
- [ ] Google Sheets 更新: "/release"

#### 📁 **ワークスペース必須ディレクトリ**

- [ ] `${TRACKER_ID}/extraction/` (抽出結果)
- [ ] `${TRACKER_ID}/quality/` (品質レポート)
- [ ] `${TRACKER_ID}/dashboard/` (HTML ダッシュボード)
- [ ] `${TRACKER_ID}/tests/` (テスト結果)

#### 🎯 **標準ダッシュボード要件（2025-08-08制定）**

**全トラッカー完了時の必須ダッシュボード仕様:**

- ✅ **URL形式**: `http://100.123.241.106:8088/tracker/{TRACKER_ID}`
- ✅ **画像表示**: Base64埋め込み方式（完全なデータ、切り詰め禁止）
- ✅ **品質評価**: 高品質・中品質・低品質バッジ表示
- ✅ **統計情報**: 総画像数、品質スコア、成功画像数、要改善数
- ✅ **自動生成**: `run_quality_workflow.sh`で自動実行

**実装確認項目:**
```bash
# ダッシュボード生成確認
ls ${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/dashboard/dashboard.html

# ファイルサイズ確認（2MB以上で画像正常埋め込み）
du -h ${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/dashboard/dashboard.html

# アクセス確認（認証情報詳細はPRINCIPLE.mdを参照）
# curl -u admin:[PASSWORD] http://100.123.241.106:8088/tracker/${TRACKER_ID}
```

**技術仕様:**
- **生成システム**: `features/common/dashboard_generator.py`
- **テンプレート**: 統一HTML5 + CSS3デザイン
- **セキュリティ**: VPN + Basic認証で保護
- **互換性**: 51個の既存ダッシュボードと統合

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
6. ✅ Google Sheets のステータス更新（"/release"）
7. ✅ **次トラッカー開始前に、現トラッカーの全出力物確認完了**

#### ⚠️ **違反防止策**

- **Claude は複数のトラッカーを同時に扱わない**
- **現在のトラッカーが/release に到達するまで、次のトラッカーは開始しない**
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
5. 画像が抽出できてるか？(extention以下に画像が存在するか？)
6. ダッシュボードにデータが正しく表示されているか？
7. Google Sheetsが正しく更新されているか？
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

## 🚨 **技術的困難時の対処方針**

### 🛑 **重要原則: ユーザー相談の義務化**

**技術的困難に遭遇した場合、作業を停止してユーザーに確認を求めること**

#### ⚠️ **技術的困難の定義**

以下の状況では必ずユーザーに相談：

1. **実装エラー**: 予期しないエラーが2回以上連続発生
2. **デモ実装の誘惑**: 実際の機能ではなくデモファイルでお茶を濁したくなった時
3. **処理時間の限界**: 2分タイムアウトなど技術制約に直面
4. **依存関係の破綻**: モジュール・パス・インポートエラーの連鎖
5. **仕様の曖昧性**: 要件が不明確で判断に迷う場合

#### 🚫 **絶対禁止行為**

- **デモファイル・プレースホルダー出力**: 実際の機能実装を放棄してテキストファイル等で代替
- **エラー隠蔽**: エラーを無視して強引に先に進む
- **独断回避**: ユーザー相談なしに仕様変更・簡略化を行う
- **未完了での完了報告**: 機能実装が不完全な状態で「完了」と報告

#### ✅ **正しい対処手順**

1. **作業停止**: 現在の作業を一旦停止
2. **状況整理**: 何が起こっているか、何を試したかを整理
3. **選択肢提示**: 可能な解決策を複数提示
4. **ユーザー相談**: 明確に「技術的困難のためユーザー確認が必要」と伝える
5. **指示待ち**: ユーザーの判断・指示を待つ

#### 📋 **相談時の必須項目**

```
🚨 技術的困難のため作業を停止しました

【状況】: 具体的に何が起こっているか
【試行済み】: これまでに試した解決策
【エラー内容】: 具体的なエラーメッセージ・状況
【選択肢】: 
  1. 選択肢A: 詳細説明
  2. 選択肢B: 詳細説明  
  3. 選択肢C: 詳細説明

どの選択肢で進めるべきでしょうか？
```

#### 🎯 **OPTET-011での学習事項（重大バグ事例）**

- **問題**: デモテキストファイルを出力して「完了」と報告
- **原因**: 技術的困難（モジュールエラー）を独断で回避
- **結果**: ユーザーから「重大バグ」として指摘
- **教訓**: **実際の機能実装以外は未完了。デモ・プレースホルダーは絶対禁止**

#### 🔒 **品質保証の絶対条件**

- **実際の処理結果**: デモではなく実際のSAM+YOLO抽出結果
- **実際のファイル出力**: テキストファイルではなく画像ファイル
- **実際の機能動作**: プレースホルダーではなく動作する機能
- **完全な実装**: 部分実装での完了報告は厳禁

## プロジェクト概要

[PRINCIPLE.md](PRINCIPLE.md)を参照

## 📚 ドキュメント参照ポリシー

**基本原則**: 「ドキュメントを作る努力よりも、削っても真の意味や仕様は通せるようにまとめる」

プロジェクト内の情報は以下の統一リファレンスを参照してください：

### 統一リファレンス一覧

- **🔗 Google Sheets 進捗管理**: [`docs/google_sheets_reference.md`](docs/google_sheets_reference.md) を参照
- **⚙️ GitHub Actions 統合**: [`docs/github_actions_reference.md`](docs/github_actions_reference.md) を参照
- **📋 技術仕様**: [`docs/technical_specifications.md`](docs/technical_specifications.md) を参照
- **🎯 トラッカーID標準化**: [`docs/tracker_id_standardization_report.md`](docs/tracker_id_standardization_report.md) を参照
- **📝 トラッカー命名ガイドライン**: [`docs/tracker_naming_guidelines.md`](docs/tracker_naming_guidelines.md) を参照
- **📦 依存関係管理**: [`docs/dependency_reference.md`](docs/dependency_reference.md) を参照
- **🏗️ プロジェクト構造**: [`docs/project_structure_reference.md`](docs/project_structure_reference.md) を参照

### 参照形式

他のドキュメントでは以下の形式で統一リファレンスを参照：

```markdown
**GitHub Actions 設定については**: `docs/github_actions_reference.md` を参照
**技術仕様の詳細については**: `docs/technical_specifications.md` を参照
**依存関係管理については**: `docs/dependency_reference.md` を参照
```

### メンテナンス原則

1. **情報の一元化**: 各分野の情報は 1 つの統一リファレンスに集約
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

- **Week N 成果物**: `*week*`パターンのファイル（Week 4 成果等）
- **stable 版ファイル**: `*stable*`パターンのファイル
- **品質改善機能**: `*quality*`, `*evaluation*`, `*sci*`パターン
- **core/features 実装**: システム中核・カスタム機能

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

#### 🎯 推奨実行方法（現在使用中）

```bash
# 【推奨】新アーキテクチャ抽出コマンド（単体画像）
python features/extraction/commands/extract_character.py input_image.jpg -o output_dir/

# 【推奨】新アーキテクチャ抽出コマンド（バッチ処理）
python features/extraction/commands/extract_character.py input_dir/ -o output_dir/ --batch

# 【推奨】デフォルトパス使用（v0.8.1以降）
# 入力: /mnt/c/AItools/lora/train/yado/org/kana05/
# 出力: /mnt/c/AItools/lora/train/yado/tracker-workspace/{トラッカーID}/
python features/extraction/commands/extract_character.py --batch  # デフォルトパスで実行

# 【緊急時】インタラクティブ抽出（自動処理失敗時の救済・100%成功率）
python features/extraction/commands/quick_interactive.py image.jpg --points 750,1000,pos 800,1200,pos
```

#### 🔧 専用・特殊用途

```bash
# 【大規模バッチ】完全自動パイプライン実行（大量処理向け）
python tools/core/run_auto_pipeline.py

# 【実績使用】現在のトラッカータスクで使用中（OPTET-010で実証済み）
python tools/core/sam_yolo_character_segment.py --mode reproduce-auto \
  --input_dir /mnt/c/AItools/lora/train/yado/org/kana05/ \
  --output_dir ${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/extraction/
```

#### 📦 データセット特化（レガシー・保守モード）

```bash
# 【kana08専用】強化安定版バッチ処理（レジューム対応）
python tools/batch/kana08_enhanced_stable_batch.py \
  --input_dir /path/to/images \
  --output_dir /path/to/output \
  --resume  # 中断箇所から再開
```

#### 💡 使用指針

- **日常的な抽出**: `features/extraction/commands/extract_character.py`
- **トラッカータスク**: `tools/core/sam_yolo_character_segment.py --mode reproduce-auto`
- **自動処理失敗時**: `features/extraction/commands/quick_interactive.py`
- **大規模処理**: `tools/core/run_auto_pipeline.py`
- **kana08データセット**: `tools/batch/kana08_enhanced_stable_batch.py`

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

## Google Sheets 連携（進捗管理）

```bash
# 進捗管理CLIツール
python tools/progress_tracker/cli.py status  # 現在の進捗確認
python tools/progress_tracker/cli.py update  # 進捗更新

# 自動更新フック設定
export GOOGLE_SHEETS_HOOK_ENABLED=true
# 実行時に自動的にGoogle Sheetsに進捗が反映される
```

## バッチ処理とレジューム機能

### 🚀 現在のトラッカーワークフロー

```bash
# トラッカータスク用実際の抽出パイプライン
python tools/scripts/${TRACKER_ID}_real_extraction.py

# または汎用SAM+YOLO抽出
python tools/core/sam_yolo_character_segment.py --mode reproduce-auto \
  --input_dir /mnt/c/AItools/lora/train/yado/org/kana05/ \
  --output_dir ${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/extraction/ \
  --score_threshold 0.07
```

### 📦 データセット特化バッチ処理

```bash
# 【kana08専用】強化安定版バッチ処理（レジューム対応）
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
└── plan/                  # 今後の実装計画書（未実装）
```

### 処理フロー

1. **YOLO 検出**: キャラクター候補の境界ボックス検出（閾値 0.07、アニメ特化調整済み）
2. **SAM 精密分割**: YOLO の結果をプロンプトとした高精度セグメンテーション
3. **品質評価**: 5 つの評価手法（balanced, confidence, size, fullbody, central）から最適選択
4. **改善処理**: A 評価以外に対して適応的マスク拡張・手足切断防止処理
5. **後処理**: マスク適用、背景除去、リサイズ等

## 開発ガイドライン

### 必須モデルファイル

- `sam_vit_h_4b8939.pth` - SAM ViT-H モデル（2.6GB）
- `yolov8n.pt`, `yolov8x.pt` - YOLO v8 モデル
- これらのファイルは `.gitignore` で除外されている

### コード品質基準

- **black**: バージョン 23.\* 必須、100 文字行制限
- **isort**: バージョン 5.12.0 必須
- **flake8**: `.flake8` 設定に従う
- **mypy**: 型チェック（setup.py, notebooks 除く）

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

### YOLO + SAM 2 段階アプローチ

- **YOLO**: 高速なキャラクター候補検出
- **SAM**: YOLO ボックスをプロンプトとした精密セグメンテーション
- この組み合わせで速度と精度を両立

### 品質評価システム（5 段階）

1. **balanced** - バランス重視（推奨）
2. **confidence_priority** - 信頼度優先
3. **size_priority** - サイズ優先
4. **fullbody_priority** - 全身検出優先
5. **central_priority** - 中心位置優先

### インタラクティブ抽出システム

- **GUI 版**: `features/evaluation/utils/interactive_assistant.py`（X11 環境必要）
- **CLI 版**: `features/extraction/commands/quick_interactive.py`
- 自動処理失敗時の手動介入により 100%成功率を達成

## 最新の修正内容（v0.8.1）

### 🐛 修正された問題

#### BatchMemoryManagerスコープエラー（重要修正）
- **問題**: 内部関数からのデコレータ呼び出し時にスコープエラーが発生
- **修正**: `features/common/memory_optimizer.py`のデコレータ実装を修正
- **影響**: バッチ処理の成功率が12.8%から87.2%に大幅改善（680%向上）

#### 画像拡張子の不整合修正
- **問題**: 処理結果は`.jpg`で出力されるが、内部チェックが`.png`を期待
- **修正**: `features/extraction/commands/extract_character.py`の出力拡張子を`.jpg`に統一
- **影響**: 成功した処理が「失敗」として誤判定される問題を解消

#### 面積比閾値の調整
- **変更**: `min_area_ratio`を0.01から0.005に変更
- **効果**: より小さなマスクも受け入れ、抽出成功率を向上

### 📊 改善結果
- **修正前**: 5/39枚抽出成功（12.8%）
- **修正後**: 34/39枚抽出成功（87.2%）
- **改善率**: 6.8倍の成功率向上

## トラブルシューティング

### よくある問題

```bash
# python -> python3

# CUDA利用不可 → PyTorch CUDA版再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# メモリ不足 → 軽量モデル使用
export YOLO_MODEL=yolov8n.pt  # デフォルトはyolov8x.pt

# モデルファイル不在 → 手動ダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 権限エラー → 実行権限付与
chmod +x bin/shell/*.sh

# BatchMemoryManagerエラー（v0.8.1で修正済み）
# 症状: "free variable 'BatchMemoryManager' referenced before assignment"
# 解決: v0.8.1以降では自動的に修正されています

# 拡張子不整合エラー（v0.8.1で修正済み）
# 症状: 正常に抽出されているが「失敗」と判定される
# 解決: v0.8.1以降では自動的に修正されています
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

- **処理成功率**: 96.7% (148/153 画像)
- **品質評価**: balanced 手法で 30%、size_priority 手法で 40%の成功率
- **Phase 3 インタラクティブ**: 100%成功率（従来自動処理 0%から大幅改善）
- **平均品質スコア**: 0.742（範囲: 0.482-0.938）

## 重要な制約事項

- **GPU 推奨**: CPU 処理は極めて遅い（8GB VRAM 以上推奨）
- **メモリ制限**: RAM 2GB、VRAM 8GB、処理時間 5 分/画像で安定性管理
- **バッチ処理**: 5-8 秒/画像、大規模データセットでは数時間要する
- **レジューム機能**: 処理中断時の再開をサポート
- **アニメ特化**: YOLO 閾値 0.07 にアニメキャラクター向け調整済み
