# CLAUDE.md

日本語で応答してください。Claude Code実行時の必須規約です。

## 🎯 Lost-in-the-Middle問題解決方針（KIRO-010）

**情報過多による規約違反とdangerous permission問題の根本解決**を目的とし、必須規約のみ記載。

## ⛔ 承認必須ルール（2025-09-15制定）

**Phase移行時・破壊的操作前は必ず停止し「承認をお願いします」と明記してユーザーの返答を待つこと。承認なしでの続行は規約違反。**

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

## ⚠️ 重要: 強制ワークフロー実行システム（KIRO-006実装）

**このプロジェクトでは強制ワークフロー実行システムを採用しています**

### 🚀 ワークフロー実行コマンド

```bash
# 新規トラッカー起票
python tools/workflow/workflow_cli.py plan {TRACKER_ID} "概要" "詳細" "作者名"

# ワークフロー状態管理開始
python tools/workflow/workflow_cli.py create {TRACKER_ID}

# 現在のステップ指示確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}

# ステップ実行
python tools/workflow/workflow_cli.py step {TRACKER_ID}

# ワークフロー状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}
```

### 🛡️ システム特徴

- **SQLiteベース状態管理**: 厳密なフェーズ・ステップ管理
- **承認ゲートシステム**: 人間の承認が必要なステップで自動ブロック
- **非冪等的動作制御**: AIの一貫した動作を保証
- **検証ベース制御**: 各ステップの完了条件を自動検証

### 📋 実行手順

1. **plan**コマンドでGoogle Sheetsに起票
2. **create**コマンドでローカルワークフロー開始
3. **instructions** → **step**を繰り返してフェーズを進行
4. 承認が必要なステップでは自動的に待機状態になる

## 🛡️ 入力検証必須要件

**存在しない入力ディレクトリでの処理実行は厳禁**

- 事前検証: 全ての入力パス存在確認必須
- 即座終了: 不正パス検出時の処理停止
- **詳細**: `docs/checklists/input_path_validation_checklist.md` を参照

## 🚨 システム破壊防止ルール

**新規実装前に必ず既存の類似機能を調査し、既存システムの活用を最優先とすること**

- 既存統合システム（`integrated_dashboard_server.py`）の存在確認必須
- 個別ファイル変更前の影響範囲確認必須
- **詳細**: `docs/checklists/dashboard_quality_checklist.md` を参照

## 🚨 統合ダッシュボード必須手順

**品質ワークフロー完了時の必須確認**:
```bash
cp {TRACKER_WORKSPACE}/{TRACKER_ID}/dashboard/dashboard.html {TRACKER_WORKSPACE}/{TRACKER_ID}/index.html
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/refresh
```
**詳細**: `docs/quick-guides/integrated_dashboard_operations.md` を参照

## 🚨 技術的困難時の対処方針

**技術的困難に遭遇した場合、作業を停止してユーザーに確認を求めること**

### 絶対禁止行為
- デモファイル・プレースホルダー出力での代替
- エラー隠蔽や独断での仕様変更・簡略化
- 未完了での完了報告

### 品質保証の絶対条件
- 実際の処理結果（SAM+YOLO抽出結果）
- 実際のファイル出力（画像ファイル）
- 完全な機能実装

## 🚨 Git操作安全性ガイドライン

### 絶対禁止コマンド
`git add .` `git add -A` `git add --all`

**理由**: deprecated/フォルダに33,036ファイルの大量未追跡ファイル存在

### 推奨操作
- 個別ファイル指定: `git add specific_file.py`
- 確認後実行: `git status` → 個別add → commit

## 🔄 サブエージェント活用ルール（KIRO-024: アンチパターン回避）

**公式ドキュメント**: https://code.claude.com/docs/ja/sub-agents

### ⚠️ 暗黙的制約（常に適用）

**Claudeは以下のタスクを受けた場合、明示的な指示がなくてもサブエージェント（Task tool）を使用すること：**

1. **調査・探索タスク**: コードベース調査、ファイル検索、エラー原因調査
2. **大量ファイル読み込み**: 5ファイル以上読む必要がある場合
3. **並列可能な複数タスク**: 独立した複数の作業を同時実行
4. **別件発生時**: ワークフロー実行中に別問題が発生した場合

**理由**: Lost-in-the-Middle問題により、この制約が忘れられる可能性があるため、
Claudeはこのルールを会話開始時に内部的に再確認すること。

### 組み込みサブエージェント

| タイプ | 用途 |
|--------|------|
| **Explore** | コードベース検索・分析（Haiku、高速） |
| **Plan** | プランモード中のコンテキスト収集 |
| **general-purpose** | 複雑なマルチステップタスク |

### 推奨フレーズ（ルールが忘れられた時のバックアップ）

- 「サブエージェントを使って調査して」
- 「並列でサブエージェントに依頼して」

### 禁止事項

- メインコンテキストで大量のファイルを直接読み込む
- 無関係なタスクを同一セッションで混在させる（キッチンシンク）

## 最重要原則

[PRINCIPLE.md](PRINCIPLE.md)を参照

## 📚 ドキュメント参照ポリシー

**基本原則**: 情報の一元化・重複排除・参照徹底

### 統一リファレンス一覧
- **Google Sheets**: `docs/integrations/external/google_sheets_reference.md`
- **技術仕様**: `docs/technical_specifications.md`
- **依存関係管理**: `docs/dependency_reference.md`
- **プロジェクト構造**: `docs/project_structure_reference.md`

### フォルダ構造原則
- `docs/checklists/`: 実行時確認項目
- `docs/templates/`: 作業用フォーマット  
- `docs/workflows/`: プロセス説明

## 重要なセキュリティ原則

**画像ファイルは秘匿情報として扱うこと**

- ❌ 画像ファイルの commit 禁止
- ❌ プロジェクトルート直下への画像出力禁止
- ✅ `/mnt/c/AItools/segment-anything/` 直下以外への出力必須
- ✅ `.gitignore` での画像関連パス完全除外

## 重要ファイル保護原則

**移動・削除前に必ずユーザー確認が必要なファイル:**
- `*week*`, `*stable*`, `*quality*`, `*evaluation*`, `core/features*`パターン

**必須プロセス:**
1. ファイル内容の詳細精査
2. ユーザーへの事前確認
3. 段階的実行（一括移行禁止）

## 主要コマンド

### 環境セットアップ
```bash
python -m venv sam-env
source sam-env/bin/activate
pip install -e .[dev]
```

### キャラクター抽出実行
```bash
# 推奨: バッチ処理（デフォルトパス）
python features/extraction/commands/extract_character.py --batch

# 単体画像処理
python features/extraction/commands/extract_character.py input_image.jpg -o output_dir/

# 緊急時: インタラクティブ抽出（100%成功率）
python features/extraction/commands/quick_interactive.py image.jpg
```

### Google Sheets 連携
```bash
python tools/progress_tracker/cli.py status
python tools/progress_tracker/cli.py update
```

**詳細コマンド**: `docs/workflows/batch_extraction_template.md` を参照

## プロジェクト構造・テスト・バッチ処理

**プロジェクト構造**: `docs/project_structure_reference.md` を参照
**テスト実行**: `./bin/shell/linter.sh`、`pytest tests/`
**バッチ処理**: `tools/batch/kana08_enhanced_stable_batch.py`、`--resume`オプション対応

### 処理フロー

1. **YOLO 検出**: キャラクター候補の境界ボックス検出（閾値 0.07、アニメ特化調整済み）
2. **SAM 精密分割**: YOLO の結果をプロンプトとした高精度セグメンテーション
3. **品質評価**: 5 つの評価手法（balanced, confidence, size, fullbody, central）から最適選択
4. **改善処理**: A 評価以外に対して適応的マスク拡張・手足切断防止処理
5. **後処理**: マスク適用、背景除去、リサイズ等

## 技術仕様・開発ガイドライン

**詳細技術仕様**: `docs/technical_specifications.md` を参照
**開発ガイドライン**: `docs/workflows/batch_extraction_template.md` を参照
**トラブルシューティング**: `docs/troubleshooting.md` を参照

### 必須要件
- **GPU**: 8GB VRAM以上推奨
- **モデル**: `sam_vit_h_4b8939.pth`、`yolov8x.pt`
- **品質チェック**: `./bin/shell/linter.sh`
