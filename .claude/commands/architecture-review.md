# /architecture-review - アーキテクチャ整合性確認

## 概要

設計変更時にプロジェクトのアーキテクチャ原則との整合性を確認する。

## 実行手順

### 1. 基本原則チェック

以下の原則に違反していないか確認せよ:

1. **単一責任の原則 (SRP)**: 1クラス/関数 = 1責任
2. **DRY原則**: 同じコードが複数箇所に存在しないか
3. **YAGNI原則**: 今必要でない機能を作っていないか
4. **依存関係の方向**: 上位モジュール → 下位モジュールのみ

### 2. プロジェクト固有アーキテクチャチェック

このプロジェクトの階層構造に従っているか確認せよ:

```
core/           → Meta実装（変更禁止）
features/       → カスタム実装（変更可）
tools/          → 実行スクリプト
tests/          → テストコード
```

**確認項目**:
- [ ] `core/` 配下のファイルを変更していないか
- [ ] 新機能は `features/` に配置されているか
- [ ] 実行スクリプトは `tools/` に配置されているか
- [ ] テストは `tests/unit/` または `tests/integration/` に配置されているか

### 3. 依存関係チェック

以下のコマンドで循環依存を確認せよ:

```bash
# 対象モジュールのimportを確認
grep -r "from features" features/ --include="*.py" | head -20
grep -r "from core" features/ --include="*.py" | head -20
```

**禁止される依存**:
- `core/` → `features/` （逆方向依存）
- `features/module_a` ↔ `features/module_b` （循環依存）

### 4. 既存パターンとの整合性

類似機能が既に存在しないか確認せよ:

```bash
# 類似クラス/関数の検索
grep -r "class.*Extractor" features/ --include="*.py"
grep -r "def.*extract" features/ --include="*.py"
```

## 出力フォーマット

```markdown
## アーキテクチャレビュー結果

### 基本原則
- [ ] SRP: OK / 違反あり（詳細: ...）
- [ ] DRY: OK / 違反あり（詳細: ...）
- [ ] YAGNI: OK / 違反あり（詳細: ...）
- [ ] 依存方向: OK / 違反あり（詳細: ...）

### プロジェクト構造
- [ ] core/未変更: OK / 違反あり
- [ ] features/配置: OK / 不適切
- [ ] tests/配置: OK / 不適切

### 依存関係
- [ ] 循環依存: なし / あり（詳細: ...）

### 総合判定
✅ 整合性あり / ⚠️ 要修正（理由: ...）
```

## 関連Skill

- `/impact-analysis`: 変更の影響範囲調査
- `/pre-implementation-check`: 実装前の総合チェック
