# /impact-analysis - 影響度調査

## 概要

コード変更による影響範囲を調査し、リスク評価を行う。

## 実行手順

### 1. 変更対象の特定

変更するファイル・関数・クラスを明確にせよ:

```
対象: {ファイルパス}
変更内容: {追加/修正/削除}
```

### 2. 直接参照の調査

以下のコマンドで参照箇所を特定せよ:

```bash
# クラス/関数名での検索
grep -rn "TargetClassName" --include="*.py" .
grep -rn "target_function_name" --include="*.py" .

# インポート文の検索
grep -rn "from module import TargetClass" --include="*.py" .
grep -rn "import module" --include="*.py" .
```

### 3. テストカバレッジの確認

対象コードに対するテストが存在するか確認せよ:

```bash
# 関連テストファイルの検索
find tests/ -name "*target*" -type f
grep -rn "TargetClass" tests/ --include="*.py"
```

### 4. 削除時の追加確認

コードを削除する場合、以下を必ず確認せよ:

```bash
# 全プロジェクトでの使用箇所
grep -rn "削除対象名" . --include="*.py" | grep -v __pycache__

# 設定ファイルでの参照
grep -rn "削除対象名" . --include="*.json" --include="*.yaml" --include="*.toml"
```

### 5. リスク評価

**高リスク**:
- 3つ以上のモジュールが依存
- 外部APIとの連携箇所
- テストカバレッジなし

**中リスク**:
- 1-2モジュールが依存
- 内部処理のみ
- テストカバレッジ一部あり

**低リスク**:
- 依存なし（新規追加・独立機能）
- テストカバレッジ十分

## 出力フォーマット

```markdown
## 影響度調査レポート

### 変更対象
- ファイル: `{path}`
- 対象: `{class/function名}`
- 変更種別: 追加 / 修正 / 削除

### 依存関係
| 参照元ファイル | 参照箇所 | 影響度 |
|---------------|---------|-------|
| `path/file.py` | L123 | 高/中/低 |

### テストカバレッジ
- 既存テスト: あり / なし
- テストファイル: `tests/unit/test_xxx.py`

### リスク評価
**総合リスク**: 高 / 中 / 低

**理由**:
- {評価理由1}
- {評価理由2}

### 推奨アクション
1. {必要なアクション}
2. {必要なアクション}
```

## 関連Skill

- `/architecture-review`: 変更がアーキテクチャに適合するか確認
- `/test-first`: テストが不足している場合の対応
