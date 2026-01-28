# /test-first - TDD ガイド

## 概要
Test-Driven Development (TDD) のガイドライン。

## TODO: 以下の内容を実装予定

### TDD サイクル
1. **Red**: 失敗するテストを先に書く
2. **Green**: テストを通す最小限の実装
3. **Refactor**: コードを整理

### テスト作成ガイド（案）
- ユニットテストの配置場所
- テスト命名規則
- モック/スタブの使用方針
- カバレッジ目標

### プロジェクト固有のテスト実行方法
```bash
# ユニットテスト
pytest tests/unit/

# 特定ファイル
pytest tests/tools/workflow/test_xxx.py -v
```

---
*このSkillは空のプレースホルダーです。後で内容を実装してください。*
