# Claude for GitHub 使用ガイド

## 🎯 基本的な使い方

### Issue作成での自動実装
```
Title: Add validation function
Body: @claude please implement a function that validates image dimensions (minimum 512x512)
```

### PR内での修正依頼
```
@claude please fix the linting errors in this PR
```

### レビューコメントでの改善
```
@claude please add error handling to this function
```

## 💡 効果的な使い方

### ✅ 適切なタスク例
- **テスト追加**: `@claude add unit tests for extract_character function`
- **ドキュメント**: `@claude add docstrings to yolo_wrapper.py`
- **バグ修正**: `@claude fix the import error in features/common/hooks/start.py`
- **Linting**: `@claude fix flake8 errors in this file`

### ❌ 適さないタスク例
- 複雑な調査が必要な実装
- アーキテクチャ変更
- 大規模リファクタリング
- GPU処理やモデル評価

## 🔄 推奨ワークフロー

### 1. メインタスク（Claude Code）
```
Claude Code Session:
- 複雑な問題の分析
- 新機能の実装
- 調査・ヒアリング
```

### 2. 派生タスク（GitHub Issues）
```
実装完了後に小タスクをIssue化:
- Issue: "@claude add tests for new threshold function"
- Issue: "@claude update README with new usage examples"  
- Issue: "@claude fix linting in modified files"
```

## 📋 コマンドリファレンス

### 基本コマンド
- `@claude implement this` - Issue内容に基づいて実装
- `@claude fix this` - 問題を修正
- `@claude add tests` - テストを追加
- `@claude update docs` - ドキュメントを更新

### 詳細指定
- `@claude please add type hints to all functions in this file`
- `@claude implement a utility function that resizes images to 512x512`
- `@claude fix the failing test in test_yolo_wrapper.py`

## ⚡ 期待される効果

### 開発速度向上
- 単純作業の自動化
- 手作業でのテスト追加が不要
- Linting修正の自動化

### コード品質向上
- 一貫したコードスタイル
- 自動テストカバレッジ向上
- ドキュメントの整備

### 作業履歴の可視化
- すべての変更がPR/Issue履歴に残る
- レビュープロセスの標準化
- チーム開発への拡張容易

## 🚨 注意事項

### セキュリティ
- APIキーは絶対にコードに含めない
- GitHub Secretsのみ使用

### 品質管理
- 生成されたコードは必ずレビュー
- 複雑なロジックは手動確認
- テストは実際に実行して確認

### トラブルシューティング
- 生成されたコードでエラーが出た場合
  - PR内で `@claude please fix the error: [エラー内容]`
- 期待と違う実装の場合
  - `@claude please modify this to [具体的な要求]`

## 💾 バックアップ

もし問題が発生した場合：
```bash
./rollback-github-actions.sh
```

従来のPROGRESS_TRACKER.md + Claude Code開発に戻せます。