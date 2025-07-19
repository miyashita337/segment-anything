# GitHub Actions 統合ガイド

## 概要

このプロジェクトでは、GitHub Actionsを**補助ツール**として導入しています。
主な開発はローカルで行い、CIは構文チェックと品質保証のみを担当します。

## 現在の開発フロー

### 1. 従来のワークフロー（継続使用）
```
PROGRESS_TRACKER.md → ローカル開発 → 画像処理テスト → 人間評価
```

### 2. GitHub Actions補助（新規追加）
```
Push/PR → 自動構文チェック → フィードバック
```

## セットアップ状況

### ✅ 完了済み
- バックアップタグ: `pre-github-actions-v0.3.6`
- ロールバックスクリプト: `rollback-github-actions.sh`
- 基本CI設定: `.github/workflows/basic-ci.yml`

### 🚧 今後の予定
- Claude Code統合（`@claude`メンション）
- Issue自動化
- テスト結果可視化

## 使い方

### 通常の開発（変更なし）
```bash
# PROGRESS_TRACKER.mdを確認
cat PROGRESS_TRACKER.md

# ローカルで開発・テスト
python extract_kaname03.py --quality_method balanced

# コミット・プッシュ
git add -A
git commit -m "実装内容"
git push
```

### CI結果の確認
1. GitHubのActionsタブを開く
2. 最新の実行結果を確認
3. エラーがあれば修正

## トラブルシューティング

### GitHub Actionsが邪魔な場合
```bash
# 一時的に無効化
git rm -r .github/workflows
git commit -m "Disable GitHub Actions temporarily"
```

### 完全にロールバック
```bash
./rollback-github-actions.sh
```

## FAQ

**Q: なぜGPU処理をGitHub Actionsで実行しないの？**
A: GitHub ActionsにはGPUがないため、実画像処理はローカルのみ

**Q: PROGRESS_TRACKER.mdは使い続ける？**
A: はい、当面は並行運用します

**Q: いつIssueに移行する？**
A: 1-2日試してみて便利だったら段階的に移行

## 評価期間（1-2日）

以下の点を評価してください：
- [ ] 構文エラーの早期発見に役立つか？
- [ ] 開発速度は向上するか？
- [ ] 余計な手間が増えていないか？

問題があれば即座にロールバック可能です。