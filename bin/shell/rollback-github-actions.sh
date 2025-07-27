#!/bin/bash
# GitHub Actions導入前の状態に戻すスクリプト

echo "=================================="
echo "GitHub Actions導入前の状態に戻します..."
echo "=================================="

# 現在のブランチを確認
CURRENT_BRANCH=$(git branch --show-current)
echo "現在のブランチ: $CURRENT_BRANCH"

# 変更を保存
if [[ -n $(git status --porcelain) ]]; then
    echo "未コミットの変更を一時保存します..."
    git stash push -m "GitHub Actions rollback stash"
fi

# タグの状態に戻る
echo "pre-github-actions-v0.3.6 の状態に戻ります..."
git checkout pre-github-actions-v0.3.6

# GitHub Actions関連ブランチを削除（存在する場合）
if git show-ref --verify --quiet refs/heads/github-actions-integration; then
    echo "github-actions-integrationブランチを削除します..."
    git branch -D github-actions-integration
fi

# GitHub Actions関連ファイルを削除（存在する場合）
if [ -d ".github/workflows" ]; then
    echo "GitHub Actionsワークフローを削除します..."
    rm -rf .github/workflows
fi

echo ""
echo "✅ ロールバック完了！"
echo "=================================="
echo "PROGRESS_TRACKER.mdベースの開発に戻りました。"
echo ""
echo "もし以前の作業を続ける場合:"
echo "  git checkout main"
echo "  git stash pop  # (未コミットの変更があった場合)"
echo ""