# Windows版Claude CodeでGitHub Actions統合が失敗する問題とManual Setup解決法

## はじめに

Claude Code（Anthropic公式CLI）とGitHub Actionsを統合することで、Issueやコメントに`@claude`とメンションするだけで自動的にコード実装を行えるようになります。しかし、Windows環境では`/install-github-app`コマンドが正常に動作しないという問題があります。

この記事では、その問題の詳細と回避方法（Manual Setup）について解説します。

## 問題の詳細

### `/install-github-app`コマンドで発生する問題

Windows版Claude Codeで`/install-github-app`を実行すると、以下の問題が発生します：

![Claude Code install問題1](https://storage.googleapis.com/zenn-user-upload/ef0e0b47c43f-20250720.png)

#### 1. URL表示の問題
- 外部URLが正しく処理されない
- パス文字列（`"C:\Users\...\スクリーンショット.png"`）が混入してリンクが途切れる

![Claude Code URL問題](https://storage.googleapis.com/zenn-user-upload/36e0c0b8117e-20250720.png)

#### 2. 認証フローの中断
- 手動でURLを修正してアクセスしても、承認ボタンを押した後の遷移が失敗
- Claude側がレスポンスを待ち続けて無限ループ状態になる

![認証画面問題](https://storage.googleapis.com/zenn-user-upload/58c0bd35b6a4-20250720.png)

#### 3. Authentication Codeの活用不可
- Authentication Codeは表示されるものの、Claude Codeがそれを受け取る方法が不明
- 手動入力の手段も提供されていない

![Authentication Code](https://storage.googleapis.com/zenn-user-upload/fa97590f11b9-20250720.png)

## 解決方法：Manual Setup

幸い、[claude-code-action](https://github.com/grll/claude-code-action)では Manual Setup（手動セットアップ）が用意されています。

### 1. GitHub Appのインストール

まず、以下のURLから手動でGitHub Appをインストールします：

```
https://github.com/apps/claude
```

1. "Install" ボタンをクリック
2. 対象リポジトリを選択（例：`miyashita337/segment-anything`）
3. 必要な権限を確認して承認

### 2. Claude認証情報の取得

Windows環境でのClaude認証情報は以下の場所に保存されています：

**WSL環境の場合**：
```bash
~/.claude/.credentials.json
```

**認証情報の確認**：
```bash
cat ~/.claude/.credentials.json
```

ファイル内容例：
```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-...",
    "refreshToken": "sk-ant-ort01-...",
    "expiresAt": 1753038241303,
    "scopes": ["user:inference", "user:profile"],
    "subscriptionType": "max"
  }
}
```

### 3. GitHub Secretsの設定

GitHubリポジトリの設定ページで Secrets を追加します：

`Settings` → `Secrets and variables` → `Actions`

![GitHub Secrets設定画面](path/to/secrets-screenshot.png)

追加が必要なSecrets：

| Secret名 | 取得元 | 説明 |
|----------|--------|------|
| `CLAUDE_ACCESS_TOKEN` | `accessToken` | Claude APIアクセストークン |
| `CLAUDE_REFRESH_TOKEN` | `refreshToken` | リフレッシュトークン |
| `CLAUDE_EXPIRES_AT` | `expiresAt` | トークン有効期限 |
| `SECRETS_ADMIN_PAT` | 手動作成 | GitHub Personal Access Token（オプション） |

### 4. ワークフローファイルの配置

`.github/workflows/claude.yml`を作成（または確認）：

```yaml
name: Claude PR Assistant

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned, labeled]
  pull_request_review:
    types: [submitted]

jobs:
  claude-code-action:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review' && contains(github.event.review.body, '@claude')) ||
      (github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Claude PR Action
        uses: grll/claude-code-action@beta
        with:
          use_oauth: true
          claude_access_token: ${{ secrets.CLAUDE_ACCESS_TOKEN }}
          claude_refresh_token: ${{ secrets.CLAUDE_REFRESH_TOKEN }}
          claude_expires_at: ${{ secrets.CLAUDE_EXPIRES_AT }}
          secrets_admin_pat: ${{ secrets.SECRETS_ADMIN_PAT }}
          timeout_minutes: "60"
```

## 動作確認

セットアップ完了後、以下の方法で動作確認ができます：

### 1. Issue作成テスト
新しいIssueを作成し、本文に`@claude`を含めます：

```markdown
@claude 新しい機能を実装してください

以下の要件で実装をお願いします：
- 機能A
- 機能B
- テストも含めて
```

### 2. コメントテスト
既存のIssueやPRにコメントします：

```markdown
@claude このコードをリファクタリングしてください
```

### 3. GitHub Actionsの確認
リポジトリの「Actions」タブでワークフローの実行状況を確認できます。

## トラブルシューティング

### ワークフローが起動しない
- [ ] GitHub Appがインストールされているか確認
- [ ] Secretsが正しく設定されているか確認
- [ ] ワークフローファイルの構文エラーをチェック

### 認証エラーが発生する
- [ ] Claude認証情報の有効期限をチェック（`expiresAt`）
- [ ] Claude Codeで`/login`を実行して再認証
- [ ] 新しい認証情報でSecretsを更新

### トークン更新の必要性
Claude認証トークンは定期的な更新が必要です：

1. Claude Codeで`/login`コマンドを実行
2. `~/.claude/.credentials.json`から新しい認証情報を取得
3. GitHub Secretsを更新

## まとめ

Windows版Claude Codeでは`/install-github-app`コマンドに問題がありますが、Manual Setupにより同等の機能を実現できます。

### Manual Setupの利点
- ✅ 確実に動作する
- ✅ 認証フローの問題を回避
- ✅ 詳細な制御が可能
- ✅ トラブルシューティングが容易

### 注意点
- 🔧 手動での設定が必要
- 🔄 認証情報の定期更新が必要
- 🔒 Secretsの適切な管理が重要

Windows環境でClaude CodeとGitHub Actionsの統合を検討している方は、この方法をお試しください。

## 参考リンク

- [Claude Code Action公式リポジトリ](https://github.com/grll/claude-code-action)
- [GitHub Secrets documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Anthropic Claude Code](https://claude.ai/code)

---

この記事が Windows版Claude Code ユーザーの役に立てば幸いです。質問や改善点があれば、コメントでお知らせください。