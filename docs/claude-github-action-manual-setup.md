# Claude Code GitHub Action Manual Setup Guide (Windows)

**作成日**: 2025-07-20  
**対象**: Windows版Claude Codeでの`/install-github-app`コマンドが機能しない問題の回避方法

## 問題の概要

Windows版Claude Codeでは、`/install-github-app`コマンドを実行すると以下の問題が発生します：
- URLが正しく処理されず、外部リンクに余分な文字が含まれる
- 認証フローが完了してもClaude Codeがレスポンスを待ち続ける
- Authentication Codeが表示されても使用方法が不明

## Manual Setup手順

### 1. GitHub Appのインストール

1. ブラウザで以下のURLにアクセス：
   ```
   https://github.com/apps/claude
   ```

2. "Install" ボタンをクリック

3. インストール先のリポジトリとして `miyashita337/segment-anything` を選択

4. 必要な権限を確認して承認

### 2. Claude認証情報の取得

Windows環境では、Claude認証情報は以下の場所に保存されています：

**WSL環境の場合**:
```bash
~/.claude/.credentials.json
```

**取得方法**:
```bash
cat ~/.claude/.credentials.json
```

このファイルから以下の値を取得します：
- `accessToken` → `CLAUDE_ACCESS_TOKEN`として使用
- `refreshToken` → `CLAUDE_REFRESH_TOKEN`として使用
- `expiresAt` → `CLAUDE_EXPIRES_AT`として使用

### 3. GitHubリポジトリへのSecrets追加

1. GitHubリポジトリページで `Settings` → `Secrets and variables` → `Actions` に移動

2. 以下のSecretsを追加：

   | Secret名 | 値 |
   |----------|-----|
   | `CLAUDE_ACCESS_TOKEN` | credentials.jsonの`accessToken`の値 |
   | `CLAUDE_REFRESH_TOKEN` | credentials.jsonの`refreshToken`の値 |
   | `CLAUDE_EXPIRES_AT` | credentials.jsonの`expiresAt`の値 |
   | `SECRETS_ADMIN_PAT` | (オプション) GitHub Personal Access Token |

### 4. Personal Access Token (PAT)の作成（オプション）

自動的にSecretsを更新したい場合：

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token (classic)" をクリック
3. 以下の権限を付与：
   - `repo` (フルアクセス)
   - `admin:org` → `read:org`
4. トークンを生成し、`SECRETS_ADMIN_PAT`として追加

### 5. ワークフローファイルの確認

`.github/workflows/claude.yml`が既に存在することを確認。必要に応じて以下の内容で更新：

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

## 動作テスト

1. 新しいIssueを作成し、本文に `@claude` を含める
2. 既存のIssue/PRにコメントで `@claude` をメンション
3. GitHub Actionsタブでワークフローの実行を確認

## トラブルシューティング

### ワークフローが起動しない場合
- Secretsが正しく設定されているか確認
- GitHub Appがリポジトリにインストールされているか確認
- ワークフローファイルの構文エラーがないか確認

### 認証エラーが発生する場合
- Claude認証情報が最新か確認（`expiresAt`の期限を確認）
- 必要に応じてClaude Codeで再ログイン後、新しい認証情報を取得

### トークンの更新
Claude認証トークンは定期的に更新が必要です。期限が切れた場合：
1. Claude Codeで `/login` コマンドを実行
2. 新しい認証情報を取得
3. GitHub Secretsを更新

## セキュリティ上の注意

- 認証情報は絶対に公開リポジトリにコミットしない
- Secretsは暗号化されて保存されるが、適切なアクセス制御を維持
- Personal Access Tokenは必要最小限の権限のみ付与

## 参考リンク

- [Claude Code Action公式リポジトリ](https://github.com/grll/claude-code-action)
- [GitHub Secrets documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)