# GitHub Actions ワークフロー

このディレクトリには、プロジェクトのCI/CDワークフローが含まれています。

## 現在のワークフロー

### basic-ci.yml
- **目的**: 構文チェックと基本的な品質保証
- **トリガー**: Push/PR時
- **内容**:
  - flake8による構文チェック
  - Pythonモジュールのimportテスト
  - GPU不要の軽量テストのみ

## 重要な注意事項

⚠️ **画像処理テストはローカルでのみ実行**
- GitHub ActionsではGPUが利用できないため、実際の画像処理はローカルで実行
- CIは構文エラーの早期発見が主目的

## ローカルテストの実行方法

```bash
# 実際の画像処理テスト
python extract_kana03.py --quality_method balanced

# ユニットテスト
python -m pytest tests/

# linterの実行
./linter.sh
```

## Claude for GitHub について

### ⚠️ 重要な修正
以前作成していた `claude.yml` ワークフローファイルは**削除しました**。

**理由**: Claude for GitHubは**GitHub App**として動作するため、ワークフローファイルは不要です。

### 正しい使用方法

#### 1. GitHub App の確認
- Repository → Settings → Integrations
- "Claude for GitHub" がインストールされていることを確認

#### 2. 使用方法
```
# Issue作成または既存Issueにコメント
@claude please add docstrings to extract_character.py

# PRでのレビュー依頼
@claude please fix the linting errors in this file

# 具体的な実装依頼
@claude implement a function that validates image dimensions
```

#### 3. 動作の流れ
1. Issue/PRで `@claude` メンション
2. Claude GitHub App が自動実行
3. 新しいブランチ + PR作成（自動実装の場合）
4. レビュー → マージ

## トラブルシューティング

### Claude for GitHub が動作しない場合
1. **GitHub App のインストール確認**
   - Marketplace から "Claude for GitHub" を再インストール
   - リポジトリ権限を明示的に付与

2. **ANTHROPIC_API_KEY 確認**
   - Repository → Settings → Secrets and variables → Actions
   - 正しいAPIキーが設定されているか確認

3. **Issueでのメンション方法**
   - `@claude` の後に具体的な指示を記載
   - 例: `@claude add type hints to this function`

## ロールバック方法

もしGitHub Actionsが不要になった場合:
```bash
./rollback-github-actions.sh
```