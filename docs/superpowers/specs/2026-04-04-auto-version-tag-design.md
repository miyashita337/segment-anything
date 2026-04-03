# Auto Version Tag Design

PRマージ時の自動バージョンタグ付けシステム。

## 要件

- PRがmainにマージされたタイミングで自動的にバージョンをbumpし、gitタグを作成する
- マージ方法を問わない（GitHub UI / Claude Code / CLI）
- Conventional Commitsからbump種別を自動判定する
- GitHub Releaseも自動生成する
- Claude Code `/ship`との二重タグを防止する

## バージョン源泉

- `VERSION`ファイル（プロジェクトルート）が唯一の源泉
- 初期値: `0.3.0`（既存setup.pyの値を引き継ぎ）
- `setup.py`はVERSIONファイルから読み取る: `version=open("VERSION").read().strip()`

## GitHub Actions ワークフロー

### ファイル

`.github/workflows/auto-version-tag.yml`

### トリガー

```yaml
on:
  pull_request:
    types: [closed]
    branches: [main]
```

### 処理フロー

1. `merged == true`を確認（closeのみでは発火しない）
2. マージされたPRのコミットメッセージを取得
3. Conventional Commitsを解析してbump種別を判定:
   - `BREAKING CHANGE` or `!:` → major
   - `feat:` → minor
   - それ以外（`fix:`, `refactor:`, `docs:`, `chore:`等）→ patch
   - 複数種別がある場合は最も大きいbumpを採用
4. 現在のVERSIONを読み、bump後のバージョンを計算
5. 同じタグが既に存在すればスキップ（二重防止）
6. VERSIONファイルを更新 → コミット → タグpush
7. `gh release create`でリリースノート自動生成

### パーミッション

`contents: write`のみ。

### 二重防止ロジック

```bash
if git rev-parse "v${NEW_VERSION}" >/dev/null 2>&1; then
  echo "Tag v${NEW_VERSION} already exists, skipping"
  exit 0
fi
```

## Claude Code `/ship`との連携

| マージ経路 | 動作 |
|---|---|
| Claude Code `/ship` | VERSION bump + tag + push → Actionsはタグ存在を検知しスキップ |
| GitHub UI | ユーザーがMerge → Actionsが自動でVERSION bump + tag + release作成 |

どちらのパスでも必ず1回だけタグが付く。

## bump判定の詳細

コミットメッセージの先頭を正規表現でマッチ:

```
^feat(\(.+\))?!:  → major (breaking)
^.+!:             → major (breaking)
BREAKING CHANGE:  → major (body内)
^feat(\(.+\))?:   → minor
^(fix|refactor|docs|style|test|chore|perf|ci)(\(.+\))?:  → patch
```

PRに複数コミットがある場合、全コミットを走査し最大のbumpを採用。

## 汎用化（Phase 2 — スコープ外）

- Issue #114 で追跡
- Reusable Workflow (`workflow_call`) として切り出し
- 共通リポジトリに配置、各リポジトリから参照

## 決定事項

| 項目 | 決定 |
|---|---|
| バージョン源泉 | VERSIONファイル |
| bump判定 | Conventional Commits自動解析 |
| リリースノート | GitHub Release自動生成 |
| 初期バージョン | 0.3.0 |
| アプローチ | カスタムGitHub Actions（依存ゼロ） |
