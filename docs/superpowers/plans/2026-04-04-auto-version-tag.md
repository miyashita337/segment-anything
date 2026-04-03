# Auto Version Tag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PRマージ時に Conventional Commits から自動でバージョンをbumpし、gitタグとGitHub Releaseを作成するシステム。

**Architecture:** VERSIONファイルを唯一のバージョン源泉とし、GitHub Actions（PR merge トリガー）でbump判定→タグ作成→Release生成を行う。Claude Code `/ship` との二重防止ロジックを含む。

**Tech Stack:** GitHub Actions, bash, gh CLI

---

## File Structure

| File | Responsibility |
|---|---|
| `VERSION` (create) | バージョン番号の唯一の源泉 |
| `setup.py` (modify) | VERSIONファイルから読み取るように変更 |
| `.github/workflows/auto-version-tag.yml` (create) | PRマージ時の自動バージョンタグ付けワークフロー |
| `tests/test_version_bump.sh` (create) | bump判定ロジックのテストスクリプト |

---

### Task 1: VERSION ファイル作成と setup.py 修正

**Files:**
- Create: `VERSION`
- Modify: `setup.py:10`

- [ ] **Step 1: VERSION ファイルを作成**

```
0.3.0
```

末尾改行あり、それ以外の内容なし。

- [ ] **Step 2: setup.py を修正して VERSION から読み取る**

`setup.py` の `version="0.3.0"` を以下に変更:

```python
setup(
    name="segment_anything",
    version=open("VERSION").read().strip(),
    install_requires=[],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
```

- [ ] **Step 3: 動作確認**

Run: `python -c "import setup; print('OK')" 2>/dev/null; python setup.py --version`
Expected: `0.3.0`

- [ ] **Step 4: コミット**

```bash
git add VERSION setup.py
git commit -m "chore: add VERSION file as single source of truth for version"
```

---

### Task 2: bump判定ロジックのテストスクリプト作成

**Files:**
- Create: `tests/test_version_bump.sh`

- [ ] **Step 1: テストスクリプトを作成**

```bash
#!/usr/bin/env bash
set -euo pipefail

# bump_version関数をソースする（Task 3で作成するワークフローから抽出）
# テスト用に関数を直接定義

determine_bump() {
  local commits="$1"
  local bump="patch"

  if echo "$commits" | grep -qE "BREAKING CHANGE:|^.+!:"; then
    bump="major"
  elif echo "$commits" | grep -qE "^feat(\(.+\))?:"; then
    if [ "$bump" != "major" ]; then
      bump="minor"
    fi
  fi

  echo "$bump"
}

bump_version() {
  local version="$1"
  local bump="$2"
  local major minor patch

  IFS='.' read -r major minor patch <<< "$version"

  case "$bump" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "${major}.$((minor + 1)).0" ;;
    patch) echo "${major}.${minor}.$((patch + 1))" ;;
  esac
}

# テスト実行
PASS=0
FAIL=0

assert_eq() {
  local test_name="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    echo "PASS: $test_name"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $test_name (expected=$expected, actual=$actual)"
    FAIL=$((FAIL + 1))
  fi
}

# determine_bump テスト
assert_eq "fix → patch" "patch" "$(determine_bump "fix: typo")"
assert_eq "feat → minor" "minor" "$(determine_bump "feat: add feature")"
assert_eq "feat(scope) → minor" "minor" "$(determine_bump "feat(api): add endpoint")"
assert_eq "feat!: → major" "major" "$(determine_bump "feat!: breaking change")"
assert_eq "fix!: → major" "major" "$(determine_bump "fix!: breaking fix")"
assert_eq "BREAKING CHANGE in body → major" "major" "$(determine_bump "feat: something
BREAKING CHANGE: old API removed")"
assert_eq "refactor → patch" "patch" "$(determine_bump "refactor: cleanup")"
assert_eq "docs → patch" "patch" "$(determine_bump "docs: update readme")"
assert_eq "chore → patch" "patch" "$(determine_bump "chore: update deps")"

# bump_version テスト
assert_eq "patch bump" "0.3.1" "$(bump_version "0.3.0" "patch")"
assert_eq "minor bump" "0.4.0" "$(bump_version "0.3.0" "minor")"
assert_eq "major bump" "1.0.0" "$(bump_version "0.3.0" "major")"
assert_eq "patch from 1.2.3" "1.2.4" "$(bump_version "1.2.3" "patch")"
assert_eq "minor from 1.2.3" "1.3.0" "$(bump_version "1.2.3" "minor")"
assert_eq "major from 1.2.3" "2.0.0" "$(bump_version "1.2.3" "major")"

# 複数コミットで最大bump採用
MULTI_COMMITS="fix: typo
feat: add feature
docs: update"
assert_eq "mixed commits → minor (max)" "minor" "$(determine_bump "$MULTI_COMMITS")"

MULTI_BREAKING="fix: typo
feat!: breaking
docs: update"
assert_eq "mixed with breaking → major (max)" "major" "$(determine_bump "$MULTI_BREAKING")"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
```

- [ ] **Step 2: 実行権限付与して実行**

Run: `chmod +x tests/test_version_bump.sh && bash tests/test_version_bump.sh`
Expected: 全テストPASS、exit code 0

- [ ] **Step 3: コミット**

```bash
git add tests/test_version_bump.sh
git commit -m "test: add version bump logic tests"
```

---

### Task 3: GitHub Actions ワークフロー作成

**Files:**
- Create: `.github/workflows/auto-version-tag.yml`

- [ ] **Step 1: ワークフローファイルを作成**

```yaml
name: Auto Version Tag

on:
  pull_request:
    types: [closed]
    branches: [main]

permissions:
  contents: write

jobs:
  version-tag:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Get PR commits
        id: commits
        run: |
          COMMITS=$(gh pr view ${{ github.event.pull_request.number }} \
            --json commits --jq '.commits[].messageHeadline' 2>/dev/null || \
            git log --format="%s" ${{ github.event.pull_request.base.sha }}..${{ github.event.pull_request.head.sha }})
          echo "messages<<EOF" >> $GITHUB_OUTPUT
          echo "$COMMITS" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Determine bump type
        id: bump
        run: |
          COMMITS="${{ steps.commits.outputs.messages }}"

          BUMP="patch"
          if echo "$COMMITS" | grep -qE "BREAKING CHANGE:|^.+!:"; then
            BUMP="major"
          elif echo "$COMMITS" | grep -qE "^feat(\(.+\))?:"; then
            BUMP="minor"
          fi

          echo "type=$BUMP" >> $GITHUB_OUTPUT
          echo "Bump type: $BUMP"

      - name: Calculate new version
        id: version
        run: |
          CURRENT=$(cat VERSION)
          BUMP="${{ steps.bump.outputs.type }}"

          IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

          case "$BUMP" in
            major) NEW="$((MAJOR + 1)).0.0" ;;
            minor) NEW="${MAJOR}.$((MINOR + 1)).0" ;;
            patch) NEW="${MAJOR}.${MINOR}.$((PATCH + 1))" ;;
          esac

          echo "current=$CURRENT" >> $GITHUB_OUTPUT
          echo "new=$NEW" >> $GITHUB_OUTPUT
          echo "Version: $CURRENT -> $NEW ($BUMP)"

      - name: Check for existing tag
        id: check
        run: |
          NEW="${{ steps.version.outputs.new }}"
          if git rev-parse "v${NEW}" >/dev/null 2>&1; then
            echo "Tag v${NEW} already exists, skipping"
            echo "skip=true" >> $GITHUB_OUTPUT
          else
            echo "skip=false" >> $GITHUB_OUTPUT
          fi

      - name: Update VERSION and create tag
        if: steps.check.outputs.skip == 'false'
        run: |
          NEW="${{ steps.version.outputs.new }}"

          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          echo "$NEW" > VERSION
          git add VERSION
          git commit -m "chore: bump version to $NEW"
          git tag "v${NEW}"
          git push origin main --follow-tags

      - name: Create GitHub Release
        if: steps.check.outputs.skip == 'false'
        run: |
          NEW="${{ steps.version.outputs.new }}"
          gh release create "v${NEW}" \
            --title "v${NEW}" \
            --generate-notes \
            --target main
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 2: YAML構文チェック**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/auto-version-tag.yml')); print('YAML OK')"`
Expected: `YAML OK`

注: PyYAML がなければ `pip install pyyaml` を実行。

- [ ] **Step 3: コミット**

```bash
git add .github/workflows/auto-version-tag.yml
git commit -m "feat: add auto version tag workflow on PR merge"
```

---

### Task 4: 統合テスト（ローカル検証）

**Files:** なし（既存ファイルの検証のみ）

- [ ] **Step 1: VERSION ファイルの内容確認**

Run: `cat VERSION`
Expected: `0.3.0`

- [ ] **Step 2: setup.py がVERSIONから読めることを確認**

Run: `python -c "exec(open('setup.py').read())"`
Expected: エラーなし

- [ ] **Step 3: bump テストが全パスすることを再確認**

Run: `bash tests/test_version_bump.sh`
Expected: 全テストPASS

- [ ] **Step 4: ワークフローYAMLの構文確認**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/auto-version-tag.yml')); print('YAML OK')"`
Expected: `YAML OK`

- [ ] **Step 5: 設計ドキュメントをコミットに含める**

```bash
git add docs/superpowers/specs/2026-04-04-auto-version-tag-design.md
git add docs/superpowers/plans/2026-04-04-auto-version-tag.md
git commit -m "docs: add auto version tag design spec and implementation plan"
```

---

### Task 5: 実環境テスト（PRを作ってマージ）

**Files:** なし（GitHub上での動作確認）

- [ ] **Step 1: フィーチャーブランチを作成してPR作成**

```bash
git checkout -b feat/auto-version-tag
git push -u origin feat/auto-version-tag
gh pr create --title "feat: add auto version tag system" --body "## Summary
- VERSION file as single source of truth
- GitHub Actions workflow for auto version bump + tag on PR merge
- Conventional Commits based bump detection (major/minor/patch)
- Duplicate tag prevention for Claude Code /ship compatibility

## Test plan
- [ ] VERSION file contains 0.3.0
- [ ] setup.py reads from VERSION
- [ ] Bump logic tests pass (tests/test_version_bump.sh)
- [ ] Workflow YAML is valid
- [ ] After merge: tag v0.4.0 is created (this PR contains feat:)
- [ ] After merge: GitHub Release is generated"
```

- [ ] **Step 2: PR をマージ**

GitHub UI で Merge ボタンを押す、または:
```bash
gh pr merge --squash
```

- [ ] **Step 3: Actions の実行を確認**

Run: `gh run list --limit 3`
Expected: `Auto Version Tag` ワークフローが実行されている

- [ ] **Step 4: タグとリリースが作成されたか確認**

Run: `git fetch --tags && git tag -l`
Expected: `v0.4.0` が存在（この PR は `feat:` を含むため minor bump）

Run: `gh release view v0.4.0`
Expected: リリースノートが表示される
