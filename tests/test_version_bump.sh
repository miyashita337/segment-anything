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
