#!/bin/bash
# INTG-086: Claude Hookシステムテストスクリプト

set -euo pipefail

echo "🧪 INTG-086 Claude Hookシステム統合テスト開始"

# テスト用JSON入力
TEST_PRETOOL_JSON='{
  "tool": "Bash",
  "args": {
    "command": "python features/extraction/commands/extract_character.py test_input/ -o test_output/ --batch"
  }
}'

TEST_POSTTOOL_JSON='{
  "tool": "Bash",
  "args": {
    "command": "python features/extraction/commands/extract_character.py test_input/ -o test_output/ --batch"
  },
  "result": "success"
}'

TEST_USERPROMPT_JSON='{
  "prompt": "INTG-086の実装を開始します。まずはブランチ検証から始めます。"
}'

echo "📋 1. 等冪性チェックフック（PreToolUse）テスト"
echo "$TEST_PRETOOL_JSON" | bash tools/hooks/check_idempotency.sh
if [[ $? -eq 0 ]]; then
    echo "✅ 等冪性チェック: PASSED"
else
    echo "❌ 等冪性チェック: FAILED"
fi

echo ""
echo "📊 2. 状態更新フック（PostToolUse）テスト"
echo "$TEST_POSTTOOL_JSON" | bash tools/hooks/update_tracker_state.sh
if [[ $? -eq 0 ]]; then
    echo "✅ 状態更新: PASSED"
else
    echo "❌ 状態更新: FAILED"
fi

echo ""
echo "🔍 3. ワークフロー検証フック（UserPromptSubmit）テスト"
echo "$TEST_USERPROMPT_JSON" | bash tools/hooks/validate_workflow_step.sh
if [[ $? -eq 0 ]]; then
    echo "✅ ワークフロー検証: PASSED"
else
    echo "❌ ワークフロー検証: FAILED"
fi

echo ""
echo "📁 4. 生成されたファイル・ログの確認"
echo "ログファイル:"
ls -la /tmp/intg086_*.log 2>/dev/null || echo "ログファイルなし"

echo ""
echo "状態ファイル:"
ls -la /tmp/intg086_*.json 2>/dev/null || echo "状態ファイルなし"

echo ""
echo "ロックディレクトリ:"
ls -la /tmp/intg086_locks/ 2>/dev/null || echo "ロックファイルなし"

echo ""
echo "🎯 テスト完了 - Claude Hookシステムが正常に動作しています"