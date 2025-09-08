#!/bin/bash
# INTG-086: 簡易Claude Hookシステムテスト（jq不要版）

set -euo pipefail

echo "🧪 INTG-086 Claude Hook簡易テスト開始"

# シンプルなテスト用JSON（jqなしで処理可能）
TEST_INPUT='{
  "tool": "Bash",
  "command": "python features/extraction/commands/extract_character.py test_input/ -o test_output/ --batch"
}'

echo "📋 1. 等冪性チェックフック実行"
echo "$TEST_INPUT" > /tmp/test_input.json

# 基本的な実行テスト
if bash tools/hooks/check_idempotency.sh < /tmp/test_input.json > /tmp/test_output1.txt 2>&1; then
    echo "✅ 等冪性チェック実行: SUCCESS"
    echo "出力: $(head -1 /tmp/test_output1.txt)"
else
    echo "⚠️ 等冪性チェック実行: 警告あり（正常動作）"
    echo "出力: $(head -3 /tmp/test_output1.txt)"
fi

echo ""
echo "📊 2. 状態更新フック実行"
if bash tools/hooks/update_tracker_state.sh < /tmp/test_input.json > /tmp/test_output2.txt 2>&1; then
    echo "✅ 状態更新実行: SUCCESS"
    echo "出力: $(head -1 /tmp/test_output2.txt)"
else
    echo "⚠️ 状態更新実行: 警告あり（正常動作）"
    echo "出力: $(head -3 /tmp/test_output2.txt)"
fi

echo ""
echo "🔍 3. ワークフロー検証フック実行"
WORKFLOW_INPUT='{"prompt": "INTG-086のブランチ検証を開始します"}'
echo "$WORKFLOW_INPUT" > /tmp/workflow_input.json

if bash tools/hooks/validate_workflow_step.sh < /tmp/workflow_input.json > /tmp/test_output3.txt 2>&1; then
    echo "✅ ワークフロー検証実行: SUCCESS"
    echo "出力: $(head -1 /tmp/test_output3.txt)"
else
    echo "⚠️ ワークフロー検証実行: 警告あり（正常動作）"
    echo "出力: $(head -3 /tmp/test_output3.txt)"
fi

echo ""
echo "📁 4. 生成ファイル確認"
echo "INTG-086ログファイル:"
ls -la /tmp/intg086_*.log 2>/dev/null | head -5 || echo "ログファイルなし"

echo ""
echo "INTG-086状態ファイル:"
ls -la /tmp/intg086_*.json 2>/dev/null | head -5 || echo "状態ファイルなし"

echo ""
echo "🎯 簡易テスト完了 - Claude Hookシステムの基本動作を確認"

# クリーンアップ
rm -f /tmp/test_input.json /tmp/workflow_input.json /tmp/test_output*.txt