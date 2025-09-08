#!/bin/bash
# INTG-086: 修正版ワークフローHookシステム統合テスト

set -euo pipefail

echo "🧪 INTG-086 修正版ワークフローHookシステム統合テスト開始"

# テスト用JSON入力（改良版）
TEST_WORKFLOW_COMPLIANCE_JSON='{
  "tool": "Bash",
  "args": {
    "command": "python features/extraction/commands/extract_character.py /mnt/c/AItools/lora/train/yado/org/kana05/ -o /mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-086/extraction/ --batch"
  }
}'

TEST_PROGRESS_UPDATE_JSON='{
  "tool": "Bash",
  "args": {
    "command": "python tools/progress_tracker/cli.py status INTG-086"
  },
  "result": "success"
}'

TEST_TRACKER_CONTEXT_JSON='{
  "prompt": "INTG-086の進捗を確認して次のステップを教えてください。現在Phase 1の計画フェーズです。"
}'

echo ""
echo "📋 1. ワークフローコンプライアンス検証Hook（PreToolUse）テスト"
echo "$TEST_WORKFLOW_COMPLIANCE_JSON" | bash tools/hooks/validate_workflow_compliance.sh 2>&1 | head -10
echo "ステータス: $(echo $?)"

echo ""
echo "📊 2. ワークフロー進捗更新Hook（PostToolUse）テスト"
echo "$TEST_PROGRESS_UPDATE_JSON" | bash tools/hooks/update_workflow_progress.sh 2>&1 | head -10
echo "ステータス: $(echo $?)"

echo ""
echo "🔍 3. トラッカーコンテキスト分析Hook（UserPromptSubmit）テスト"
echo "$TEST_TRACKER_CONTEXT_JSON" | bash tools/hooks/analyze_tracker_context.sh 2>&1 | head -10
echo "ステータス: $(echo $?)"

echo ""
echo "📁 4. workspace_config.pyを使用したパス取得テスト"
python3 -c "
import sys
sys.path.insert(0, '/mnt/c/AItools/segment-anything')
try:
    from config.workspace_config import WorkspaceConfig
    workspace = WorkspaceConfig.get_tracker_workspace('INTG-086')
    print('✅ ワークスペースパス取得成功:', workspace)
    
    import os
    workflow_dir = workspace / '.workflow'
    if workflow_dir.exists():
        print('✅ .workflowディレクトリ存在確認')
        for file in workflow_dir.glob('*.json'):
            print(f'  📄 状態ファイル: {file.name}')
    else:
        print('⚠️ .workflowディレクトリが存在しません')
        
except Exception as e:
    print('❌ エラー:', str(e))
"

echo ""
echo "🔧 5. Hookファイル存在確認"
echo "Hook設定ファイル: .claude/hooks.json"
if [[ -f ".claude/hooks.json" ]]; then
    echo "✅ 存在"
    echo "📝 Hook設定内容:"
    cat .claude/hooks.json | jq '.hooks | keys[]'
else
    echo "❌ 存在しません"
fi

echo ""
echo "Hook実行スクリプト:"
for script in tools/hooks/validate_workflow_compliance.sh tools/hooks/update_workflow_progress.sh tools/hooks/analyze_tracker_context.sh; do
    if [[ -x "$script" ]]; then
        echo "✅ $script （実行可能）"
    else
        echo "❌ $script （実行不可またはファイルなし）"
    fi
done

echo ""
echo "📋 6. チェックリストJSONファイル確認"
if [[ -f "tools/hooks/workflow_checklists.json" ]]; then
    echo "✅ workflow_checklists.json 存在"
    echo "📊 チェックリスト項目数:"
    jq '
        def count_recursive:
            if type == "object" then
                if has("items") and (.items | type == "array") then
                    .items | length
                elif has("phases") then
                    .phases | [.[] | count_recursive] | add
                else
                    [.[] | count_recursive] | add // 0
                end
            elif type == "array" then
                [.[] | count_recursive] | add // 0
            else
                0
            end;
        . | count_recursive
    ' tools/hooks/workflow_checklists.json
else
    echo "❌ workflow_checklists.json が存在しません"
fi

echo ""
echo "🎯 統合テスト完了 - INTG-086修正版Hook系統の基本動作を確認"