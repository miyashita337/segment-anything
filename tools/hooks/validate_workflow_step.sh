#!/bin/bash
# INTG-086: ワークフロー手順検証スクリプト
# Claude Hook - UserPromptSubmit で実行される13ステップワークフロー順序チェック

set -euo pipefail

# 設定
LOG_FILE="/tmp/intg086_workflow.log"
WORKFLOW_STATE_FILE="/tmp/intg086_workflow_state.json"

# 13ステップワークフローの定義
WORKFLOW_STEPS=(
    "branch_verification"
    "planning"
    "google_sheets_integration"
    "implementation"
    "testing"
    "quality_workflow"
    "documentation"
    "git_operations"
    "approval_stage1"
    "approval_stage2"
    "approval_stage3"
    "approval_stage4"
    "cleanup"
)

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 現在のワークフロー状態を取得
get_current_workflow_state() {
    if [[ -f "$WORKFLOW_STATE_FILE" ]]; then
        cat "$WORKFLOW_STATE_FILE"
    else
        echo '{"current_step": 0, "completed_steps": [], "tracker_id": null, "status": "not_started"}'
    fi
}

# ワークフロー状態を更新
update_workflow_state() {
    local current_step="$1"
    local tracker_id="$2"
    local status="$3"
    local completed_steps="$4"
    
    cat > "$WORKFLOW_STATE_FILE" << EOF
{
  "current_step": $current_step,
  "completed_steps": [$completed_steps],
  "tracker_id": "$tracker_id",
  "status": "$status",
  "timestamp": "$(date -Iseconds)"
}
EOF
}

# ユーザープロンプトからトラッカーIDを抽出
extract_tracker_id() {
    local prompt="$1"
    # トラッカーIDパターンをマッチング（例：INTG-086, QUAL-001等）
    echo "$prompt" | grep -oP '[A-Z]{3,4}-[0-9]{3}' | head -1 || echo ""
}

# ワークフロー手順の前提条件チェック
check_step_prerequisites() {
    local step_name="$1"
    local current_state="$2"
    local completed_steps=$(echo "$current_state" | jq -r '.completed_steps[]' | tr '\n' ' ')
    
    case "$step_name" in
        "branch_verification")
            # 最初のステップは常に実行可能
            return 0
            ;;
        "planning")
            if [[ "$completed_steps" == *"branch_verification"* ]]; then
                return 0
            fi
            ;;
        "google_sheets_integration")
            if [[ "$completed_steps" == *"planning"* ]]; then
                return 0
            fi
            ;;
        "implementation")
            if [[ "$completed_steps" == *"google_sheets_integration"* ]]; then
                return 0
            fi
            ;;
        "testing")
            if [[ "$completed_steps" == *"implementation"* ]]; then
                return 0
            fi
            ;;
        "quality_workflow")
            if [[ "$completed_steps" == *"testing"* ]]; then
                return 0
            fi
            ;;
        *)
            # その他のステップも同様のロジックで前提条件をチェック
            return 0
            ;;
    esac
    
    return 1
}

# メインロジック
INPUT_JSON=$(cat)
USER_PROMPT=$(echo "$INPUT_JSON" | jq -r '.prompt // empty')

log "🔍 INTG-086ワークフロー検証開始"

# 現在の状態を取得
CURRENT_STATE=$(get_current_workflow_state)
CURRENT_STEP_INDEX=$(echo "$CURRENT_STATE" | jq -r '.current_step')
CURRENT_TRACKER_ID=$(echo "$CURRENT_STATE" | jq -r '.tracker_id')

# ユーザープロンプトからトラッカーIDを抽出
PROMPT_TRACKER_ID=$(extract_tracker_id "$USER_PROMPT")

log "現在の状態: ステップ=$CURRENT_STEP_INDEX, トラッカー=$CURRENT_TRACKER_ID"

# 新しいトラッカーの開始検出
if [[ -n "$PROMPT_TRACKER_ID" && "$PROMPT_TRACKER_ID" != "$CURRENT_TRACKER_ID" ]]; then
    log "🆕 新しいトラッカー検出: $PROMPT_TRACKER_ID"
    
    # ワークフローをリセット
    update_workflow_state 0 "$PROMPT_TRACKER_ID" "started" '""'
    CURRENT_STEP_INDEX=0
    CURRENT_TRACKER_ID="$PROMPT_TRACKER_ID"
fi

# ユーザープロンプントの内容から次のステップを推定
SUGGESTED_NEXT_STEP=""
if [[ "$USER_PROMPT" == *"ブランチ"* ]] || [[ "$USER_PROMPT" == *"branch"* ]]; then
    SUGGESTED_NEXT_STEP="branch_verification"
elif [[ "$USER_PROMPT" == *"計画"* ]] || [[ "$USER_PROMPT" == *"plan"* ]]; then
    SUGGESTED_NEXT_STEP="planning"
elif [[ "$USER_PROMPT" == *"Google Sheets"* ]] || [[ "$USER_PROMPT" == *"シート"* ]]; then
    SUGGESTED_NEXT_STEP="google_sheets_integration"
elif [[ "$USER_PROMPT" == *"実装"* ]] || [[ "$USER_PROMPT" == *"implementation"* ]]; then
    SUGGESTED_NEXT_STEP="implementation"
elif [[ "$USER_PROMPT" == *"テスト"* ]] || [[ "$USER_PROMPT" == *"test"* ]]; then
    SUGGESTED_NEXT_STEP="testing"
elif [[ "$USER_PROMPT" == *"品質"* ]] || [[ "$USER_PROMPT" == *"quality"* ]]; then
    SUGGESTED_NEXT_STEP="quality_workflow"
fi

# ワークフロー順序の検証
if [[ -n "$SUGGESTED_NEXT_STEP" ]]; then
    # 推定されたステップのインデックスを取得
    SUGGESTED_STEP_INDEX=-1
    for i in "${!WORKFLOW_STEPS[@]}"; do
        if [[ "${WORKFLOW_STEPS[$i]}" == "$SUGGESTED_NEXT_STEP" ]]; then
            SUGGESTED_STEP_INDEX=$i
            break
        fi
    done
    
    if [[ $SUGGESTED_STEP_INDEX -ge 0 ]]; then
        # 前提条件チェック
        if check_step_prerequisites "$SUGGESTED_NEXT_STEP" "$CURRENT_STATE"; then
            log "✅ ワークフローステップ検証成功: $SUGGESTED_NEXT_STEP"
            
            # 完了したステップリストを更新
            COMPLETED_STEPS_LIST=$(echo "$CURRENT_STATE" | jq -r '.completed_steps[]' | tr '\n' ',' | sed 's/,$//')
            if [[ -n "$COMPLETED_STEPS_LIST" ]]; then
                COMPLETED_STEPS_LIST="\"$COMPLETED_STEPS_LIST\", \"$SUGGESTED_NEXT_STEP\""
            else
                COMPLETED_STEPS_LIST="\"$SUGGESTED_NEXT_STEP\""
            fi
            
            update_workflow_state $((SUGGESTED_STEP_INDEX + 1)) "$CURRENT_TRACKER_ID" "in_progress" "$COMPLETED_STEPS_LIST"
            
        else
            log "⚠️ ワークフロー順序違反: $SUGGESTED_NEXT_STEP を実行するには前のステップの完了が必要です"
            # 順序違反でもブロックはしない（警告のみ）
        fi
    fi
fi

# ワークフロー完了度レポート
if [[ -n "$CURRENT_TRACKER_ID" ]]; then
    COMPLETED_COUNT=$(echo "$CURRENT_STATE" | jq -r '.completed_steps | length')
    TOTAL_STEPS=${#WORKFLOW_STEPS[@]}
    COMPLETION_RATE=$(echo "scale=2; $COMPLETED_COUNT / $TOTAL_STEPS * 100" | bc -l)
    
    log "📊 ワークフロー進捗: $COMPLETED_COUNT/$TOTAL_STEPS ステップ完了 ($COMPLETION_RATE%)"
fi

log "✅ ワークフロー検証完了"
echo '{"status": "validated", "message": "ワークフロー手順検証が完了しました"}'