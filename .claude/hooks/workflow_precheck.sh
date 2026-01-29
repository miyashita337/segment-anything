#!/bin/bash
# workflow_precheck.sh - PreToolUse Hook for Workflow State Verification
# KIRO-016: Layer 1 Hook Implementation
#
# Exit codes:
#   0 - Allow tool execution
#   2 - Block tool execution (approval required or invalid state)
#
# Failsafe principle: On error, default to allow (exit 0)

set -o pipefail

# Configuration
WORKFLOW_DB="${WORKFLOW_DB:-workflow_state.db}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMEOUT_SECONDS=5

# Log function (stderr to avoid interfering with tool output)
# Yellow color for visibility
log() {
    echo -e "\033[33m[workflow_precheck] $1\033[0m" >&2
}

# Read tool input from stdin (Claude Code passes JSON)
TOOL_INPUT=$(cat)
TOOL_NAME=$(echo "$TOOL_INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
COMMAND=$(echo "$TOOL_INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# KIRO-016: 確認系コマンドはブロック対象外
# status, list, git status などの読み取り専用コマンドは常に許可
is_readonly_command() {
    local cmd="$1"
    # workflow_cli.py の読み取り専用サブコマンド
    if [[ "$cmd" =~ workflow_cli\.py.*(status|list|instructions|help) ]]; then
        return 0
    fi
    # git の読み取り専用コマンド
    if [[ "$cmd" =~ ^git\ (status|log|diff|show|branch) ]]; then
        return 0
    fi
    # sqlite3 の読み取り専用コマンド (SELECT)
    if [[ "$cmd" =~ sqlite3.*SELECT ]]; then
        return 0
    fi
    # その他の読み取り専用コマンド
    if [[ "$cmd" =~ ^(ls|cat|head|tail|grep|find|pwd|echo|which|type) ]]; then
        return 0
    fi
    return 1
}

# Bash以外のツール、または読み取り専用コマンドは早期リターン（ブロックしない）
if [[ "$TOOL_NAME" != "Bash" ]]; then
    exit 0
fi

if [[ -n "$COMMAND" ]] && is_readonly_command "$COMMAND"; then
    log "INFO: Read-only command detected, allowing execution"
    exit 0
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    log "WARNING: jq not found, skipping workflow check"
    exit 0
fi

# Check if sqlite3 is available
if ! command -v sqlite3 &> /dev/null; then
    log "WARNING: sqlite3 not found, skipping workflow check"
    exit 0
fi

# Check if workflow database exists
# Support both relative and absolute paths
if [[ "$WORKFLOW_DB" == /* ]]; then
    DB_PATH="$WORKFLOW_DB"
else
    DB_PATH="${PROJECT_ROOT}/${WORKFLOW_DB}"
fi
if [[ ! -f "$DB_PATH" ]]; then
    log "INFO: Workflow database not found, allowing execution"
    exit 0
fi

# Get current active workflow (with timeout)
# Note: workflow_states table doesn't have 'status' column
# We get the most recent workflow by updated_at
get_active_workflow() {
    timeout "$TIMEOUT_SECONDS" sqlite3 "$DB_PATH" \
        "SELECT tracker_id, current_phase, current_step, requires_approval
         FROM workflow_states
         ORDER BY updated_at DESC
         LIMIT 1;" 2>/dev/null
}

# KIRO-016: シングルトン抽出制御
# 抽出ステップにあるワークフローが存在するかチェック（並列抽出防止）
check_extraction_in_progress() {
    local count=$(timeout "$TIMEOUT_SECONDS" sqlite3 "$DB_PATH" \
        "SELECT COUNT(*) FROM workflow_states
         WHERE current_step IN ('subagent_extraction', 'subagent_validation');" 2>/dev/null)
    echo "${count:-0}"
}

# シングルトン抽出制御: 抽出処理中は無条件でブロック
EXTRACTION_COUNT=$(check_extraction_in_progress)
if [[ "$EXTRACTION_COUNT" -gt 0 ]]; then
    log "BLOCKED: 抽出処理が実行中です（シングルトン制御）"
    echo "" >&2
    echo "⚠️ 抽出処理が他のワークフローで実行中のため、新しい処理をブロックしました" >&2
    echo "   GPUリソース保護のため、抽出処理は同時に1つのみ実行可能です" >&2
    echo "" >&2
    echo "確認コマンド: python tools/workflow/workflow_cli.py status" >&2
    exit 2
fi

# Main check
WORKFLOW_INFO=$(get_active_workflow)

if [[ -z "$WORKFLOW_INFO" ]]; then
    log "INFO: No active workflow found, allowing execution"
    exit 0
fi

# Parse workflow info
IFS='|' read -r TRACKER_ID CURRENT_PHASE CURRENT_STEP REQUIRES_APPROVAL <<< "$WORKFLOW_INFO"

# Check if approval exists (used for multiple checks below)
APPROVAL_EXISTS=$(timeout "$TIMEOUT_SECONDS" sqlite3 "$DB_PATH" \
    "SELECT COUNT(*) FROM approvals
     WHERE tracker_id = '$TRACKER_ID'
     AND step_id = '$CURRENT_STEP'
     AND status = 'approved';" 2>/dev/null)

# NOTE: 抽出ステップのシングルトン制御は上部で実行済み
# ここに到達した場合、現在のワークフローは抽出ステップではない

# Check if approval is required for current step (general case)
if [[ "$REQUIRES_APPROVAL" == "1" ]]; then
    if [[ "$APPROVAL_EXISTS" != "1" ]]; then
        log "BLOCKED: Workflow $TRACKER_ID at step $CURRENT_STEP requires approval"
        echo "Workflow $TRACKER_ID is waiting for approval at step: $CURRENT_STEP" >&2
        echo "Please approve using: python tools/workflow/workflow_cli.py approve $TRACKER_ID" >&2
        exit 2
    fi
fi

# Check phase/step consistency
if [[ -z "$CURRENT_PHASE" ]] || [[ -z "$CURRENT_STEP" ]]; then
    log "WARNING: Invalid workflow state detected, allowing execution (failsafe)"
    exit 0
fi

log "INFO: Workflow $TRACKER_ID at $CURRENT_PHASE/$CURRENT_STEP - OK"
exit 0
