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
get_active_workflow() {
    timeout "$TIMEOUT_SECONDS" sqlite3 "$DB_PATH" \
        "SELECT tracker_id, current_phase, current_step, requires_approval
         FROM workflow_states
         WHERE status = 'active'
         ORDER BY updated_at DESC
         LIMIT 1;" 2>/dev/null
}

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

# KIRO-016: 抽出関連ステップの強制承認チェック
# subagent_extraction または subagent_validation ステップは承認なしで進めることができない
EXTRACTION_STEPS="subagent_extraction|subagent_validation"
if [[ "$CURRENT_STEP" =~ $EXTRACTION_STEPS ]]; then
    if [[ "$APPROVAL_EXISTS" != "1" ]]; then
        log "BLOCKED: 画像抽出関連ステップには承認が必須です"
        echo "" >&2
        echo "⚠️ 画像抽出ステップは承認なしで進めることができません" >&2
        echo "   ユーザーに確認してください：" >&2
        echo "   - 画像抽出を実行しますか？" >&2
        echo "   - スキップする場合はユーザーの明示的な指示が必要です" >&2
        echo "" >&2
        echo "承認コマンド: python tools/workflow/workflow_cli.py approve $TRACKER_ID" >&2
        exit 2
    fi
fi

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
