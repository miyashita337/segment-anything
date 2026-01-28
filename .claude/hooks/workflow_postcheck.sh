#!/bin/bash
# workflow_postcheck.sh - PostToolUse Hook for Completion Verification
# KIRO-016: Layer 1 Hook Implementation
#
# This hook runs after Bash tool execution to:
# 1. Verify completion conditions
# 2. Detect errors and suggest Skill usage
# 3. Log workflow-related command results
#
# Exit codes:
#   0 - Success (always returns 0 for PostToolUse)

set -o pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMEOUT_SECONDS=5

# Log function (stderr)
log() {
    echo "[workflow_postcheck] $1" >&2
}

# Read tool output from stdin (if available)
TOOL_OUTPUT=""
if [[ ! -t 0 ]]; then
    TOOL_OUTPUT=$(cat)
fi

# Detect workflow_cli.py execution
if echo "$TOOL_OUTPUT" | grep -q "workflow_cli.py"; then

    # Check for errors in output
    if echo "$TOOL_OUTPUT" | grep -qiE "(Error:|Exception:|Traceback|failed|エラー)"; then
        log "ERROR: workflow_cli.py execution error detected"
        echo "" >&2
        echo "---" >&2
        echo "New error pattern detected in workflow_cli.py execution." >&2
        echo "Consider adding this pattern to troubleshooting Skill:" >&2
        echo "  /workflow-troubleshoot" >&2
        echo "" >&2
        echo "Or check the workflow guide:" >&2
        echo "  python tools/workflow/workflow_cli.py guide" >&2
        echo "---" >&2
    fi

    # Check for specific workflow errors
    if echo "$TOOL_OUTPUT" | grep -q "requires approval"; then
        log "INFO: Approval required - suggesting approval command"
        echo "" >&2
        echo "Tip: Use 'python tools/workflow/workflow_cli.py approvals' to see pending approvals" >&2
    fi

    if echo "$TOOL_OUTPUT" | grep -q "not found"; then
        log "INFO: Resource not found - suggesting creation"
        echo "" >&2
        echo "Tip: Use 'python tools/workflow/workflow_cli.py create <TRACKER_ID>' to create workflow" >&2
    fi

    # Suggest Skills based on next step
    # KIRO-016: 画像抽出フェーズの強制確認警告
    if echo "$TOOL_OUTPUT" | grep -q "次のステップ: subagent_extraction"; then
        echo "" >&2
        echo "⚠️ ========================================" >&2
        echo "⚠️ 画像抽出フェーズに入ります" >&2
        echo "⚠️ ========================================" >&2
        echo "" >&2
        echo "以下を確認してください：" >&2
        echo "  1. このタスクで画像抽出が必要ですか？" >&2
        echo "  2. 入力ディレクトリは正しいですか？" >&2
        echo "" >&2
        echo "⚠️ Claudeが勝手にスキップすることは禁止されています" >&2
        echo "⚠️ 必ずユーザーの承認を得てください" >&2
        echo "───────────────────────────────────────────" >&2
    fi

    if echo "$TOOL_OUTPUT" | grep -q "次のステップ: implementation"; then
        log "INFO: Implementation phase - suggesting pre-implementation skills"
        echo "" >&2
        echo "───────────────────────────────────────────────────" >&2
        echo "💡 実装フェーズに入りました。推奨Skill:" >&2
        echo "   /pre-implementation-check  - 実装前チェックリスト" >&2
        echo "   /impact-analysis          - 影響度調査" >&2
        echo "   /test-first               - TDD ガイド" >&2
        echo "───────────────────────────────────────────────────" >&2
    fi

    if echo "$TOOL_OUTPUT" | grep -q "次のステップ: testing"; then
        log "INFO: Testing phase - suggesting test skills"
        echo "" >&2
        echo "───────────────────────────────────────────────────" >&2
        echo "💡 テストフェーズに入りました。推奨Skill:" >&2
        echo "   /test-first               - TDD ガイド" >&2
        echo "───────────────────────────────────────────────────" >&2
    fi

    if echo "$TOOL_OUTPUT" | grep -q "次のステップ: review"; then
        log "INFO: Review phase - suggesting architecture review"
        echo "" >&2
        echo "───────────────────────────────────────────────────" >&2
        echo "💡 レビューフェーズに入りました。推奨Skill:" >&2
        echo "   /architecture-review      - アーキテクト的確認" >&2
        echo "───────────────────────────────────────────────────" >&2
    fi
fi

# Detect general Python errors
if echo "$TOOL_OUTPUT" | grep -qE "^Traceback \(most recent call last\):"; then
    log "ERROR: Python traceback detected"
    echo "" >&2
    echo "---" >&2
    echo "Python error detected. Suggestions:" >&2
    echo "  1. Check virtual environment: source sam-env/bin/activate" >&2
    echo "  2. Verify dependencies: pip list | grep -E 'google|sqlite'" >&2
    echo "  3. Check configuration: python tools/progress_tracker/cli.py check-config" >&2
    echo "---" >&2
fi

# Detect git-related issues
if echo "$TOOL_OUTPUT" | grep -qE "(fatal:|error:.*git|not a git repository)"; then
    log "WARNING: Git error detected"
    echo "" >&2
    echo "Git issue detected. Check:" >&2
    echo "  - Current directory: pwd" >&2
    echo "  - Git status: git status" >&2
    echo "  - Branch: git branch --show-current" >&2
fi

# Detect permission issues
if echo "$TOOL_OUTPUT" | grep -qE "(Permission denied|EACCES)"; then
    log "WARNING: Permission error detected"
    echo "" >&2
    echo "Permission issue detected. Try:" >&2
    echo "  chmod +x <script_path>" >&2
fi

# Pass through the original output
if [[ -n "$TOOL_OUTPUT" ]]; then
    echo "$TOOL_OUTPUT"
fi

# Always exit 0 for PostToolUse (informational only)
exit 0
