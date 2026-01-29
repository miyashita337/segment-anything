#!/bin/bash
# SessionEnd: Markdown形式の可読ログを生成

set -e

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // "unknown"')
TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')

LOG_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/transcripts/logs"
mkdir -p "$LOG_DIR"

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  OUTPUT_LOG="$LOG_DIR/session_${SESSION_ID}_$(date +%Y%m%d).md"

  {
    echo "# Session Log: $SESSION_ID"
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "## Transcript"
    echo '```jsonl'
    cat "$TRANSCRIPT_PATH"
    echo '```'
  } > "$OUTPUT_LOG"

  echo "Log saved: $OUTPUT_LOG" >&2
fi

exit 0
