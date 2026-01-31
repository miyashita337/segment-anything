#!/bin/bash
# PreCompact: トランスクリプトをアーカイブ

set -e

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // "unknown"')
TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')
TRIGGER=$(echo "$HOOK_INPUT" | jq -r '.trigger // "manual"')

ARCHIVE_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/transcripts/archive"
mkdir -p "$ARCHIVE_DIR"

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  ARCHIVE_FILE="$ARCHIVE_DIR/${SESSION_ID}_${TRIGGER}_$(date +%Y%m%d_%H%M%S).jsonl"
  cp "$TRANSCRIPT_PATH" "$ARCHIVE_FILE"
  echo "Archived: $ARCHIVE_FILE" >&2
fi

exit 0
