#!/bin/bash
# PreCompact/SessionEnd: トランスクリプトを自動アーカイブ

set -e

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // "unknown"')
TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')
TRIGGER=$(echo "$HOOK_INPUT" | jq -r '.trigger // "manual"')

# アーカイブディレクトリ
ARCHIVE_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/transcripts/archive"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILENAME="${SESSION_ID}_${TRIGGER}_${TIMESTAMP}.jsonl"

mkdir -p "$ARCHIVE_DIR"

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  cp "$TRANSCRIPT_PATH" "$ARCHIVE_DIR/$BACKUP_FILENAME"
  echo "Archived: $ARCHIVE_DIR/$BACKUP_FILENAME" >&2
fi

exit 0
