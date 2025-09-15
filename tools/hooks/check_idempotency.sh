#!/bin/bash
# INTG-086: 等冪性チェックスクリプト
# Claude Hook - PreToolUse で実行される重複実行防止システム

set -euo pipefail

# ログ設定
LOG_FILE="/tmp/intg086_idempotency.log"
LOCK_DIR="/tmp/intg086_locks"
STATE_FILE="/tmp/intg086_state.json"

# ロックディレクトリ作成
mkdir -p "$LOCK_DIR"

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# JSON入力の解析
INPUT_JSON=$(cat)
TOOL=$(echo "$INPUT_JSON" | jq -r '.tool // empty')
TOOL_ARGS=$(echo "$INPUT_JSON" | jq -r '.args // empty')

log "🔍 INTG-086等冪性チェック開始: Tool=$TOOL"

# 危険なコマンドパターンをチェック
if [[ "$TOOL" == "Bash" ]]; then
    COMMAND=$(echo "$TOOL_ARGS" | jq -r '.command // empty')
    
    # 抽出処理の重複実行チェック
    if [[ "$COMMAND" == *"extract_character.py"* ]]; then
        EXTRACTION_LOCK="$LOCK_DIR/extraction_running.lock"
        
        if [[ -f "$EXTRACTION_LOCK" ]]; then
            LOCK_PID=$(cat "$EXTRACTION_LOCK" 2>/dev/null || echo "")
            if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
                log "❌ 抽出処理が既に実行中です (PID: $LOCK_PID)"
                echo '{"allow": false, "reason": "抽出処理が既に実行中のため、重複実行を防止しました"}'
                exit 1
            else
                log "⚠️ 古いロックファイルを削除します"
                rm -f "$EXTRACTION_LOCK"
            fi
        fi
        
        # 新しいロックを作成
        echo $$ > "$EXTRACTION_LOCK"
        log "🔒 抽出処理ロック作成: PID=$$"
    fi
    
    # 品質ワークフローの重複チェック
    if [[ "$COMMAND" == *"run_quality_workflow.sh"* ]]; then
        QUALITY_LOCK="$LOCK_DIR/quality_running.lock"
        
        if [[ -f "$QUALITY_LOCK" ]]; then
            LOCK_PID=$(cat "$QUALITY_LOCK" 2>/dev/null || echo "")
            if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
                log "❌ 品質ワークフローが既に実行中です (PID: $LOCK_PID)"
                echo '{"allow": false, "reason": "品質ワークフローが既に実行中のため、重複実行を防止しました"}'
                exit 1
            else
                rm -f "$QUALITY_LOCK"
            fi
        fi
        
        echo $$ > "$QUALITY_LOCK"
        log "🔒 品質ワークフローロック作成: PID=$$"
    fi
fi

# Edit/Write操作の一致性チェック
if [[ "$TOOL" == "Edit" ]] || [[ "$TOOL" == "Write" ]] || [[ "$TOOL" == "MultiEdit" ]]; then
    FILE_PATH=$(echo "$TOOL_ARGS" | jq -r '.file_path // empty')
    
    # 重要ファイルの同時編集防止
    if [[ "$FILE_PATH" == *".py" ]] || [[ "$FILE_PATH" == *".sh" ]] || [[ "$FILE_PATH" == *".json" ]]; then
        FILE_LOCK="$LOCK_DIR/$(basename "$FILE_PATH").lock"
        
        if [[ -f "$FILE_LOCK" ]]; then
            LOCK_TIME=$(stat -c %Y "$FILE_LOCK" 2>/dev/null || echo "0")
            CURRENT_TIME=$(date +%s)
            
            # 5分以上古いロックは無効とする
            if (( CURRENT_TIME - LOCK_TIME > 300 )); then
                log "⚠️ 古いファイルロックを削除: $FILE_PATH"
                rm -f "$FILE_LOCK"
            else
                log "❌ ファイルが編集中です: $FILE_PATH"
                echo '{"allow": false, "reason": "ファイルが他の処理で編集中のため、同時編集を防止しました"}'
                exit 1
            fi
        fi
        
        touch "$FILE_LOCK"
        log "📝 ファイル編集ロック作成: $FILE_PATH"
    fi
fi

# 状態情報を記録
CURRENT_STATE=$(cat <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "tool": "$TOOL",
  "pid": $$,
  "action": "pre_tool_use_check"
}
EOF
)

echo "$CURRENT_STATE" > "$STATE_FILE"

log "✅ 等冪性チェック完了 - 実行を許可"
echo '{"allow": true, "message": "等冪性チェック完了 - 安全に実行できます"}'