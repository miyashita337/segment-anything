#!/bin/bash
# INTG-086: ワークフローコンプライアンス検証スクリプト
# Claude Hook - PreToolUse で実行される5つのドキュメント準拠チェック

set -euo pipefail

# ログ設定（動的・workspace_config.py活用版）
LOG_FILE="/tmp/intg086_workflow_compliance.log"  # デフォルト値
CHECKLISTS_FILE="$(dirname $0)/workflow_checklists.json"

# 動的ログファイル設定関数
setup_log_file() {
    local tracker_id="$1"
    local workspace_path="$2"
    
    if [[ -n "$tracker_id" ]] && [[ -n "$workspace_path" ]] && [[ -d "$workspace_path" ]]; then
        LOG_DIR="$workspace_path/.workflow/logs"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/workflow_compliance.log"
    else
        LOG_FILE="/tmp/intg086_workflow_compliance.log"
    fi
}

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# workspace_config.pyを使用したワークスペースパス取得
get_workspace_path() {
    local tracker_id="$1"
    local workspace_path
    
    workspace_path=$(python3 -c "
import sys
sys.path.insert(0, '/mnt/c/AItools/segment-anything')
try:
    from config.workspace_config import WorkspaceConfig
    tracker_id = '$tracker_id'
    workspace = WorkspaceConfig.get_tracker_workspace(tracker_id)
    print(workspace)
except Exception as e:
    print('ERROR:', str(e), file=sys.stderr)
    sys.exit(1)
" 2>&1)
    
    if [[ "$workspace_path" == ERROR:* ]]; then
        log "❌ workspace_config.py エラー: ${workspace_path#ERROR:}"
        return 1
    fi
    
    echo "$workspace_path"
}

# トラッカーIDの抽出
extract_tracker_id() {
    local input="$1"
    # コマンドまたはファイルパスからトラッカーIDを抽出
    echo "$input" | grep -oP '[A-Z]{3,4}-[0-9]{3}' | head -1 || echo ""
}

# メイン処理
INPUT_JSON=$(cat)
TOOL=$(echo "$INPUT_JSON" | jq -r '.tool // empty')
TOOL_ARGS=$(echo "$INPUT_JSON" | jq -r '.args // empty')

log "🔍 INTG-086 ワークフローコンプライアンス検証開始: Tool=$TOOL"

# コマンドまたはファイルパスからトラッカーID抽出
COMMAND=$(echo "$TOOL_ARGS" | jq -r '.command // empty' 2>/dev/null || echo "")
FILE_PATH=$(echo "$TOOL_ARGS" | jq -r '.file_path // empty' 2>/dev/null || echo "")
TRACKER_ID=$(extract_tracker_id "$COMMAND $FILE_PATH")

if [[ -z "$TRACKER_ID" ]]; then
    log "⚠️ トラッカーIDが検出できません。汎用チェックモードで実行"
    # トラッカーID不明時は基本チェックのみ実行
    echo '{"allow": true, "message": "トラッカーID不明のため基本検証のみ"}'
    exit 0
fi

log "📋 トラッカーID検出: $TRACKER_ID"

# ワークスペースパス取得
WORKSPACE_PATH=$(get_workspace_path "$TRACKER_ID")
if [[ $? -ne 0 ]] || [[ -z "$WORKSPACE_PATH" ]]; then
    log "❌ エラー: ワークスペースパスを取得できません"
    cat <<EOF
{
  "allow": false,
  "reason": "ワークスペースパス取得失敗",
  "error": "workspace_config.pyからトラッカー $TRACKER_ID のパスを取得できません",
  "action": "workspace_config.pyの設定とトラッカーIDを確認してください"
}
EOF
    exit 1
fi

# ワークスペース存在確認
if [[ ! -d "$WORKSPACE_PATH" ]]; then
    log "⚠️ ワークスペースディレクトリが存在しません: $WORKSPACE_PATH"
    log "📁 ディレクトリを作成します..."
    mkdir -p "$WORKSPACE_PATH/.workflow"
fi

# 動的ログファイル設定（ワークスペースパス確定後）
setup_log_file "$TRACKER_ID" "$WORKSPACE_PATH"
log "📁 ログファイル設定: $LOG_FILE"

# 状態管理ディレクトリ
WORKFLOW_STATE_DIR="$WORKSPACE_PATH/.workflow"
mkdir -p "$WORKFLOW_STATE_DIR"

# 現在のワークフロー状態読み込み
CHECKLIST_STATUS_FILE="$WORKFLOW_STATE_DIR/checklist_status.json"
if [[ ! -f "$CHECKLIST_STATUS_FILE" ]]; then
    # 初期状態ファイル作成
    cat > "$CHECKLIST_STATUS_FILE" <<EOF
{
  "tracker_id": "$TRACKER_ID",
  "created_at": "$(date -Iseconds)",
  "phase_0_5_branch": false,
  "phase_1_planning": {
    "sam_env_check": false,
    "google_sheets_sync": false,
    "sow_creation": false
  },
  "phase_2_implementation": {
    "started": false,
    "approval": false
  },
  "phase_3_quality": {
    "workflow_executed": false,
    "dashboard_created": false
  }
}
EOF
    log "📝 新規チェックリスト状態ファイル作成: $CHECKLIST_STATUS_FILE"
fi

# 特定のチェック実行
CURRENT_STATUS=$(cat "$CHECKLIST_STATUS_FILE")

# Phase 0.5: ブランチ検証（最重要）
if [[ "$TOOL" == "Bash" ]] && [[ "$COMMAND" == *"git"* ]]; then
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    
    if [[ "$CURRENT_BRANCH" != feature/* ]]; then
        log "❌ CRITICAL: feature/ブランチではありません - 現在: $CURRENT_BRANCH"
        log "⚠️ ユーザー確認が必要です - ブランチ問題を検出しました"
        cat <<EOF
{
  "allow": false,
  "reason": "ブランチ検証失敗 - ユーザー確認要求",
  "severity": "CRITICAL",
  "error": "作業ブランチがfeature/で始まっていません",
  "current_branch": "$CURRENT_BRANCH",
  "required_pattern": "feature/$TRACKER_ID",
  "action": "git checkout -b feature/$TRACKER_ID を実行してください",
  "user_confirmation_required": true,
  "continue_anyway": false
}
EOF
        exit 1
    fi
    
    # ブランチ検証成功を記録
    echo "$CURRENT_STATUS" | jq '.phase_0_5_branch = true' > "$CHECKLIST_STATUS_FILE"
    log "✅ ブランチ検証成功: $CURRENT_BRANCH"
fi

# 入力パス検証（input_path_validation準拠）
if [[ "$COMMAND" == *"extract_character.py"* ]] || [[ "$COMMAND" == *"run_quality_workflow.sh"* ]]; then
    INPUT_PATH=$(echo "$COMMAND" | grep -oP '(?<=-i\s|--input_dir\s)[^\s]+|(?<=^)[^\s]+(?=/[^/\s]+\.(jpg|png))' | head -1 || echo "")
    
    if [[ -n "$INPUT_PATH" ]] && [[ ! -d "$INPUT_PATH" ]] && [[ ! -f "$INPUT_PATH" ]]; then
        log "❌ エラー: 入力ディレクトリが存在しません: $INPUT_PATH"
        log "⚠️ ユーザー確認が必要です - パス検証問題を検出しました"
        cat <<EOF
{
  "allow": false,
  "reason": "入力パス検証失敗 - ユーザー確認要求",
  "severity": "HIGH",
  "error": "入力ディレクトリが存在しません",
  "path": "$INPUT_PATH",
  "action": "正しいパスを指定してください。代替案の提案は禁止されています。",
  "user_confirmation_required": true,
  "continue_anyway": false,
  "suggestions": [
    "パスの確認: ls $(dirname '$INPUT_PATH')",
    "正しいパスの再指定",
    "必要に応じてディレクトリ作成"
  ]
}
EOF
        exit 1
    fi
    
    if [[ -n "$INPUT_PATH" ]]; then
        log "✅ 入力パス検証成功: $INPUT_PATH"
    fi
fi

# 品質ワークフロー実行チェック
if [[ "$COMMAND" == *"run_quality_workflow.sh"* ]]; then
    # Phase 1が完了していることを確認
    PHASE_1_COMPLETE=$(echo "$CURRENT_STATUS" | jq -r '
        if .phase_1_planning.sam_env_check and 
           .phase_1_planning.google_sheets_sync and 
           .phase_1_planning.sow_creation 
        then "true" else "false" end
    ')
    
    if [[ "$PHASE_1_COMPLETE" != "true" ]]; then
        log "⚠️ 警告: Phase 1が未完了で品質ワークフロー実行を試みています"
        log "💡 ユーザー確認要求: ワークフロー順序問題を検出しました"
        
        # Phase 1の詳細状況確認
        PHASE_1_STATUS=$(echo "$CURRENT_STATUS" | jq -r '{
            sam_env: .phase_1_planning.sam_env_check,
            google_sheets: .phase_1_planning.google_sheets_sync,
            sow: .phase_1_planning.sow_creation
        }')
        
        cat <<EOF
{
  "allow": true,
  "reason": "ワークフロー順序警告 - ユーザー確認推奨",
  "severity": "MEDIUM",
  "warning": "Phase 1が未完了で品質ワークフロー実行を試みています",
  "phase_1_status": $PHASE_1_STATUS,
  "user_confirmation_required": true,
  "continue_anyway": true,
  "recommendations": [
    "Phase 1完了後の実行を推奨",
    "未完了項目: sam_env_check, google_sheets_sync, sow_creation の確認",
    "強制実行する場合は十分注意してください"
  ]
}
EOF
        return 0
    fi
fi

# 状態更新
UPDATE_TIME=$(date -Iseconds)
echo "$CURRENT_STATUS" | jq --arg time "$UPDATE_TIME" '.last_checked = $time' > "$CHECKLIST_STATUS_FILE"

log "✅ ワークフローコンプライアンス検証完了"
echo '{"allow": true, "message": "ワークフロー検証成功", "tracker_id": "'$TRACKER_ID'", "workspace": "'$WORKSPACE_PATH'"}'