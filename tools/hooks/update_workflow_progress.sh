#!/bin/bash
# INTG-086: ワークフロー進捗更新スクリプト
# Claude Hook - PostToolUse で実行される進捗記録・品質検証

set -euo pipefail

# ログ設定（動的・workspace_config.py活用版）
LOG_FILE="/tmp/intg086_workflow_progress.log"  # デフォルト値
CHECKLISTS_FILE="$(dirname $0)/workflow_checklists.json"

# 動的ログファイル設定関数
setup_log_file() {
    local tracker_id="$1"
    local workspace_path="$2"
    
    if [[ -n "$tracker_id" ]] && [[ -n "$workspace_path" ]] && [[ -d "$workspace_path" ]]; then
        LOG_DIR="$workspace_path/.workflow/logs"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/workflow_progress.log"
    else
        LOG_FILE="/tmp/intg086_workflow_progress.log"
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
    echo "$input" | grep -oP '[A-Z]{3,4}-[0-9]{3}' | head -1 || echo ""
}

# ダッシュボード品質検証
validate_dashboard_quality() {
    local workspace="$1"
    local extraction_result="$workspace/extraction_result.json"
    
    if [[ ! -f "$extraction_result" ]]; then
        log "⚠️ extraction_result.json が存在しません"
        return 1
    fi
    
    # 必須フィールドチェック
    local validation_result=$(python3 -c "
import json
import sys

try:
    with open('$extraction_result', 'r') as f:
        data = json.load(f)
    
    required_fields = [
        'tracker_id', 'total_images', 'successful_extractions',
        'average_quality_score', 'statistical_analysis', 'extraction_results'
    ]
    
    missing = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    
    if missing:
        print('MISSING:', ','.join(missing))
        sys.exit(1)
    
    # 品質スコアチェック
    success_rate = data['successful_extractions'] / data['total_images'] if data['total_images'] > 0 else 0
    
    result = {
        'valid': True,
        'success_rate': success_rate,
        'total_images': data['total_images'],
        'successful': data['successful_extractions']
    }
    
    print(json.dumps(result))
    
except Exception as e:
    print('ERROR:', str(e))
    sys.exit(1)
" 2>&1)
    
    if [[ "$validation_result" == ERROR:* ]] || [[ "$validation_result" == MISSING:* ]]; then
        log "❌ ダッシュボード品質検証失敗: $validation_result"
        return 1
    fi
    
    echo "$validation_result"
    return 0
}

# メイン処理
INPUT_JSON=$(cat)
TOOL=$(echo "$INPUT_JSON" | jq -r '.tool // empty')
TOOL_ARGS=$(echo "$INPUT_JSON" | jq -r '.args // empty')
TOOL_RESULT=$(echo "$INPUT_JSON" | jq -r '.result // empty')

log "📊 INTG-086 ワークフロー進捗更新開始: Tool=$TOOL"

# コマンドまたはファイルパスからトラッカーID抽出
COMMAND=$(echo "$TOOL_ARGS" | jq -r '.command // empty' 2>/dev/null || echo "")
FILE_PATH=$(echo "$TOOL_ARGS" | jq -r '.file_path // empty' 2>/dev/null || echo "")
TRACKER_ID=$(extract_tracker_id "$COMMAND $FILE_PATH")

if [[ -z "$TRACKER_ID" ]]; then
    log "⚠️ トラッカーIDが検出できません"
    echo '{"status": "skipped", "message": "トラッカーID不明のためスキップ"}'
    exit 0
fi

log "📋 トラッカーID: $TRACKER_ID"

# ワークスペースパス取得
WORKSPACE_PATH=$(get_workspace_path "$TRACKER_ID")
if [[ $? -ne 0 ]] || [[ -z "$WORKSPACE_PATH" ]]; then
    log "❌ ワークスペースパス取得失敗"
    echo '{"status": "error", "message": "ワークスペースパス取得失敗"}'
    exit 1
fi

# 動的ログファイル設定（ワークスペースパス確定後）
setup_log_file "$TRACKER_ID" "$WORKSPACE_PATH"
log "📁 ログファイル設定: $LOG_FILE"

# 状態管理ディレクトリ
WORKFLOW_STATE_DIR="$WORKSPACE_PATH/.workflow"
mkdir -p "$WORKFLOW_STATE_DIR"

# チェックリスト状態ファイル
CHECKLIST_STATUS_FILE="$WORKFLOW_STATE_DIR/checklist_status.json"
PHASE_PROGRESS_FILE="$WORKFLOW_STATE_DIR/phase_progress.json"

# 現在の状態読み込み（なければ初期化）
if [[ -f "$CHECKLIST_STATUS_FILE" ]]; then
    CURRENT_STATUS=$(cat "$CHECKLIST_STATUS_FILE")
else
    CURRENT_STATUS='{}'
fi

# コマンド実行結果に基づく進捗更新
UPDATE_MADE=false

# sam-env確認完了
if [[ "$COMMAND" == *"source sam-env/bin/activate"* ]] || [[ "$COMMAND" == *"echo \$VIRTUAL_ENV"* ]]; then
    CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq '.phase_1_planning.sam_env_check = true')
    log "✅ sam-env確認完了を記録"
    UPDATE_MADE=true
fi

# Google Sheets連携完了
if [[ "$COMMAND" == *"progress_tracker/cli.py"* ]]; then
    CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq '.phase_1_planning.google_sheets_sync = true')
    log "✅ Google Sheets連携完了を記録"
    UPDATE_MADE=true
fi

# 抽出処理完了
if [[ "$COMMAND" == *"extract_character.py"* ]]; then
    CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq '.phase_2_implementation.started = true')
    log "✅ 抽出処理実行を記録"
    UPDATE_MADE=true
    
    # 出力ディレクトリの品質チェック
    OUTPUT_DIR=$(echo "$COMMAND" | grep -oP '(?<=-o\s|--output_dir\s)[^\s]+' | head -1 || echo "")
    if [[ -n "$OUTPUT_DIR" ]] && [[ -d "$OUTPUT_DIR" ]]; then
        FILE_COUNT=$(find "$OUTPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        log "📊 抽出結果: $FILE_COUNT ファイル生成"
        CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq --arg count "$FILE_COUNT" '.extraction_stats.file_count = ($count | tonumber)')
    fi
fi

# 品質ワークフロー完了
if [[ "$COMMAND" == *"run_quality_workflow.sh"* ]]; then
    CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq '.phase_3_quality.workflow_executed = true')
    log "✅ 品質ワークフロー実行を記録"
    UPDATE_MADE=true
    
    # ダッシュボード品質検証
    if DASHBOARD_RESULT=$(validate_dashboard_quality "$WORKSPACE_PATH"); then
        log "✅ ダッシュボード品質検証成功"
        CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq '.phase_3_quality.dashboard_created = true')
        CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq --argjson result "$DASHBOARD_RESULT" '.dashboard_quality = $result')
    else
        log "⚠️ ダッシュボード品質検証失敗"
    fi
fi

# ファイル編集の記録
if [[ "$TOOL" == "Edit" ]] || [[ "$TOOL" == "Write" ]]; then
    if [[ "$FILE_PATH" == *"$TRACKER_ID"* ]]; then
        CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq --arg file "$FILE_PATH" '.edited_files += [$file]')
        log "📝 ファイル編集記録: $FILE_PATH"
        UPDATE_MADE=true
    fi
fi

# 更新があれば保存
if [[ "$UPDATE_MADE" == "true" ]]; then
    UPDATE_TIME=$(date -Iseconds)
    CURRENT_STATUS=$(echo "$CURRENT_STATUS" | jq --arg time "$UPDATE_TIME" '.last_updated = $time')
    echo "$CURRENT_STATUS" > "$CHECKLIST_STATUS_FILE"
    log "💾 チェックリスト状態を更新"
fi

# フェーズ進捗の計算と更新
calculate_phase_progress() {
    local status="$1"
    
    local phases=()
    
    # Phase 0.5
    if [[ $(echo "$status" | jq -r '.phase_0_5_branch // false') == "true" ]]; then
        phases+=('{"id": 0.5, "name": "branch_verification", "status": "completed"}')
    else
        phases+=('{"id": 0.5, "name": "branch_verification", "status": "pending"}')
    fi
    
    # Phase 1
    local phase1_items=$(echo "$status" | jq -r '.phase_1_planning // {} | [.sam_env_check, .google_sheets_sync, .sow_creation] | map(. // false) | map(if . then 1 else 0 end) | add')
    if [[ "$phase1_items" == "3" ]]; then
        phases+=('{"id": 1, "name": "planning", "status": "completed"}')
    elif [[ "$phase1_items" == "0" ]]; then
        phases+=('{"id": 1, "name": "planning", "status": "pending"}')
    else
        phases+=('{"id": 1, "name": "planning", "status": "in_progress"}')
    fi
    
    # Phase 2
    if [[ $(echo "$status" | jq -r '.phase_2_implementation.started // false') == "true" ]]; then
        phases+=('{"id": 2, "name": "implementation", "status": "in_progress"}')
    else
        phases+=('{"id": 2, "name": "implementation", "status": "pending"}')
    fi
    
    # Phase 3
    if [[ $(echo "$status" | jq -r '.phase_3_quality.workflow_executed // false') == "true" ]]; then
        if [[ $(echo "$status" | jq -r '.phase_3_quality.dashboard_created // false') == "true" ]]; then
            phases+=('{"id": 3, "name": "quality", "status": "completed"}')
        else
            phases+=('{"id": 3, "name": "quality", "status": "in_progress"}')
        fi
    else
        phases+=('{"id": 3, "name": "quality", "status": "pending"}')
    fi
    
    # JSON配列として出力
    echo "{\"phases\": [$(IFS=,; echo "${phases[*]}")]}"
}

# フェーズ進捗を計算して保存
PHASE_PROGRESS=$(calculate_phase_progress "$CURRENT_STATUS")
echo "$PHASE_PROGRESS" > "$PHASE_PROGRESS_FILE"

# 進捗サマリー生成
COMPLETED_COUNT=$(echo "$PHASE_PROGRESS" | jq '[.phases[] | select(.status == "completed")] | length')
TOTAL_PHASES=4
PROGRESS_PERCENT=$(echo "scale=1; $COMPLETED_COUNT * 100 / $TOTAL_PHASES" | bc)

log "📊 全体進捗: $COMPLETED_COUNT/$TOTAL_PHASES フェーズ完了 ($PROGRESS_PERCENT%)"

# 結果返却
cat <<EOF
{
  "status": "completed",
  "message": "ワークフロー進捗更新完了",
  "tracker_id": "$TRACKER_ID",
  "progress": {
    "completed_phases": $COMPLETED_COUNT,
    "total_phases": $TOTAL_PHASES,
    "percentage": $PROGRESS_PERCENT
  }
}
EOF