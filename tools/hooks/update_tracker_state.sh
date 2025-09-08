#!/bin/bash
# INTG-086: トラッカー状態更新スクリプト
# Claude Hook - PostToolUse で実行される品質検証・進捗管理システム

set -euo pipefail

# 設定
LOG_FILE="/tmp/intg086_tracker_state.log"
STATE_DIR="/tmp/intg086_tracker_states"
LOCK_DIR="/tmp/intg086_locks"
QUALITY_THRESHOLDS_FILE="/tmp/intg086_quality_thresholds.json"

# ディレクトリ作成
mkdir -p "$STATE_DIR" "$LOCK_DIR"

# 品質閾値設定
cat > "$QUALITY_THRESHOLDS_FILE" << 'EOF'
{
  "extraction_success_rate": 0.85,
  "dashboard_completeness": 0.95,
  "workflow_consistency": 0.90,
  "minimum_extracted_files": 3
}
EOF

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 品質評価関数
evaluate_extraction_quality() {
    local output_dir="$1"
    local success_count=0
    local total_count=0
    
    if [[ -d "$output_dir" ]]; then
        for file in "$output_dir"/*.jpg "$output_dir"/*.png; do
            if [[ -f "$file" ]]; then
                ((total_count++))
                # ファイルサイズチェック（0バイトでなければ成功とみなす）
                if [[ -s "$file" ]]; then
                    ((success_count++))
                fi
            fi
        done 2>/dev/null
    fi
    
    local success_rate=0
    if (( total_count > 0 )); then
        success_rate=$(echo "scale=3; $success_count / $total_count" | bc -l)
    fi
    
    echo "{ \"success_count\": $success_count, \"total_count\": $total_count, \"success_rate\": $success_rate }"
}

# ダッシュボード完成度評価
evaluate_dashboard_completeness() {
    local tracker_workspace="$1"
    local components=("index.html" "progress.json" "extraction_report.html" "quality_metrics.json")
    local existing_count=0
    
    for component in "${components[@]}"; do
        if [[ -f "$tracker_workspace/$component" ]]; then
            ((existing_count++))
        fi
    done
    
    local completeness=$(echo "scale=3; $existing_count / ${#components[@]}" | bc -l)
    echo "{ \"existing_count\": $existing_count, \"total_components\": ${#components[@]}, \"completeness\": $completeness }"
}

# JSON入力解析
INPUT_JSON=$(cat)
TOOL=$(echo "$INPUT_JSON" | jq -r '.tool // empty')
TOOL_ARGS=$(echo "$INPUT_JSON" | jq -r '.args // empty')
TOOL_RESULT=$(echo "$INPUT_JSON" | jq -r '.result // empty')

log "📊 INTG-086状態更新開始: Tool=$TOOL"

# 抽出処理完了後の品質チェック
if [[ "$TOOL" == "Bash" ]]; then
    COMMAND=$(echo "$TOOL_ARGS" | jq -r '.command // empty')
    
    if [[ "$COMMAND" == *"extract_character.py"* ]]; then
        # 抽出ロック削除
        EXTRACTION_LOCK="$LOCK_DIR/extraction_running.lock"
        if [[ -f "$EXTRACTION_LOCK" ]]; then
            LOCK_PID=$(cat "$EXTRACTION_LOCK" 2>/dev/null || echo "")
            if [[ "$LOCK_PID" == "$$" ]]; then
                rm -f "$EXTRACTION_LOCK"
                log "🔓 抽出処理ロック解除: PID=$$"
            fi
        fi
        
        # 出力ディレクトリから品質評価
        OUTPUT_DIR=$(echo "$COMMAND" | grep -oP '\-o\s+\K[^\s]+' || echo "")
        if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
            QUALITY_RESULT=$(evaluate_extraction_quality "$OUTPUT_DIR")
            SUCCESS_RATE=$(echo "$QUALITY_RESULT" | jq -r '.success_rate')
            SUCCESS_COUNT=$(echo "$QUALITY_RESULT" | jq -r '.success_count')
            
            # 品質閾値チェック
            MIN_SUCCESS_RATE=$(jq -r '.extraction_success_rate' "$QUALITY_THRESHOLDS_FILE")
            MIN_FILES=$(jq -r '.minimum_extracted_files' "$QUALITY_THRESHOLDS_FILE")
            
            if (( $(echo "$SUCCESS_RATE >= $MIN_SUCCESS_RATE" | bc -l) )) && (( SUCCESS_COUNT >= MIN_FILES )); then
                log "✅ 抽出品質合格: 成功率=$SUCCESS_RATE (閾値=$MIN_SUCCESS_RATE), ファイル数=$SUCCESS_COUNT"
                QUALITY_STATUS="PASSED"
            else
                log "⚠️ 抽出品質不足: 成功率=$SUCCESS_RATE (閾値=$MIN_SUCCESS_RATE), ファイル数=$SUCCESS_COUNT"
                QUALITY_STATUS="FAILED"
            fi
            
            # 状態ファイル更新
            STATE_FILE="$STATE_DIR/extraction_$(date +%s).json"
            cat > "$STATE_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "tool": "$TOOL",
  "command": "$COMMAND",
  "output_directory": "$OUTPUT_DIR",
  "quality_result": $QUALITY_RESULT,
  "quality_status": "$QUALITY_STATUS",
  "thresholds_met": $(if [[ "$QUALITY_STATUS" == "PASSED" ]]; then echo "true"; else echo "false"; fi)
}
EOF
            
            log "📝 抽出状態記録: $STATE_FILE"
        fi
    fi
    
    # 品質ワークフロー完了処理
    if [[ "$COMMAND" == *"run_quality_workflow.sh"* ]]; then
        QUALITY_LOCK="$LOCK_DIR/quality_running.lock"
        if [[ -f "$QUALITY_LOCK" ]]; then
            LOCK_PID=$(cat "$QUALITY_LOCK" 2>/dev/null || echo "")
            if [[ "$LOCK_PID" == "$$" ]]; then
                rm -f "$QUALITY_LOCK"
                log "🔓 品質ワークフローロック解除: PID=$$"
            fi
        fi
    fi
fi

# Edit/Write操作完了後の処理
if [[ "$TOOL" == "Edit" ]] || [[ "$TOOL" == "Write" ]] || [[ "$TOOL" == "MultiEdit" ]]; then
    FILE_PATH=$(echo "$TOOL_ARGS" | jq -r '.file_path // empty')
    
    # ファイルロック削除
    if [[ -n "$FILE_PATH" ]]; then
        FILE_LOCK="$LOCK_DIR/$(basename "$FILE_PATH").lock"
        if [[ -f "$FILE_LOCK" ]]; then
            rm -f "$FILE_LOCK"
            log "📝 ファイル編集ロック解除: $FILE_PATH"
        fi
        
        # ダッシュボードファイルの場合、完成度評価
        if [[ "$FILE_PATH" == *"index.html"* ]] || [[ "$FILE_PATH" == *"progress.json"* ]]; then
            TRACKER_WORKSPACE=$(dirname "$FILE_PATH")
            DASHBOARD_RESULT=$(evaluate_dashboard_completeness "$TRACKER_WORKSPACE")
            COMPLETENESS=$(echo "$DASHBOARD_RESULT" | jq -r '.completeness')
            
            MIN_COMPLETENESS=$(jq -r '.dashboard_completeness' "$QUALITY_THRESHOLDS_FILE")
            
            if (( $(echo "$COMPLETENESS >= $MIN_COMPLETENESS" | bc -l) )); then
                log "✅ ダッシュボード品質合格: 完成度=$COMPLETENESS (閾値=$MIN_COMPLETENESS)"
                DASHBOARD_STATUS="PASSED"
            else
                log "⚠️ ダッシュボード品質不足: 完成度=$COMPLETENESS (閾値=$MIN_COMPLETENESS)"
                DASHBOARD_STATUS="FAILED"
            fi
            
            # 状態記録
            STATE_FILE="$STATE_DIR/dashboard_$(date +%s).json"
            cat > "$STATE_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "tool": "$TOOL",
  "file_path": "$FILE_PATH",
  "tracker_workspace": "$TRACKER_WORKSPACE",
  "dashboard_result": $DASHBOARD_RESULT,
  "dashboard_status": "$DASHBOARD_STATUS"
}
EOF
            
            log "📊 ダッシュボード状態記録: $STATE_FILE"
        fi
    fi
fi

# 現在の全体的品質状況レポート生成
generate_quality_report() {
    local latest_extraction_state=$(ls -t "$STATE_DIR"/extraction_*.json 2>/dev/null | head -1)
    local latest_dashboard_state=$(ls -t "$STATE_DIR"/dashboard_*.json 2>/dev/null | head -1)
    
    local extraction_status="UNKNOWN"
    local dashboard_status="UNKNOWN"
    
    if [[ -n "$latest_extraction_state" && -f "$latest_extraction_state" ]]; then
        extraction_status=$(jq -r '.quality_status' "$latest_extraction_state")
    fi
    
    if [[ -n "$latest_dashboard_state" && -f "$latest_dashboard_state" ]]; then
        dashboard_status=$(jq -r '.dashboard_status' "$latest_dashboard_state")
    fi
    
    # 全体品質評価
    local overall_status="PASSED"
    if [[ "$extraction_status" == "FAILED" ]] || [[ "$dashboard_status" == "FAILED" ]]; then
        overall_status="FAILED"
    elif [[ "$extraction_status" == "UNKNOWN" ]] || [[ "$dashboard_status" == "UNKNOWN" ]]; then
        overall_status="PARTIAL"
    fi
    
    cat > "/tmp/intg086_quality_report.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "extraction_status": "$extraction_status",
  "dashboard_status": "$dashboard_status", 
  "overall_status": "$overall_status",
  "latest_extraction_state": "$latest_extraction_state",
  "latest_dashboard_state": "$latest_dashboard_state"
}
EOF
    
    log "📋 品質レポート生成完了: 全体状況=$overall_status"
}

generate_quality_report

log "✅ トラッカー状態更新完了"
echo '{"status": "completed", "message": "品質検証・状態更新が完了しました"}'