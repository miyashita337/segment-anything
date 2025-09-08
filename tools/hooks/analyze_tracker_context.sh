#!/bin/bash
# INTG-086: トラッカーコンテキスト分析スクリプト
# Claude Hook - UserPromptSubmit で実行されるトラッカー状況分析・次ステップ提案

set -euo pipefail

# ログ設定（動的・workspace_config.py活用版）
LOG_FILE="/tmp/intg086_context_analysis.log"  # デフォルト値

# 動的ログファイル設定関数
setup_log_file() {
    local tracker_id="$1"
    local workspace_path="$2"
    
    if [[ -n "$tracker_id" ]] && [[ -n "$workspace_path" ]] && [[ -d "$workspace_path" ]]; then
        LOG_DIR="$workspace_path/.workflow/logs"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/context_analysis.log"
    else
        LOG_FILE="/tmp/intg086_context_analysis.log"
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

# トラッカーIDの抽出（複数パターン対応）
extract_tracker_id() {
    local prompt="$1"
    
    # パターン1: 直接的なトラッカーID（INTG-086, QUAL-001等）
    local direct_id=$(echo "$prompt" | grep -oP '[A-Z]{3,4}-[0-9]{3}' | head -1 || echo "")
    
    if [[ -n "$direct_id" ]]; then
        echo "$direct_id"
        return 0
    fi
    
    # パターン2: ファイルパスからの推測
    local path_id=$(echo "$prompt" | grep -oP '(?<=/)[A-Z]{3,4}-[0-9]{3}(?=/)' | head -1 || echo "")
    
    if [[ -n "$path_id" ]]; then
        echo "$path_id"
        return 0
    fi
    
    echo ""
}

# フェーズ判定
determine_current_phase() {
    local status="$1"
    
    # Phase 0.5チェック
    if [[ $(echo "$status" | jq -r '.phase_0_5_branch // false') == "false" ]]; then
        echo "0.5"
        return 0
    fi
    
    # Phase 1チェック
    local phase1_complete=$(echo "$status" | jq -r '
        if .phase_1_planning.sam_env_check and 
           .phase_1_planning.google_sheets_sync and 
           .phase_1_planning.sow_creation 
        then "true" else "false" end
    ')
    
    if [[ "$phase1_complete" == "false" ]]; then
        echo "1"
        return 0
    fi
    
    # Phase 2チェック
    if [[ $(echo "$status" | jq -r '.phase_2_implementation.started // false') == "false" ]]; then
        echo "2"
        return 0
    fi
    
    # Phase 3チェック
    if [[ $(echo "$status" | jq -r '.phase_3_quality.workflow_executed // false') == "false" ]]; then
        echo "3"
        return 0
    fi
    
    # 完了
    echo "completed"
}

# 5つのドキュメントチェック実行
check_five_documents_compliance() {
    local workspace="$1"
    local tracker_id="$2"
    local phase="$3"
    
    local compliance_result='{
        "unified_tracker_template": false,
        "tracker_workflow_checklist": false,
        "input_path_validation": false,
        "dashboard_quality": false,
        "output_directory_config": false
    }'
    
    # 1. unified_tracker_template チェック
    if [[ -f "docs/workflows/templates/unified_tracker_template.md" ]]; then
        compliance_result=$(echo "$compliance_result" | jq '.unified_tracker_template = true')
    fi
    
    # 2. tracker_workflow_checklist チェック
    if [[ -f "docs/workflows/checklists/tracker_workflow_checklist.md" ]]; then
        compliance_result=$(echo "$compliance_result" | jq '.tracker_workflow_checklist = true')
    fi
    
    # 3. input_path_validation チェック
    if [[ -f "docs/checklists/input_path_validation_checklist.md" ]]; then
        compliance_result=$(echo "$compliance_result" | jq '.input_path_validation = true')
    fi
    
    # 4. dashboard_quality チェック
    if [[ -f "docs/checklists/dashboard_quality_checklist.md" ]]; then
        compliance_result=$(echo "$compliance_result" | jq '.dashboard_quality = true')
    fi
    
    # 5. output_directory_config チェック
    if [[ -f "docs/workflows/output_directory_config.md" ]]; then
        compliance_result=$(echo "$compliance_result" | jq '.output_directory_config = true')
    fi
    
    log "📋 5つのドキュメントコンプライアンスチェック完了"
    echo "$compliance_result"
}

# フェーズ別詳細チェック
perform_phase_specific_checks() {
    local phase="$1"
    local tracker_id="$2"
    local workspace="$3"
    local status="$4"
    
    case "$phase" in
        "0.5")
            # ブランチチェック
            local current_branch=$(git branch --show-current 2>/dev/null || echo "")
            echo "{\"branch_check\": {\"current\": \"$current_branch\", \"required\": \"feature/$tracker_id\", \"valid\": $(if [[ \"$current_branch\" == feature/* ]]; then echo 'true'; else echo 'false'; fi)}}"
            ;;
        "1")
            # Phase 1 チェック詳細
            local phase1_details=$(echo "$status" | jq '{
                sam_env_check: .phase_1_planning.sam_env_check,
                google_sheets_sync: .phase_1_planning.google_sheets_sync,
                sow_creation: .phase_1_planning.sow_creation,
                completion_rate: ([.phase_1_planning.sam_env_check, .phase_1_planning.google_sheets_sync, .phase_1_planning.sow_creation] | map(if . then 1 else 0 end) | add) / 3 * 100
            }')
            echo "$phase1_details"
            ;;
        "2")
            # Phase 2 実装チェック
            local phase2_details=$(echo "$status" | jq '{
                implementation_started: .phase_2_implementation.started,
                extraction_stats: (.extraction_stats // {}),
                edited_files_count: (.edited_files // [] | length)
            }')
            echo "$phase2_details"
            ;;
        "3")
            # Phase 3 品質チェック
            local dashboard_exists="false"
            if [[ -f "$workspace/extraction_result.json" ]]; then
                dashboard_exists="true"
            fi
            
            local phase3_details=$(echo "$status" | jq --arg dashboard "$dashboard_exists" '{
                workflow_executed: .phase_3_quality.workflow_executed,
                dashboard_created: .phase_3_quality.dashboard_created,
                dashboard_file_exists: ($dashboard | test("true")),
                quality_validated: (.dashboard_quality // {})
            }')
            echo "$phase3_details"
            ;;
        *)
            echo "{\"phase_check\": \"not_implemented\"}"
            ;;
    esac
}

# 次ステップ提案（強化版）
suggest_next_steps() {
    local phase="$1"
    local tracker_id="$2"
    local workspace="$3"
    local status="$4"
    
    case "$phase" in
        "0.5")
            echo "🚨 Phase 0.5: ブランチ検証が必要"
            echo "  次のステップ: git checkout -b feature/$tracker_id"
            echo "  重要度: CRITICAL（これが完了するまで他の作業は禁止）"
            ;;
        "1")
            local missing_items=()
            if [[ $(echo "$status" | jq -r '.phase_1_planning.sam_env_check // false') == "false" ]]; then
                missing_items+=("sam-env仮想環境確認")
            fi
            if [[ $(echo "$status" | jq -r '.phase_1_planning.google_sheets_sync // false') == "false" ]]; then
                missing_items+=("Google Sheets連携")
            fi
            if [[ $(echo "$status" | jq -r '.phase_1_planning.sow_creation // false') == "false" ]]; then
                missing_items+=("SOW作成")
            fi
            
            echo "📋 Phase 1: 計画・準備フェーズ"
            echo "  未完了項目: ${missing_items[*]}"
            echo "  次のステップ:"
            echo "    1. sam-env仮想環境確認: source sam-env/bin/activate"
            echo "    2. Google Sheets連携: python tools/progress_tracker/cli.py status $tracker_id"
            echo "    3. SOW（作業範囲確定書）作成"
            ;;
        "2")
            echo "🔧 Phase 2: 実装フェーズ"
            echo "  次のステップ: 実装作業開始"
            echo "  注意: Phase 1の承認が完了していることを確認してください"
            
            # 抽出統計情報がある場合は表示
            local file_count=$(echo "$status" | jq -r '.extraction_stats.file_count // 0')
            if [[ "$file_count" -gt 0 ]]; then
                echo "  📊 抽出済み: $file_count ファイル"
            fi
            ;;
        "3")
            echo "✅ Phase 3: 品質ワークフロー"
            echo "  次のステップ: ./tools/scripts/run_quality_workflow.sh $tracker_id"
            echo "  最終段階: ダッシュボード生成・品質検証"
            
            # ダッシュボード状況確認
            if [[ -f "$workspace/extraction_result.json" ]]; then
                echo "  📊 ダッシュボード: 生成済み"
            else
                echo "  ⚠️ ダッシュボード: 未生成"
            fi
            ;;
        "completed")
            echo "🎉 全フェーズ完了"
            echo "  完了作業: Git操作（コミット・プッシュ・PR作成）"
            ;;
        *)
            echo "⚠️ フェーズ状態不明"
            ;;
    esac
}

# メイン処理
INPUT_JSON=$(cat)
USER_PROMPT=$(echo "$INPUT_JSON" | jq -r '.prompt // empty')

log "🔍 INTG-086 トラッカーコンテキスト分析開始"

# トラッカーIDの抽出
TRACKER_ID=$(extract_tracker_id "$USER_PROMPT")

if [[ -z "$TRACKER_ID" ]]; then
    log "⚠️ トラッカーIDが検出されませんでした"
    log "📝 ユーザープロンプト: $(echo "$USER_PROMPT" | head -c 100)..."
    echo '{"status": "no_tracker", "message": "トラッカーIDが検出されませんでした"}'
    exit 0
fi

log "📋 トラッカーID検出: $TRACKER_ID"

# ワークスペースパス取得
WORKSPACE_PATH=$(get_workspace_path "$TRACKER_ID")
if [[ $? -ne 0 ]] || [[ -z "$WORKSPACE_PATH" ]]; then
    log "⚠️ ワークスペースパス取得失敗 - 新規トラッカーの可能性"
    echo "{\"status\": \"new_tracker\", \"tracker_id\": \"$TRACKER_ID\", \"message\": \"新規トラッカーを検出。Phase 0.5（ブランチ検証）から開始してください。\"}"
    exit 0
fi

# 動的ログファイル設定（ワークスペースパス取得後）
setup_log_file "$TRACKER_ID" "$WORKSPACE_PATH"
log "📁 ログファイル設定: $LOG_FILE"

# 状態ファイルの確認
WORKFLOW_STATE_DIR="$WORKSPACE_PATH/.workflow"
CHECKLIST_STATUS_FILE="$WORKFLOW_STATE_DIR/checklist_status.json"

if [[ ! -f "$CHECKLIST_STATUS_FILE" ]]; then
    log "📄 新規トラッカー: チェックリスト状態ファイルが存在しません"
    
    # ワークスペース存在するが状態ファイルがない場合
    if [[ -d "$WORKSPACE_PATH" ]]; then
        echo "{\"status\": \"existing_tracker_no_state\", \"tracker_id\": \"$TRACKER_ID\", \"message\": \"既存のワークスペースが見つかりましたが、状態管理ファイルがありません。Phase 0.5から開始することをお勧めします。\"}"
    else
        echo "{\"status\": \"new_tracker\", \"tracker_id\": \"$TRACKER_ID\", \"message\": \"新規トラッカーを検出\"}"
    fi
    exit 0
fi

# 現在の状態分析
CURRENT_STATUS=$(cat "$CHECKLIST_STATUS_FILE")
CURRENT_PHASE=$(determine_current_phase "$CURRENT_STATUS")

log "📊 現在のフェーズ: $CURRENT_PHASE"

# ユーザープロンプトの意図分析
PROMPT_INTENT=""
if [[ "$USER_PROMPT" == *"開始"* ]] || [[ "$USER_PROMPT" == *"start"* ]]; then
    PROMPT_INTENT="start"
elif [[ "$USER_PROMPT" == *"確認"* ]] || [[ "$USER_PROMPT" == *"status"* ]] || [[ "$USER_PROMPT" == *"進捗"* ]]; then
    PROMPT_INTENT="status_check"
elif [[ "$USER_PROMPT" == *"実装"* ]] || [[ "$USER_PROMPT" == *"implementation"* ]]; then
    PROMPT_INTENT="implementation"
elif [[ "$USER_PROMPT" == *"品質"* ]] || [[ "$USER_PROMPT" == *"quality"* ]]; then
    PROMPT_INTENT="quality"
elif [[ "$USER_PROMPT" == *"完了"* ]] || [[ "$USER_PROMPT" == *"complete"* ]]; then
    PROMPT_INTENT="completion"
fi

log "🎯 プロンプト意図: $PROMPT_INTENT"

# 5つのドキュメントコンプライアンスチェック実行
DOCUMENTS_COMPLIANCE=$(check_five_documents_compliance "$WORKSPACE_PATH" "$TRACKER_ID" "$CURRENT_PHASE")
log "📋 ドキュメントコンプライアンス完了"

# フェーズ別詳細チェック実行
PHASE_SPECIFIC_CHECK=$(perform_phase_specific_checks "$CURRENT_PHASE" "$TRACKER_ID" "$WORKSPACE_PATH" "$CURRENT_STATUS")
log "🔍 フェーズ別詳細チェック完了"

# レコメンデーション生成（強化版）
RECOMMENDATIONS=$(suggest_next_steps "$CURRENT_PHASE" "$TRACKER_ID" "$WORKSPACE_PATH" "$CURRENT_STATUS")

# フェーズ進捗詳細（統合版）
PHASE_DETAILS=$(cat <<EOF
{
  "tracker_id": "$TRACKER_ID",
  "current_phase": "$CURRENT_PHASE",
  "workspace_path": "$WORKSPACE_PATH",
  "prompt_intent": "$PROMPT_INTENT",
  "documents_compliance": $DOCUMENTS_COMPLIANCE,
  "phase_specific_check": $PHASE_SPECIFIC_CHECK,
  "phase_status": {
    "0.5": $(echo "$CURRENT_STATUS" | jq '.phase_0_5_branch // false'),
    "1": {
      "sam_env": $(echo "$CURRENT_STATUS" | jq '.phase_1_planning.sam_env_check // false'),
      "google_sheets": $(echo "$CURRENT_STATUS" | jq '.phase_1_planning.google_sheets_sync // false'),
      "sow": $(echo "$CURRENT_STATUS" | jq '.phase_1_planning.sow_creation // false')
    },
    "2": {
      "started": $(echo "$CURRENT_STATUS" | jq '.phase_2_implementation.started // false')
    },
    "3": {
      "workflow_executed": $(echo "$CURRENT_STATUS" | jq '.phase_3_quality.workflow_executed // false'),
      "dashboard_created": $(echo "$CURRENT_STATUS" | jq '.phase_3_quality.dashboard_created // false')
    }
  },
  "recommendations": "$RECOMMENDATIONS"
}
EOF
)

# 結果出力
echo "$PHASE_DETAILS" | jq '.'

# 詳細ログ
log "📋 コンテキスト分析完了"
log "   トラッカーID: $TRACKER_ID"
log "   現在フェーズ: $CURRENT_PHASE"
log "   ワークスペース: $WORKSPACE_PATH"
log "   プロンプト意図: $PROMPT_INTENT"

echo '{"status": "completed", "message": "トラッカーコンテキスト分析完了"}'