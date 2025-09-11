#!/bin/bash
# INTG-087: セマンティックワークフロー検証スクリプト
# Claude Hook - PreToolUse で実行されるセマンティック判定システム
# ハードコーディング排除・ドキュメント構造変更対応

set -euo pipefail

# ログ設定
LOG_FILE="/tmp/intg087_semantic_compliance.log"
PYTHON_VALIDATOR_PATH="/mnt/c/AItools/segment-anything/tools/workflow/semantic_workflow_validator.py"

# 動的ログファイル設定関数
setup_log_file() {
    local tracker_id="$1"
    local workspace_path="$2"
    
    if [[ -n "$tracker_id" ]] && [[ -n "$workspace_path" ]] && [[ -d "$workspace_path" ]]; then
        LOG_DIR="$workspace_path/.workflow/logs"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/semantic_compliance.log"
    else
        LOG_FILE="/tmp/intg087_semantic_compliance.log"
    fi
}

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# セマンティック検証Python実行
run_semantic_validation() {
    local command="$1"
    local tracker_id="$2"
    local workspace_path="$3"
    
    # Python環境確認
    if [[ ! -f "$PYTHON_VALIDATOR_PATH" ]]; then
        log "❌ エラー: セマンティック検証システムが見つかりません: $PYTHON_VALIDATOR_PATH"
        return 1
    fi
    
    # セマンティック検証実行
    local validation_result
    validation_result=$(python3 -c "
import sys
import json
sys.path.insert(0, '/mnt/c/AItools/segment-anything')

try:
    from tools.workflow.semantic_workflow_validator import validate_command_execution
    
    command = '''$command'''
    result = validate_command_execution(command)
    
    # 結果をJSONで出力
    output = {
        'result': result.result.value,
        'reason': result.reason,
        'required_actions': result.required_actions,
        'blocking_factors': result.blocking_factors,
        'semantic_analysis': result.semantic_analysis
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))

except Exception as e:
    error_output = {
        'result': 'error',
        'reason': f'セマンティック検証エラー: {str(e)}',
        'required_actions': ['セマンティック検証システムの確認が必要'],
        'blocking_factors': ['validation_system_error']
    }
    print(json.dumps(error_output, ensure_ascii=False, indent=2))
    sys.exit(1)

" 2>&1)
    
    echo "$validation_result"
}

# トラッカーIDの抽出
extract_tracker_id() {
    local input="$1"
    echo "$input" | grep -oP '[A-Z]{3,4}-[0-9]{3}' | head -1 || echo ""
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

# Hook統合設定の確認
check_hook_integration_settings() {
    local config_file="/mnt/c/AItools/segment-anything/config/execution_rules.yaml"
    
    if [[ ! -f "$config_file" ]]; then
        log "⚠️ 警告: execution_rules.yaml が見つかりません。レガシーモードで実行"
        return 1
    fi
    
    # Hook統合が有効かチェック
    local hook_enabled
    hook_enabled=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(config.get('hook_integration', {}).get('enabled', False))
except Exception:
    print('False')
" 2>/dev/null || echo "False")
    
    [[ "$hook_enabled" == "True" ]]
}

# レガシーHook実行（フォールバック）
run_legacy_validation() {
    log "🔄 レガシー検証モードで実行中..."
    
    # 元のvalidate_workflow_compliance.shを呼び出し
    local legacy_script="/mnt/c/AItools/segment-anything/tools/hooks/validate_workflow_compliance.sh"
    
    if [[ -f "$legacy_script" ]]; then
        bash "$legacy_script"
    else
        # 最小限の検証
        echo '{"allow": true, "message": "レガシー検証: 基本チェックのみ実行"}'
    fi
}

# メイン処理
main() {
    INPUT_JSON=$(cat)
    TOOL=$(echo "$INPUT_JSON" | jq -r '.tool // empty')
    TOOL_ARGS=$(echo "$INPUT_JSON" | jq -r '.args // empty')
    
    log "🧠 INTG-087 セマンティックワークフロー検証開始: Tool=$TOOL"
    
    # コマンド抽出
    COMMAND=$(echo "$TOOL_ARGS" | jq -r '.command // empty' 2>/dev/null || echo "")
    FILE_PATH=$(echo "$TOOL_ARGS" | jq -r '.file_path // empty' 2>/dev/null || echo "")
    TRACKER_ID=$(extract_tracker_id "$COMMAND $FILE_PATH")
    
    # Hook統合設定確認
    if ! check_hook_integration_settings; then
        log "📋 セマンティック統合未設定。レガシーモードで実行"
        run_legacy_validation
        return $?
    fi
    
    log "🚀 セマンティック検証モード有効"
    
    # トラッカーID確認
    if [[ -z "$TRACKER_ID" ]]; then
        log "⚠️ トラッカーIDが検出できません。汎用セマンティック検証実行"
        TRACKER_ID=""
        WORKSPACE_PATH=""
    else
        log "📋 トラッカーID検出: $TRACKER_ID"
        
        # ワークスペースパス取得
        WORKSPACE_PATH=$(get_workspace_path "$TRACKER_ID")
        if [[ $? -ne 0 ]] || [[ -z "$WORKSPACE_PATH" ]]; then
            log "❌ 警告: ワークスペースパス取得失敗。汎用モードで続行"
            WORKSPACE_PATH=""
        else
            setup_log_file "$TRACKER_ID" "$WORKSPACE_PATH"
            log "📁 ログファイル更新: $LOG_FILE"
        fi
    fi
    
    # セマンティック検証実行
    if [[ -n "$COMMAND" ]]; then
        log "🔍 セマンティック検証実行: $COMMAND"
        
        VALIDATION_RESULT=$(run_semantic_validation "$COMMAND" "$TRACKER_ID" "$WORKSPACE_PATH")
        if [[ $? -ne 0 ]]; then
            log "❌ セマンティック検証システムエラー。レガシーモードにフォールバック"
            run_legacy_validation
            return $?
        fi
        
        # 結果解析
        RESULT_TYPE=$(echo "$VALIDATION_RESULT" | jq -r '.result')
        REASON=$(echo "$VALIDATION_RESULT" | jq -r '.reason')
        REQUIRED_ACTIONS=$(echo "$VALIDATION_RESULT" | jq -r '.required_actions[]?' 2>/dev/null || echo "")
        BLOCKING_FACTORS=$(echo "$VALIDATION_RESULT" | jq -r '.blocking_factors[]?' 2>/dev/null || echo "")
        
        log "📊 セマンティック検証結果: $RESULT_TYPE"
        log "💭 判定理由: $REASON"
        
        # 判定結果に基づく処理
        case "$RESULT_TYPE" in
            "allowed")
                log "✅ セマンティック検証成功: コマンド実行許可"
                cat <<EOF
{
  "allow": true,
  "message": "セマンティック検証成功",
  "semantic_result": "$RESULT_TYPE",
  "reason": "$REASON",
  "tracker_id": "$TRACKER_ID",
  "workspace": "$WORKSPACE_PATH",
  "validation_type": "semantic"
}
EOF
                ;;
                
            "blocked")
                log "🚫 セマンティック検証失敗: コマンド実行阻止"
                if [[ -n "$BLOCKING_FACTORS" ]]; then
                    log "🚧 阻止要因: $BLOCKING_FACTORS"
                fi
                if [[ -n "$REQUIRED_ACTIONS" ]]; then
                    log "📝 必要アクション: $REQUIRED_ACTIONS"
                fi
                
                cat <<EOF
{
  "allow": false,
  "reason": "$REASON",
  "semantic_result": "$RESULT_TYPE",
  "blocking_factors": $(echo "$VALIDATION_RESULT" | jq '.blocking_factors'),
  "required_actions": $(echo "$VALIDATION_RESULT" | jq '.required_actions'),
  "severity": "HIGH",
  "user_confirmation_required": true,
  "continue_anyway": false,
  "validation_type": "semantic"
}
EOF
                ;;
                
            "requires_approval")
                log "⏳ セマンティック検証: 承認待ち状態"
                log "📝 必要承認: $REQUIRED_ACTIONS"
                
                cat <<EOF
{
  "allow": false,
  "reason": "$REASON",
  "semantic_result": "$RESULT_TYPE",
  "required_actions": $(echo "$VALIDATION_RESULT" | jq '.required_actions'),
  "severity": "MEDIUM",
  "user_confirmation_required": true,
  "continue_anyway": false,
  "validation_type": "semantic",
  "approval_guidance": [
    "必要な承認を取得してください",
    "承認取得後に再実行してください"
  ]
}
EOF
                ;;
                
            "phase_mismatch")
                log "🔄 セマンティック検証: フェーズ不適合"
                log "⚠️ フェーズ要件: $REQUIRED_ACTIONS"
                
                cat <<EOF
{
  "allow": false,
  "reason": "$REASON",
  "semantic_result": "$RESULT_TYPE",
  "required_actions": $(echo "$VALIDATION_RESULT" | jq '.required_actions'),
  "severity": "MEDIUM",
  "user_confirmation_required": true,
  "continue_anyway": false,
  "validation_type": "semantic",
  "phase_guidance": [
    "必要なフェーズを完了してください",
    "ワークフロー順序を確認してください"
  ]
}
EOF
                ;;
                
            "error")
                log "❌ セマンティック検証システムエラー。レガシーモードにフォールバック"
                run_legacy_validation
                ;;
                
            *)
                log "⚠️ 未知のセマンティック検証結果: $RESULT_TYPE"
                log "🔄 レガシーモードにフォールバック"
                run_legacy_validation
                ;;
        esac
        
    else
        log "⚠️ 検証対象コマンドが空です。基本検証で許可"
        echo '{"allow": true, "message": "検証対象コマンドなし", "validation_type": "basic"}'
    fi
}

# スクリプト実行
main "$@"