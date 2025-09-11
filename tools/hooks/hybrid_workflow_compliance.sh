#!/bin/bash
# INTG-087: ハイブリッドワークフロー検証システム
# 厳格な検証（validate_workflow）+ セマンティック補完（semantic_workflow）
# 品質保証を維持しながら柔軟性を追加

set -euo pipefail

# ログ設定
LOG_FILE="/tmp/hybrid_workflow_compliance.log"
SCRIPT_DIR="$(dirname "$0")"

# 動的ログファイル設定関数
setup_log_file() {
    local tracker_id="$1"
    local workspace_path="$2"
    
    if [[ -n "$tracker_id" ]] && [[ -n "$workspace_path" ]] && [[ -d "$workspace_path" ]]; then
        LOG_DIR="$workspace_path/.workflow/logs"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/hybrid_compliance.log"
    else
        LOG_FILE="/tmp/hybrid_workflow_compliance.log"
    fi
}

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 検証モード判定（環境変数による切り替え）
VALIDATION_MODE="${VALIDATION_MODE:-hybrid}"
log "🔄 ハイブリッド検証システム起動: Mode=$VALIDATION_MODE"

# 入力JSONを保存（両スクリプトで使用）
INPUT_JSON=$(cat)
TEMP_INPUT_FILE="/tmp/hybrid_input_$$.json"
echo "$INPUT_JSON" > "$TEMP_INPUT_FILE"

# クリーンアップ関数
cleanup() {
    rm -f "$TEMP_INPUT_FILE"
}
trap cleanup EXIT

# 検証結果格納変数
STRICT_RESULT=""
SEMANTIC_RESULT=""
FINAL_ALLOW=true
FINAL_MESSAGES=()
VALIDATION_DETAILS=()

# モード別処理
case "$VALIDATION_MODE" in
    "strict")
        # 厳格検証のみ
        log "📏 厳格検証モード（validate_workflow_compliance.sh のみ）"
        STRICT_RESULT=$(cat "$TEMP_INPUT_FILE" | bash "$SCRIPT_DIR/validate_workflow_compliance.sh" 2>&1)
        STRICT_EXIT_CODE=$?
        
        if [[ $STRICT_EXIT_CODE -ne 0 ]]; then
            log "❌ 厳格検証失敗"
            echo "$STRICT_RESULT"
            exit 1
        fi
        
        log "✅ 厳格検証成功"
        echo "$STRICT_RESULT"
        exit 0
        ;;
        
    "semantic")
        # セマンティック検証のみ
        log "🧠 セマンティック検証モード（semantic_workflow_compliance.sh のみ）"
        SEMANTIC_RESULT=$(cat "$TEMP_INPUT_FILE" | bash "$SCRIPT_DIR/semantic_workflow_compliance.sh" 2>&1)
        SEMANTIC_EXIT_CODE=$?
        
        if [[ $SEMANTIC_EXIT_CODE -ne 0 ]]; then
            log "❌ セマンティック検証失敗"
            echo "$SEMANTIC_RESULT"
            exit 1
        fi
        
        log "✅ セマンティック検証成功"
        echo "$SEMANTIC_RESULT"
        exit 0
        ;;
        
    "hybrid"|*)
        # ハイブリッドモード（デフォルト）
        log "🔀 ハイブリッド検証モード（厳格 + セマンティック補完）"
        
        # ========================================
        # 第1段階: 厳格な検証（必須・ブロッキング）
        # ========================================
        log "📏 第1段階: 厳格検証実行中..."
        STRICT_RESULT=$(cat "$TEMP_INPUT_FILE" | bash "$SCRIPT_DIR/validate_workflow_compliance.sh" 2>&1)
        STRICT_EXIT_CODE=$?
        
        if [[ $STRICT_EXIT_CODE -ne 0 ]]; then
            log "❌ 厳格検証失敗 - 実行をブロック"
            
            # 厳格検証の失敗は即座にブロック
            # セマンティック検証は実行しない（品質優先）
            echo "$STRICT_RESULT"
            exit 1
        fi
        
        log "✅ 第1段階: 厳格検証成功"
        VALIDATION_DETAILS+=("strict_validation: passed")
        
        # ========================================
        # 第2段階: セマンティック補完検証（警告のみ）
        # ========================================
        log "🧠 第2段階: セマンティック補完検証実行中..."
        
        # セマンティック検証実行（エラーでも処理継続）
        SEMANTIC_RESULT=$(cat "$TEMP_INPUT_FILE" | bash "$SCRIPT_DIR/semantic_workflow_compliance.sh" 2>&1 || true)
        SEMANTIC_EXIT_CODE=$?
        
        if [[ $SEMANTIC_EXIT_CODE -ne 0 ]]; then
            log "⚠️ セマンティック検証で警告検出（ブロックしない）"
            VALIDATION_DETAILS+=("semantic_validation: warning")
            
            # セマンティック検証の警告内容を解析
            if echo "$SEMANTIC_RESULT" | grep -q "BLOCKED"; then
                FINAL_MESSAGES+=("⚠️ セマンティック分析: 潜在的な問題を検出しました")
                FINAL_MESSAGES+=("  推奨事項: $(echo "$SEMANTIC_RESULT" | grep -oP '"reason":\s*"[^"]*"' | cut -d'"' -f4)")
            fi
        else
            log "✅ 第2段階: セマンティック検証も成功"
            VALIDATION_DETAILS+=("semantic_validation: passed")
            
            # セマンティック検証の追加情報があれば取得
            if echo "$SEMANTIC_RESULT" | grep -q "semantic_analysis"; then
                FINAL_MESSAGES+=("💡 セマンティック分析: 追加の洞察を提供")
            fi
        fi
        
        # ========================================
        # 最終結果の生成
        # ========================================
        log "📊 ハイブリッド検証完了"
        
        # JSON形式で結果を返却
        cat <<EOF
{
  "allow": true,
  "mode": "hybrid",
  "validation_results": {
    "strict": "passed",
    "semantic": "$([ $SEMANTIC_EXIT_CODE -eq 0 ] && echo "passed" || echo "warning")"
  },
  "message": "ハイブリッド検証成功（厳格検証合格）",
  "details": [
    $(printf '"%s"' "${VALIDATION_DETAILS[@]}" | paste -sd,)
  ],
  "recommendations": [
    $(printf '"%s"' "${FINAL_MESSAGES[@]}" | paste -sd,)
  ]
}
EOF
        exit 0
        ;;
esac