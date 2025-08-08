#!/bin/bash
# =================================================================
# スマートタスク振り分けシステム
# =================================================================
# 目的: タスクを自動分析し、適切なAI（Gemini/Claude）に振り分け
# 効果: 最適なAI選択で品質とコストを両立
#
# 使用方法:
#   ./smart_task_dispatcher.sh "task_description"
#
# 例:
#   ./smart_task_dispatcher.sh "Create a simple README file"
#   ./smart_task_dispatcher.sh "Implement complex algorithm"
# =================================================================

set -euo pipefail

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="./task_dispatch_results"
GEMINI_API_KEY=""

# ログ関数
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
    echo -e "${BLUE}[$(timestamp)]${NC} ℹ️  $1"
}

log_success() {
    echo -e "${GREEN}[$(timestamp)]${NC} ✅ $1"
}

log_warning() {
    echo -e "${YELLOW}[$(timestamp)]${NC} ⚠️  $1"
}

log_error() {
    echo -e "${RED}[$(timestamp)]${NC} ❌ $1"
}

log_dispatch() {
    echo -e "${PURPLE}[$(timestamp)]${NC} 🎯 $1"
}

# 使用方法表示
usage() {
    cat << EOF
使用方法: $0 "TASK_DESCRIPTION" [OPTIONS]

必須引数:
    TASK_DESCRIPTION    実行したいタスクの説明

オプション:
    --explain          振り分け理由を詳細表示
    --force-ai AI      強制的に指定AIを使用 (gemini, claude)
    --output-dir DIR   出力ディレクトリ
    --help            このヘルプを表示

タスク振り分け基準:
    📝 Gemini適用: 文書作成、簡単なコード、設定ファイル
    🧠 Claude適用: 複雑な実装、リファクタリング、設計

例:
    $0 "Create project documentation"     # → Gemini
    $0 "Implement advanced algorithm"     # → Claude
    $0 "Generate simple test function"    # → Gemini
EOF
    exit 0
}

# タスク分類システム
classify_task() {
    local task="$1"
    local explain_mode="${2:-false}"
    
    # Gemini適用タスクパターン
    local gemini_patterns=(
        # 文書系
        "create.*documentation|write.*readme|generate.*doc"
        "create.*comment|add.*comment|write.*comment"
        "generate.*example|create.*example|write.*example"
        "create.*template|generate.*template"
        
        # 簡単なコード系
        "simple.*function|basic.*function|create.*function"
        "simple.*script|basic.*script|write.*script"
        "create.*test.*simple|simple.*test"
        "generate.*config|create.*config|write.*config"
        
        # データ処理系
        "parse.*json|read.*json|write.*json"
        "parse.*csv|read.*csv|write.*csv"
        "convert.*format|transform.*data"
        
        # 日本語パターン
        "ドキュメント.*作成|文書.*作成|説明.*作成"
        "README.*作成|コメント.*作成|例.*作成"
        "簡単.*関数|基本.*関数|単純.*関数"
        "設定.*作成|設定.*ファイル|テンプレート.*作成"
    )
    
    # Claude必須タスクパターン
    local claude_patterns=(
        # 複雑な実装系
        "complex.*implement|advanced.*implement|sophisticated.*implement"
        "refactor|refactoring|restructure|architecture"
        "algorithm.*implement|optimize.*algorithm|performance.*improve"
        "design.*pattern|implement.*pattern|architectural.*design"
        
        # 統合・連携系
        "integrate.*system|connect.*api|implement.*api"
        "database.*design|schema.*design|migration"
        "security.*implement|authentication|authorization"
        
        # 分析・デバッグ系
        "debug.*complex|analyze.*performance|troubleshoot"
        "code.*review|quality.*analysis|security.*analysis"
        
        # 日本語パターン
        "複雑.*実装|高度.*実装|アーキテクチャ.*設計"
        "リファクタリング|最適化|パフォーマンス.*改善"
        "統合.*システム|API.*実装|データベース.*設計"
        "セキュリティ.*実装|認証.*実装|分析"
    )
    
    local score_gemini=0
    local score_claude=0
    local matched_patterns=()
    
    # Geminiパターンマッチング
    for pattern in "${gemini_patterns[@]}"; do
        if echo "$task" | grep -Eqi "$pattern"; then
            score_gemini=$((score_gemini + 1))
            if [[ "$explain_mode" == true ]]; then
                matched_patterns+=("Gemini: $pattern")
            fi
        fi
    done
    
    # Claudeパターンマッチング
    for pattern in "${claude_patterns[@]}"; do
        if echo "$task" | grep -Eqi "$pattern"; then
            score_claude=$((score_claude + 2))  # Claudeパターンは重み2倍
            if [[ "$explain_mode" == true ]]; then
                matched_patterns+=("Claude: $pattern")
            fi
        fi
    done
    
    # 長さベースの調整
    local task_length=${#task}
    if [[ $task_length -gt 100 ]]; then
        score_claude=$((score_claude + 1))
    fi
    
    # 判定結果
    local decision
    local confidence
    
    if [[ $score_claude -gt $score_gemini ]]; then
        decision="claude"
        confidence=$(echo "scale=2; $score_claude / ($score_claude + $score_gemini + 1)" | bc -l)
    elif [[ $score_gemini -gt 0 ]]; then
        decision="gemini"
        confidence=$(echo "scale=2; $score_gemini / ($score_claude + $score_gemini + 1)" | bc -l)
    else
        # デフォルト: 短いタスクはGemini、長いタスクはClaude
        if [[ $task_length -lt 50 ]]; then
            decision="gemini"
            confidence="0.50"
        else
            decision="claude"
            confidence="0.60"
        fi
    fi
    
    # 説明モードの場合
    if [[ "$explain_mode" == true ]]; then
        echo "=== タスク分類結果 ==="
        echo "タスク: $task"
        echo "文字数: $task_length"
        echo "Geminiスコア: $score_gemini"
        echo "Claudeスコア: $score_claude"
        echo "決定: $decision (信頼度: $confidence)"
        echo ""
        echo "マッチしたパターン:"
        for pattern in "${matched_patterns[@]}"; do
            echo "  - $pattern"
        done
        echo ""
    fi
    
    echo "$decision:$confidence"
}

# Geminiタスク実行
execute_gemini_task() {
    local task="$1"
    local output_file="$2"
    
    log_dispatch "Geminiでタスクを実行中..."
    
    # 設定ファイルからAPIキー取得
    local gemini_key=$(python3 -c "
import sys
sys.path.append('/mnt/c/AItools/segment-anything')
try:
    from features.common.api_config import get_api_config
    config = get_api_config()
    key = config.get_gemini_api_key()
    print(key or '')
except:
    print('')
" 2>/dev/null)
    
    export GEMINI_API_KEY="$gemini_key"
    
    if "$SCRIPT_DIR/gemini_helper.sh" "$task" --output-dir "$OUTPUT_DIR" > "$output_file.path" 2>&1; then
        local result_path=$(cat "$output_file.path")
        
        # 結果を統一形式でコピー
        if [[ -f "$result_path" ]]; then
            cp "$result_path" "$output_file"
            rm -f "$output_file.path"
            
            log_success "Geminiタスク実行完了"
            return 0
        else
            log_error "Gemini結果ファイルが見つかりません: $result_path"
            return 1
        fi
    else
        log_error "Geminiタスク実行に失敗しました"
        return 1
    fi
}

# Claudeタスク実行（指示生成）
execute_claude_task() {
    local task="$1"
    local output_file="$2"
    
    log_dispatch "Claude Code用の指示を生成中..."
    
    cat > "$output_file" << EOF
# Claude Code 実行指示

## タスク概要
$task

## 実行理由
このタスクは以下の理由でClaude Codeでの実行が適切と判定されました：
- 複雑性や重要性が高い
- 高品質な結果が要求される
- Geminiでは十分な品質が期待できない

## 実行要件
1. **品質重視**: 実装の正確性と保守性を最優先
2. **エラーハンドリング**: 適切な例外処理を含める
3. **ドキュメント**: 必要に応じてコメント・説明を追加
4. **テスト**: 可能な場合はテストケースも含める

## 次のステップ
1. この指示をClaude Codeに提示
2. 結果を確認・検証
3. 必要に応じて品質改善

---
*自動生成日時: $(timestamp)*
*生成システム: Smart Task Dispatcher*
EOF

    log_success "Claude Code指示生成完了"
    log_info "Claude Codeでの手動実行が必要です"
    
    return 0
}

# メイン処理
main() {
    local task_description=""
    local explain_mode=false
    local force_ai=""
    
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --explain)
                explain_mode=true
                shift
                ;;
            --force-ai)
                force_ai="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help)
                usage
                ;;
            *)
                if [[ -z "$task_description" ]]; then
                    task_description="$1"
                else
                    log_error "複数のタスク説明が指定されました: $1"
                    usage
                fi
                shift
                ;;
        esac
    done
    
    # タスク説明必須チェック
    if [[ -z "$task_description" ]]; then
        log_error "タスク説明が指定されていません"
        usage
    fi
    
    # 出力ディレクトリ作成
    mkdir -p "$OUTPUT_DIR"
    
    log_info "スマートタスク振り分け開始"
    log_info "タスク: $task_description"
    
    # 強制AI指定の場合
    if [[ -n "$force_ai" ]]; then
        log_dispatch "強制AI指定: $force_ai"
        local decision="$force_ai"
        local confidence="1.00"
    else
        # タスク分類実行
        local classification_result=$(classify_task "$task_description" "$explain_mode")
        local decision=$(echo "$classification_result" | cut -d: -f1)
        local confidence=$(echo "$classification_result" | cut -d: -f2)
    fi
    
    # 出力ファイル設定
    local timestamp_str=$(date '+%Y%m%d_%H%M%S')
    local output_file="$OUTPUT_DIR/result_${decision}_${timestamp_str}.txt"
    
    log_dispatch "振り分け結果: $decision (信頼度: $confidence)"
    
    # AI実行
    case "$decision" in
        gemini)
            if execute_gemini_task "$task_description" "$output_file"; then
                log_success "タスク完了: $output_file"
                echo "✅ Geminiで処理完了: $output_file"
            else
                log_warning "Gemini実行失败，Claude Codeにフォールバック"
                execute_claude_task "$task_description" "$output_file"
                echo "⚠️ Claude Code実行が必要: $output_file"
            fi
            ;;
        claude)
            execute_claude_task "$task_description" "$output_file"
            echo "🧠 Claude Code実行が必要: $output_file"
            ;;
        *)
            log_error "無効な振り分け結果: $decision"
            exit 1
            ;;
    esac
    
    # 統計更新（オプション）
    if [[ -f "$OUTPUT_DIR/dispatch_stats.log" ]]; then
        echo "$(timestamp),$decision,$confidence,$task_description" >> "$OUTPUT_DIR/dispatch_stats.log"
    else
        echo "timestamp,ai,confidence,task" > "$OUTPUT_DIR/dispatch_stats.log"
        echo "$(timestamp),$decision,$confidence,$task_description" >> "$OUTPUT_DIR/dispatch_stats.log"
    fi
}

# エラーハンドリング
trap 'log_error "エラーが発生しました（行 $LINENO）"; exit 1' ERR

# メイン処理実行
main "$@"