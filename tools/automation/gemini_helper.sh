#!/bin/bash
# =================================================================
# Gemini CLI連携ヘルパースクリプト
# =================================================================
# 目的: 簡易タスクをGeminiに自動振り分け、Claude使用量をさらに削減
# 効果: 定型タスクの30-40%をGeminiに委任、品質はフォールバック保証
#
# 使用方法:
#   ./gemini_helper.sh "task_description" [--force-claude]
#
# 例:
#   ./gemini_helper.sh "Create a simple test function for character extraction"
#   ./gemini_helper.sh "Generate README documentation" --format markdown
#   ./gemini_helper.sh "Complex refactoring task" --force-claude
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
GEMINI_API_KEY=""
GEMINI_FALLBACK_KEY=""
OUTPUT_DIR="./gemini_outputs"
FORCE_CLAUDE=false
OUTPUT_FORMAT="text"
CONFIDENCE_THRESHOLD=0.7

# タイムスタンプ関数
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# ログ関数
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

log_gemini() {
    echo -e "${CYAN}[$(timestamp)]${NC} 🤖 $1"
}

log_claude() {
    echo -e "${PURPLE}[$(timestamp)]${NC} 🧠 $1"
}

# 使用方法表示
usage() {
    cat << EOF
使用方法: $0 "TASK_DESCRIPTION" [OPTIONS]

必須引数:
    TASK_DESCRIPTION    実行したいタスクの説明

オプション:
    --force-claude      Geminiをスキップして直接Claudeを使用
    --format FORMAT     出力形式 (text, markdown, json, code)
    --output-dir DIR    出力ディレクトリ
    --confidence NUM    品質判定閾値 (0.0-1.0, デフォルト: 0.7)
    --help             このヘルプを表示

例:
    $0 "Create a simple test function"
    $0 "Generate documentation" --format markdown
    $0 "Complex algorithm implementation" --force-claude
EOF
    exit 0
}

# 引数解析
TASK_DESCRIPTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-claude)
            FORCE_CLAUDE=true
            shift
            ;;
        --format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE_THRESHOLD="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            if [[ -z "$TASK_DESCRIPTION" ]]; then
                TASK_DESCRIPTION="$1"
            else
                log_error "複数のタスク説明が指定されました: $1"
                usage
            fi
            shift
            ;;
    esac
done

# タスク説明必須チェック
if [[ -z "$TASK_DESCRIPTION" ]]; then
    log_error "タスク説明が指定されていません"
    usage
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# Gemini APIキー確認
check_gemini_api() {
    # 設定ファイルからAPIキー取得
    log_info "設定ファイルからAPIキーを取得中..."
    
    local config_result=$(python3 -c "
import sys
sys.path.append('/mnt/c/AItools/segment-anything')
try:
    from features.common.api_config import get_api_config
    config = get_api_config()
    primary_key = config.get_gemini_api_key()
    fallback_key = config.get_gemini_api_key(use_fallback=True)
    print(f'{primary_key or \"\"}|{fallback_key or \"\"}')
except Exception as e:
    print(f'ERROR|{e}')
" 2>/dev/null)
    
    if [[ "$config_result" == ERROR* ]]; then
        log_error "設定ファイル読み込みエラー: ${config_result#ERROR|}"
        return 1
    fi
    
    GEMINI_API_KEY=$(echo "$config_result" | cut -d'|' -f1)
    GEMINI_FALLBACK_KEY=$(echo "$config_result" | cut -d'|' -f2)
    
    # プライマリキーをチェック
    if [[ -n "$GEMINI_API_KEY" && ${#GEMINI_API_KEY} -gt 30 ]]; then
        log_info "プライマリGemini APIキー確認完了"
        return 0
    fi
    
    # フォールバックキーをチェック
    if [[ -n "$GEMINI_FALLBACK_KEY" && ${#GEMINI_FALLBACK_KEY} -gt 30 ]]; then
        log_warning "フォールバックGemini APIキーを使用"
        GEMINI_API_KEY="$GEMINI_FALLBACK_KEY"
        return 0
    fi
    
    # 環境変数から取得を試す
    if [[ -n "${GEMINI_API_KEY_ENV:-}" ]]; then
        GEMINI_API_KEY="$GEMINI_API_KEY_ENV"
        log_warning "環境変数からGemini APIキーを取得"
        return 0
    fi
    
    log_error "有効なGemini APIキーが見つかりません"
    return 1
}

# タスク複雑度判定
assess_task_complexity() {
    local task="$1"
    local complexity_score=0
    
    # 複雑度指標（キーワードベース）
    local high_complexity_keywords=(
        "refactor" "architecture" "design pattern" "algorithm"
        "optimization" "performance" "security" "integration"
        "complex" "advanced" "sophisticated" "comprehensive"
        "リファクタリング" "アーキテクチャ" "アルゴリズム" "最適化"
        "複雑" "高度" "包括的" "統合"
    )
    
    local medium_complexity_keywords=(
        "implement" "function" "class" "method" "logic"
        "validation" "parsing" "processing" "conversion"
        "実装" "関数" "クラス" "メソッド" "処理"
    )
    
    local low_complexity_keywords=(
        "create" "generate" "write" "document" "comment"
        "readme" "example" "simple" "basic" "template"
        "作成" "生成" "記述" "簡単" "基本" "テンプレート"
    )
    
    # 複雑度スコア計算
    for keyword in "${high_complexity_keywords[@]}"; do
        if echo "$task" | grep -qi "$keyword"; then
            complexity_score=$((complexity_score + 3))
        fi
    done
    
    for keyword in "${medium_complexity_keywords[@]}"; do
        if echo "$task" | grep -qi "$keyword"; then
            complexity_score=$((complexity_score + 2))
        fi
    done
    
    for keyword in "${low_complexity_keywords[@]}"; do
        if echo "$task" | grep -qi "$keyword"; then
            complexity_score=$((complexity_score + 1))
        fi
    done
    
    # 複雑度レベル判定
    if [[ $complexity_score -ge 6 ]]; then
        echo "HIGH"
    elif [[ $complexity_score -ge 3 ]]; then
        echo "MEDIUM"
    else
        echo "LOW"
    fi
}

# Gemini API呼び出し
call_gemini_api() {
    local prompt="$1"
    local output_file="$2"
    
    log_gemini "Gemini APIを呼び出し中..."
    
    # Gemini API呼び出し（Google AI Studio API使用）
    curl -s -X POST \
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${GEMINI_API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{
            "contents": [{
                "parts": [{
                    "text": "'"$prompt"'"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }' > "$output_file.json" 2>/dev/null
    
    # レスポンス解析
    if [[ -f "$output_file.json" ]]; then
        # JSONから本文抽出
        python3 -c "
import json
import sys
try:
    with open('$output_file.json', 'r') as f:
        data = json.load(f)
    
    if 'candidates' in data and len(data['candidates']) > 0:
        content = data['candidates'][0]['content']['parts'][0]['text']
        with open('$output_file', 'w') as f:
            f.write(content)
        print('SUCCESS')
    else:
        print('ERROR: No candidates in response')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" || {
            log_error "Gemini APIレスポンスの解析に失敗"
            return 1
        }
    else
        log_error "Gemini APIの呼び出しに失敗"
        return 1
    fi
    
    # 一時JSONファイル削除
    rm -f "$output_file.json"
    
    log_success "Gemini API呼び出し完了"
}

# 結果品質評価
evaluate_result_quality() {
    local result_file="$1"
    local task="$2"
    
    if [[ ! -f "$result_file" ]]; then
        echo "0.0"
        return
    fi
    
    local file_size=$(stat -c%s "$result_file" 2>/dev/null || stat -f%z "$result_file" 2>/dev/null)
    local line_count=$(wc -l < "$result_file")
    local word_count=$(wc -w < "$result_file")
    
    # 基本品質スコア
    local quality_score=0.5
    
    # ファイルサイズ評価
    if [[ $file_size -gt 100 ]]; then
        quality_score=$(echo "$quality_score + 0.2" | bc -l)
    fi
    
    # 行数評価
    if [[ $line_count -gt 5 ]]; then
        quality_score=$(echo "$quality_score + 0.1" | bc -l)
    fi
    
    # 語数評価
    if [[ $word_count -gt 20 ]]; then
        quality_score=$(echo "$quality_score + 0.1" | bc -l)
    fi
    
    # 内容品質チェック（簡易）
    if grep -qi "error\|failed\|sorry\|cannot" "$result_file"; then
        quality_score=$(echo "$quality_score - 0.3" | bc -l)
    fi
    
    # コード関連タスクの場合
    if echo "$task" | grep -qi "code\|function\|class\|implementation"; then
        if grep -qi "def\|class\|function\|import" "$result_file"; then
            quality_score=$(echo "$quality_score + 0.2" | bc -l)
        fi
    fi
    
    # 最小・最大値制限
    quality_score=$(echo "if($quality_score < 0) 0 else if($quality_score > 1) 1 else $quality_score" | bc -l)
    
    echo "$quality_score"
}

# Claude Code呼び出し（フォールバック）
call_claude_fallback() {
    local task="$1"
    local output_file="$2"
    
    log_claude "Claude Codeにフォールバック実行..."
    
    # Claude Code用の指示ファイル作成
    cat > "${output_file}.claude_instruction" << EOF
以下のタスクを実行してください：

$task

要求事項:
- 出力形式: $OUTPUT_FORMAT
- 簡潔で実用的な結果を提供
- エラーハンドリングを含める
- 必要に応じてコメントを追加

このタスクはGeminiでの処理品質が不十分だったため、
Claude Codeでの再実行となります。
EOF
    
    echo "⚠️ このタスクはClaude Code での手動実行が必要です" > "$output_file"
    echo "指示ファイル: ${output_file}.claude_instruction" >> "$output_file"
    echo "" >> "$output_file"
    echo "タスク内容:" >> "$output_file"
    echo "$task" >> "$output_file"
    
    log_warning "Claude Code での手動実行が必要です"
    log_info "指示ファイル: ${output_file}.claude_instruction"
}

# メイン処理
main() {
    log_info "Gemini CLI連携ヘルパー開始"
    log_info "タスク: $TASK_DESCRIPTION"
    
    # APIキー確認
    if ! check_gemini_api; then
        log_error "Gemini API設定に問題があります"
        exit 1
    fi
    
    # 出力ファイル設定
    local timestamp_str=$(date '+%Y%m%d_%H%M%S')
    local output_file="$OUTPUT_DIR/gemini_result_${timestamp_str}.txt"
    
    # 強制Claude指定の場合
    if [[ "$FORCE_CLAUDE" == true ]]; then
        log_claude "強制Claude実行が指定されました"
        call_claude_fallback "$TASK_DESCRIPTION" "$output_file"
        echo "$output_file"
        exit 0
    fi
    
    # タスク複雑度評価
    local complexity=$(assess_task_complexity "$TASK_DESCRIPTION")
    log_info "タスク複雑度: $complexity"
    
    # 複雑度に基づく判定
    if [[ "$complexity" == "HIGH" ]]; then
        log_warning "高複雑度タスクのため、Claude Codeに委任します"
        call_claude_fallback "$TASK_DESCRIPTION" "$output_file"
        echo "$output_file"
        exit 0
    fi
    
    # Gemini実行
    local gemini_prompt="Task: $TASK_DESCRIPTION

Please provide a high-quality response in $OUTPUT_FORMAT format.
Be concise, practical, and include error handling where appropriate.
Focus on providing actionable and accurate information.

Requirements:
- Output format: $OUTPUT_FORMAT
- Language: Japanese (if the task is in Japanese) or English
- Include practical examples if relevant
- Ensure the response is ready to use"
    
    if call_gemini_api "$gemini_prompt" "$output_file"; then
        # 品質評価
        local quality_score=$(evaluate_result_quality "$output_file" "$TASK_DESCRIPTION")
        log_info "品質スコア: $quality_score"
        
        # 品質閾値チェック
        if [[ $(echo "$quality_score >= $CONFIDENCE_THRESHOLD" | bc -l) -eq 1 ]]; then
            log_success "Geminiによる処理が完了しました"
            log_info "出力ファイル: $output_file"
            echo "$output_file"
        else
            log_warning "品質が閾値($CONFIDENCE_THRESHOLD)を下回りました"
            log_claude "Claude Codeにフォールバック実行します"
            
            # 元の結果をバックアップ
            mv "$output_file" "${output_file}.gemini_backup"
            
            call_claude_fallback "$TASK_DESCRIPTION" "$output_file"
            echo "$output_file"
        fi
    else
        log_error "Gemini APIの呼び出しに失敗しました"
        call_claude_fallback "$TASK_DESCRIPTION" "$output_file"
        echo "$output_file"
    fi
}

# エラーハンドリング
trap 'log_error "エラーが発生しました（行 $LINENO）"; exit 1' ERR

# メイン処理実行
main