#!/bin/bash
"""
ワークフローテスト実行スクリプト

統合ワークフローテストシステム（Level 1-4）の実行用シェルスクリプト
"""

set -euo pipefail

# スクリプトディレクトリ取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# カラー出力設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 設定デフォルト値
LEVEL="all"
VERBOSE=true
STOP_ON_FAILURE=false
OUTPUT_FILE=""
PYTHON_CMD="python3"

# ヘルプ表示
show_help() {
    echo -e "${WHITE}ワークフローテスト実行スクリプト${NC}"
    echo
    echo -e "${CYAN}使用方法:${NC}"
    echo "  $0 [OPTIONS]"
    echo
    echo -e "${CYAN}オプション:${NC}"
    echo -e "  ${YELLOW}--level LEVEL${NC}     テストレベル指定 (level_1|level_2|level_3|level_4|all) [デフォルト: all]"
    echo -e "  ${YELLOW}--quiet${NC}           簡潔な出力モード"
    echo -e "  ${YELLOW}--stop-on-failure${NC} 失敗時に停止"
    echo -e "  ${YELLOW}--output FILE${NC}     結果をファイルに出力"
    echo -e "  ${YELLOW}--python PYTHON${NC}   Pythonコマンド指定 [デフォルト: python3]"
    echo -e "  ${YELLOW}--help${NC}            このヘルプを表示"
    echo
    echo -e "${CYAN}例:${NC}"
    echo "  $0                           # 全レベル実行"
    echo "  $0 --level level_1           # Level 1のみ実行"
    echo "  $0 --quiet --stop-on-failure # 簡潔出力、失敗時停止"
    echo "  $0 --output results.json     # 結果をJSONファイルに保存"
    echo
    echo -e "${CYAN}テストレベル:${NC}"
    echo -e "  ${GREEN}level_1${NC}  基本ワークフローテスト（入力検証、トラッカーID検証）"
    echo -e "  ${GREEN}level_2${NC}  品質ワークフローテスト（SAM+YOLO抽出、品質評価）"
    echo -e "  ${GREEN}level_3${NC}  統計分析ワークフローテスト（Cohen's d、Google Sheets）"
    echo -e "  ${GREEN}level_4${NC}  承認ワークフローテスト（Pushover通知、承認プロセス）"
    echo -e "  ${GREEN}all${NC}      全テストレベルを順次実行"
}

# 引数解析
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --level)
                LEVEL="$2"
                shift 2
                ;;
            --quiet)
                VERBOSE=false
                shift
                ;;
            --stop-on-failure)
                STOP_ON_FAILURE=true
                shift
                ;;
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --python)
                PYTHON_CMD="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 不明なオプション: $1${NC}"
                echo "ヘルプを表示するには --help を使用してください。"
                exit 1
                ;;
        esac
    done
}

# 環境チェック
check_environment() {
    echo -e "${BLUE}🔍 環境チェック中...${NC}"
    
    # プロジェクトルート確認
    if [[ ! -f "${PROJECT_ROOT}/CLAUDE.md" ]]; then
        echo -e "${RED}❌ プロジェクトルートが正しくありません: ${PROJECT_ROOT}${NC}"
        exit 1
    fi
    
    # Pythonコマンド確認
    if ! command -v "${PYTHON_CMD}" &> /dev/null; then
        echo -e "${RED}❌ Pythonコマンドが見つかりません: ${PYTHON_CMD}${NC}"
        echo "正しいPythonコマンドを --python オプションで指定してください。"
        exit 1
    fi
    
    # Python version確認
    PYTHON_VERSION=$(${PYTHON_CMD} --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python: ${PYTHON_VERSION}${NC}"
    
    # 必要なPythonパッケージ確認
    local required_packages=("pytest" "pytest-mock")
    for package in "${required_packages[@]}"; do
        if ! ${PYTHON_CMD} -c "import ${package}" &> /dev/null; then
            echo -e "${YELLOW}⚠️  パッケージ不足: ${package}${NC}"
            echo "インストール: pip install ${package}"
        else
            echo -e "${GREEN}✅ ${package}${NC}"
        fi
    done
    
    # テストディレクトリ構造確認
    local test_files=(
        "tests/test_runner.py"
        "tests/workflow/test_basic_workflow.py"
        "tests/workflow/test_quality_workflow.py"
        "tests/workflow/test_statistical_workflow.py"
        "tests/workflow/test_approval_workflow.py"
    )
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${test_file}" ]]; then
            echo -e "${GREEN}✅ ${test_file}${NC}"
        else
            echo -e "${RED}❌ テストファイル不足: ${test_file}${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ 環境チェック完了${NC}\n"
}

# テストレベル検証
validate_test_level() {
    local valid_levels=("level_1" "level_2" "level_3" "level_4" "all")
    local is_valid=false
    
    for valid_level in "${valid_levels[@]}"; do
        if [[ "${LEVEL}" == "${valid_level}" ]]; then
            is_valid=true
            break
        fi
    done
    
    if [[ "${is_valid}" == false ]]; then
        echo -e "${RED}❌ 無効なテストレベル: ${LEVEL}${NC}"
        echo "有効なレベル: ${valid_levels[*]}"
        exit 1
    fi
}

# テスト前クリーンアップ
cleanup_before_test() {
    echo -e "${BLUE}🧹 テスト前クリーンアップ中...${NC}"
    
    # テスト用一時ディレクトリクリーンアップ
    local temp_dirs=(
        "${PROJECT_ROOT}/tests/fixtures"
        "${PROJECT_ROOT}/test_temp"
        "${PROJECT_ROOT}/workspace_test"
    )
    
    for temp_dir in "${temp_dirs[@]}"; do
        if [[ -d "${temp_dir}" ]]; then
            rm -rf "${temp_dir}"
            echo -e "${GREEN}✅ クリーンアップ: ${temp_dir}${NC}"
        fi
    done
    
    # Pytestキャッシュクリーンアップ
    if [[ -d "${PROJECT_ROOT}/.pytest_cache" ]]; then
        rm -rf "${PROJECT_ROOT}/.pytest_cache"
        echo -e "${GREEN}✅ Pytestキャッシュクリーンアップ${NC}"
    fi
    
    # Python __pycache__ クリーンアップ
    find "${PROJECT_ROOT}/tests" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✅ Python キャッシュクリーンアップ${NC}"
    
    echo -e "${GREEN}✅ クリーンアップ完了${NC}\n"
}

# テスト実行
run_tests() {
    echo -e "${WHITE}🚀 ワークフローテスト実行開始${NC}"
    echo -e "${CYAN}テストレベル: ${LEVEL}${NC}"
    echo -e "${CYAN}詳細出力: ${VERBOSE}${NC}"
    echo -e "${CYAN}失敗時停止: ${STOP_ON_FAILURE}${NC}"
    if [[ -n "${OUTPUT_FILE}" ]]; then
        echo -e "${CYAN}出力ファイル: ${OUTPUT_FILE}${NC}"
    fi
    echo

    # Pythonパス設定
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    
    # テストランナー引数構築
    local runner_args=("--level" "${LEVEL}")
    
    if [[ "${VERBOSE}" == true ]]; then
        runner_args+=("--verbose")
    fi
    
    if [[ "${STOP_ON_FAILURE}" == true ]]; then
        runner_args+=("--stop-on-failure")
    fi
    
    if [[ -n "${OUTPUT_FILE}" ]]; then
        runner_args+=("--output" "${OUTPUT_FILE}")
    fi
    
    # テスト実行
    cd "${PROJECT_ROOT}"
    
    local start_time=$(date +%s)
    
    if ${PYTHON_CMD} tests/test_runner.py "${runner_args[@]}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo
        echo -e "${GREEN}✅ ワークフローテスト完了 (${duration}秒)${NC}"
        
        # 結果ファイル確認
        if [[ -n "${OUTPUT_FILE}" && -f "${OUTPUT_FILE}" ]]; then
            echo -e "${CYAN}📁 結果ファイル: ${OUTPUT_FILE}${NC}"
            
            # ファイルサイズ表示
            local file_size=$(du -h "${OUTPUT_FILE}" | cut -f1)
            echo -e "${CYAN}📊 ファイルサイズ: ${file_size}${NC}"
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo
        echo -e "${RED}❌ ワークフローテスト失敗 (${duration}秒)${NC}"
        return 1
    fi
}

# テスト後クリーンアップ
cleanup_after_test() {
    echo
    echo -e "${BLUE}🧹 テスト後クリーンアップ中...${NC}"
    
    # 一時ファイル削除（オプション）
    local cleanup_patterns=(
        "${PROJECT_ROOT}/tests/fixtures/mock_*.json"
        "${PROJECT_ROOT}/test_temp"
        "${PROJECT_ROOT}/.pytest_cache"
    )
    
    for pattern in "${cleanup_patterns[@]}"; do
        # パターンマッチしたファイル・ディレクトリを削除
        for item in ${pattern}; do
            if [[ -e "${item}" ]]; then
                rm -rf "${item}"
                echo -e "${GREEN}✅ クリーンアップ: $(basename "${item}")${NC}"
            fi
        done
    done
    
    echo -e "${GREEN}✅ クリーンアップ完了${NC}"
}

# シグナルハンドリング
trap cleanup_after_test EXIT

# メイン処理
main() {
    # 引数解析
    parse_arguments "$@"
    
    # 実行情報表示
    echo -e "${WHITE}📋 ワークフローテスト実行スクリプト${NC}"
    echo -e "${CYAN}プロジェクトルート: ${PROJECT_ROOT}${NC}"
    echo
    
    # 環境チェック
    check_environment
    
    # テストレベル検証
    validate_test_level
    
    # テスト前クリーンアップ
    cleanup_before_test
    
    # テスト実行
    if run_tests; then
        echo -e "${GREEN}🎉 全ワークフローテスト成功${NC}"
        exit 0
    else
        echo -e "${RED}💥 ワークフローテスト失敗${NC}"
        exit 1
    fi
}

# メイン処理実行
main "$@"