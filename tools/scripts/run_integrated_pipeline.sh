#!/bin/bash

# 統合パイプライン実行スクリプト (Phase 3-6)
# トラッカー: INTEGRATE-3-6
# 目的: Phase 3-6統合パイプライン用ラッパースクリプト

set -euo pipefail

# スクリプトの場所を基準にプロジェクトルートを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ログ設定
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# デフォルト設定
DEFAULT_CONFIG="${PROJECT_ROOT}/config/pipeline_config.yaml"
DEFAULT_INPUT_DIR="/mnt/c/AItools/lora/train/yado/org/kana05/"
WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/tracker-workspace"

# 色付きログ関数
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1" | tee -a "${LOG_DIR}/pipeline_wrapper.log"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1" | tee -a "${LOG_DIR}/pipeline_wrapper.log"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" | tee -a "${LOG_DIR}/pipeline_wrapper.log"
}

# 使用方法表示
show_usage() {
    cat << EOF
統合パイプライン実行スクリプト (Phase 3-6)

使用方法:
    $0 <TRACKER_ID> [オプション]

必須引数:
    TRACKER_ID          トラッカーID (例: INTEGRATE-3-6)

オプション:
    --input-dir PATH    入力ディレクトリパス (デフォルト: ${DEFAULT_INPUT_DIR})
    --config PATH       設定ファイルパス (デフォルト: ${DEFAULT_CONFIG})
    --resume            レジューム実行
    --verbose           詳細ログ出力
    --dry-run           設定確認のみ（実行しない）
    --help              このヘルプを表示

例:
    # 基本実行
    $0 INTEGRATE-3-6

    # カスタム入力ディレクトリで実行
    $0 INTEGRATE-3-6 --input-dir /path/to/custom/input

    # レジューム実行
    $0 INTEGRATE-3-6 --resume

    # ドライラン（設定確認のみ）
    $0 INTEGRATE-3-6 --dry-run

環境変数:
    PYTHON_CMD          使用するPythonコマンド (デフォルト: python3)
    PIPELINE_TIMEOUT    タイムアウト秒数 (デフォルト: 1800)
    ENABLE_CUDA_CHECK   CUDA利用可能性チェック (デフォルト: true)

EOF
}

# 引数解析
TRACKER_ID=""
INPUT_DIR="${DEFAULT_INPUT_DIR}"
CONFIG_PATH="${DEFAULT_CONFIG}"
RESUME=false
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        --*)
            log_error "不明なオプション: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "${TRACKER_ID}" ]]; then
                TRACKER_ID="$1"
            else
                log_error "複数のトラッカーIDが指定されました: ${TRACKER_ID}, $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# トラッカーID必須チェック
if [[ -z "${TRACKER_ID}" ]]; then
    log_error "トラッカーIDが指定されていません"
    show_usage
    exit 1
fi

# 環境変数設定
PYTHON_CMD="${PYTHON_CMD:-python3}"
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-1800}"  # 30分
ENABLE_CUDA_CHECK="${ENABLE_CUDA_CHECK:-true}"

# 事前チェック
pre_check() {
    log_info "=== 事前チェック開始 ==="
    
    # Python存在チェック
    if ! command -v "${PYTHON_CMD}" > /dev/null 2>&1; then
        log_error "Pythonコマンドが見つかりません: ${PYTHON_CMD}"
        exit 1
    fi
    
    # 設定ファイル存在チェック
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        log_error "設定ファイルが見つかりません: ${CONFIG_PATH}"
        exit 1
    fi
    
    # パイプラインスクリプト存在チェック
    PIPELINE_SCRIPT="${PROJECT_ROOT}/tools/core/integrated_quality_pipeline.py"
    if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
        log_error "パイプラインスクリプトが見つかりません: ${PIPELINE_SCRIPT}"
        exit 1
    fi
    
    # 入力ディレクトリ存在チェック（dry-runでない場合）
    if [[ "${DRY_RUN}" != "true" ]]; then
        if [[ ! -d "${INPUT_DIR}" ]]; then
            log_error "入力ディレクトリが存在しません: ${INPUT_DIR}"
            log_error "対処方法:"
            log_error "  1. パスを確認してください: ls $(dirname "${INPUT_DIR}")"
            log_error "  2. 正しいパスを --input-dir で指定してください"
            log_error "  3. または --dry-run で設定確認のみ実行してください"
            exit 1
        fi
        
        # 画像ファイル存在チェック
        if ! find "${INPUT_DIR}" -name "*.jpg" -o -name "*.png" | head -1 | grep -q .; then
            log_warn "入力ディレクトリに画像ファイルが見つかりません: ${INPUT_DIR}"
        fi
    fi
    
    # ワークスペースディレクトリ作成
    WORKSPACE_DIR="${WORKSPACE_BASE}/${TRACKER_ID}"
    if [[ "${DRY_RUN}" != "true" ]]; then
        mkdir -p "${WORKSPACE_DIR}"
        log_info "ワークスペースディレクトリ準備完了: ${WORKSPACE_DIR}"
    fi
    
    # CUDA利用可能性チェック
    if [[ "${ENABLE_CUDA_CHECK}" == "true" ]]; then
        log_info "CUDA利用可能性チェック中..."
        if "${PYTHON_CMD}" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
            log_info "CUDA チェック完了"
        else
            log_warn "CUDA チェックに失敗しました（CPU処理となります）"
        fi
    fi
    
    log_info "=== 事前チェック完了 ==="
}

# 設定表示
show_config() {
    log_info "=== 実行設定 ==="
    log_info "トラッカーID: ${TRACKER_ID}"
    log_info "入力ディレクトリ: ${INPUT_DIR}"
    log_info "設定ファイル: ${CONFIG_PATH}"
    log_info "ワークスペース: ${WORKSPACE_BASE}/${TRACKER_ID}"
    log_info "レジューム実行: ${RESUME}"
    log_info "詳細ログ: ${VERBOSE}"
    log_info "ドライラン: ${DRY_RUN}"
    log_info "Pythonコマンド: ${PYTHON_CMD}"
    log_info "タイムアウト: ${PIPELINE_TIMEOUT}秒"
    log_info "================="
}

# メイン実行
main() {
    local start_time=$(date +%s)
    local log_file="${LOG_DIR}/${TRACKER_ID}_integrated_pipeline_wrapper.log"
    
    log_info "統合パイプライン実行開始: ${TRACKER_ID}"
    log_info "実行ログ: ${log_file}"
    
    # 事前チェック
    pre_check
    
    # 設定表示
    show_config
    
    # ドライラン終了
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "ドライラン完了（実際の処理は実行されていません）"
        return 0
    fi
    
    # パイプライン実行コマンド構築
    local cmd_args=(
        "${PYTHON_CMD}"
        "${PROJECT_ROOT}/tools/core/integrated_quality_pipeline.py"
        "--config" "${CONFIG_PATH}"
        "--tracker-id" "${TRACKER_ID}"
    )
    
    if [[ "${RESUME}" == "true" ]]; then
        cmd_args+=("--resume")
    fi
    
    if [[ "${VERBOSE}" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    # 一時的に設定ファイルの入力パスを更新
    local temp_config="${PROJECT_ROOT}/config/temp_pipeline_config_${TRACKER_ID}.yaml"
    if [[ "${INPUT_DIR}" != "${DEFAULT_INPUT_DIR}" ]]; then
        log_info "一時設定ファイル作成中..."
        sed "s|default_input:.*|default_input: \"${INPUT_DIR}\"|" "${CONFIG_PATH}" > "${temp_config}"
        cmd_args[3]="${temp_config}"  # --config の値を更新
    fi
    
    # パイプライン実行
    log_info "パイプライン実行中..."
    log_info "実行コマンド: ${cmd_args[*]}"
    
    local exit_code=0
    if timeout "${PIPELINE_TIMEOUT}" "${cmd_args[@]}" 2>&1 | tee -a "${log_file}"; then
        log_info "パイプライン実行成功"
    else
        exit_code=$?
        if [[ ${exit_code} -eq 124 ]]; then
            log_error "パイプライン実行タイムアウト (${PIPELINE_TIMEOUT}秒)"
        else
            log_error "パイプライン実行失敗 (終了コード: ${exit_code})"
        fi
    fi
    
    # 一時設定ファイル削除
    if [[ -f "${temp_config}" ]]; then
        rm -f "${temp_config}"
    fi
    
    # 実行時間計算
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 結果表示
    if [[ ${exit_code} -eq 0 ]]; then
        log_info "=== 実行完了 ==="
        log_info "トラッカーID: ${TRACKER_ID}"
        log_info "実行時間: ${duration}秒"
        log_info "ワークスペース: ${WORKSPACE_BASE}/${TRACKER_ID}"
        log_info "ダッシュボード: ${WORKSPACE_BASE}/${TRACKER_ID}/dashboard/dashboard.html"
        
        # 成功通知
        if command -v windows-notify > /dev/null 2>&1; then
            windows-notify -t "Claude Code" -m "統合パイプライン完了: ${TRACKER_ID} (${duration}秒)" || true
        fi
    else
        log_error "=== 実行失敗 ==="
        log_error "トラッカーID: ${TRACKER_ID}"
        log_error "実行時間: ${duration}秒"
        log_error "終了コード: ${exit_code}"
        log_error "ログファイル: ${log_file}"
        
        # 失敗通知
        if command -v windows-notify > /dev/null 2>&1; then
            windows-notify -t "Claude Code" -m "統合パイプライン失敗: ${TRACKER_ID} (エラーコード: ${exit_code})" || true
        fi
    fi
    
    return ${exit_code}
}

# エラートラップ設定
trap 'log_error "スクリプト実行中にエラーが発生しました"; exit 1' ERR

# メイン実行
main "$@"