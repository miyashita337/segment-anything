#!/bin/bash
# バックグラウンドバッチ実行システム
#
# screen セッションでの安全な長時間実行
# - 自動ログ記録
# - プロセス監視
# - エラー時の自動通知

set -euo pipefail

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
SESSION_NAME=""
TRACKER_ID=""
INPUT_DIR=""
BATCH_SIZE=1
TIMEOUT=600
USE_GOOGLE_SHEETS=false
DRY_RUN=false

# ログディレクトリ作成
mkdir -p "$LOG_DIR"

# 関数定義
show_usage() {
    cat << EOF
バックグラウンドバッチ実行システム

使用法:
    $0 TRACKER_ID --input-dir INPUT_DIR [オプション]

必須引数:
    TRACKER_ID      トラッカーID (例: PH3-007-PRODUCTION)
    --input-dir     入力ディレクトリパス

オプション:
    --batch-size SIZE    バッチサイズ (デフォルト: 1)
    --timeout SECONDS    画像あたりタイムアウト秒 (デフォルト: 600)
    --google-sheets      Google Sheets連携を有効化
    --dry-run           実行前確認のみ
    --session-name NAME  screen セッション名 (自動生成)
    -h, --help          このヘルプを表示

例:
    # 基本実行
    $0 PH3-007-PRODUCTION --input-dir /path/to/images

    # Google Sheets連携あり
    $0 PH3-007-PRODUCTION --input-dir /path/to/images --google-sheets --timeout 900

    # ドライラン
    $0 PH3-007-PRODUCTION --input-dir /path/to/images --dry-run
EOF
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/background_runner.log"
}

check_prerequisites() {
    log_message "事前チェック開始"
    
    # screen コマンドの確認
    if ! command -v screen &> /dev/null; then
        log_message "エラー: screen コマンドがインストールされていません"
        echo "インストール方法:"
        echo "  Ubuntu/Debian: sudo apt-get install screen"
        echo "  CentOS/RHEL: sudo yum install screen"
        exit 1
    fi
    
    # Python環境の確認
    if ! command -v python3 &> /dev/null; then
        log_message "エラー: python3 が見つかりません"
        exit 1
    fi
    
    # 入力ディレクトリの確認
    if [[ ! -d "$INPUT_DIR" ]]; then
        log_message "エラー: 入力ディレクトリが存在しません: $INPUT_DIR"
        exit 1
    fi
    
    # 画像ファイルの確認
    local image_count=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
    if [[ $image_count -eq 0 ]]; then
        log_message "エラー: 処理対象画像が見つかりません: $INPUT_DIR"
        exit 1
    fi
    
    log_message "処理対象画像: ${image_count}枚"
    
    # GPU確認
    if command -v nvidia-smi &> /dev/null; then
        local gpu_status=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
        log_message "GPU状況: $gpu_status"
    else
        log_message "警告: nvidia-smi が見つかりません。GPU処理に問題がある可能性があります"
    fi
    
    log_message "事前チェック完了"
}

cleanup_existing_processes() {
    log_message "既存プロセスのクリーンアップ開始"
    
    # 残留screen セッションの確認
    local existing_sessions=$(screen -list | grep -c "$SESSION_NAME" || true)
    if [[ $existing_sessions -gt 0 ]]; then
        log_message "警告: 既存のscreen セッション '$SESSION_NAME' が見つかりました"
        read -p "既存セッションを終了しますか？ [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            screen -S "$SESSION_NAME" -X quit 2>/dev/null || true
            log_message "既存セッションを終了しました"
        else
            log_message "実行を中止します"
            exit 1
        fi
    fi
    
    # SAM関連プロセスのクリーンアップ
    local sam_processes=$(ps aux | grep -c "sam_yolo_character_segment.py" || true)
    if [[ $sam_processes -gt 1 ]]; then  # grep自身も含まれるので1より大きい場合
        log_message "警告: SAM関連プロセスが実行中です"
        ps aux | grep "sam_yolo_character_segment.py" | grep -v grep
        
        read -p "これらのプロセスを終了しますか？ [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "sam_yolo_character_segment.py" || true
            log_message "SAM関連プロセスを終了しました"
            sleep 3
        fi
    fi
    
    log_message "プロセスクリーンアップ完了"
}

generate_execution_command() {
    local cmd="python3 tools/automation/simple_batch_runner.py $TRACKER_ID"
    cmd="$cmd --input-dir '$INPUT_DIR'"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --timeout $TIMEOUT"
    
    if [[ "$USE_GOOGLE_SHEETS" == "true" ]]; then
        cmd="$cmd --google-sheets"
    fi
    
    # Pushover通知は常に有効（30分間隔）
    cmd="$cmd --pushover-interval 30"
    
    echo "$cmd"
}

show_execution_plan() {
    cat << EOF

=====================================
バックグラウンド実行プラン
=====================================

トラッカーID: $TRACKER_ID
入力ディレクトリ: $INPUT_DIR
バッチサイズ: $BATCH_SIZE
タイムアウト: ${TIMEOUT}秒/枚
Google Sheets連携: $([ "$USE_GOOGLE_SHEETS" = "true" ] && echo "有効" || echo "無効")
Pushover通知: 有効（30分間隔で進捗通知）

Screen セッション名: $SESSION_NAME
ログファイル: $LOG_DIR/${TRACKER_ID}_execution.log

実行コマンド:
$(generate_execution_command)

推定実行時間: $(calculate_estimated_time)
推定GPU使用量: 高（連続使用）

📱 Pushover通知タイミング:
  - 処理開始時
  - 25%, 50%, 75%達成時
  - 30分ごとの定期更新
  - エラー発生時
  - 処理完了時

=====================================

EOF
}

calculate_estimated_time() {
    local image_count=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
    local total_seconds=$((image_count * TIMEOUT))
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    
    if [[ $hours -gt 0 ]]; then
        echo "${hours}時間${minutes}分（最大見積もり）"
    else
        echo "${minutes}分（最大見積もり）"
    fi
}

start_background_execution() {
    log_message "バックグラウンド実行開始: $TRACKER_ID"
    
    local execution_log="$LOG_DIR/${TRACKER_ID}_execution.log"
    local cmd=$(generate_execution_command)
    
    # screen セッションでの実行
    screen -dmS "$SESSION_NAME" bash -c "
        cd '$PROJECT_ROOT'
        echo '=== バックグラウンド実行開始: $(date) ===' | tee -a '$execution_log'
        echo 'コマンド: $cmd' | tee -a '$execution_log'
        echo | tee -a '$execution_log'
        
        $cmd 2>&1 | tee -a '$execution_log'
        
        local exit_code=\$?
        echo | tee -a '$execution_log'
        echo '=== バックグラウンド実行終了: $(date) ===' | tee -a '$execution_log'
        echo '終了コード: '\$exit_code | tee -a '$execution_log'
        
        # 完了通知
        if command -v windows-notify &> /dev/null; then
            if [[ \$exit_code -eq 0 ]]; then
                windows-notify -t 'Claude Code' -m '$TRACKER_ID バックグラウンド処理完了'
            else
                windows-notify -t 'Claude Code' -m '$TRACKER_ID バックグラウンド処理失敗 (終了コード: '\$exit_code')'
            fi
        fi
        
        # セッション維持（結果確認用）
        echo 'Enterキーでセッション終了...'
        read
    "
    
    sleep 2  # セッション起動待機
    
    if screen -list | grep -q "$SESSION_NAME"; then
        log_message "✅ バックグラウンド実行開始成功"
        echo
        echo "📋 監視・操作コマンド:"
        echo "  進捗確認: screen -r $SESSION_NAME"
        echo "  デタッチ: Ctrl+A, D"
        echo "  ログ確認: tail -f $execution_log"
        echo "  セッション一覧: screen -list"
        echo
        echo "🔍 Google Sheets監視コマンド:"
        echo "  python3 tools/automation/sheets_polling_monitor.py $TRACKER_ID"
        echo
        return 0
    else
        log_message "❌ バックグラウンド実行開始失敗"
        return 1
    fi
}

# メイン処理
main() {
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input-dir)
                INPUT_DIR="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --google-sheets)
                USE_GOOGLE_SHEETS=true
                shift
                ;;
            --session-name)
                SESSION_NAME="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                echo "不明なオプション: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$TRACKER_ID" ]]; then
                    TRACKER_ID="$1"
                else
                    echo "予期しない引数: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 必須引数チェック
    if [[ -z "$TRACKER_ID" || -z "$INPUT_DIR" ]]; then
        echo "エラー: TRACKER_ID と --input-dir は必須です"
        show_usage
        exit 1
    fi
    
    # セッション名の自動生成
    if [[ -z "$SESSION_NAME" ]]; then
        SESSION_NAME="extract_${TRACKER_ID}_$(date +%s)"
    fi
    
    # 実行プラン表示
    show_execution_plan
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "🔍 ドライラン完了（実際の実行は行われませんでした）"
        exit 0
    fi
    
    # 実行確認
    read -p "この設定でバックグラウンド実行を開始しますか？ [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "実行をキャンセルしました"
        exit 0
    fi
    
    # 実行フロー
    check_prerequisites
    cleanup_existing_processes
    start_background_execution
    
    log_message "バックグラウンド実行システム完了"
}

# 引数があれば main を実行
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi