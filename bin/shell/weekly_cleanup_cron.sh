#!/bin/bash
"""
P1-021: 週次ソースコード肥大化解決システム - cron用自動実行スクリプト

【概要】
毎週日曜日午前2時に自動実行されるクリーンアップスクリプト

【設定方法】
1. crontabに以下を追加:
   0 2 * * 0 /mnt/c/AItools/segment-anything/bin/shell/weekly_cleanup_cron.sh

2. 手動テスト実行:
   ./bin/shell/weekly_cleanup_cron.sh --test

【ログ出力】
- 実行ログ: logs/weekly_cleanup/cron_YYYYMMDD_HHMMSS.log
- エラーログ: logs/weekly_cleanup/error_YYYYMMDD_HHMMSS.log
"""

# スクリプトの場所を取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ログディレクトリ設定
LOG_DIR="$PROJECT_ROOT/logs/weekly_cleanup"
mkdir -p "$LOG_DIR"

# タイムスタンプ生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/cron_$TIMESTAMP.log"
ERROR_LOG="$LOG_DIR/error_$TIMESTAMP.log"

# テストモードチェック
TEST_MODE=false
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    echo "🧪 テストモード: DRY-RUN実行"
fi

# 実行開始ログ
echo "🧹 P1-021: 週次クリーンアップ開始 - $(date)" | tee "$LOG_FILE"
echo "📁 プロジェクトルート: $PROJECT_ROOT" | tee -a "$LOG_FILE"

# 環境確認
cd "$PROJECT_ROOT" || {
    echo "❌ エラー: プロジェクトルートに移動できません: $PROJECT_ROOT" | tee -a "$ERROR_LOG"
    exit 1
}

# Python環境確認
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ エラー: Python3が見つかりません" | tee -a "$ERROR_LOG"
    exit 1
fi

echo "🐍 Python: $($PYTHON_CMD --version)" | tee -a "$LOG_FILE"

# クリーンアップスクリプト実行
CLEANUP_SCRIPT="$PROJECT_ROOT/tools/scripts/weekly_cleanup.py"

if [[ ! -f "$CLEANUP_SCRIPT" ]]; then
    echo "❌ エラー: クリーンアップスクリプトが見つかりません: $CLEANUP_SCRIPT" | tee -a "$ERROR_LOG"
    exit 1
fi

# 実行コマンド構築
if [[ "$TEST_MODE" == "true" ]]; then
    CMD="$PYTHON_CMD $CLEANUP_SCRIPT --dry-run --verbose"
    echo "🔍 テスト実行: $CMD" | tee -a "$LOG_FILE"
else
    CMD="$PYTHON_CMD $CLEANUP_SCRIPT --force --verbose"
    echo "⚡ 本実行: $CMD" | tee -a "$LOG_FILE"
fi

# 実行前のディスク使用量記録
DISK_BEFORE=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $3}')
echo "💾 実行前ディスク使用量: $DISK_BEFORE" | tee -a "$LOG_FILE"

# クリーンアップ実行
echo "🚀 クリーンアップ実行開始..." | tee -a "$LOG_FILE"
if $CMD >> "$LOG_FILE" 2>> "$ERROR_LOG"; then
    CLEANUP_EXIT_CODE=0
    echo "✅ クリーンアップ正常完了" | tee -a "$LOG_FILE"
else
    CLEANUP_EXIT_CODE=$?
    echo "❌ クリーンアップエラー (終了コード: $CLEANUP_EXIT_CODE)" | tee -a "$ERROR_LOG"
fi

# 実行後のディスク使用量記録
DISK_AFTER=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $3}')
echo "💾 実行後ディスク使用量: $DISK_AFTER" | tee -a "$LOG_FILE"

# 実行時間計算
END_TIME=$(date)
echo "⏰ 完了時刻: $END_TIME" | tee -a "$LOG_FILE"

# エラーチェック
if [[ -s "$ERROR_LOG" ]]; then
    echo "⚠️ エラーログが記録されました:" | tee -a "$LOG_FILE"
    echo "   $ERROR_LOG" | tee -a "$LOG_FILE"
    
    # エラー内容を概要で表示
    echo "📋 エラー概要:" | tee -a "$LOG_FILE"
    head -10 "$ERROR_LOG" | sed 's/^/   /' | tee -a "$LOG_FILE"
    
    if [[ $(wc -l < "$ERROR_LOG") -gt 10 ]]; then
        echo "   ... (省略)" | tee -a "$LOG_FILE"
    fi
else
    echo "✅ エラーなし" | tee -a "$LOG_FILE"
    # 空のエラーログは削除
    rm -f "$ERROR_LOG"
fi

# 通知送信（Pushover設定があれば）
PUSHOVER_CONFIG="$PROJECT_ROOT/config/pushover.json"
if [[ -f "$PUSHOVER_CONFIG" ]] && command -v python3 &> /dev/null; then
    echo "📱 Pushover通知送信試行..." | tee -a "$LOG_FILE"
    
    # 通知送信スクリプト（inline Python）
    python3 << EOF 2>> "$ERROR_LOG"
import json
import sys
import requests
from pathlib import Path

config_file = Path("$PUSHOVER_CONFIG")
if config_file.exists():
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        # 通知メッセージ構築
        if $CLEANUP_EXIT_CODE == 0:
            title = "🧹 P1-021 週次クリーンアップ完了"
            message = f"ディスク使用量: $DISK_BEFORE → $DISK_AFTER\\n完了時刻: $END_TIME"
            priority = 0
        else:
            title = "❌ P1-021 週次クリーンアップエラー"
            message = f"終了コード: $CLEANUP_EXIT_CODE\\nエラーログ: $ERROR_LOG"
            priority = 1
        
        # Pushover送信
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": config["api_token"],
                "user": config["user_key"],
                "title": title,
                "message": message,
                "priority": priority
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Pushover通知送信成功")
        else:
            print(f"⚠️ Pushover通知送信失敗: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Pushover通知送信エラー: {e}")
else:
    print("⚠️ Pushover設定ファイルが見つかりません")
EOF
fi

# 最終レポート
echo "=" | tr '=' '=' | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🧹 P1-021 週次クリーンアップ完了レポート" | tee -a "$LOG_FILE"
echo "   実行日時: $(date)" | tee -a "$LOG_FILE"
echo "   終了コード: $CLEANUP_EXIT_CODE" | tee -a "$LOG_FILE"
echo "   ログファイル: $LOG_FILE" | tee -a "$LOG_FILE"
if [[ -f "$ERROR_LOG" ]]; then
    echo "   エラーログ: $ERROR_LOG" | tee -a "$LOG_FILE"
fi
echo "=" | tr '=' '=' | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ログファイル自体の管理（30日以上古いログを削除）
find "$LOG_DIR" -name "cron_*.log" -mtime +30 -delete 2>/dev/null
find "$LOG_DIR" -name "error_*.log" -mtime +30 -delete 2>/dev/null

exit $CLEANUP_EXIT_CODE