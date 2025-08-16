#!/bin/bash
"""
PH2-004-RESOURCE: リソース管理最適化システム - シェルスクリプトインターフェース

【概要】
GPU・CPU・メモリ・ディスクリソースの統合監視・最適化システム用シェルスクリプト

【使用方法】
./bin/shell/ph2_004_resource_monitor.sh monitor    # 5分間監視
./bin/shell/ph2_004_resource_monitor.sh check      # 現在状況確認
./bin/shell/ph2_004_resource_monitor.sh optimize   # 手動最適化
./bin/shell/ph2_004_resource_monitor.sh report     # レポート生成

【自動実行設定】
crontab -e
# 1時間ごとのリソースチェック
0 * * * * /mnt/c/AItools/segment-anything/bin/shell/ph2_004_resource_monitor.sh optimize
"""

# スクリプトの場所を取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ログディレクトリ設定
LOG_DIR="$PROJECT_ROOT/logs/resource_monitoring"
mkdir -p "$LOG_DIR"

# タイムスタンプ生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/resource_monitor_$TIMESTAMP.log"

# Python環境確認
cd "$PROJECT_ROOT" || {
    echo "❌ エラー: プロジェクトルートに移動できません: $PROJECT_ROOT"
    exit 1
}

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ エラー: Python3が見つかりません"
    exit 1
fi

# メイン実行スクリプト
RESOURCE_SCRIPT="$PROJECT_ROOT/tools/scripts/ph2_004_resource_optimization.py"

if [[ ! -f "$RESOURCE_SCRIPT" ]]; then
    echo "❌ エラー: リソース監視スクリプトが見つかりません: $RESOURCE_SCRIPT"
    exit 1
fi

# 使用方法表示
show_usage() {
    echo "🔧 PH2-004-RESOURCE: リソース管理最適化システム"
    echo ""
    echo "使用方法:"
    echo "  $0 monitor    # リアルタイム監視（デフォルト5分間）"
    echo "  $0 check      # 現在のリソース状況確認"
    echo "  $0 optimize   # 手動最適化実行"
    echo "  $0 report     # レポート生成"
    echo "  $0 --help     # このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0 monitor               # 5分間監視"
    echo "  $0 check                 # 即座に状況確認"
    echo "  $0 optimize              # 最適化実行"
    echo "  $0 report                # レポート生成"
    echo ""
    echo "自動実行設定:"
    echo "  crontab -e"
    echo "  0 * * * * $0 optimize    # 1時間ごとの最適化"
    echo "  */30 * * * * $0 check    # 30分ごとのチェック"
}

# モード別実行
case "$1" in
    "monitor")
        echo "📊 PH2-004-RESOURCE: リアルタイム監視開始" | tee "$LOG_FILE"
        echo "📁 プロジェクトルート: $PROJECT_ROOT" | tee -a "$LOG_FILE"
        echo "⏰ 開始時刻: $(date)" | tee -a "$LOG_FILE"
        
        # 監視実行（デフォルト5分間）
        DURATION=${2:-300}
        echo "🕐 監視時間: ${DURATION}秒" | tee -a "$LOG_FILE"
        
        if $PYTHON_CMD "$RESOURCE_SCRIPT" monitor --duration "$DURATION" | tee -a "$LOG_FILE"; then
            echo "✅ 監視完了" | tee -a "$LOG_FILE"
            EXIT_CODE=0
        else
            echo "❌ 監視エラー" | tee -a "$LOG_FILE"
            EXIT_CODE=1
        fi
        ;;
        
    "check")
        echo "🔍 PH2-004-RESOURCE: リソース状況確認" | tee "$LOG_FILE"
        echo "⏰ 実行時刻: $(date)" | tee -a "$LOG_FILE"
        
        if $PYTHON_CMD "$RESOURCE_SCRIPT" check | tee -a "$LOG_FILE"; then
            echo "✅ 確認完了" | tee -a "$LOG_FILE"
            EXIT_CODE=0
        else
            echo "❌ 確認エラー" | tee -a "$LOG_FILE"
            EXIT_CODE=1
        fi
        ;;
        
    "optimize")
        echo "🔧 PH2-004-RESOURCE: リソース最適化実行" | tee "$LOG_FILE"
        echo "⏰ 実行時刻: $(date)" | tee -a "$LOG_FILE"
        
        # 実行前のリソース状況記録
        echo "📊 最適化前の状況:" | tee -a "$LOG_FILE"
        df -h "$PROJECT_ROOT" | tee -a "$LOG_FILE"
        free -h | tee -a "$LOG_FILE"
        
        if $PYTHON_CMD "$RESOURCE_SCRIPT" optimize | tee -a "$LOG_FILE"; then
            echo "✅ 最適化完了" | tee -a "$LOG_FILE"
            
            # 最適化後の状況記録
            echo "📊 最適化後の状況:" | tee -a "$LOG_FILE"
            df -h "$PROJECT_ROOT" | tee -a "$LOG_FILE"
            free -h | tee -a "$LOG_FILE"
            
            EXIT_CODE=0
        else
            echo "❌ 最適化エラー" | tee -a "$LOG_FILE"
            EXIT_CODE=1
        fi
        ;;
        
    "report")
        echo "📄 PH2-004-RESOURCE: レポート生成" | tee "$LOG_FILE"
        echo "⏰ 実行時刻: $(date)" | tee -a "$LOG_FILE"
        
        # レポート出力パス設定
        REPORT_OUTPUT="$LOG_DIR/resource_report_$TIMESTAMP.json"
        
        if $PYTHON_CMD "$RESOURCE_SCRIPT" report --output "$REPORT_OUTPUT" | tee -a "$LOG_FILE"; then
            echo "✅ レポート生成完了" | tee -a "$LOG_FILE"
            echo "📄 レポートファイル: $REPORT_OUTPUT" | tee -a "$LOG_FILE"
            
            # レポートサイズ表示
            if [[ -f "$REPORT_OUTPUT" ]]; then
                REPORT_SIZE=$(du -h "$REPORT_OUTPUT" | cut -f1)
                echo "📊 レポートサイズ: $REPORT_SIZE" | tee -a "$LOG_FILE"
            fi
            
            EXIT_CODE=0
        else
            echo "❌ レポート生成エラー" | tee -a "$LOG_FILE"
            EXIT_CODE=1
        fi
        ;;
        
    "--help"|"-h"|"help")
        show_usage
        exit 0
        ;;
        
    "")
        echo "❌ エラー: モードを指定してください"
        echo ""
        show_usage
        exit 1
        ;;
        
    *)
        echo "❌ エラー: 不明なモード '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac

# 実行完了ログ
echo "⏰ 完了時刻: $(date)" | tee -a "$LOG_FILE"
echo "📄 ログファイル: $LOG_FILE" | tee -a "$LOG_FILE"

# Pushover通知（設定があれば）
PUSHOVER_CONFIG="$PROJECT_ROOT/config/pushover.json"
if [[ -f "$PUSHOVER_CONFIG" ]] && command -v python3 &> /dev/null; then
    echo "📱 Pushover通知送信試行..." | tee -a "$LOG_FILE"
    
    # 通知送信スクリプト（inline Python）
    python3 << EOF 2>> "$LOG_FILE"
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
        if $EXIT_CODE == 0:
            title = "🔧 PH2-004-RESOURCE 実行完了"
            message = f"モード: $1\\n実行時刻: $(date)\\nログ: $LOG_FILE"
            priority = 0
        else:
            title = "❌ PH2-004-RESOURCE 実行エラー"
            message = f"モード: $1\\n終了コード: $EXIT_CODE\\nログ: $LOG_FILE"
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

# 最終結果出力
echo "==================================" | tee -a "$LOG_FILE"
echo "🔧 PH2-004-RESOURCE 実行完了" | tee -a "$LOG_FILE"
echo "   モード: $1" | tee -a "$LOG_FILE"
echo "   終了コード: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "   ログ: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"

# ログファイル自体の管理（30日以上古いログを削除）
find "$LOG_DIR" -name "resource_monitor_*.log" -mtime +30 -delete 2>/dev/null

exit $EXIT_CODE