#!/bin/bash
# ダッシュボードサーバー自動監視設定スクリプト

SCRIPT_PATH="/mnt/c/AItools/segment-anything/keep_dashboard_running.sh"
CRON_LOG="/mnt/c/AItools/segment-anything/cron_dashboard.log"

echo "🚀 ダッシュボードサーバー自動監視設定"
echo "=================================="

# 現在のcrontab確認
echo "📋 現在のcrontab:"
crontab -l 2>/dev/null || echo "crontabが設定されていません"

echo ""
echo "🔧 推奨設定:"
echo "以下のコマンドを実行してcronジョブを追加してください："
echo ""
echo "crontab -e"
echo ""
echo "そして以下の行を追加："
echo "# ダッシュボードサーバー監視（5分毎）"
echo "*/5 * * * * $SCRIPT_PATH >> $CRON_LOG 2>&1"
echo ""
echo "または、一括設定する場合："
echo "(crontab -l 2>/dev/null; echo \"*/5 * * * * $SCRIPT_PATH >> $CRON_LOG 2>&1\") | crontab -"
echo ""
echo "🌐 アクセスURL: https://100.123.241.106/tracker"
echo "📊 Basic認証: admin / dashboard2025!"