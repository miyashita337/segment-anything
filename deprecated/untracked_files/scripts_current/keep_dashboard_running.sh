#!/bin/bash
# ダッシュボードサーバー永続起動スクリプト

SERVER_SCRIPT="/mnt/c/AItools/segment-anything/integrated_dashboard_server.py"
LOG_FILE="/mnt/c/AItools/segment-anything/dashboard_server.log"
PID_FILE="/mnt/c/AItools/segment-anything/dashboard_server.pid"

# サーバーが動作中かチェック
check_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            return 0  # 動作中
        fi
    fi
    return 1  # 停止中
}

# サーバー起動
start_server() {
    echo "$(date): ダッシュボードサーバーを起動中..."
    nohup python3 "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "$(date): サーバー起動完了 PID: $(cat $PID_FILE)"
    echo "$(date): アクセス: https://100.123.241.106/tracker"
}

# メイン処理
if check_server; then
    echo "$(date): ダッシュボードサーバーは既に動作中です (PID: $(cat $PID_FILE))"
    echo "$(date): アクセス: https://100.123.241.106/tracker"
else
    echo "$(date): ダッシュボードサーバーが停止しています。再起動します..."
    start_server
fi

# サーバー状態確認
sleep 3
if check_server; then
    echo "$(date): ✅ サーバー正常動作中"
    echo "$(date): 🌐 https://100.123.241.106/tracker でアクセス可能"
else
    echo "$(date): ❌ サーバー起動に失敗しました"
    tail -10 "$LOG_FILE"
fi