#!/bin/bash

echo "📱 ngrokトンネルセットアップスクリプト"
echo "=================================="

# ngrokがインストールされているか確認
if ! command -v ngrok &> /dev/null; then
    echo "⚠️ ngrokがインストールされていません"
    echo "以下のコマンドでインストールしてください:"
    echo ""
    echo "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
    echo "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list"
    echo "sudo apt update && sudo apt install ngrok"
    echo ""
    echo "その後、ngrokアカウントを作成して認証トークンを設定:"
    echo "ngrok authtoken YOUR_AUTH_TOKEN"
    exit 1
fi

echo "✅ ngrokがインストールされています"
echo ""
echo "🚀 ngrokトンネルを開始します..."
echo "Basic認証: admin / integrate36"
echo ""
echo "注意: 無料版ngrokはBasic認証をサポートしていません"
echo "代わりにサーバー側でBasic認証を処理しています"
echo ""
echo "トンネル開始中..."

# ngrokトンネル開始（Basic認証はサーバー側で処理）
ngrok http 8080