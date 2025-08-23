#!/bin/bash
# QCC-021 進捗モニタリングスクリプト（10分ごと実行）

while true; do
    echo "==================== $(date '+%Y-%m-%d %H:%M:%S') ===================="
    echo "🔍 QCC-021 バッチ抽出進捗確認"
    
    # プロセス確認
    if ps aux | grep -E "extract_character.*kana" | grep -v grep > /dev/null; then
        echo "✅ プロセス実行中"
        ps aux | grep -E "extract_character.*kana" | grep -v grep | awk '{print "   PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%"}'
    else
        echo "❌ プロセス停止"
    fi
    
    # 抽出画像数
    extracted_count=$(find /mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-021-EXTENDED/extraction -name "*.jpg" | wc -l)
    echo "📊 現在の抽出数: ${extracted_count}枚"
    
    # 目標までの残り
    remaining=$((379 - extracted_count))
    echo "🎯 目標まで: ${remaining}枚"
    
    # ログの最後5行
    if [ -f "/mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-021-EXTENDED/extraction/kana03_extraction.log" ]; then
        echo "📝 最新ログ:"
        tail -3 /mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-021-EXTENDED/extraction/kana03_extraction.log | sed 's/^/   /'
    fi
    
    echo "================================================================="
    
    # 10分待機
    sleep 600
done