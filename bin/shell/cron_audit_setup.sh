#!/bin/bash
"""
日次監査システムセットアップスクリプト
毎日20:00にパス準拠性監査を自動実行・Pushover通知
"""

# 設定
PROJECT_ROOT="/mnt/c/AItools/segment-anything"
PYTHON_CMD="python3"
AUDIT_SCRIPT="$PROJECT_ROOT/tools/audit_path_compliance.py"
CRON_TIME="0 20 * * *"  # 毎日20:00

# 実行環境確認
check_environment() {
    echo "🔍 実行環境確認中..."
    
    if [ ! -f "$AUDIT_SCRIPT" ]; then
        echo "❌ 監査スクリプトが見つかりません: $AUDIT_SCRIPT"
        exit 1
    fi
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "❌ Python3が見つかりません"
        exit 1
    fi
    
    echo "✅ 実行環境OK"
}

# crontab設定
setup_cron() {
    echo "⏰ crontab設定中..."
    
    # 既存のエントリ削除
    crontab -l 2>/dev/null | grep -v "audit_path_compliance.py" | crontab -
    
    # 新しいエントリ追加
    (crontab -l 2>/dev/null; echo "$CRON_TIME cd $PROJECT_ROOT && $PYTHON_CMD $AUDIT_SCRIPT >> /tmp/audit_log.txt 2>&1") | crontab -
    
    echo "✅ crontab設定完了"
    echo "📅 実行スケジュール: 毎日20:00"
}

# 手動テスト実行
test_execution() {
    echo "🧪 手動テスト実行..."
    
    cd "$PROJECT_ROOT"
    $PYTHON_CMD "$AUDIT_SCRIPT" --no-pushover
    
    echo "✅ テスト完了"
}

# Pushover設定確認
check_pushover_config() {
    echo "📱 Pushover設定確認..."
    
    PUSHOVER_CONFIG="$PROJECT_ROOT/config/pushover.json"
    
    if [ ! -f "$PUSHOVER_CONFIG" ]; then
        echo "⚠️ Pushover設定ファイルが見つかりません: $PUSHOVER_CONFIG"
        echo "💡 通知を有効にするには、設定ファイルを作成してください"
        
        # テンプレート作成
        cat > "$PUSHOVER_CONFIG.template" << EOF
{
    "api_token": "YOUR_API_TOKEN_HERE",
    "user_key": "YOUR_USER_KEY_HERE"
}
EOF
        echo "📋 テンプレート作成: $PUSHOVER_CONFIG.template"
    else
        echo "✅ Pushover設定ファイル確認済み"
    fi
}

# メイン実行
main() {
    echo "🚀 日次監査システムセットアップ開始"
    echo "=" * 50
    
    check_environment
    check_pushover_config
    
    # オプション処理
    case "${1:-setup}" in
        "setup")
            setup_cron
            echo "🎉 セットアップ完了！"
            echo "   次回実行: 今日の20:00"
            ;;
        "test")
            test_execution
            ;;
        "remove")
            echo "🗑️ crontab削除中..."
            crontab -l 2>/dev/null | grep -v "audit_path_compliance.py" | crontab -
            echo "✅ 削除完了"
            ;;
        "status")
            echo "📋 現在のcrontab設定:"
            crontab -l 2>/dev/null | grep "audit_path_compliance.py" || echo "設定なし"
            ;;
        *)
            echo "使用方法: $0 [setup|test|remove|status]"
            echo "  setup  : crontab設定（デフォルト）"
            echo "  test   : 手動テスト実行"
            echo "  remove : crontab削除"
            echo "  status : 現在の設定確認"
            ;;
    esac
}

main "$@"