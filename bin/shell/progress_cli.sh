#!/bin/bash
"""
進捗管理CLI統一実行スクリプト
Google API接続問題の自動診断・修復機能付き
"""

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# プロジェクトルート検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAM_ENV_PYTHON="$PROJECT_ROOT/sam-env/bin/python3"
PROGRESS_CLI="$PROJECT_ROOT/tools/progress_tracker/cli.py"
HEALTH_CHECKER="$PROJECT_ROOT/tools/utils/google_api_health_check.py"

echo -e "${BLUE}=== 進捗管理CLI (自動診断付き) ===${NC}"

# 引数チェック
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}使用方法:${NC}"
    echo "  $0 status                    # 進捗状況確認"
    echo "  $0 create TRACKER_ID --description \"説明\"  # 新規タスク作成"
    echo "  $0 update TRACKER_ID STATUS   # ステータス更新"
    echo "  $0 list                      # タスク一覧"
    echo "  $0 health-check              # Google API接続診断"
    echo ""
    echo -e "${BLUE}例:${NC}"
    echo "  $0 create QI-003 --description \"新機能実装\""
    echo "  $0 update QI-003 \"/release\""
    exit 1
fi

# 特別コマンド: ヘルスチェック
if [ "$1" = "health-check" ]; then
    echo -e "${BLUE}Google API接続診断を実行中...${NC}"
    
    if [ ! -f "$HEALTH_CHECKER" ]; then
        echo -e "${RED}❌ ヘルスチェッカーが見つかりません: $HEALTH_CHECKER${NC}"
        exit 1
    fi
    
    # sam-env環境で実行
    if [ -f "$SAM_ENV_PYTHON" ]; then
        "$SAM_ENV_PYTHON" "$HEALTH_CHECKER" --auto-fix
    else
        echo -e "${RED}❌ sam-env環境が見つかりません${NC}"
        exit 1
    fi
    exit 0
fi

# sam-env環境確認
if [ ! -f "$SAM_ENV_PYTHON" ]; then
    echo -e "${RED}❌ sam-env環境が見つかりません: $SAM_ENV_PYTHON${NC}"
    echo "以下のコマンドでsam-env環境を作成してください:"
    echo "  cd $PROJECT_ROOT && python3 -m venv sam-env"
    exit 1
fi

# 進捗管理CLI存在確認
if [ ! -f "$PROGRESS_CLI" ]; then
    echo -e "${RED}❌ 進捗管理CLIが見つかりません: $PROGRESS_CLI${NC}"
    exit 1
fi

# Google API接続の事前チェック（軽量版）
echo -e "${BLUE}Google API接続確認中...${NC}"
CONNECTION_CHECK_RESULT=$("$SAM_ENV_PYTHON" -c "
import sys
sys.path.append('$PROJECT_ROOT')
try:
    from tools.progress_tracker.sheets_client import GoogleSheetsClient
    print('CONNECTION_OK')
except ImportError as e:
    print('IMPORT_ERROR')
    print(str(e))
except Exception as e:
    print('OTHER_ERROR')
    print(str(e))
" 2>&1)

if echo "$CONNECTION_CHECK_RESULT" | grep -q "IMPORT_ERROR"; then
    echo -e "${YELLOW}⚠️ Google APIライブラリが不足しています${NC}"
    echo -e "${BLUE}自動修復を試行中...${NC}"
    
    # 自動修復実行
    if [ -f "$HEALTH_CHECKER" ]; then
        "$SAM_ENV_PYTHON" "$HEALTH_CHECKER" --auto-fix --quiet
        
        # 再チェック
        CONNECTION_RECHECK=$("$SAM_ENV_PYTHON" -c "
import sys
sys.path.append('$PROJECT_ROOT')
try:
    from tools.progress_tracker.sheets_client import GoogleSheetsClient
    print('CONNECTION_OK')
except Exception as e:
    print('STILL_ERROR')
" 2>&1)
        
        if echo "$CONNECTION_RECHECK" | grep -q "CONNECTION_OK"; then
            echo -e "${GREEN}✅ 自動修復成功${NC}"
        else
            echo -e "${RED}❌ 自動修復失敗 - 手動でヘルスチェックを実行してください:${NC}"
            echo "  $0 health-check"
            exit 1
        fi
    else
        echo -e "${RED}❌ 自動修復ツールが見つかりません${NC}"
        exit 1
    fi
elif echo "$CONNECTION_CHECK_RESULT" | grep -q "OTHER_ERROR"; then
    echo -e "${YELLOW}⚠️ Google API接続に問題があります${NC}"
    echo -e "${BLUE}詳細診断を実行してください: $0 health-check${NC}"
else
    echo -e "${GREEN}✅ Google API接続正常${NC}"
fi

# 環境変数設定
export PROGRESS_TRACKER_SHEET_NAME="シート1"

# 進捗管理CLI実行
echo -e "${BLUE}進捗管理CLI実行中...${NC}"
echo "コマンド: $SAM_ENV_PYTHON $PROGRESS_CLI $*"

"$SAM_ENV_PYTHON" "$PROGRESS_CLI" "$@"

# 実行結果確認
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 実行完了${NC}"
else
    echo -e "${RED}❌ 実行エラー (終了コード: $EXIT_CODE)${NC}"
    echo -e "${BLUE}問題が続く場合、ヘルスチェックを実行してください: $0 health-check${NC}"
fi

exit $EXIT_CODE