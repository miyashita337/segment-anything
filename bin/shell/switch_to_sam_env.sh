#!/bin/bash
"""
SAM環境切り替えスクリプト
Google API接続問題の予防と環境統一
"""

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# プロジェクトルート検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAM_ENV_PATH="$PROJECT_ROOT/sam-env"

echo -e "${BLUE}=== SAM環境切り替えスクリプト ===${NC}"
echo -e "プロジェクトルート: $PROJECT_ROOT"

# 現在の環境確認
echo -e "\n${YELLOW}現在の環境確認:${NC}"
echo "Python実行ファイル: $(which python3)"
echo "Python環境: $VIRTUAL_ENV"

if [[ "$VIRTUAL_ENV" == *"sam-env"* ]]; then
    echo -e "${GREEN}✅ 既にsam-env環境です${NC}"
    exit 0
elif [[ "$VIRTUAL_ENV" == *"serena"* ]] || [[ "$VIRTUAL_ENV" == *"mcp"* ]]; then
    echo -e "${YELLOW}⚠️ Serena/MCP環境を検出 - sam-envに切り替えます${NC}"
else
    echo -e "${YELLOW}⚠️ 不明な環境 - sam-envに切り替えます${NC}"
fi

# sam-env存在確認
if [ ! -d "$SAM_ENV_PATH" ]; then
    echo -e "${RED}❌ sam-env環境が見つかりません: $SAM_ENV_PATH${NC}"
    echo "以下のコマンドでsam-env環境を作成してください:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv sam-env"
    echo "  source sam-env/bin/activate"
    echo "  pip install -e ."
    exit 1
fi

# 環境切り替え実行
echo -e "\n${BLUE}sam-env環境に切り替え中...${NC}"

# 切り替えコマンド生成
ACTIVATE_CMD="source $SAM_ENV_PATH/bin/activate"

# 現在のシェル検出
if [ -n "$BASH_VERSION" ]; then
    SHELL_TYPE="bash"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_TYPE="zsh"
else
    SHELL_TYPE="sh"
fi

echo -e "${YELLOW}実行してください:${NC}"
echo -e "${GREEN}$ACTIVATE_CMD${NC}"

# 自動切り替え（可能な場合）
if [ "$1" = "--auto" ] || [ "$1" = "-a" ]; then
    echo -e "\n${BLUE}自動切り替えモード${NC}"
    eval "$ACTIVATE_CMD"
    
    echo -e "${GREEN}✅ sam-env環境に切り替えました${NC}"
    echo "新しいPython実行ファイル: $(which python3)"
    
    # Google APIヘルスチェック実行
    if [ -f "$PROJECT_ROOT/tools/utils/google_api_health_check.py" ]; then
        echo -e "\n${BLUE}Google APIヘルスチェック実行中...${NC}"
        python3 "$PROJECT_ROOT/tools/utils/google_api_health_check.py" --auto-fix
    fi
else
    echo -e "\n${YELLOW}手動切り替えの場合、以下をコピー&ペーストしてください:${NC}"
    echo -e "${GREEN}$ACTIVATE_CMD${NC}"
    echo -e "\n自動切り替えの場合: $0 --auto"
fi

echo -e "\n${BLUE}=== 完了 ===${NC}"