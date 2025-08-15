#!/bin/bash

# segment-anything環境確認・自動切り替えスクリプト
# 用途: セッション継続時の環境管理不備を防止
# 作成日: 2025-08-02

set -e

echo "🔍 segment-anything環境確認開始..."

# 現在のディレクトリ確認
if [ "$PWD" = "/mnt/c/AItools/segment-anything" ]; then
    echo "✅ 正しいプロジェクトディレクトリです: $PWD"
    
    # 仮想環境確認・切り替え
    if [[ "$VIRTUAL_ENV" != *"sam-env"* ]]; then
        echo "⚠️  警告: sam-env環境が有効化されていません"
        echo "現在の環境: ${VIRTUAL_ENV:-'(なし)'}"
        echo "🔄 sam-env環境に切り替え中..."
        
        # Windows/Linux両対応
        if [ -f "sam-env/bin/activate" ]; then
            source sam-env/bin/activate
            echo "✅ Windows版sam-env環境を有効化しました"
        elif [ -f "sam-env/bin/activate" ]; then
            source sam-env/bin/activate
            echo "✅ Linux版sam-env環境を有効化しました"
        else
            echo "❌ エラー: sam-env環境が見つかりません"
            echo "sam-envディレクトリを確認してください"
            exit 1
        fi
    else
        echo "✅ sam-env環境が既に有効化されています"
    fi
    
    # 環境情報表示
    echo "📋 現在の環境情報:"
    echo "   仮想環境: $VIRTUAL_ENV"
    echo "   Python: $(which python)"
    
    # 必須パッケージ確認
    echo "🔍 必須パッケージ確認中..."
    
    # PyTorch確認
    if python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        echo "   ✅ PyTorch: OK"
        
        # CUDA確認
        if python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}')" 2>/dev/null; then
            echo "   ✅ CUDA: OK"
        else
            echo "   ⚠️  CUDA: 確認できません"
        fi
    else
        echo "   ❌ PyTorch: インストールされていません"
    fi
    
    # Google Auth確認
    if python -c "import google.auth; print('Google Auth: OK')" 2>/dev/null; then
        echo "   ✅ Google Auth: OK"
    else
        echo "   ❌ Google Auth: インストールされていません"
        echo "      pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    fi
    
    # Ultralytics確認
    if python -c "import ultralytics; print('Ultralytics: OK')" 2>/dev/null; then
        echo "   ✅ Ultralytics: OK"
    else
        echo "   ❌ Ultralytics: インストールされていません"
    fi
    
    echo "🎉 環境確認完了"
    
else
    echo "❌ 警告: segment-anythingディレクトリではありません"
    echo "現在のディレクトリ: $PWD"
    echo "正しいディレクトリ: /mnt/c/AItools/segment-anything"
fi

echo ""
echo "💡 使用方法:"
echo "   source bin/shell/check_env.sh"
echo "   または"
echo "   ./bin/shell/check_env.sh"