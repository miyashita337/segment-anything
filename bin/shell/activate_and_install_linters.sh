#!/bin/bash
# Linter依存関係のインストールスクリプト

echo "🐍 仮想環境の有効化..."

# Windowsの場合
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]; then
    source sam-env/bin/activate
else
    source sam-env/bin/activate
fi

echo "✅ 仮想環境有効化完了: $(which python)"

echo "📦 Linter依存関係のインストール..."
pip install black==23.* isort==5.12.0 mypy flake8

echo "🔍 インストール確認..."
echo "Black: $(black --version)"
echo "isort: $(isort --version)"
echo "mypy: $(mypy --version)"
echo "flake8: $(flake8 --version)"

echo "✅ 完了！"