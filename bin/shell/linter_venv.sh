#!/bin/bash
# 仮想環境を使用したlinterスクリプト

# 仮想環境のPythonを使用（WSL/Windows対応）
PYTHON="sam-env/bin/python3"

echo "🐍 Using Python: $PYTHON"

# バージョンチェック
{
  $PYTHON -m black --version | grep -E "23\." > /dev/null
} || {
  echo "Linter requires 'black==23.*' !"
  exit 1
}

ISORT_VERSION=$($PYTHON -m isort --version-number)
if [[ "$ISORT_VERSION" != 5.12* ]]; then
  echo "Linter requires isort==5.12.0 !"
  exit 1
fi

echo "Running isort ..."
$PYTHON -m isort . --atomic

echo "Running black ..."
$PYTHON -m black -l 100 .

echo "Running flake8 ..."
$PYTHON -m flake8 .

echo "Running mypy..."
$PYTHON -m mypy --exclude 'setup.py|notebooks' .

echo "✅ Linting complete!"