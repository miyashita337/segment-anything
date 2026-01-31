# /test-first - TDDガイドライン

## 概要

Test-Driven Development (TDD) に基づく実装手順。

## Red-Green-Refactorサイクル

### 1. Red: 失敗するテストを書く

実装前に、期待する動作を定義するテストを作成せよ:

```python
# tests/unit/test_新機能.py
import pytest

def test_新機能_正常系():
    """期待する動作を記述"""
    result = target_function(input_data)
    assert result == expected_output

def test_新機能_異常系():
    """エラーケースを記述"""
    with pytest.raises(ValueError):
        target_function(invalid_input)
```

**テスト実行で失敗を確認**:
```bash
pytest tests/unit/test_新機能.py -v
```

### 2. Green: テストを通す最小限の実装

テストが通る最小限のコードを実装せよ:

```bash
# テスト実行
pytest tests/unit/test_新機能.py -v
```

**原則**: 余計な機能を追加しない（YAGNI）

### 3. Refactor: コードを整理

テストが通った状態で、コードを改善せよ:

- 重複の排除
- 命名の改善
- 構造の整理

**リファクタリング後もテストが通ることを確認**:
```bash
pytest tests/unit/test_新機能.py -v
```

## テスト配置規則

```
tests/
├── unit/           # 単体テスト（単一モジュールのみ）
│   └── test_xxx.py
├── integration/    # 統合テスト（複数モジュール連携）
│   └── test_xxx_integration.py
├── workflow/       # ワークフローテスト
│   └── test_xxx_workflow.py
└── conftest.py     # 共通フィクスチャ
```

**命名規則**:
- 単体テスト: `test_{機能名}.py`
- 統合テスト: `test_{機能名}_integration.py`
- テスト関数: `test_{対象}_{条件}_{期待結果}()`

## テスト実行コマンド

```bash
# 全テスト実行
pytest tests/ -v

# 単体テストのみ
pytest tests/unit/ -v

# 統合テストのみ
pytest tests/integration/ -v

# 特定ファイル
pytest tests/unit/test_xxx.py -v

# 特定テスト関数
pytest tests/unit/test_xxx.py::test_function_name -v

# カバレッジ付き
pytest tests/ --cov=features --cov-report=term-missing

# 失敗時に停止
pytest tests/ -x

# 詳細出力
pytest tests/ -v --tb=short
```

## Linter実行

テスト作成後、コード品質チェックを実行せよ:

```bash
./bin/shell/linter.sh
```

## テスト作成テンプレート

```python
"""
{機能名}のテスト

対象: {対象モジュール/クラス/関数}
"""
import pytest
from features.xxx import TargetClass


class TestTargetClass:
    """TargetClassのテスト"""

    def test_正常系_基本動作(self):
        """基本的な動作を確認"""
        target = TargetClass()
        result = target.method(valid_input)
        assert result == expected

    def test_正常系_境界値(self):
        """境界値での動作を確認"""
        target = TargetClass()
        result = target.method(boundary_input)
        assert result == expected

    def test_異常系_無効入力(self):
        """無効入力でのエラーを確認"""
        target = TargetClass()
        with pytest.raises(ValueError, match="expected error message"):
            target.method(invalid_input)

    def test_異常系_None入力(self):
        """None入力でのエラーを確認"""
        target = TargetClass()
        with pytest.raises(TypeError):
            target.method(None)
```

## 関連Skill

- `/pre-implementation-check`: 実装前の総合チェック
- `/impact-analysis`: テスト対象の影響範囲確認
