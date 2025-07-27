# 出力ディレクトリ設定

**最終更新**: 2025-07-26  
**バージョン**: 1.0

## 📋 概要

このドキュメントは、Segment Anythingプロジェクトにおける出力ディレクトリの標準構成と命名規則を定義します。すべての抽出結果、品質レポート、テスト結果は整理された形で保存されます。

## 🗂 標準ディレクトリ構造

### ベースディレクトリ
```
/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/
```

### ディレクトリ構成
```
workspace/
├── extractions/              # 抽出パイプライン結果
│   ├── phase1_test/         # Phase 1テスト結果
│   ├── baseline/            # ベースライン結果
│   ├── experiments/         # 実験的な抽出結果
│   └── production/          # 本番用抽出結果
│
├── quality_reports/         # 品質評価レポート
│   ├── unified/            # 統合品質レポート
│   │   ├── phase1/        # Phase 1用
│   │   ├── baseline/      # ベースライン用
│   │   └── daily/         # 日次レポート
│   │
│   ├── dashboards/         # ダッシュボードHTML
│   │   └── comparisons/    # 比較分析
│   │
│   └── raw/               # 生データ・中間ファイル
│
├── test_results/           # テスト結果
│   ├── unit_tests/        # 単体テスト
│   ├── integration_tests/ # 統合テスト
│   └── benchmarks/        # ベンチマーク結果
│
└── temp/                   # 一時ファイル（定期削除対象）
```

## 🛠 使用方法

### 環境変数での設定
```bash
# デフォルトの基本パス
export WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace"

# 個別設定（オプション）
export EXTRACTION_OUTPUT_BASE="${WORKSPACE_BASE}/extractions"
export QUALITY_REPORT_BASE="${WORKSPACE_BASE}/quality_reports"
export TEST_OUTPUT_BASE="${WORKSPACE_BASE}/test_results"
```

### Pythonコードでの利用
```python
import os
from pathlib import Path

# 環境変数から取得、なければデフォルト値使用
WORKSPACE_BASE = Path(os.getenv(
    'WORKSPACE_BASE', 
    '/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace'
))

# 各種出力パス
EXTRACTION_OUTPUT = WORKSPACE_BASE / 'extractions'
QUALITY_REPORT_OUTPUT = WORKSPACE_BASE / 'quality_reports'
TEST_OUTPUT = WORKSPACE_BASE / 'test_results'

# ディレクトリ作成
def ensure_output_dir(category: str, subcategory: str = None) -> Path:
    """出力ディレクトリを確保"""
    base_map = {
        'extraction': EXTRACTION_OUTPUT,
        'quality': QUALITY_REPORT_OUTPUT,
        'test': TEST_OUTPUT
    }
    
    output_dir = base_map.get(category, WORKSPACE_BASE / 'temp')
    if subcategory:
        output_dir = output_dir / subcategory
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

## 📏 命名規則

### ディレクトリ名
- 小文字とアンダースコア使用: `phase1_test`, `baseline_kana05`
- 日付を含む場合: `YYYYMMDD_description` (例: `20250726_phase1_test`)

### ファイル名
- 抽出結果: `{dataset}_{id}_extracted.{ext}` (例: `kana05_0001_extracted.jpg`)
- レポート: `{type}_report_{dataset}_{timestamp}.json`
- ダッシュボード: `dashboard_{comparison_type}_{timestamp}.html`

## 🚫 禁止事項

1. **プロジェクトルートへの直接出力禁止**
2. **test_*やresults_*などの曖昧な名前の使用禁止**
3. **workspace外への出力は原則禁止**（特別な理由がある場合は文書化）

## 🔄 マイグレーション

既存のファイルは以下のルールで移行:
- `results_phase1_test/` → `workspace/extractions/phase1_test/`
- `test_results_improved/` → `workspace/extractions/improved/`
- 散在するJSONレポート → `workspace/quality_reports/unified/`

## 📝 変更履歴

- 2025-07-26: v1.0 初版作成