# ドキュメントリファクタリング移行マッピング

**作成日**: 2025-09-03  
**KIRO-001**: ドキュメント構造最適化プロジェクト

## 📋 移行方針

### 絶対移動禁止（ルート維持）
- `README.md` - GitHub自動読み込み
- `CLAUDE.md` - Claude Code直接参照  
- `PRINCIPLE.md` - プロジェクト核心原則

### 慎重検討必要（トラッカーシステム関連）
- `docs/checklists/tracker_workflow_checklist.md` - 参照10箇所
- `docs/templates/unified_tracker_template.md` - 参照3箇所
- `docs/google_sheets_reference.md` - Google Sheets連携

## 🗂️ Phase 1: 低リスクファイル移行（Week 1）

### reports/完了報告書移行
| 現在のパス | 移行先 | 参照数 | リスク |
|-----------|--------|--------|--------|
| `docs/results/TEST-001/completion_report.md` | `docs/reports/completion/TEST-001_completion_report.md` | 0 | 低 |
| `p1_005_completion_report.md` | `docs/reports/completion/p1_005_completion_report.md` | 0 | 低 |
| `PH2-001_ROOT_CAUSE_ANALYSIS.md` | `docs/reports/completion/PH2-001_root_cause_analysis.md` | 0 | 低 |

### archive/過去ファイル移行
| 現在のパス | 移行先 | 参照数 | リスク |
|-----------|--------|--------|--------|
| `exec_log01.md` | `docs/archive/historical/exec_log01.md` | 0 | 低 |
| `BACKUP_PLAN.md` | `docs/archive/historical/backup_plan.md` | 2 | 低 |
| `folder_structure.md` | `docs/archive/historical/folder_structure.md` | 0 | 低 |
| `docs/LEGACY_PROGRAM_ARCHIVE.md` | `docs/archive/deprecated/legacy_program_archive.md` | 0 | 低 |

### analysis/分析レポート移行
| 現在のパス | 移行先 | 参照数 | リスク |
|-----------|--------|--------|--------|
| `gemini_competition_context.md` | `docs/reports/analysis/gemini_competition_context.md` | 0 | 低 |
| `gpt4o_consultation_summary.md` | `docs/reports/analysis/gpt4o_consultation_summary.md` | 0 | 低 |

## 🗂️ Phase 2: 中リスクファイル移行（Week 2）

### technical/仕様書移行
| 現在のパス | 移行先 | 参照数 | リスク |
|-----------|--------|--------|--------|
| `docs/technical/specifications/system_spec.md` | `docs/technical/specifications/system_docs/technical/specifications/system_spec.md` | 15+ | 中 |
| `docs/technical_specifications.md` | `docs/technical/specifications/technical_specifications.md` | 8 | 中 |
| `docs/objective_metrics_specification.md` | `docs/technical/specifications/objective_metrics_docs/technical/specifications/system_spec.md` | 2 | 中 |

### development/品質関連移行
| 現在のパス | 移行先 | 参照数 | リスク |
|-----------|--------|--------|--------|
| `docs/development/quality/standards.md` | `docs/development/quality/standards.md` | 5 | 中 |
| `docs/reports/quality/qc_comprehensive_report.md` | `docs/reports/quality/qc_comprehensive_report.md` | 3 | 中 |
| `docs/QUALITY-FIX-001_analysis_and_solution.md` | `docs/reports/quality/quality_fix_001_analysis.md` | 1 | 中 |

## 🗂️ Phase 3: 高リスクファイル（特別配慮）

### トラッカーシステム専用領域（新規作成）
```
docs/
├── tracker-system/              # トラッカー専用領域
│   ├── checklists/             # 13ステップチェックリスト
│   │   └── tracker_workflow_checklist.md
│   ├── templates/              # 統合テンプレート
│   │   └── unified_tracker_template.md
│   └── workspace/              # 各トラッカーの作業領域
│       └── KIRO-001/
│           └── SOW.md
```

### Google Sheets連携領域（新規作成）
```
docs/
├── integrations/               # 外部連携
│   ├── google-sheets/         # Sheets関連
│   │   ├── google_sheets_reference.md
│   │   └── google_sheets_setup.md
│   └── github-actions/        # Actions設定
│       └── github_actions_reference.md
```

## 📊 影響分析サマリー

### 参照数統計
- 総参照数: 656箇所
- 高リスクファイル（10箇所以上）: 3ファイル
- 中リスクファイル（3-9箇所）: 8ファイル
- 低リスクファイル（0-2箇所）: 70+ファイル

### リスク評価
- **Phase 1（低リスク）**: 即座に移行可能、影響最小
- **Phase 2（中リスク）**: 参照更新必要、慎重に実施
- **Phase 3（高リスク）**: 特別配慮必要、専用領域作成推奨

## 🔧 実装手順

### Step 1: ディレクトリ構造作成
```bash
# 新ディレクトリ構造作成
mkdir -p docs/{core,development/{setup,workflows,quality,progress}}
mkdir -p docs/{technical/{specifications,api,performance}}
mkdir -p docs/{reports/{quality,completion,analysis}}
mkdir -p docs/{archive/{deprecated,historical}}
mkdir -p docs/{tracker-system/{checklists,templates,workspace}}
mkdir -p docs/{integrations/{google-sheets,github-actions}}
```

### Step 2: 参照更新スクリプト
```python
# tools/update_references.py として作成予定
import re
import os

MAPPING = {
    "docs/technical/specifications/system_spec.md": "docs/technical/specifications/system_docs/technical/specifications/system_spec.md",
    "docs/development/quality/standards.md": "docs/development/quality/standards.md",
    # ... 他のマッピング
}

def update_references(file_path, mapping):
    """ファイル内の参照を更新"""
    # 実装詳細は別途作成
    pass
```

## 📅 実行スケジュール

| フェーズ | 期間 | 対象ファイル数 | リスクレベル |
|---------|------|---------------|-------------|
| Phase 1 | Day 1-3 | 10ファイル | 低 |
| Phase 2 | Day 4-7 | 15ファイル | 中 |
| Phase 3 | Day 8-10 | 5ファイル | 高 |
| 検証 | Day 11-12 | 全体 | - |

## ✅ 成功条件

1. **参照リンク切れ**: 0件
2. **トラッカーワークフロー継続性**: 100%維持
3. **ドキュメント発見時間**: 50%短縮
4. **重複情報**: 70%削減

---

**承認者**: [ユーザー承認待ち]  
**実行開始**: 承認後即座