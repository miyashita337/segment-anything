# ドキュメントリファクタリング計画書

**作成日**: 2025-09-01  
**対象プロジェクト**: segment-anything v0.9.27  
**目的**: ドキュメント体系の整理・構造化による保守性・アクセス性の向上

## 🎯 **現状の問題点**

### **1. 情報の散在・重複**
- ルート直下に22個のMarkdownファイルが散在
- 同じ情報が複数ファイルに重複記載
- 参照関係が複雑で情報を探しにくい

### **2. 階層構造の不明確さ**
- 重要度・用途別の分類が不十分
- 一時的なファイルと恒久的なファイルが混在
- アクセス頻度による整理がされていない

## 📋 **リファクタリング方針**

### **重要な制約事項**
以下のファイルは機能的制約により**ルート直下に保持**：
- **README.md**: GitHubページで自動読み込みのため
- **CLAUDE.md**: Claude Codeが直接参照するため
- **PRINCIPLE.md**: プロジェクト核心原則として直接アクセス必要

## 🏗️ **新ディレクトリ構造**

### **Phase 1: 用途別ディレクトリ構造**

```
docs/
├── core/                    # 核心ドキュメント（常時参照）
│   ├── ARCHITECTURE.md     # システム設計（新規作成）
│   └── QUICK_START.md      # クイックスタート（新規作成）
│
├── development/            # 開発関連
│   ├── setup/             # 環境構築
│   │   ├── installation.md
│   │   └── configuration.md
│   ├── workflows/         # 開発プロセス
│   │   ├── development_guide.md
│   │   └── testing_guide.md
│   ├── quality/           # 品質管理
│   │   ├── standards.md   # PROJECT_SETTINGS.md移動先
│   │   └── qa_process.md
│   └── progress/          # 進捗管理
│       ├── current_phase.md
│       ├── task_management.md
│       └── history.md
│
├── technical/             # 技術仕様
│   ├── specifications/    # 仕様書
│   │   └── system_spec.md # spec.md移動先
│   ├── api/              # API仕様
│   └── performance/      # パフォーマンス
│
├── reports/              # レポート・分析
│   ├── quality/          # 品質レポート
│   │   └── qc_comprehensive_report.md # QC_COMPREHENSIVE_REPORT.md移動先
│   ├── completion/       # 完了報告
│   │   ├── p1_005_completion_report.md
│   │   └── other_completion_reports.md
│   └── analysis/         # 分析結果
│       ├── gemini_competition_context.md
│       └── gpt4o_consultation_summary.md
│
└── archive/              # アーカイブ
    ├── deprecated/       # 廃止予定
    └── historical/       # 履歴保存
        └── exec_log01.md

# ルート直下（移動しない）
README.md                   # GitHub自動読み込み用（保持）
CLAUDE.md                   # Claude Code参照用（保持）
PRINCIPLE.md                # プロジェクト核心原則（保持）
```

## 📋 **ファイル移行計画**

### **移動しない（ルート直下保持）**
```yaml
保持理由付きファイル:
  - README.md              # GitHub自動読み込み
  - CLAUDE.md              # Claude Code参照
  - PRINCIPLE.md           # プロジェクト核心原則
```

### **移行対象ファイル**

#### **優先度1（即座移行）**
```yaml
技術仕様:
  - spec.md → docs/technical/specifications/system_spec.md

開発標準:
  - PROJECT_SETTINGS.md → docs/development/quality/standards.md

品質レポート:
  - QC_COMPREHENSIVE_REPORT.md → docs/reports/quality/qc_comprehensive_report.md
```

#### **優先度2（1週間以内）**
```yaml
分析・コンサルテーション:
  - gemini_competition_context.md → docs/reports/analysis/gemini_competition_context.md
  - gpt4o_consultation_summary.md → docs/reports/analysis/gpt4o_consultation_summary.md

完了報告:
  - p1_005_completion_report.md → docs/reports/completion/p1_005_completion_report.md
  - PH2-001_ROOT_CAUSE_ANALYSIS.md → docs/reports/completion/ph2_001_root_cause_analysis.md
```

#### **優先度3（2週間以内）**
```yaml
大型ファイル分離:
  - PROGRESS_TRACKER.md → 以下に分離
    - docs/development/progress/current_phase.md
    - docs/development/progress/task_management.md
    - docs/development/progress/history.md

アーカイブ移行:
  - exec_log01.md → docs/archive/historical/exec_log01.md
  - BACKUP_PLAN.md → docs/archive/historical/backup_plan.md
  - folder_structure.md → docs/archive/historical/folder_structure.md
```

## 🛠️ **実装手順**

### **Step 1: 新ディレクトリ構造作成（30分）**
```bash
# 新ディレクトリ構造作成
mkdir -p docs/{core,development/{setup,workflows,quality,progress},technical/{specifications,api,performance},reports/{quality,completion,analysis},archive/{deprecated,historical}}

# ナビゲーションファイル作成
touch docs/README.md
touch docs/core/ARCHITECTURE.md
touch docs/core/QUICK_START.md
```

### **Step 2: 優先度1ファイル移行（1時間）**
```bash
# 技術仕様移行
mv spec.md docs/technical/specifications/system_spec.md

# 開発標準移行
mv PROJECT_SETTINGS.md docs/development/quality/standards.md

# 品質レポート移行
mv QC_COMPREHENSIVE_REPORT.md docs/reports/quality/qc_comprehensive_report.md
```

### **Step 3: ナビゲーションシステム構築（1時間）**
```markdown
# docs/README.md作成内容
## 🚀 はじめに
- [システム設計](core/ARCHITECTURE.md)
- [クイックスタート](core/QUICK_START.md)

## 🔧 開発者向け
- [環境構築](development/setup/)
- [開発ワークフロー](development/workflows/)
- [品質管理](development/quality/)
- [進捗管理](development/progress/)

## 📊 技術情報
- [システム仕様](technical/specifications/)
- [API仕様](technical/api/)
- [パフォーマンス](technical/performance/)

## 📈 レポート・分析
- [品質レポート](reports/quality/)
- [完了報告](reports/completion/)
- [分析結果](reports/analysis/)
```

### **Step 4: README.md更新（30分）**
```markdown
# README.md末尾に追加
## 📚 ドキュメント

詳細なドキュメントは以下を参照してください：
- **[ドキュメントハブ](docs/)** - 全ドキュメントの索引
- **[開発原則](PRINCIPLE.md)** - プロジェクトの基本原則
- **[AI協働ガイド](CLAUDE.md)** - Claude Code連携情報
```

### **Step 5: 参照リンク更新（2時間）**
```bash
# 全ファイルの参照リンク更新
# 例：spec.md → docs/technical/specifications/system_spec.md
# 例：PROJECT_SETTINGS.md → docs/development/quality/standards.md
```

## 📊 **期待される効果**

### **短期効果（1-2週間）**
- **ドキュメント発見性の向上**: 用途別分類により目的のファイルを素早く発見
- **情報重複の削減**: 参照ベースシステムにより重複情報を削減
- **新規参加者の理解促進**: 構造化されたドキュメント体系により学習コスト削減

### **中期効果（1-2ヶ月）**
- **保守コストの削減**: 情報の一元化により更新作業を効率化
- **情報の一貫性向上**: 参照ベースにより情報の矛盾を防止
- **開発効率の向上**: 必要な情報への迅速なアクセス

### **長期効果（3ヶ月以上）**
- **知識の体系化**: プロジェクト知識の構造化・継承
- **プロジェクトの持続可能性向上**: ドキュメント管理の効率化
- **品質管理の効率化**: 品質関連情報の集約・管理

## ⚠️ **注意点・リスク対策**

### **移行時のリスク**
- **既存の参照リンク切れ**: 他ファイルからの参照が無効になる可能性
- **情報の一時的な混乱**: 移行期間中の情報アクセス困難
- **開発作業への影響**: ドキュメント参照の一時的な支障

### **対策**
- **段階的移行**: 優先度別の段階的移行により影響を最小化
- **旧ファイルの一定期間保持**: 移行完了まで旧ファイルを保持
- **移行ガイドの作成**: 移行内容を明記したガイド作成
- **参照リンクの一括更新**: 移行後の参照リンク一括修正

## 🎯 **成功指標**

### **定量的指標**
- **ドキュメント発見時間**: 50%短縮目標
- **重複情報**: 70%削減目標
- **参照リンク切れ**: 0件維持

### **定性的指標**
- **新規参加者の理解度**: アンケート評価向上
- **開発者の満足度**: ドキュメント使用体験向上
- **保守作業効率**: ドキュメント更新作業時間短縮

## 📅 **実行スケジュール**

### **Week 1: 基盤構築**
- Day 1: 新ディレクトリ構造作成
- Day 2-3: 優先度1ファイル移行
- Day 4-5: ナビゲーションシステム構築

### **Week 2: 本格移行**
- Day 1-3: 優先度2ファイル移行
- Day 4-5: 参照リンク更新・検証

### **Week 3: 完了・検証**
- Day 1-2: 優先度3ファイル移行
- Day 3-4: 全体検証・微調整
- Day 5: 移行完了・ドキュメント公開

---

**作成者**: Kiro AI Assistant  
**承認者**: [ユーザー承認待ち]  
**実行開始予定**: 承認後即座