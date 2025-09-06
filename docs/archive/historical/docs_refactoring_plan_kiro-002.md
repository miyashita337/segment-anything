# ドキュメントリファクタリング計画書 KIRO-002

**作成日**: 2025-09-06  
**対象プロジェクト**: segment-anything v0.9.27+  
**前回作業**: feature/kiro-001 完了済み  
**目的**: 新規開発により再増加したドキュメントの再最適化

## 📊 **現在の状況調査結果**

### **全体状況**
feature/kiro-001での最適化後、新規開発により再びドキュメントが増加していることを確認：

#### **ファイル数の変化**
- **ルート直下Markdownファイル**: 15個（前回調査時22個から減少）
- **docsディレクトリ**: 82個のファイル（大幅に構造化済み）
- **新規追加ファイル**: 多数のfix系・kiro系Pythonファイル

#### **新規追加された問題ファイル**
```yaml
ルート直下の新規問題ファイル:
  - docs_refactoring_plan.md          # 前回の計画書（移動対象）
  - QUAL-044-002_completion_report.md # 新しい完了報告書
  - README_Phase3.md                  # Phase3関連ドキュメント
  - priority_analysis_report.md       # 分析レポート
  - lora_training_evaluation.md       # 評価レポート

新規Python修正ファイル（ルート直下）:
  - fix_kiro_dashboard_*.py (6個)     # kiro関連修正スクリプト
  - fix_dashboard_all_images_path.py  # ダッシュボード修正
  - fix_qcc021_extended.py           # QCC修正

新規ログ・設定ファイル:
  - extraction_QUAL-044-002.log      # 新しい抽出ログ
  - =0.10.0                          # 不明なファイル（削除対象？）
```

### 🏗️ **docs構造の現状**

#### **良好な点**
- ✅ 基本的なディレクトリ構造は整備済み
- ✅ `docs/tracker-workspace/KIRO-001/` でトラッカー管理実装済み
- ✅ アーカイブ・レポート分類は機能している

#### **新たな問題点**
```yaml
構造的問題:
  - docs直下に50個以上のファイルが散在
  - 新しいQUAL-044関連ファイルが適切に分類されていない
  - development_notes/ が新規作成されているが整理不十分

分類不備:
  - 技術仕様ファイルの一部がdocs直下に残存
  - 完了報告書の分散（reports/completion/ と docs直下）
  - ワークフロー関連ファイルの重複
```

## 🎯 **KIRO-002 リファクタリング方針**

### **重要な制約事項（継続）**
以下のファイルは機能的制約により**ルート直下に保持**：
- **README.md**: GitHubページで自動読み込みのため
- **CLAUDE.md**: Claude Codeが直接参照するため
- **PRINCIPLE.md**: プロジェクト核心原則として直接アクセス必要

## 🚨 **KIRO-002で対応すべき課題**

### **優先度1: ルート直下の緊急整理**
```yaml
即座移動対象:
  - docs_refactoring_plan.md → docs/archive/historical/docs_refactoring_plan_kiro-001.md
  - QUAL-044-002_completion_report.md → docs/reports/completion/
  - README_Phase3.md → docs/development/phases/
  - priority_analysis_report.md → docs/reports/analysis/
  - lora_training_evaluation.md → docs/reports/analysis/

削除検討:
  - fix_kiro_dashboard_*.py (6個) → 一時的修正スクリプト、アーカイブ後削除
  - =0.10.0 → 不明ファイル、削除

ログファイル整理:
  - extraction_QUAL-044-002.log → logs/extraction/
  - その他*.log → 適切なlogsサブディレクトリ
```

### **優先度2: docs内の再整理**
```yaml
docs直下散在ファイル整理（50個以上）:
  - 技術仕様: technical/specifications/
  - 完了報告: reports/completion/
  - 分析結果: reports/analysis/
  - ワークフロー: workflows/
  - 開発ノート: development/notes/

重複解消:
  - ワークフロー関連の重複ファイル統合
  - 完了報告書の統一的管理
  - 技術仕様の一元化

QUAL-044関連統合:
  - development_notes/qual-044-* → reports/completion/QUAL-044/
  - 関連ファイルの一箇所集約
```

### **優先度3: 新規カテゴリの検討**
```yaml
新規必要カテゴリ:
  - docs/fixes/           # 修正スクリプト・パッチ管理
  - docs/development/phases/  # Phase別ドキュメント
  - docs/development/notes/   # 開発ノート統合
  - logs/extraction/      # 抽出ログ専用
  - logs/dashboard/       # ダッシュボードログ
```

## 🏗️ **更新されたディレクトリ構造**

### **KIRO-002 新構造**
```
docs/
├── core/                    # 核心ドキュメント
│   ├── ARCHITECTURE.md     # システム設計
│   └── QUICK_START.md      # クイックスタート
│
├── development/            # 開発関連
│   ├── setup/             # 環境構築
│   ├── workflows/         # 開発プロセス
│   ├── quality/           # 品質管理
│   ├── progress/          # 進捗管理
│   ├── phases/            # 【新規】Phase別ドキュメント
│   └── notes/             # 【新規】開発ノート統合
│
├── technical/             # 技術仕様
│   ├── specifications/    # 仕様書
│   ├── api/              # API仕様
│   └── performance/      # パフォーマンス
│
├── reports/              # レポート・分析
│   ├── quality/          # 品質レポート
│   ├── completion/       # 完了報告
│   │   └── QUAL-044/     # 【新規】QUAL-044関連統合
│   └── analysis/         # 分析結果
│
├── fixes/                # 【新規】修正スクリプト・パッチ管理
│   ├── dashboard/        # ダッシュボード修正
│   ├── kiro/            # kiro関連修正
│   └── archive/         # 適用済み修正のアーカイブ
│
├── archive/              # アーカイブ
│   ├── deprecated/       # 廃止予定
│   └── historical/       # 履歴保存
│
├── tracker-system/       # トラッカー専用領域
│   ├── checklists/       # 13ステップチェックリスト
│   ├── templates/        # 統合テンプレート
│   └── workspace/        # 各トラッカー作業領域
│
└── integrations/         # 外部連携
    ├── google-sheets/    # Sheets関連
    └── github-actions/   # Actions設定

# ルート直下（移動しない）
README.md                 # GitHub自動読み込み用（保持）
CLAUDE.md                 # Claude Code参照用（保持）
PRINCIPLE.md              # プロジェクト核心原則（保持）

# 新規ログ構造
logs/
├── extraction/           # 抽出ログ
├── dashboard/           # ダッシュボードログ
└── system/             # システムログ
```

## 🛠️ **KIRO-002 実装手順**

### **Phase 1: 緊急整理（1日）**

#### **Step 1-1: 新規ディレクトリ作成（15分）**
```bash
# 新規カテゴリ作成
mkdir -p docs/fixes/{dashboard,kiro,archive}
mkdir -p docs/development/{phases,notes}
mkdir -p docs/reports/completion/QUAL-044
mkdir -p logs/{extraction,dashboard,system}
```

#### **Step 1-2: ルート直下問題ファイル移動（30分）**
```bash
# 計画書アーカイブ
mv docs_refactoring_plan.md docs/archive/historical/docs_refactoring_plan_kiro-001.md

# 完了報告書移動
mv QUAL-044-002_completion_report.md docs/reports/completion/QUAL-044/

# Phase3ドキュメント移動
mv README_Phase3.md docs/development/phases/

# 分析レポート移動
mv priority_analysis_report.md docs/reports/analysis/
mv lora_training_evaluation.md docs/reports/analysis/
```

#### **Step 1-3: 修正スクリプト整理（30分）**
```bash
# kiro関連修正スクリプト移動
mv fix_kiro_dashboard_*.py docs/fixes/kiro/
mv fix_dashboard_all_images_path.py docs/fixes/dashboard/
mv fix_qcc021_extended.py docs/fixes/archive/

# ログファイル移動
mv extraction_QUAL-044-002.log logs/extraction/
mv dashboard_server.log logs/dashboard/

# 不明ファイル削除
rm -f "=0.10.0"
```

### **Phase 2: docs内再構造化（2-3日）**

#### **Step 2-1: docs直下ファイル分類（1日）**
```yaml
技術仕様系:
  - technical_specifications.md → technical/specifications/
  - dependency_reference.md → technical/specifications/
  - CONFIG_VARIABLES_REFERENCE.md → technical/specifications/

完了報告系:
  - phase_a1_completion_report.md → reports/completion/
  - phase_a2_plan.md → reports/completion/
  - P1-017_critical_usability_improvements.md → reports/completion/

分析系:
  - improvement_phase2_progress.md → reports/analysis/
  - v043_implementation_summary.md → reports/analysis/
  - version_management_strategy.md → reports/analysis/

開発ノート系:
  - development_notes/* → development/notes/
```

#### **Step 2-2: QUAL-044関連統合（半日）**
```bash
# QUAL-044関連ファイル統合
mv docs/development_notes/qual-044-* docs/reports/completion/QUAL-044/
```

#### **Step 2-3: 重複ファイル統合（半日）**
```yaml
ワークフロー重複:
  - 類似機能のワークフローファイル統合
  - 古いバージョンのアーカイブ移動

完了報告重複:
  - 同一プロジェクトの複数報告書統合
  - 履歴として必要なもののみ保持
```

### **Phase 3: 参照関係修正・検証（1-2日）**

#### **Step 3-1: 参照リンク更新（1日）**
```bash
# 移動したファイルの参照リンク一括更新
# 例：docs_refactoring_plan.md → docs/archive/historical/docs_refactoring_plan_kiro-001.md
# 例：QUAL-044-002_completion_report.md → docs/reports/completion/QUAL-044/
```

#### **Step 3-2: ナビゲーション更新（半日）**
```markdown
# docs/README.md更新
## 🔧 開発者向け
- [環境構築](development/setup/)
- [開発ワークフロー](development/workflows/)
- [品質管理](development/quality/)
- [進捗管理](development/progress/)
- [Phase別ドキュメント](development/phases/)     # 新規追加
- [開発ノート](development/notes/)               # 新規追加

## 🛠️ 修正・パッチ管理                            # 新規セクション
- [ダッシュボード修正](fixes/dashboard/)
- [kiro関連修正](fixes/kiro/)
- [適用済みアーカイブ](fixes/archive/)
```

#### **Step 3-3: 動作確認・最終調整（半日）**
```bash
# リンク切れチェック
# ナビゲーション動作確認
# 重要ファイルのアクセス確認
```

## 📊 **期待される効果**

### **短期効果（1週間）**
- **ルート直下**: 15個 → 3個（README.md, CLAUDE.md, PRINCIPLE.md）
- **docs構造**: より明確な分類による発見性向上
- **修正スクリプト管理**: 体系的な修正履歴管理

### **中期効果（1ヶ月）**
- **保守性**: 重複解消による更新コスト削減
- **開発効率**: 必要な情報への迅速なアクセス
- **品質向上**: 修正パッチの体系的管理

### **長期効果（3ヶ月）**
- **知識継承**: 構造化された開発ノート・修正履歴
- **プロジェクト持続性**: 効率的なドキュメント管理体制
- **新規参加者支援**: 明確な情報構造による学習促進

## ⚠️ **KIRO-002 特有の注意点**

### **修正スクリプト管理**
- **一時的修正**: fixes/以下で管理、適用後はarchive/へ
- **恒久的修正**: 適切なコードベースに統合
- **履歴保持**: 修正理由・適用日時の記録

### **QUAL-044関連の統合**
- **関連ファイル集約**: 散在していたファイルを一箇所に統合
- **参照関係保持**: 他ドキュメントからの参照を維持
- **完了報告の体系化**: 類似プロジェクトとの整合性確保

### **ログファイル管理**
- **カテゴリ別分類**: extraction, dashboard, system
- **サイズ管理**: 大容量ログの定期的なアーカイブ
- **アクセス制御**: 機密情報を含むログの適切な管理

## 🎯 **成功指標**

### **定量的指標**
- **ルート直下ファイル数**: 15個 → 3個（80%削減）
- **docs直下ファイル数**: 50個 → 10個以下（80%削減）
- **重複ファイル**: 完全解消
- **参照リンク切れ**: 0件維持

### **定性的指標**
- **ドキュメント発見時間**: 60%短縮目標
- **修正履歴の追跡性**: 100%向上
- **新規開発者の理解度**: アンケート評価向上

## 📅 **KIRO-002 実行スケジュール**

### **Week 1: 緊急整理・基盤構築**
- Day 1: Phase 1完全実行（緊急整理）
- Day 2-3: Phase 2-1（docs直下ファイル分類）
- Day 4-5: Phase 2-2,2-3（統合・重複解消）

### **Week 2: 参照関係修正・検証**
- Day 1-2: Phase 3-1（参照リンク更新）
- Day 3: Phase 3-2（ナビゲーション更新）
- Day 4-5: Phase 3-3（動作確認・調整）

### **Week 3: 最終検証・運用開始**
- Day 1-2: 全体検証・微調整
- Day 3-4: ドキュメント・運用ガイド作成
- Day 5: KIRO-002完了・運用開始

## 📝 **KIRO-001からの学習事項**

### **成功要因**
- **段階的アプローチ**: リスクを最小化した段階的実装
- **バックアップ戦略**: 完全なバックアップによる安全性確保
- **トラッカーシステム保護**: 既存ワークフローの保護

### **改善点**
- **新規ファイル対策**: 継続的な監視・分類システム
- **修正スクリプト管理**: 体系的な修正履歴管理
- **自動化検討**: 定期的な構造チェック・警告システム

## 🔄 **継続的改善計画**

### **KIRO-003以降の予防策**
```yaml
監視システム:
  - ルート直下ファイル数の定期チェック
  - 新規ファイル分類の自動提案
  - 重複ファイル検出アラート

自動化検討:
  - ファイル分類の半自動化
  - 参照リンク更新の自動化
  - ドキュメント品質チェック
```

---

**作成者**: Kiro AI Assistant  
**前回作業**: feature/kiro-001 (完了)  
**対象ブランチ**: feature/kiro-002  
**実行開始予定**: ユーザー承認後即座