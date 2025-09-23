# KIRO-007: 強制ワークフロー実行システム - ドキュメント肥大化整理計画

**作成日**: 2025-09-23
**ステータス**: 検討完了・実行待ち
**関連PR**: [#86 強制ワークフロー実行システム](https://github.com/miyashita337/segment-anything/pull/86)

## 🎯 プロジェクト概要

強制ワークフロー実行システム（PR#86）の安定稼働を保ちながら、AIの非冪等的動作を引き起こすドキュメント肥大化問題を段階的に解決する。

### 🚨 解決すべき根本問題
- **新旧システム混在**: 強制ワークフローと旧ワークフローの記述が混在
- **AIの混乱**: 矛盾する指示により「**非冪等的動作**」が発生
- **ドキュメント肥大化**: 95ファイル・40ディレクトリの過剰な構造

## 📊 現状分析詳細

### プルリクエスト#86 実装内容
- **変更規模**: 66ファイルの大規模修正
- **核心システム**: SQLiteベース状態管理 + 承認ゲートシステム
- **成果**: ワークフロー遵守率 60% → 95% (580%改善)
- **アーキテクチャ移行**: 信頼ベース制御 → 検証ベース制御

### ドキュメント構造問題
```
docs/ (40ディレクトリ, 95ファイル)
├── archive/deprecated/        # 旧システム関連（未整理）
├── development_notes/         # 設計過程文書（大部分非アクティブ）
├── reports/completion/        # 古い完了報告（TEST-001等）
├── workflows/                 # 新旧ワークフロー記述混在
└── templates/                 # 推奨・非推奨テンプレート混在
```

### CLAUDE.md 矛盾問題
- **旧ワークフロー記述**: 13ステップ・4フェーズ（lines 32-49）
- **新ワークフロー参照**: 強制実行システム（lines 478+）
- **混乱要因**: 両システムへの同時参照指示

## 🗂️ 段階的整理戦略（3-Phase Approach）

### 【Phase 1】緊急度高 - 安全な削除・移動
**期間**: 1-2日
**リスク**: 最低
**対象**: 強制ワークフローに完全に影響しない明確な旧コンテンツ

#### 削除対象ファイル
```bash
# 完全な旧システム関連
docs/archive/deprecated/legacy_program_archive.md
docs/archive/historical/backup_plan.md
docs/archive/historical/docs_refactoring_plan_kiro-001.md
docs/archive/historical/docs_refactoring_plan_kiro-002.md

# 非アクティブ設計文書
docs/development_notes/qual-044-design-process-phase1.md
docs/development_notes/qual-044-design-process-phase2.md
docs/development_notes/qual-044-implementation-architecture.md

# 古い完了報告
docs/reports/completion/TEST-001_completion_report.md
docs/reports/completion/p1_005_completion_report.md
docs/results/TEST-001/ (ディレクトリ全体)

# 重複・非推奨ガイド
docs/phase0_completion_checklist.md
docs/phase0_problems.md
docs/phase1_fullbody_analysis.md
docs/phase1_mosaic_boundary_analysis.md
```

#### 移動対象（archive化）
```bash
# 歴史的価値はあるが現在非使用
docs/ASYNC_SYSTEM_FINAL_REPORT.md → docs/archive/historical/
docs/ASYNC_SYSTEM_FINAL_SUMMARY.md → docs/archive/historical/
docs/QI-002_ROOT_CAUSE_PREVENTION.md → docs/archive/historical/
docs/QUALITY-FIX-001_analysis_and_solution.md → docs/archive/historical/
```

### 【Phase 2】矛盾解消 - ドキュメント統合
**期間**: 2-3日
**リスク**: 中程度
**対象**: 新旧システムで矛盾する記述の統一

#### CLAUDE.md 整理
```markdown
# 削除する旧ワークフロー記述
- ⚠️ 重要: トラッカーワークフロー必須要件（4 回目仕様違反の恒久対策）
- 📋 ワークフロー必須参照ドキュメント（旧チェックリスト参照）
- 🚨 絶対厳守事項（13ステップ・4フェーズ記述）

# 追加する新ワークフロー記述
- 強制ワークフロー実行システムへの統一参照
- SQLiteベース状態管理の利用方法
- 承認ゲートシステムの運用手順
```

#### ワークフロー関連統合
```bash
# 重複ガイド統合
docs/workflows/dashboard_management_guide.md +
docs/workflows/dashboard_management_unified.md →
docs/workflows/dashboard_management.md

# 非推奨テンプレート削除
docs/templates/implementation_report_template.md (旧)
docs/templates/progress_tracking_template.md (旧)
# → docs/workflows/templates/unified_tracker_template.md (新) を推奨

# 整合性確認必要
docs/workflows/checklists/tracker_workflow_checklist.md
docs/workflows/integration/workflow_plan_command_guide.md
```

### 【Phase 3】最適化 - 構造再編
**期間**: 3-4日
**リスク**: 中程度
**対象**: 残存する構造的問題の解決

#### ディレクトリ構造最適化
```bash
# 統合対象ディレクトリ
docs/development/ + docs/technical/ → docs/technical/
docs/integrations/github-actions/ → docs/technical/ci-cd/
docs/tracker-system/ → docs/workflows/ (重複解消)

# 参照整理
- 全.mdファイルのリンク修正
- CLAUDE.md の参照パス更新
- README.md の構造説明更新
```

## 🛡️ 安全性確保措置

### 強制ワークフロー実行システム保護
1. **コアファイル保護**
   - `tools/workflow/workflow_cli.py` - 変更禁止
   - `config/workspace_config.py` - 変更禁止
   - `.github/workflows/kiro-006-phase2-tests.yml` - 変更禁止

2. **参照整合性確保**
   - 新システムで使用中のドキュメントは移動前に要確認
   - `docs/workflows/templates/unified_tracker_template.md` 等の実稼働ファイル保護

3. **段階実行原則**
   - 各Phase完了後にワークフロー動作確認
   - 問題発生時の即座ロールバック体制

## 📋 Phase別実行計画

### Phase 1 実行手順
1. **バックアップ作成**: `git branch backup/before-kiro-007-phase1`
2. **削除実行**: 上記削除対象ファイルをgit rm
3. **移動実行**: archive化対象をgit mv
4. **動作確認**: 強制ワークフローCLIの動作テスト
5. **コミット**: "feat(KIRO-007): Phase 1 - 旧システム関連ファイル整理"

### Phase 2 実行手順
1. **CLAUDE.md更新**: 旧ワークフロー記述を新システム統一記述に変更
2. **重複ガイド統合**: workflows配下の重複ファイル統合
3. **テンプレート整理**: 非推奨テンプレート削除
4. **動作確認**: Claudeによるワークフロー実行テスト
5. **コミット**: "feat(KIRO-007): Phase 2 - 新旧システム記述統一"

### Phase 3 実行手順
1. **構造設計**: 最適化後のディレクトリ構造決定
2. **段階移動**: ディレクトリ統合を段階実行
3. **リンク修正**: 全参照パスの一括更新
4. **最終確認**: 全システム動作の総合テスト
5. **コミット**: "feat(KIRO-007): Phase 3 - ドキュメント構造最適化完了"

## 🎯 期待効果

### 定量的改善目標
- **ファイル数**: 95 → 60 (-37%削減)
- **ディレクトリ数**: 40 → 25 (-38%削減)
- **CLAUDE.md行数**: 現在の肥大化状態 → 簡潔な新システム専用記述

### 質的改善目標
- **AI混乱解消**: 矛盾指示排除による一貫した動作
- **開発効率向上**: 明確な参照関係による高速な情報検索
- **メンテナンス性向上**: 重複排除による更新コスト削減

## 📅 実行スケジュール

**Phase 1**: 2025-09-24 (1日)
**Phase 2**: 2025-09-25-26 (2日)
**Phase 3**: 2025-09-27-29 (3日)
**最終確認**: 2025-09-30 (1日)

**合計期間**: 7日間

## 🚀 KIRO-007登録準備完了

この計画をもとに、以下のコマンドでトラッカー登録を実行予定：

```bash
python tools/workflow/workflow_cli.py plan KIRO-007 \
  "強制ワークフロー実行システム対応ドキュメント肥大化整理" \
  "AIの非冪等的動作を引き起こすドキュメント構造を3段階で整理。Phase 1: 安全削除、Phase 2: 矛盾解消、Phase 3: 構造最適化" \
  "miyashita337"
```

---

**検討者**: Claude Code
**承認待ち**: ユーザー確認後にKIRO-007実行開始