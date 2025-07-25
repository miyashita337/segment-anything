# バージョン管理戦略

**作成日**: 2025-07-24  
**対象**: segment-anything プロジェクト全体

## 🎯 現在の状況分析

### 現在のバージョン状態
```yaml
current_status:
  spec.md_version: "v0.4.0"
  project_phase: "客観的評価システム移行期"  
  major_change: "主観的評価 → 客観的3指標システム"
  system_name: "SAMベースキャラクター抽出システム"
```

## 📊 バージョニング方針

### 推奨：v1.0.0 への移行

**理由**：
1. **システム成熟度**: 基本機能が安定稼働
2. **評価システム確立**: 客観的3指標システムの導入
3. **プロダクション対応**: 実用レベルの品質達成

```yaml
version_strategy:
  next_version: "v1.0.0"
  naming_convention: "セマンティックバージョニング"
  
  version_meaning:
    major: "1" # 基本システム完成（SAM+YOLO+客観評価）
    minor: "0" # 客観的評価システム初期版
    patch: "0" # 初回リリース
```

### バージョン進行計画

#### Phase 1: v1.0.x系列（基盤確立期）
```yaml
v1.0.0:
  target_date: "2025-08-07"
  main_features:
    - "客観的3指標システム（PLA/SCI/PLE）"
    - "自動品質評価パイプライン"
    - "基本マイルストーン追跡"
  quality_targets:
    pla_mean: 0.75
    sci_mean: 0.70
    ple_minimum: 0.10

v1.1.0:
  target_date: "2025-08-21"
  main_features:
    - "詳細統計レポート機能"
    - "アラートシステム実装"
    - "週次トレンド分析"
  
v1.2.0:
  target_date: "2025-09-04"
  main_features:
    - "マイルストーン自動追跡"
    - "継続改善サイクル自動化"
    - "パフォーマンス最適化"
```

#### Phase 2: v2.0.x系列（高度化期）
```yaml
v2.0.0:
  target_date: "2025-09-25"
  main_features:
    - "Claude風統合推論システム"
    - "多層特徴抽出エンジン"
    - "適応的品質調整機能"
  quality_targets:
    pla_mean: 0.85
    sci_mean: 0.80
    ple_minimum: 0.15
```

## 🏷️ プロジェクト名称の明確化

### 正式名称
```yaml
project_identity:
  official_name: "SAM-based Character Extraction System"
  japanese_name: "SAMベースキャラクター抽出システム"
  abbreviation: "SCES"
  
  clarification:
    not_sam_system: "SAMシステムではない"
    but_uses_sam: "SAMを活用するシステム"
    primary_purpose: "LoRA学習用画像生成"
```

### システム定義
```yaml
system_definition:
  core_technology: "Meta's Segment Anything Model (SAM)"
  additional_components:
    - "Ultralytics YOLO (物体検出)"
    - "OpenCV (画像処理)"
    - "MediaPipe (姿勢推定)"
    - "独自客観評価システム"
  
  system_type: "複合システム"
  classification: "SAM活用システム"
```

## 📁 リポジトリ構造とバージョン管理

### ファイルのバージョン表記統一
```yaml
file_versioning:
  spec.md: "v1.0.0 (2025-08-07更新予定)"
  PRINCIPLE.md: "バージョン番号なし（普遍的原則）"
  universal_specifications.md: "バージョン番号なし（不変仕様）"
  
  workflow_docs:
    automated_evaluation_framework.md: "v2.0 (客観評価版)"
    quality_evaluation_guide.md: "v2.0 (客観評価版)"
    PROGRESS_TRACKER.md: "v1.0 (新規作成)"
```

### 後方互換性管理
```yaml
compatibility_management:
  deprecated_files:
    location: "deprecated/"
    naming: "[original_name]_v[old_version]_deprecated.md"
    retention: "1年間保持"
  
  migration_guides:
    location: "docs/migration/"
    content: "旧版から新版への移行手順"
    
  breaking_changes:
    documentation: "CHANGELOG.md"
    notification: "事前1週間通知"
```

## 🚀 バージョンアップ実行計画

### Step 1: v1.0.0準備（即座実行）
```bash
# 現在のspec.mdを更新
sed -i 's/v0.4.0/v1.0.0/' /mnt/c/AItools/segment-anything/spec.md

# CHANGELOGの作成
cat > /mnt/c/AItools/segment-anything/CHANGELOG.md << EOF
# Changelog

## [v1.0.0] - 2025-08-07
### Added
- 客観的3指標評価システム（PLA/SCI/PLE）
- 自動品質評価パイプライン
- マイルストーン追跡システム

### Changed
- 主観的評価から客観的数値評価に全面移行
- evaluation frameworkの根本的書き換え

### Deprecated
- 旧主観的評価システム（v0.4.0まで）

## [v0.4.0] - 2025-07-21
### Added
- 基本SAM+YOLO統合システム
- バッチ処理機能
- 基本品質評価
EOF
```

### Step 2: 廃止予定ファイルの整理
```bash
# deprecatedディレクトリの作成
mkdir -p /mnt/c/AItools/segment-anything/deprecated

# 旧版ファイルの移動（必要に応じて）
# 現在は新システムなので移動対象なし
```

### Step 3: バージョン情報の統一
```bash
# 全ドキュメントのバージョン表記確認・更新
find /mnt/c/AItools/segment-anything/docs -name "*.md" -exec grep -l "v0\." {} \;
# 発見されたファイルを順次更新
```

## 📋 日次タスクでのバージョン管理

### 開発者の日次確認事項
```yaml
daily_version_tasks:
  version_consistency_check:
    - "spec.mdのバージョンが最新か？"
    - "新機能がCHANGELOGに記録されているか？"
    
  compatibility_verification:
    - "普遍的仕様に準拠しているか？"
    - "破壊的変更がないか？"
    
  documentation_sync:
    - "ドキュメントバージョンが実装と一致するか？"
    - "移行ガイドが必要な変更があるか？"
```

### 週次バージョン確認
```yaml
weekly_version_reviews:
  milestone_progress:
    - "現在のバージョンでマイルストーン達成可能か？"
    - "次バージョンの計画に変更が必要か？"
    
  quality_metrics_review:
    - "品質目標がバージョン計画と整合しているか？"
    - "パフォーマンス指標が期待値内か？"
```

## 🔄 継続的バージョン管理

### 自動化可能な管理項目
```python
# バージョン管理自動化スクリプト（実装推奨）
def check_version_consistency():
    """バージョン整合性の自動チェック"""
    spec_version = extract_version_from_spec()
    changelog_version = extract_latest_changelog_version()
    
    if spec_version != changelog_version:
        alert("バージョン不整合検出")
    
def update_all_version_references(new_version):
    """全ファイルのバージョン参照を更新"""
    files_to_update = [
        "spec.md",
        "README.md", 
        "docs/workflows/*.md"
    ]
    
    for file_path in files_to_update:
        update_version_in_file(file_path, new_version)
```

---

**結論**: v1.0.0への移行を推奨します。現在のシステムは客観的評価システムの導入により、実用レベルの成熟度に達しています。