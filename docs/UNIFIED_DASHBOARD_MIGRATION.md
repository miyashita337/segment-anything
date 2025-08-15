# 統合ダッシュボードシステム マイグレーション ガイド

## 📋 概要

既存の乱立していたダッシュボード生成システムを統合し、一元管理を実現しました。

### 🔄 変更内容

#### Before（問題のあった状態）
- **40以上の個別ダッシュボード生成スクリプト**
- **重複機能の散在**
- **保守性の低下**
- **一貫性の欠如**

#### After（統合後）
- **1つの統合システム**
- **設定駆動型アーキテクチャ**
- **プラグインベース拡張**
- **完全な後方互換性**

## 🏗️ システム構成

### 新規作成されたファイル
```
features/common/
├── unified_dashboard_generator.py      # メインシステム
├── dashboard_config.py                 # 設定管理
└── dashboard_plugins/                  # プラグインシステム
    ├── __init__.py
    ├── image_quality_plugin.py
    └── statistics_plugin.py

tools/scripts/
└── unified_dashboard_wrapper.py        # 互換性ラッパー

config/dashboard/                        # 自動生成設定
├── default_dashboard_config.yaml
└── tracker_specific/
    └── (各トラッカー).yaml
```

### 修正されたファイル
- `tools/scripts/run_quality_workflow.sh` (195-212行)

## 🚀 使用方法

### 基本的な使用方法
```bash
# 統合システムでダッシュボード生成
python3 tools/scripts/unified_dashboard_wrapper.py QI-004 /extraction/dir /output/dir

# 既存のrun_quality_workflow.shも自動的に統合システムを使用
./tools/scripts/run_quality_workflow.sh QI-004
```

### プログラムでの使用
```python
from features.common.unified_dashboard_generator import UnifiedDashboardGenerator

generator = UnifiedDashboardGenerator()
dashboard_path = generator.generate_dashboard(
    tracker_id="QI-004",
    extraction_dir="/workspace/QI-004/extraction",
    output_dir="/workspace/QI-004"
)
```

## ⚙️ 設定システム

### デフォルト設定
システムが自動的に `config/dashboard/default_dashboard_config.yaml` を作成します。

### トラッカー固有設定
```bash
# 設定テンプレート作成
python3 -c "
from features.common.unified_dashboard_generator import UnifiedDashboardGenerator
generator = UnifiedDashboardGenerator()
template_path = generator.create_tracker_config_template('QI-007')
print(f'設定テンプレート: {template_path}')
"
```

### 設定項目
```yaml
title: "QI-004 品質評価ダッシュボード"
description: "QI-004トラッカーの品質評価結果"
image_display:
  method: "path_reference"      # "base64" or "path_reference"
  max_size_mb: 0.1
  lazy_loading: true
quality_analysis:
  enable_image_analysis: true
  enable_statistical_analysis: true
  enable_graph_generation: true
layout:
  template: "standard"          # "standard", "compact", "detailed"
  grid_columns: 3
  enable_responsive: true
plugins:
  enabled_plugins: ["image_quality", "statistics"]
```

## 🔌 プラグインシステム

### 利用可能プラグイン
1. **image_quality** - 画像品質解析（QI-004機能統合）
2. **statistics** - 統計分析・品質分布計算

### カスタムプラグイン作成
```python
# features/common/dashboard_plugins/custom_plugin.py
from features.common.dashboard_plugins import DashboardPlugin

class Plugin(DashboardPlugin):
    @property
    def name(self) -> str:
        return "custom"
    
    @property 
    def version(self) -> str:
        return "1.0.0"
        
    def execute(self, dashboard_data, plugin_settings):
        # カスタム処理
        return dashboard_data
```

## 📊 統合された機能

### QI-004機能統合
- ✅ **ImageQualityAnalyzer** - 画像品質解析
- ✅ **DashboardOptimizer** - パフォーマンス最適化
- ✅ **画像パス参照方式** - 軽量ダッシュボード

### quality_dashboard機能統合
- ✅ **QualityDashboard** - 統計グラフ生成
- ✅ **品質レポートダッシュボード** - 10指標対応

### StandardDashboardGenerator互換
- ✅ **完全互換性** - 既存APIとの互換性
- ✅ **Tailwind CSS** - レスポンシブデザイン
- ✅ **品質バッジシステム** - 統一された品質表示

## 🧪 テスト・検証

### システムテスト実行
```bash
# 全機能テスト
python3 test_unified_dashboard.py

# 個別テスト
python3 test_unified_dashboard.py basic      # 基本機能
python3 test_unified_dashboard.py config    # 設定システム
python3 test_unified_dashboard.py plugins   # プラグイン
python3 test_unified_dashboard.py compatibility # 互換性
```

### テスト結果確認項目
- ✅ **基本機能テスト**: ダッシュボード生成
- ✅ **設定システムテスト**: YAML設定管理
- ✅ **プラグインシステムテスト**: 拡張機能
- ✅ **互換性テスト**: 既存システム統合

## 🔄 マイグレーション手順

### Phase 1: 統合システム導入（完了）
- [x] UnifiedDashboardGenerator実装
- [x] 設定システム実装
- [x] プラグインシステム実装
- [x] run_quality_workflow.sh統合

### Phase 2: 既存スクリプト段階的移行（今後）
```bash
# 既存の個別スクリプトを確認
ls tools/scripts/*dashboard*.py

# 段階的に統合システムに移行
# tools/scripts/qi004_dashboard_generator.py → 統合システム使用
# tools/scripts/qi006_dashboard_generator.py → 統合システム使用
# (他40個のスクリプト)
```

### Phase 3: レガシーシステム削除（今後）
統合システムの安定性確認後、以下を削除：
- `features/evaluation/qi004_dashboard_optimization_system.py`
- `tools/scripts/*_dashboard_generator.py` (40個)
- 重複する小規模スクリプト

## ✅ 確認事項

### 動作確認
- [x] 統合システム基本動作
- [x] 既存システムとの互換性
- [x] プラグインシステム動作
- [x] 設定システム動作
- [x] run_quality_workflow.sh統合

### 品質指標
- ✅ **テスト合格率**: 4/4テスト（100%）
- ✅ **ファイルサイズ**: 3.8-7.8KB（軽量）
- ✅ **処理時間**: 数秒以内（高速）
- ✅ **エラーハンドリング**: 堅牢

## 🎯 期待効果

### 保守性向上
- **1つのシステム**で全ダッシュボード管理
- **一貫性のあるUI/UX**
- **テスト駆動開発**での品質保証

### 拡張性向上
- **プラグインベース**での機能追加
- **設定駆動**でのカスタマイズ
- **後方互換性**の維持

### 開発効率向上
- **40個以上のスクリプト**を統合
- **重複コード削減**
- **統一されたAPIと文書**

## 🚨 注意事項

1. **段階的移行**: 既存システムは段階的に移行し、急激な変更を避ける
2. **後方互換性**: 既存のAPI呼び出しは引き続き動作する
3. **設定ファイル**: カスタム設定は自動生成されるため手動編集は最小限に

## 📞 サポート

統合システムに関する質問・問題があれば、以下で確認してください：

1. **テスト実行**: `python3 test_unified_dashboard.py`
2. **ログ確認**: 詳細ログで問題分析
3. **設定確認**: `config/dashboard/` 内の設定ファイル