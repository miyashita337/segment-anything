# Phase 3-6 統合パイプライン実装ガイド

## 概要

Phase 3-6を統合した単一の堅牢なパイプラインシステム。状態管理、レジューム機能、Web対応ダッシュボード生成を含む包括的なワークフローを提供します。

## 設計原則

### 安全性とロバスト性優先
- **運用コストよりも安全性を重視**
- **蓄積された修正による矛盾を防ぐ設計**
- **多層バリデーション**
- **詳細なエラーメッセージ**

### 状態管理とレジューム機能
- **チェックポイントベースの状態永続化**
- **処理中断時の安全な再開**
- **部分完了状態の適切な管理**

## 統合フェーズ構成

### Phase 3: 実装後品質確認
- **入力パスバリデーション**（必須）
- **設定ファイル検証**
- **依存関係チェック**

### Phase 4: キャラクター抽出実行
- **YOLO+SAM パイプライン実行**
- **品質評価システム統合**
- **プログレス監視**

### Phase 5: 品質評価・レポート生成
- **多次元品質指標算出**
- **成功/失敗分析**
- **詳細レポート生成**

### Phase 6: ダッシュボード生成（新規）
- **Web アクセス可能なHTMLダッシュボード**
- **リアルタイム品質指標表示**
- **結果画像ギャラリー**
- **インタラクティブ分析機能**

## 実装アーキテクチャ

### コアコンポーネント

#### 1. 統合パイプライン管理 (`tools/core/integrated_quality_pipeline.py`)
```python
class IntegratedQualityPipeline:
    """Phase 3-6を統合した堅牢なパイプライン"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.state_manager = StateManager()
        self.validator = ValidationEngine()
    
    def execute_pipeline(self, resume: bool = False) -> PipelineResult:
        """統合パイプライン実行"""
        # Phase 3: バリデーション
        # Phase 4: 抽出実行  
        # Phase 5: 品質評価
        # Phase 6: ダッシュボード生成
```

#### 2. 状態管理システム
```python
class StateManager:
    """パイプライン状態の永続化と復旧"""
    
    def save_checkpoint(self, phase: str, data: dict):
        """チェックポイント保存"""
    
    def load_checkpoint(self) -> Optional[dict]:
        """チェックポイント復旧"""
```

#### 3. バリデーション エンジン
```python
class ValidationEngine:
    """多層入力バリデーション"""
    
    def validate_input_paths(self, paths: List[str]) -> ValidationResult:
        """入力パス存在チェック"""
    
    def validate_configuration(self, config: dict) -> ValidationResult:
        """設定ファイル整合性チェック"""
```

### 設定ファイル (`config/pipeline_config.yaml`)

```yaml
# 統合パイプライン設定
pipeline:
  name: "integrated_phase3_6"
  version: "1.0.0"

# Phase定義
phases:
  phase3:
    name: "品質確認"
    validation:
      - input_paths_exist
      - config_integrity
      - dependencies_available
  
  phase4:
    name: "抽出実行"
    extractor: "sam_yolo_character_segment"
    quality_method: "balanced"
    
  phase5:
    name: "品質評価"
    metrics:
      - extraction_success_rate
      - quality_score_distribution
      - processing_time_analysis
  
  phase6:
    name: "ダッシュボード生成"
    dashboard:
      template: "quality_dashboard_template.html"
      output_format: "html"
      web_server:
        enabled: true
        port: 8080

# パス設定
paths:
  default_input: "/mnt/c/AItools/lora/train/yado/org/kana05/"
  workspace_base: "/mnt/c/AItools/lora/train/yado/tracker-workspace/"
  
# エラー処理
error_handling:
  max_retries: 3
  timeout_seconds: 600
  recovery_strategies:
    - "retry_with_lower_quality"
    - "skip_problematic_images"
    - "fallback_to_manual_mode"
```

## 入力バリデーション強化

### 必須チェック項目

1. **入力パス存在確認**
   ```bash
   ❌ エラー: 入力ディレクトリが存在しません
      パス: /specified/path
   
   🔧 対処方法:
      1. パスの確認: ls /parent/directory
      2. 正しいパスの指定
      3. 必要に応じてディレクトリ作成
   ```

2. **設定ファイル整合性**
3. **依存関係可用性**
4. **出力ディレクトリ書き込み権限**

## ダッシュボード機能 (Phase 6)

### HTML ダッシュボード生成

#### 主要機能
- **品質指標の可視化**
- **抽出結果画像ギャラリー**
- **リアルタイム処理状況**
- **エラー分析レポート**

#### Web アクセス機能
- **組み込み HTTP サーバー**
- **ポート設定可能**
- **セキュリティ基本設定**

### ダッシュボード テンプレート (`templates/quality_dashboard_template.html`)

```html
<!DOCTYPE html>
<html>
<head>
    <title>統合品質ダッシュボード - {{tracker_id}}</title>
    <meta charset="utf-8">
    <!-- Chart.js, Bootstrap等のライブラリ -->
</head>
<body>
    <div class="container">
        <h1>品質分析結果: {{tracker_id}}</h1>
        
        <!-- 品質指標サマリー -->
        <div class="metrics-summary">
            <div class="metric-card">
                <h3>抽出成功率</h3>
                <span class="metric-value">{{success_rate}}%</span>
            </div>
            <!-- 他の指標 -->
        </div>
        
        <!-- 結果画像ギャラリー -->
        <div class="image-gallery">
            <!-- 動的に生成 -->
        </div>
        
        <!-- 品質分布チャート -->
        <canvas id="qualityChart"></canvas>
    </div>
</body>
</html>
```

## 実行フロー

### 1. 基本実行
```bash
# 新規実行
python tools/core/integrated_quality_pipeline.py \
  --config config/pipeline_config.yaml \
  --tracker-id INTEGRATE-3-6

# レジューム実行
python tools/core/integrated_quality_pipeline.py \
  --config config/pipeline_config.yaml \
  --tracker-id INTEGRATE-3-6 \
  --resume
```

### 2. ラッパースクリプト
```bash
# 簡易実行
./tools/scripts/run_integrated_pipeline.sh INTEGRATE-3-6

# 高度な実行オプション
./tools/scripts/run_integrated_pipeline.sh INTEGRATE-3-6 \
  --input-dir /custom/input/path \
  --quality-method size_priority \
  --enable-dashboard-server
```

## エラー処理と復旧

### エラー分類
1. **設定エラー**: 不正な設定ファイル
2. **入力エラー**: 存在しないパス
3. **処理エラー**: 抽出処理失敗
4. **出力エラー**: ディスク容量不足等

### 復旧戦略
1. **自動リトライ**: 一時的エラー対応
2. **品質レベル下げ**: リソース不足対応
3. **部分スキップ**: 問題画像の除外
4. **手動モード**: 完全自動化失敗時

## 成功基準

### Phase 3
- ✅ 全入力パス存在確認
- ✅ 設定ファイル整合性確認
- ✅ 依存関係利用可能確認

### Phase 4
- ✅ 抽出パイプライン完了
- ✅ 最低70%の成功率達成
- ✅ 品質評価データ生成

### Phase 5
- ✅ 品質レポート生成完了
- ✅ 詳細分析データ出力
- ✅ エラー分析完了

### Phase 6
- ✅ HTMLダッシュボード生成
- ✅ Web アクセス確認
- ✅ 全データ可視化完了

## 保守性とスケーラビリティ

### 設計特徴
- **設定駆動型**: YAML設定での柔軟な制御
- **プラグイン対応**: 新しい品質評価手法の追加容易
- **モジュラー構成**: 各Phaseの独立性
- **テスタビリティ**: 単体・統合テスト対応

### 将来拡張ポイント
- **Phase 7**: 自動改善提案
- **Phase 8**: A/Bテスト機能
- **クラウド連携**: 大規模処理対応
- **機械学習統合**: 品質予測モデル

## 関連ドキュメント

- **技術仕様詳細**: `docs/technical_specifications.md`
- **出力ディレクトリ規約**: `docs/workflows/OUTPUT_DIRECTORY_UNIFIED.md`
- **品質評価ガイド**: `docs/workflows/quality_evaluation_guide.md`
- **ダッシュボード管理**: `docs/workflows/dashboard_management_unified.md`