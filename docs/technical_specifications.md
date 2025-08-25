# 技術仕様統一リファレンス

**作成日**: 2025-07-28  
**重要度**: 最高  
**目的**: プロジェクトの技術仕様情報を一元化した統一リファレンス

---

## 📋 このドキュメントについて

このドキュメントは、プロジェクトの技術仕様に関する**唯一の正式な参照元**です。  
すべての技術仕様関連の情報は、この統一リファレンスを参照してください。

**⚠️ 重要**: 他のドキュメントで技術仕様を扱う場合は、「詳細: `docs/technical_specifications.md` を参照」と記載してください。

---

## 🔒 変更禁止の普遍的仕様

### 1. 核心技術スタック

```yaml
# 絶対変更禁止の技術基盤
core_technologies:
  sam_model: "segment_anything"  # Meta SAM - 変更不可
  yolo_detection: "ultralytics"  # YOLO物体検出 - 変更不可
  opencv_processing: "cv2"       # OpenCV画像処理 - 変更不可
  python_runtime: ">=3.8,<3.12" # Python実行環境 - 変更不可
  cuda_requirement: "必須"        # GPU処理要件 - 変更不可
```

### 2. 基本アーキテクチャパターン

```yaml
# 設計パターン（変更禁止）
architecture_pattern:
  detection_flow: "YOLO → SAM → 品質評価 → 後処理"
  input_format: "画像ファイル（jpg/png/webp）"
  output_format: "抽出済み画像 + マスクデータ"
  processing_mode: "バッチ処理優先"
  quality_assurance: "多段階品質評価"
```

### 3. セキュリティ原則

```yaml
# セキュリティ要件（絶対遵守）
security_principles:
  image_confidentiality: "画像ファイルは機密情報扱い"
  no_commit_images: "画像ファイルのcommit絶対禁止"
  output_path_restriction: "segment-anything/直下への出力禁止"
  gitignore_enforcement: "画像関連パスの完全除外"
  
# 推奨出力パス（固定）
safe_output_paths:
  - "/mnt/c/AItools/lora/train/yado/expanded/"
  - "/mnt/c/AItools/lora/train/yado/test_batches/"
  - "/mnt/c/AItools/lora/train/yado/visualizations/"
```

### 4. ファイル構造原則

```yaml
# ディレクトリ構造（基本配置固定）
directory_structure:
  core_implementation: "core/"           # Meta Facebook実装
  custom_features: "features/"           # カスタム実装
  executable_tools: "tools/"             # 実行スクリプト
  test_suites: "tests/"                  # テストスイート
  documentation: "docs/"                 # ドキュメント
  model_files: "*.pth, *.pt"           # プロジェクトルート
```

### 5. ダッシュボード仕様システム

**実装ベース**: QUAL-040シンプル実装（features/common/dashboard_generator.py）

```yaml
# ダッシュボード生成仕様（QUAL-040実装準拠）
dashboard_specifications:
  version: "2.0.0"
  implementation_approach: "シンプル・実用性重視"
  
  # 時刻表示（リアルタイム）
  timestamp_policy:
    format: "YYYY-MM-DD HH:MM:SS"
    display: "リアルタイム生成時刻"  # datetime.now()使用
    timezone: "Asia/Tokyo"
  
  # 数値フォーマット統一
  number_formatting:
    quality_scores:
      decimal_places: 3
      format: "0.000"
      example: "0.850"
    percentages:
      decimal_places: 1
      suffix: "%"
  
  # 品質バッジ基準（4段階評価）
  quality_badge_thresholds:
    high_quality: ">= 0.8"    # 高品質: 緑色バッジ
    medium_quality: ">= 0.6"  # 中品質: 黄色バッジ
    low_quality: ">= 0.4"     # 低品質: 橙色バッジ
    poor_quality: "< 0.4"     # 要改善: 赤色バッジ
  
  # HTML構造（Tailwind CSS）
  html_structure:
    framework: "Tailwind CSS (CDN)"
    layout: "レスポンシブグリッドレイアウト"
    sections:
      header: "bg-white rounded-lg shadow-md p-6 mb-8"
      statistics_grid: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      quality_distribution: "grid grid-cols-2 md:grid-cols-4 gap-4"
      image_gallery: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
  
  # データソース（シンプル化）
  data_processing:
    input_format: "extraction_result.json"
    statistical_calculation: "基本統計のみ（品質分布計算）"
    image_path_format: "/{tracker_id}/extraction/{filename}"
```

**技術実装**:
- **シンプル生成**: 190行のコンパクトな実装（旧608行から68%削減）
- **リアルタイム表示**: datetime.now()による自然な時刻表示
- **直接HTML生成**: テンプレートエンジンなしの直接的f-string生成
- **基本統計のみ**: extraction_result.jsonベースの品質分布計算
- **保守性重視**: 理解しやすく、修正しやすい実装

---

## 📊 客観的評価指標システム

### 設計原則

**目的**: 人間評価の主観性を排除し、完全客観的・ブレない評価システムの構築

#### 基本哲学
1. **完全自動化**: 人間の介入なしで計測可能
2. **再現性**: 同じ入力に対して常に同じ結果
3. **学術的根拠**: 既存研究で検証済みの手法
4. **継続的監視**: 日次/時間次での進捗追跡

### 核心3指標システム

#### 指標1: Pixel-Level Accuracy (PLA)
**目的**: ピクセル単位での抽出精度を客観測定

```python
def calculate_pla(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    IoU (Intersection over Union) ベースの客観的指標
    
    Returns:
        float: 0.0-1.0のPLAスコア
    """
    pred_binary = (predicted_mask > 0.5).astype(np.uint8)
    gt_binary = (ground_truth_mask > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)
```

**評価基準**:
```yaml
PLA評価レベル:
  優秀: 0.90+ # 商用レベル
  良好: 0.80-0.89 # 実用レベル  
  普通: 0.70-0.79 # 改善余地あり
  要改善: 0.60-0.69 # 問題あり
  不良: <0.60 # 使用不可
```

#### 指標2: Semantic Completeness Index (SCI)
**目的**: キャラクター構造の意味的完全性を客観評価

```python
def calculate_sci(extracted_image: np.ndarray, 
                  face_detector: Any, 
                  pose_estimator: Any) -> float:
    """
    人体構造の完全性を多角的に客観評価
    
    Returns:
        float: 0.0-1.0のSCIスコア
    """
    completeness_score = 0.0
    
    # 1. 顔検出率 (30% weight)
    face_confidence = detect_face_confidence(extracted_image, face_detector)
    completeness_score += min(face_confidence, 1.0) * 0.3
    
    # 2. 肢体完全性 (40% weight)  
    limb_completeness = calculate_limb_completeness(extracted_image, pose_estimator)
    completeness_score += limb_completeness * 0.4
    
    # 3. 輪郭連続性 (30% weight)
    contour_continuity = measure_contour_continuity(extracted_image)
    completeness_score += contour_continuity * 0.3
    
    return min(completeness_score, 1.0)
```

**評価基準**:
```yaml
SCI評価レベル:
  完全: 0.85+ # 構造的に完璧
  ほぼ完全: 0.70-0.84 # 軽微な欠損のみ
  部分的: 0.50-0.69 # 重要部位の一部欠損
  不完全: 0.30-0.49 # 重大な構造欠損  
  破綻: <0.30 # 構造として成立していない
```

#### 指標3: Progressive Learning Efficiency (PLE)
**目的**: 継続的改善の効率性を客観測定（スクラップ&ビルド防止）

```python
def calculate_ple(current_results: List[float], 
                  historical_results: List[float],
                  time_window: int = 10) -> float:
    """
    継続的学習効率の測定
    
    Returns:
        float: -1.0 to 1.0のPLEスコア（負値は退行）
    """
    if len(current_results) < time_window or len(historical_results) < time_window:
        return 0.0
    
    recent_avg = np.mean(current_results[-time_window:])
    baseline_avg = np.mean(historical_results[:time_window])
    
    # 1. 改善率 (40% weight)
    improvement_rate = (recent_avg - baseline_avg) / baseline_avg if baseline_avg != 0 else 0.0
    
    # 2. 安定性 (30% weight)
    recent_std = np.std(current_results[-time_window:])
    stability = 1.0 - min(recent_std, 1.0)
    
    # 3. 効率性 (30% weight)
    trial_efficiency = improvement_rate / (len(current_results) / 100.0) if len(current_results) > 0 else 0.0
    
    ple_score = (improvement_rate * 0.4 + stability * 0.3 + trial_efficiency * 0.3)
    return max(-1.0, min(1.0, ple_score))
```

**評価基準**:
```yaml
PLE評価レベル:
  高効率学習: 0.15+ # 効率的な継続改善
  標準学習: 0.05-0.14 # 通常の改善ペース
  低効率学習: 0.00-0.04 # 改善が遅い
  停滞: -0.05-0.00 # 改善が見られない
  退行: <-0.05 # 性能が悪化している
```

---

## 🔄 変更可能な実装詳細

### パラメータ・設定値

```yaml
# 調整可能な設定（実装改善に伴い変更OK）
adjustable_parameters:
  yolo_confidence_threshold: "現在0.07（調整可能）"
  sam_model_variant: "vit_h/vit_l/vit_b（選択可能）"
  batch_size: "メモリに応じて調整可能"
  quality_thresholds: "目標値に応じて調整可能"
  
# 品質目標値（段階的向上）
quality_targets:
  pla_targets: "0.75 → 0.80 → 0.85（段階的向上）"
  sci_targets: "0.70 → 0.75 → 0.80（段階的向上）"
  ple_targets: "0.10 → 0.12 → 0.15（段階的向上）"
```

### アルゴリズム改善領域

```yaml
# 改善可能な領域
improvable_areas:
  preprocessing_methods: "前処理手法の改良"
  postprocessing_steps: "後処理ステップの追加"
  quality_calculation_details: "品質計算の詳細改善"
  performance_optimization: "実行速度の最適化"
  error_handling_enhancement: "エラー処理の強化"
```

---

## 📈 学術的根拠

### PLA (Pixel-Level Accuracy)
- **COCO Dataset**: 物体検出の国際標準
- **Pascal VOC**: セグメンテーション評価の基準
- **Medical Image Analysis**: 医療分野での確立手法

### SCI (Semantic Completeness Index)
- **MediaPipe Pose**: Google Research の人体姿勢推定
- **OpenPose**: CMU発の姿勢推定の標準実装
- **Human Pose Estimation**: 人体構造解析の確立手法

### PLE (Progressive Learning Efficiency)
- **Continual Learning**: 継続学習の効率性評価
- **Online Learning**: オンライン学習の性能指標
- **Model Performance Tracking**: MLOpsでの標準手法

---

## 🔧 実装アーキテクチャ

### メインクラス構造

```python
class ObjectiveEvaluationSystem:
    """完全客観的評価システムのメインクラス"""
    
    def __init__(self):
        self.pla_calculator = PLACalculator()
        self.sci_calculator = SCICalculator()
        self.ple_tracker = PLETracker()
        self.academic_metrics = AcademicMetricsBundle()
        self.progress_monitor = ProgressMonitor()
    
    def evaluate_batch_objective(self, results_path: str) -> ObjectiveReport:
        """バッチの完全客観評価"""
        pass
    
    def track_daily_progress(self) -> ProgressReport:
        """日次進捗追跡"""
        pass
    
    def generate_milestone_report(self) -> MilestoneReport:
        """マイルストーン達成度評価"""
        pass
```

---

## 📋 普遍性チェックリスト

新機能・変更実装時は以下を必ず確認：

### ✅ 技術スタック確認
- [ ] SAM・YOLO・OpenCVの使用を維持しているか？
- [ ] Python 3.8-3.12環境で動作するか？
- [ ] CUDA環境での動作を前提としているか？

### ✅ セキュリティ確認
- [ ] 画像ファイルをcommitしていないか？
- [ ] 出力パスが安全な場所に設定されているか？
- [ ] .gitignoreで画像ファイルが除外されているか？

### ✅ アーキテクチャ確認
- [ ] YOLO→SAM→品質評価の流れを維持しているか？
- [ ] バッチ処理パターンを踏襲しているか？
- [ ] 3指標システム（PLA/SCI/PLE）を活用しているか？

### ✅ 互換性確認
- [ ] 既存のspec.mdとの整合性があるか？
- [ ] PRINCIPLE.mdのセキュリティ原則に準拠しているか？
- [ ] 客観的品質評価を使用しているか？

---

## 🚨 緊急時対応

### 普遍的仕様違反の発見時
1. **即座に実装を停止**
2. **違反内容を `docs/issues/` に記録**
3. **原因分析と修正計画を策定**
4. **修正後に普遍性チェックリストで再確認**

### 仕様変更の検討が必要な場合
1. **仕様変更の必要性を文書化**
2. **影響範囲の詳細分析**
3. **段階的移行計画の策定**
4. **全テストスイートでの動作確認**

---

## 🎯 現在の目標値

### 最新の品質目標
```yaml
current_targets:
  pla_target: 0.80  # 実用レベル
  sci_target: 0.75  # ほぼ完全レベル
  ple_target: 0.12  # 標準学習レベル
  
benchmark_results:
  processing_success_rate: "96.7% (148/153画像)"
  average_quality_score: "0.742（範囲: 0.482-0.938）"
  interactive_success_rate: "100%（手動介入時）"
```

---

**重要**: この技術仕様統一リファレンスは、プロジェクトの技術的根幹を成す重要な文書です。  
変更は慎重に検討し、必ず全体影響を評価してから実行してください。

**更新履歴**:
- 2025-07-28: 統一リファレンス作成（`objective_metrics_specification.md` + `universal_specifications.md` 統合）