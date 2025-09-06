# 客観的評価指標仕様書

**⚠️ 重要**: この情報は [`docs/technical_specifications.md`](technical_specifications.md) に統合されました。  
**最新情報については**: `docs/technical_specifications.md` を参照してください。

**最終更新**: 2025-07-24  
**目的**: 人間評価の主観性を排除し、完全客観的・ブレない評価システムの構築

## 🎯 設計原則

### 問題認識
- **主観性の排除**: A-F評価の人間判断によるブレを根絶
- **進捗の可視化**: スクラップ&ビルドの繰り返しを防止
- **客観的基準**: 学術論文準拠の再現可能な指標

### 設計哲学
1. **完全自動化**: 人間の介入なしで計測可能
2. **再現性**: 同じ入力に対して常に同じ結果
3. **学術的根拠**: 既存研究で検証済みの手法
4. **継続的監視**: 日次/時間次での進捗追跡

## 📊 核心3指標システム

### 指標1: Pixel-Level Accuracy (PLA)
**目的**: ピクセル単位での抽出精度を客観測定

#### 計算式
```python
def calculate_pla(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    IoU (Intersection over Union) ベースの客観的指標
    
    Args:
        predicted_mask: 予測されたマスク (binary array)
        ground_truth_mask: 正解マスク (binary array)
    
    Returns:
        float: 0.0-1.0のPLAスコア
    """
    # バイナリマスクに正規化
    pred_binary = (predicted_mask > 0.5).astype(np.uint8)
    gt_binary = (ground_truth_mask > 0.5).astype(np.uint8)
    
    # IoU計算
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)
```

#### 評価基準
```yaml
PLA評価レベル:
  優秀: 0.90+ # 商用レベル
  良好: 0.80-0.89 # 実用レベル  
  普通: 0.70-0.79 # 改善余地あり
  要改善: 0.60-0.69 # 問題あり
  不良: <0.60 # 使用不可
```

#### 学術的根拠
- **COCO Dataset**: 物体検出の国際標準
- **Pascal VOC**: セグメンテーション評価の基準
- **Medical Image Analysis**: 医療分野での確立手法

### 指標2: Semantic Completeness Index (SCI)
**目的**: キャラクター構造の意味的完全性を客観評価

#### 計算式
```python
def calculate_sci(extracted_image: np.ndarray, 
                  face_detector: Any, 
                  pose_estimator: Any) -> float:
    """
    人体構造の完全性を多角的に客観評価
    
    Args:
        extracted_image: 抽出されたキャラクター画像
        face_detector: 顔検出モデル (OpenCV DNN/MediaPipe)
        pose_estimator: 姿勢推定モデル (MediaPipe Pose/OpenPose)
    
    Returns:
        float: 0.0-1.0のSCIスコア
    """
    completeness_score = 0.0
    
    # 1. 顔検出率 (30% weight)
    face_confidence = detect_face_confidence(extracted_image, face_detector)
    face_score = min(face_confidence, 1.0)
    completeness_score += face_score * 0.3
    
    # 2. 肢体完全性 (40% weight)  
    limb_completeness = calculate_limb_completeness(extracted_image, pose_estimator)
    completeness_score += limb_completeness * 0.4
    
    # 3. 輪郭連続性 (30% weight)
    contour_continuity = measure_contour_continuity(extracted_image)
    completeness_score += contour_continuity * 0.3
    
    return min(completeness_score, 1.0)

def detect_face_confidence(image: np.ndarray, face_detector: Any) -> float:
    """顔検出信頼度の計測"""
    detections = face_detector.detectMultiScale(image)
    if len(detections) == 0:
        return 0.0
    
    # 最も大きな顔の信頼度を使用
    largest_face = max(detections, key=lambda x: x[2] * x[3])
    return min(largest_face[4] if len(largest_face) > 4 else 0.8, 1.0)

def calculate_limb_completeness(image: np.ndarray, pose_estimator: Any) -> float:
    """肢体完全性の計測"""
    pose_results = pose_estimator.process(image)
    if not pose_results.pose_landmarks:
        return 0.0
    
    # 重要な関節点の検出率
    critical_landmarks = [
        # 顔
        pose_estimator.PoseLandmark.NOSE,
        pose_estimator.PoseLandmark.LEFT_EYE,
        pose_estimator.PoseLandmark.RIGHT_EYE,
        # 手
        pose_estimator.PoseLandmark.LEFT_WRIST,
        pose_estimator.PoseLandmark.RIGHT_WRIST,
        # 足
        pose_estimator.PoseLandmark.LEFT_ANKLE,
        pose_estimator.PoseLandmark.RIGHT_ANKLE
    ]
    
    detected_count = 0
    for landmark in critical_landmarks:
        if pose_results.pose_landmarks.landmark[landmark].visibility > 0.5:
            detected_count += 1
    
    return detected_count / len(critical_landmarks)

def measure_contour_continuity(image: np.ndarray) -> float:
    """輪郭連続性の計測"""
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # 輪郭検出
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0
    
    # 最大輪郭の解析
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭の滑らかさ（曲率変化率）
    smoothness = calculate_contour_smoothness(largest_contour)
    
    # 連続性（ギャップの少なさ）
    continuity = calculate_contour_gaps(largest_contour)
    
    return (smoothness + continuity) / 2.0
```

#### 評価基準
```yaml
SCI評価レベル:
  完全: 0.85+ # 構造的に完璧
  ほぼ完全: 0.70-0.84 # 軽微な欠損のみ
  部分的: 0.50-0.69 # 重要部位の一部欠損
  不完全: 0.30-0.49 # 重大な構造欠損  
  破綻: <0.30 # 構造として成立していない
```

#### 学術的根拠
- **MediaPipe Pose**: Google Research の人体姿勢推定
- **OpenPose**: CMU発の姿勢推定の標準実装
- **Human Pose Estimation**: 人体構造解析の確立手法

### 指標3: Progressive Learning Efficiency (PLE)
**目的**: 継続的改善の効率性を客観測定（スクラップ&ビルド防止）

#### 計算式
```python
def calculate_ple(current_results: List[float], 
                  historical_results: List[float],
                  time_window: int = 10) -> float:
    """
    継続的学習効率の測定
    
    Args:
        current_results: 最新の結果リスト
        historical_results: 過去の結果リスト  
        time_window: 評価時間窓（デフォルト10サンプル）
    
    Returns:
        float: -1.0 to 1.0のPLEスコア（負値は退行）
    """
    if len(current_results) < time_window or len(historical_results) < time_window:
        return 0.0
    
    # 直近の平均性能
    recent_avg = np.mean(current_results[-time_window:])
    
    # ベースライン平均性能
    baseline_avg = np.mean(historical_results[:time_window])
    
    # 1. 改善率 (40% weight)
    if baseline_avg == 0:
        improvement_rate = 0.0
    else:
        improvement_rate = (recent_avg - baseline_avg) / baseline_avg
    
    # 2. 安定性 (30% weight) - 標準偏差の逆数
    recent_std = np.std(current_results[-time_window:])
    stability = 1.0 - min(recent_std, 1.0)  # 0-1に正規化
    
    # 3. 効率性 (30% weight) - 改善量 / 試行回数
    trial_efficiency = improvement_rate / (len(current_results) / 100.0) if len(current_results) > 0 else 0.0
    
    # 重み付き平均
    ple_score = (improvement_rate * 0.4 + stability * 0.3 + trial_efficiency * 0.3)
    
    # -1.0 to 1.0 の範囲に正規化
    return max(-1.0, min(1.0, ple_score))
```

#### 評価基準
```yaml
PLE評価レベル:
  高効率学習: 0.15+ # 効率的な継続改善
  標準学習: 0.05-0.14 # 通常の改善ペース
  低効率学習: 0.00-0.04 # 改善が遅い
  停滞: -0.05-0.00 # 改善が見られない
  退行: <-0.05 # 性能が悪化している
```

#### 学術的根拠
- **Continual Learning**: 継続学習の効率性評価
- **Online Learning**: オンライン学習の性能指標
- **Model Performance Tracking**: MLOpsでの標準手法

## 📈 補助指標（学術論文準拠）

### mIoU (mean Intersection over Union)
```python
def calculate_miou(predictions: List[np.ndarray], 
                   ground_truths: List[np.ndarray]) -> float:
    """
    COCO Dataset標準の平均IoU
    
    Returns:
        float: 全画像の平均IoU
    """
    ious = []
    for pred, gt in zip(predictions, ground_truths):
        iou = calculate_pla(pred, gt)  # PLAと同じ計算
        ious.append(iou)
    
    return np.mean(ious)
```

### F1-Score for Segmentation
```python
def calculate_f1_segmentation(predicted_mask: np.ndarray, 
                             ground_truth_mask: np.ndarray) -> float:
    """
    医療画像解析標準のF1スコア
    
    Returns:
        float: Precision と Recall のハーモニック平均
    """
    pred_binary = (predicted_mask > 0.5).astype(np.uint8)
    gt_binary = (ground_truth_mask > 0.5).astype(np.uint8)
    
    # True Positive, False Positive, False Negative
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()  
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
```

### Hausdorff Distance
```python
def calculate_hausdorff_distance(predicted_mask: np.ndarray, 
                                ground_truth_mask: np.ndarray) -> float:
    """
    境界線精度の幾何学的測定
    
    Returns:
        float: 境界間の最大距離（ピクセル単位）
    """
    from scipy.spatial.distance import directed_hausdorff
    
    # 境界点の抽出
    pred_contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours, _ = cv2.findContours(ground_truth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return float('inf')
    
    pred_points = pred_contours[0].reshape(-1, 2)
    gt_points = gt_contours[0].reshape(-1, 2)
    
    # 双方向Hausdorff距離
    dist1 = directed_hausdorff(pred_points, gt_points)[0]
    dist2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(dist1, dist2)
```

## 🎯 目標設定とマイルストーン

### Phase A: 評価基盤構築（2週間）
```yaml
マイルストーン A1:
  目標: PLA測定システム完全自動化
  成功基準:
    - PLA平均値: 0.75以上
    - 計測速度: 1秒/画像以下
    - 再現性: 100%（同じ入力→同じ出力）

マイルストーン A2:
  目標: SCI計算システム実装
  成功基準:
    - SCI平均値: 0.70以上
    - 顔検出率: 90%以上
    - 肢体完全性: 80%以上
```

### Phase B: 改善システム実装（3週間）
```yaml
マイルストーン B1:
  目標: 多層特徴抽出システム
  成功基準:
    - 特徴次元数: 50→200次元
    - 特徴冗長性: 10%以下
    - 計算時間: 5秒/画像以下

マイルストーン B2:
  目標: 適応的推論エンジン
  成功基準:
    - PLE値: 0.10以上の継続改善
    - 推論パス数: 3→8パス
    - 判断一致率: 85%以上
```

### Phase C: 統合システム完成（4週間）
```yaml
マイルストーン C1:
  目標: Claude風統合パイプライン
  成功基準:
    - PLA: 0.85以上
    - SCI: 0.80以上
    - PLE: 0.15以上
    - 人間評価相関: 90%以上
```

## 🔧 実装アーキテクチャ

### クラス構造
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

class PLACalculator:
    """Pixel-Level Accuracy 計算器"""
    
    def calculate(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        pass

class SCICalculator:
    """Semantic Completeness Index 計算器"""
    
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.pose_estimator = mediapipe.solutions.pose.Pose()
    
    def calculate(self, extracted_image: np.ndarray) -> float:
        pass

class PLETracker:
    """Progressive Learning Efficiency 追跡器"""
    
    def __init__(self, history_file: str = "progress_history.json"):
        self.history_file = history_file
        self.load_history()
    
    def update_and_calculate(self, new_results: List[float]) -> float:
        pass
```

## 🚀 導入ロードマップ

### Week 1: 基盤実装
- [ ] ObjectiveEvaluationSystemクラス作成
- [ ] PLA・SCI・PLE計算エンジン実装
- [ ] 単体テストの作成と検証

### Week 2: 統合とテスト
- [ ] バッチ評価パイプライン統合
- [ ] 既存システムとの接続
- [ ] 最初のベンチマークテスト実行

### Week 3-5: 改善システム実装
- [ ] 多層特徴抽出システム
- [ ] 適応的推論エンジン
- [ ] 継続学習機能

### Week 6-9: 最終統合
- [ ] Claude風統合パイプライン
- [ ] 性能最適化
- [ ] 本番運用開始

## 📊 継続的監視システム

### 日次監視指標
```python
daily_metrics = {
    "pla_trend": [0.74, 0.75, 0.76, 0.77, 0.78],  # 5日移動平均
    "sci_trend": [0.68, 0.69, 0.71, 0.72, 0.73],
    "ple_current": 0.12,  # 現在の学習効率
    "regression_alert": False,  # 退行アラート
    "milestone_progress": 0.65  # マイルストーン進捗率
}
```

### アラートシステム
```python
class ProgressAlert:
    """進捗監視アラートシステム"""
    
    def check_regression(self, current_metrics: Dict) -> bool:
        """退行検出"""
        if current_metrics['ple_current'] < -0.05:
            self.send_alert("性能退行が検出されました")
            return True
        return False
    
    def check_stagnation(self, recent_history: List[float]) -> bool:
        """停滞検出"""
        if len(recent_history) >= 5:
            recent_variance = np.var(recent_history[-5:])
            if recent_variance < 0.001:  # ほぼ変化なし
                self.send_alert("進捗停滞が検出されました")
                return True
        return False
```

---

この客観的評価指標システムにより、人間評価の主観性を完全に排除し、ブレない・継続的な進捗測定が可能になります。