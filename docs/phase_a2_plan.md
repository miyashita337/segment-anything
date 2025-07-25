# Phase A2 実装計画書 - SCI基盤構築

**作成日**: 2025-07-26  
**フェーズ**: Phase A2 - SCI計算システム実装  
**期限**: 2025-08-14（19日間）  
**前提条件**: Phase A1完了 ✅

## 📋 プロジェクト概要

### 目的
Semantic Completeness Index（SCI）計算システムを完全実装し、抽出画像の構造完全性を客観的に評価する基盤を構築する。

### 成功基準
```yaml
Phase_A2_Success_Criteria:
  primary_targets:
    sci_mean: 0.70以上
    face_detection_rate: 0.90以上
    pose_detection_rate: 0.80以上
    processing_speed: 5秒/画像以下
  
  system_requirements:
    extraction_integration: 完全統合
    metadata_management: 自動化実現
    error_handling: 堅牢性確保
    
  quality_gates:
    unit_test_coverage: 90%以上
    integration_test_pass: 100%
    performance_benchmark: Phase A1比20%向上
```

## 🎯 主要タスクブレークダウン

### 1. 抽出結果統合評価システム（高優先度）
**期間**: 2025-07-26 ～ 2025-08-01（7日間）

#### 1.1 現状課題分析
- **問題**: 現在のシステムは元画像ベース評価のみ
- **必要**: 抽出済み画像との統合評価
- **技術課題**: ファイル対応付け、メタデータ管理

#### 1.2 設計要件
```python
class ExtractionIntegratedEvaluator:
    """抽出結果統合評価システム"""
    
    def __init__(self):
        self.metadata_manager = MetadataManager()
        self.file_matcher = FileCorrespondenceMatcher()
        self.sci_engine = EnhancedSCIEngine()
    
    def evaluate_extraction_batch(self, extraction_dir: str) -> IntegratedEvaluationReport:
        """抽出結果の統合評価"""
        # 1. ファイル対応付け
        correspondences = self.file_matcher.match_files(extraction_dir)
        
        # 2. メタデータ読み込み
        metadata = self.metadata_manager.load_batch_metadata(extraction_dir)
        
        # 3. 統合SCI評価
        sci_results = self.sci_engine.evaluate_batch_with_metadata(
            correspondences, metadata
        )
        
        return IntegratedEvaluationReport(sci_results, correspondences, metadata)
```

#### 1.3 実装フェーズ
1. **FileCorrespondenceMatcher**実装（2日）
2. **MetadataManager**実装（2日）
3. **EnhancedSCIEngine**実装（2日）
4. **統合テスト**（1日）

### 2. メタデータ管理システム（高優先度）
**期間**: 2025-07-28 ～ 2025-08-03（6日間）

#### 2.1 メタデータ仕様
```json
{
  "extraction_metadata": {
    "source_image": "kana05_0022.jpg",
    "ground_truth_mask": "kana05_0022_gt.png",
    "extracted_image": "kana05_0022_extracted.png",
    "extraction_method": "yolo_sam_v043",
    "extraction_timestamp": "2025-07-26T12:00:00Z",
    "quality_scores": {
      "yolo_confidence": 0.85,
      "sam_iou": 0.78,
      "balanced_score": 0.82
    },
    "detection_info": {
      "bounding_box": [100, 150, 400, 600],
      "character_type": "full_body",
      "pose_estimation": "standing",
      "face_detected": true
    }
  }
}
```

#### 2.2 実装コンポーネント
1. **MetadataGenerator** - 抽出時自動メタデータ生成
2. **MetadataValidator** - メタデータ整合性チェック
3. **MetadataQuery** - 高速検索・フィルタリング
4. **MetadataBackup** - バックアップ・復旧機能

### 3. SCI計算エンジン強化（高優先度）
**期間**: 2025-08-01 ～ 2025-08-08（8日間）

#### 3.1 顔検出率90%達成
**現状**: MediaPipe基本実装済み  
**課題**: 横顔、部分隠蔽、小さい顔の検出精度不足

```python
class EnhancedFaceDetector:
    """強化された顔検出システム"""
    
    def __init__(self):
        self.mediapipe_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 遠距離モデル使用
            min_detection_confidence=0.3  # 閾値を下げて検出率向上
        )
        self.cascade_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.dlib_detector = dlib.get_frontal_face_detector()
    
    def detect_faces_multi_method(self, image: np.ndarray) -> List[FaceDetection]:
        """複数手法による顔検出"""
        detections = []
        
        # MediaPipe検出
        mp_faces = self.mediapipe_detector.process(image)
        detections.extend(self._convert_mediapipe_detections(mp_faces))
        
        # OpenCV Cascade検出（横顔対応）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_faces = self.cascade_detector.detectMultiScale(gray, 1.1, 4)
        detections.extend(self._convert_cascade_detections(cascade_faces))
        
        # dlib検出（補完用）
        dlib_faces = self.dlib_detector(gray)
        detections.extend(self._convert_dlib_detections(dlib_faces))
        
        # 重複除去・統合
        return self._merge_detections(detections)
```

#### 3.2 ポーズ検出率80%達成
**現状**: MediaPipe Pose基本実装済み  
**課題**: 部分的な身体、座位、複雑なポーズの検出不足

```python
class EnhancedPoseDetector:
    """強化されたポーズ検出システム"""
    
    def __init__(self):
        self.mediapipe_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # 最高精度モデル
            enable_segmentation=True,
            min_detection_confidence=0.3
        )
        self.pose_classifier = self._load_pose_classifier()
    
    def detect_pose_comprehensive(self, image: np.ndarray) -> PoseDetectionResult:
        """包括的ポーズ検出"""
        # MediaPipeでキーポイント検出
        results = self.mediapipe_pose.process(image)
        
        if not results.pose_landmarks:
            return PoseDetectionResult(detected=False, confidence=0.0)
        
        # キーポイント可視性分析
        visibility_score = self._calculate_visibility_score(results.pose_landmarks)
        
        # ポーズ分類
        pose_category = self._classify_pose(results.pose_landmarks)
        
        # 完全性評価
        completeness_score = self._evaluate_pose_completeness(results.pose_landmarks)
        
        return PoseDetectionResult(
            detected=True,
            landmarks=results.pose_landmarks,
            visibility_score=visibility_score,
            pose_category=pose_category,
            completeness_score=completeness_score,
            confidence=min(visibility_score, completeness_score)
        )
```

### 4. パフォーマンス最適化（中優先度）
**期間**: 2025-08-08 ～ 2025-08-12（5日間）

#### 4.1 処理速度目標
- **現状**: 平均10秒/画像
- **目標**: 5秒/画像以下
- **手法**: 並列処理、キャッシュ、最適化

#### 4.2 最適化戦略
```python
class OptimizedSCIProcessor:
    """最適化されたSCI処理システム"""
    
    def __init__(self, num_workers: int = 4):
        self.face_detector = EnhancedFaceDetector()
        self.pose_detector = EnhancedPoseDetector()
        self.cache = SCIResultCache()
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
    
    def process_batch_parallel(self, image_batch: List[str]) -> List[SCIResult]:
        """並列バッチ処理"""
        # キャッシュチェック
        cached_results, uncached_images = self.cache.check_batch(image_batch)
        
        if not uncached_images:
            return cached_results
        
        # 並列処理実行
        futures = [
            self.executor.submit(self._process_single_image, img)
            for img in uncached_images
        ]
        
        # 結果収集
        new_results = [future.result() for future in futures]
        
        # キャッシュ更新
        self.cache.update_batch(uncached_images, new_results)
        
        return cached_results + new_results
```

### 5. テスト環境構築（中優先度）
**期間**: 2025-08-10 ～ 2025-08-14（5日間）

#### 5.1 テストデータセット準備
- **既存18枚正解マスク**活用
- **多様なポーズ・角度**のテストケース追加
- **エッジケース**（部分隠蔽、極端な角度等）

#### 5.2 自動テストスイート
```python
class PhaseA2TestSuite:
    """Phase A2 自動テストスイート"""
    
    def test_face_detection_accuracy(self):
        """顔検出精度テスト（目標90%）"""
        test_images = self.load_face_test_dataset()
        detector = EnhancedFaceDetector()
        
        correct_detections = 0
        total_images = len(test_images)
        
        for image, ground_truth in test_images:
            detection = detector.detect_faces_multi_method(image)
            if self._is_correct_detection(detection, ground_truth):
                correct_detections += 1
        
        accuracy = correct_detections / total_images
        assert accuracy >= 0.90, f"顔検出精度不足: {accuracy:.2%}"
    
    def test_pose_detection_accuracy(self):
        """ポーズ検出精度テスト（目標80%）"""
        test_images = self.load_pose_test_dataset()
        detector = EnhancedPoseDetector()
        
        correct_detections = 0
        total_images = len(test_images)
        
        for image, ground_truth in test_images:
            detection = detector.detect_pose_comprehensive(image)
            if self._is_correct_pose_detection(detection, ground_truth):
                correct_detections += 1
        
        accuracy = correct_detections / total_images
        assert accuracy >= 0.80, f"ポーズ検出精度不足: {accuracy:.2%}"
    
    def test_sci_calculation_performance(self):
        """SCI計算パフォーマンステスト（目標5秒/画像）"""
        test_images = self.load_performance_test_dataset()
        processor = OptimizedSCIProcessor()
        
        start_time = time.time()
        results = processor.process_batch_parallel(test_images)
        end_time = time.time()
        
        avg_time_per_image = (end_time - start_time) / len(test_images)
        assert avg_time_per_image <= 5.0, f"処理時間超過: {avg_time_per_image:.2f}秒/画像"
```

## 📅 詳細スケジュール

```yaml
Phase_A2_Schedule:
  Week1 (2025-07-26 ~ 2025-08-01):
    - 抽出結果統合評価システム設計・実装
    - FileCorrespondenceMatcher開発
    - 基本統合テスト

  Week2 (2025-08-02 ~ 2025-08-08):
    - メタデータ管理システム実装
    - 顔検出システム強化
    - ポーズ検出システム強化

  Week3 (2025-08-09 ~ 2025-08-14):
    - パフォーマンス最適化
    - 包括的テストスイート実行
    - Phase A2完了報告書作成
```

## 🚨 リスク管理

### 高リスク要因
1. **顔検出精度90%達成困難**
   - 軽減策: 複数手法組み合わせ、閾値調整
   - 代替案: 目標を85%に調整、Phase B1で改善

2. **ポーズ検出精度80%達成困難**
   - 軽減策: カスタム分類器訓練、データ拡張
   - 代替案: 部分的ポーズ検出でも評価対象とする

3. **処理速度目標未達成**
   - 軽減策: 並列処理、キャッシュ最適化
   - 代替案: クラウド処理への移行検討

### 中リスク要因
1. **メタデータ管理の複雑化**
   - 軽減策: シンプルな仕様から段階的拡張

2. **既存システムとの統合問題**
   - 軽減策: インターフェース設計の慎重な検討

## 📊 成果物・成功指標

### 必須成果物
1. **ExtractionIntegratedEvaluator** - 統合評価システム
2. **MetadataManager** - メタデータ管理システム
3. **EnhancedFaceDetector** - 強化顔検出システム
4. **EnhancedPoseDetector** - 強化ポーズ検出システム
5. **OptimizedSCIProcessor** - 最適化処理システム
6. **PhaseA2TestSuite** - 自動テストスイート

### 定量的成功指標
```yaml
Success_Metrics:
  SCI平均値: ≥ 0.70
  顔検出率: ≥ 90%
  ポーズ検出率: ≥ 80%
  処理速度: ≤ 5秒/画像
  テストカバレッジ: ≥ 90%
  統合テスト合格率: 100%
```

## 🎯 Phase A3への準備

Phase A2完了により、以下が整備される：
1. **完全な客観評価基盤**: PLA + SCI
2. **抽出結果統合システム**: 実用レベルの評価環境
3. **メタデータ駆動開発**: Phase B以降の高度機能基盤

---

**承認・開始**: Phase A2実装を上記計画に従って開始する