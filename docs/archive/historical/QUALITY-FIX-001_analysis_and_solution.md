# QUALITY-FIX-001: 抽出品質評価システム修正

## 問題分析

### 🚨 発見された主要問題

1. **真っ黒画像の高品質誤判定**
   - スコア0.8以上でも実際は抽出失敗
   - マスクは生成されているが中身が空（ピクセル値0）
   - 現在の評価は「マスクの存在」のみチェック、「内容」未検証

2. **複数キャラクター問題**
   - LoRA学習には1キャラクターが最適
   - 現在の評価は「最大キャラクター」のみ、複数検出の判定なし
   - 複数キャラでも高品質判定される

3. **部分抽出問題**
   - 全身可能でも部分抽出で高スコア
   - 人物認識範囲（bbox）の精度不足
   - SAMプロンプト生成時の境界設定問題

## 解決方針

### Phase 1: 即効性のある修正（緊急）

#### A. 真っ黒画像検出強化

```python
def validate_extraction_content(image_path: str) -> Dict[str, Any]:
    """抽出画像の内容検証"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 1. 真っ黒画像検出
    if img.shape[2] == 4:  # RGBA
        content_mask = img[:, :, 3] > 0  # アルファチャンネル
        rgb_content = img[:, :, :3][content_mask]
    else:
        content_mask = np.any(img > 0, axis=2)  # RGB全チャンネル
        rgb_content = img[content_mask]
    
    # 内容の存在確認
    if len(rgb_content) == 0:
        return {"valid": False, "reason": "empty_content", "score_penalty": -1.0}
    
    # 真っ黒判定（平均明度チェック）
    avg_brightness = np.mean(rgb_content)
    if avg_brightness < 10:  # 0-255スケールで10未満は真っ黒
        return {"valid": False, "reason": "too_dark", "score_penalty": -0.8}
    
    return {"valid": True, "brightness": avg_brightness}
```

#### B. 複数キャラクター検出

```python
def detect_multiple_characters(image_path: str, yolo_model) -> Dict[str, Any]:
    """複数キャラクター検出"""
    img = cv2.imread(image_path)
    results = yolo_model(img)
    
    person_detections = []
    for result in results:
        for detection in result.boxes:
            if detection.cls == 0:  # person class
                confidence = float(detection.conf)
                if confidence > 0.3:  # 閾値以上の検出
                    person_detections.append({
                        "bbox": detection.xyxy.tolist(),
                        "confidence": confidence
                    })
    
    character_count = len(person_detections)
    
    # LoRA最適判定
    if character_count == 0:
        return {"valid": False, "reason": "no_character", "count": 0}
    elif character_count == 1:
        return {"valid": True, "optimal_for_lora": True, "count": 1}
    else:
        return {"valid": True, "optimal_for_lora": False, "count": character_count, 
                "penalty": min(0.3 * (character_count - 1), 0.8)}
```

#### C. 全身抽出評価強化

```python
def evaluate_fullbody_extraction(bbox, image_size) -> Dict[str, Any]:
    """全身抽出評価"""
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_size
    
    # 縦横比分析（人体は縦長が期待される）
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 0
    
    # 画像に対する面積比
    bbox_area = bbox_w * bbox_h
    img_area = img_w * img_h
    area_ratio = bbox_area / img_area
    
    # 全身判定基準
    fullbody_score = 0.0
    reasons = []
    
    # 縦横比チェック（人体: 1.5-4.0が期待される）
    if 1.5 <= aspect_ratio <= 4.0:
        fullbody_score += 0.4
    else:
        reasons.append(f"aspect_ratio_{aspect_ratio:.2f}_not_human_like")
    
    # 面積比チェック（全身なら5-40%程度）
    if 0.05 <= area_ratio <= 0.4:
        fullbody_score += 0.3
    else:
        reasons.append(f"area_ratio_{area_ratio:.3f}_not_fullbody")
    
    # 位置チェック（全身なら上端が画像上部に近い）
    top_margin = y1 / img_h
    if top_margin < 0.15:  # 上端15%以内
        fullbody_score += 0.3
    else:
        reasons.append(f"top_margin_{top_margin:.3f}_not_fullbody_top")
    
    return {
        "fullbody_score": fullbody_score,
        "is_fullbody": fullbody_score >= 0.6,
        "reasons": reasons,
        "metrics": {
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "top_margin": top_margin
        }
    }
```

### Phase 2: システム改善（中期）

#### A. 統合評価システム修正

現在の`unified_quality_checker.py`に追加検証を統合：

```python
def enhanced_quality_validation(self, extraction_data: Dict, output_dir: Path) -> List[QualityMetric]:
    """強化品質検証"""
    
    enhanced_metrics = []
    
    # 抽出画像ファイル取得
    extracted_files = list(output_dir.glob("*_extracted.*"))
    
    true_success_count = 0
    lora_optimal_count = 0
    fullbody_success_count = 0
    
    for img_file in extracted_files:
        # 1. 内容検証
        content_validation = self.validate_extraction_content(str(img_file))
        
        # 2. 複数キャラクター検証
        multi_char_validation = self.detect_multiple_characters(str(img_file))
        
        # 3. 全身抽出検証（YOLOから元bbox情報取得必要）
        
        if content_validation["valid"]:
            true_success_count += 1
            
            if multi_char_validation["optimal_for_lora"]:
                lora_optimal_count += 1
    
    # 修正済み指標
    enhanced_metrics.append(QualityMetric(
        name="真の抽出成功率",
        value=true_success_count / len(extracted_files) if extracted_files else 0,
        threshold=0.7,
        status="passed" if (true_success_count / len(extracted_files) if extracted_files else 0) >= 0.7 else "failed",
        category="enhanced_validation",
        notes=f"内容検証済み: {true_success_count}/{len(extracted_files)}"
    ))
    
    enhanced_metrics.append(QualityMetric(
        name="LoRA最適化率",
        value=lora_optimal_count / len(extracted_files) if extracted_files else 0,
        threshold=0.8,
        status="passed" if (lora_optimal_count / len(extracted_files) if extracted_files else 0) >= 0.8 else "failed",
        category="enhanced_validation", 
        notes=f"1キャラクター最適: {lora_optimal_count}/{len(extracted_files)}"
    ))
    
    return enhanced_metrics
```

#### B. 新指標システム追加

```python
# 新しい指標カテゴリ
ENHANCED_THRESHOLDS = {
    "content_validity_rate": 0.85,      # 内容有効率
    "lora_optimization_rate": 0.80,     # LoRA最適化率  
    "fullbody_completion_rate": 0.60,   # 全身完成率
    "brightness_consistency": 0.7,      # 明度一貫性
    "character_count_accuracy": 0.90    # キャラ数精度
}
```

### Phase 3: 人物認識精度向上（長期）

#### A. YOLOプロンプト精度向上

```python
def improved_yolo_prompting(image, confidence_threshold=0.07):
    """改良YOLOプロンプト生成"""
    
    # 1. 複数信頼度での検出
    results_high = yolo_model(image, conf=confidence_threshold * 2)  # 高信頼度
    results_low = yolo_model(image, conf=confidence_threshold * 0.5)  # 低信頼度
    
    # 2. アンサンブル判定
    # 高信頼度で検出できた場合はそれを採用
    # そうでない場合は低信頼度結果を慎重に評価
    
    # 3. 形状分析による後処理
    # 人体らしい形状（縦横比、面積比）でフィルタリング
    
    return optimized_prompts
```

#### B. SAM境界設定最適化

```python
def adaptive_sam_prompting(yolo_bbox, image_shape):
    """適応的SAM境界設定"""
    
    # 1. 人体部位推定に基づく境界拡張
    # 頭部、手足が切れないよう動的に境界調整
    
    # 2. アニメキャラクター特化調整
    # 髪の毛、装飾品を考慮した拡張率
    
    # 3. 全身/上半身の自動判定
    # 画像内の位置とサイズから最適抽出範囲を決定
    
    return optimized_sam_prompts
```

## 実装優先順位

### 🔥 緊急（1週間以内）
1. 真っ黒画像検出機能追加
2. 複数キャラクター検出機能追加  
3. 既存unified_quality_checker.pyへの統合

### ⚡ 高優先（2-3週間）
1. 全身抽出評価システム
2. 新指標システムの完全実装
3. テストスイート作成

### 📈 中優先（1-2ヶ月）
1. YOLOプロンプト精度向上
2. SAM境界設定最適化
3. 機械学習による品質予測モデル

## 期待される改善効果

- **真の高品質率**: 30% → 80%
- **LoRA最適化率**: 40% → 85%
- **ユーザー満足度**: 大幅向上
- **手動確認作業**: 80%削減
