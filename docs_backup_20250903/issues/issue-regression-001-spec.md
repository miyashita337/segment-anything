# 技術仕様書 - ISSUE-REG-001

**ISSUE ID**: ISSUE-REG-001  
**タイトル**: v0.4.0全身抽出回帰問題修正  
**作成日**: 2025-07-21  
**最終更新**: 2025-07-21

## 🎯 修正目標

### 主要目標
1. **全身抽出機能の復活**: v0.3.5レベル（90%以上成功率）の達成
2. **品質基準達成**: B評価50%以上の達成
3. **P1-006機能保持**: 可能な限りSolid Fill Detection機能を維持

### 副次目標
- 背景誤検出率の維持（5%以下）
- 処理速度の劣化防止
- コードの保守性確保

## 🔧 修正対象ファイル

### 1. enhanced_solid_fill_processor.py (最重要)
**ファイルパス**: `/features/evaluation/utils/enhanced_solid_fill_processor.py`

#### 修正箇所
```python
# 現在のコード（346行目付近）
edge_threshold = 0.1
if (centroid_y < h * edge_threshold or centroid_y > h * (1 - edge_threshold) or
    centroid_x < w * edge_threshold or centroid_x > w * (1 - edge_threshold)):
    return 'background'

# 修正後のコード
def _get_adaptive_edge_threshold(self, image_shape: Tuple[int, int]) -> float:
    """適応的エッジ閾値の計算"""
    h, w = image_shape
    # 最大24px、最小4px、デフォルト3%
    pixel_margin = max(min(24, w * 0.03), 4)
    return pixel_margin / w

def _classify_region_type(self, mask: np.ndarray, image: np.ndarray, 
                        color: Tuple[int, int, int],
                        yolo_boxes: Optional[List] = None,
                        sam_masks: Optional[List] = None) -> str:
    """YOLO/SAM情報を考慮した領域分類"""
    # 適応的エッジ閾値
    edge_threshold = self._get_adaptive_edge_threshold(mask.shape)
    
    # YOLO/SAMボックス内かチェック
    if self._is_in_detection_area(mask, yolo_boxes, sam_masks):
        # 検出領域内なら背景分類を抑制
        edge_penalty = 0.3  # 軽減
    else:
        edge_penalty = 1.0  # 通常
```

#### 追加メソッド
- `_get_adaptive_edge_threshold()`: 適応的エッジ閾値計算
- `_is_in_detection_area()`: YOLO/SAM領域との重複判定
- `_apply_weighted_classification()`: 重み付き分類システム

### 2. extract_character.py (中重要)
**ファイルパス**: `/features/extraction/commands/extract_character.py`

#### 修正箇所（336-349行目付近）
```python
# 現在のコード
processed_image_path = processor.preprocess_for_difficult_pose(
    image_path,
    enable_manga_preprocessing=True,
    enable_effect_removal=effect_removal,
    enable_panel_split=panel_split,
    solid_fill_detection=solid_fill_detection
)

# 修正後のコード
processed_image_path = processor.preprocess_for_difficult_pose(
    image_path,
    enable_manga_preprocessing=True,
    enable_effect_removal=effect_removal,
    enable_panel_split=panel_split,
    solid_fill_detection=solid_fill_detection,
    yolo_boxes=yolo_detections,  # YOLO結果を渡す
    sam_masks=sam_preliminary_masks  # SAM予備結果を渡す
)
```

#### フォールバック機構追加
```python
# 結果検証とフォールバック
if result.get('success') and result.get('mask') is not None:
    mask_height = np.sum(result['mask'], axis=0).max()
    yolo_height = max([box[3] - box[1] for box in yolo_detections]) if yolo_detections else 0
    
    # 高さ比較による品質チェック
    if yolo_height > 0 and mask_height < yolo_height * 0.75:
        if verbose:
            print(f"⚠️ 抽出結果が小さすぎます（{mask_height/yolo_height:.1%}）。フォールバック実行中...")
        
        # Solid fill無効でリトライ
        fallback_result = processor.preprocess_for_difficult_pose(
            image_path,
            solid_fill_detection=False,
            **other_params
        )
        
        # IoUが高い方を選択
        if self._compare_iou(result, fallback_result, yolo_detections):
            result = fallback_result
```

### 3. difficult_pose.py (低重要)
**ファイルパス**: `/features/evaluation/utils/difficult_pose.py`

#### 修正箇所
```python
def preprocess_for_difficult_pose(self, image_path: str, output_path: Optional[str] = None, 
                                  enable_manga_preprocessing: bool = False,
                                  enable_effect_removal: bool = False,
                                  enable_panel_split: bool = False,
                                  enable_solid_fill_detection: bool = False,
                                  yolo_boxes: Optional[List] = None,
                                  sam_masks: Optional[List] = None) -> str:
    """前処理にYOLO/SAM情報を追加"""
    
    if enable_solid_fill_detection:
        # Solid fill processorにYOLO/SAM情報を渡す
        solid_processor = EnhancedSolidFillProcessor()
        processed_image = solid_processor.process_with_context(
            image, yolo_boxes, sam_masks
        )
```

## 🔢 パラメータ調整案

### Solid Fill Detection パラメータ
```yaml
# 現在の設定 → 修正後設定

エッジ処理:
  edge_threshold: 0.10 → adaptive(0.03, cap=24px, floor=4px)

色均一性:
  uniformity_threshold: 0.95 → 0.92
  sigma_L_threshold: 3 → 6
  sigma_ab_threshold: 4 → 8

領域サイズ:
  min_region_size: 100px → 0.002 * image_area
  min_region_area: 0.005 → 0.002

境界品質:
  sharpness_threshold: 0.4 → 0.25
  compactness_weight: 1.0 → 0.6
  uniformity_weight: 1.0 → 0.6

分類重み:
  edge_position_weight: 1.0 → 0.7
  color_similarity_weight: 1.0 → 0.8
  yolo_sam_prior_weight: 0.0 → 1.2 (新規)
```

### ランキングアルゴリズム
```python
# 現在: boundary_quality優先
region_score = boundary_quality

# 修正後: 面積×キャラクター確率優先
region_score = (
    region.area * region.character_probability * 0.6 +
    region.boundary_quality * 0.4
)
```

## 🧪 テスト要件

### 1. 回帰テストセット
- **データセット**: kana08（26画像）
- **基準画像**: C:\AItools\lora\train\yado\clipped_boundingbox\kana08_0_4_0
- **比較対象**: v0.3.5、v0.4.0、v0.4.1（修正版）

### 2. 性能指標
```yaml
必須指標:
  - 全身抽出成功率: ≥90%
  - B評価達成率: ≥50% 
  - 背景誤検出率: ≤5%

監視指標:
  - 処理時間: v0.4.0比±20%以内
  - メモリ使用量: v0.4.0比+10%以内
  - A評価率: 可能な限り向上

品質分類:
  - A評価: 完璧な全身抽出
  - B評価: 軽微な問題があるが使用可能
  - C評価: 中程度の問題（手足の一部欠損等）
  - D評価以下: 使用困難
```

### 3. テスト手順
1. **事前準備**
   ```bash
   # テスト環境準備
   cd /mnt/c/AItools/segment-anything
   git checkout v0.3.5
   python tools/test_batch_extraction.py --dataset kana08 --output v0.3.5_results
   
   git checkout v0.4.0
   python tools/test_batch_extraction.py --dataset kana08 --output v0.4.0_results
   ```

2. **修正版テスト**
   ```bash
   git checkout issue-reg-001-fix
   python tools/test_batch_extraction.py --dataset kana08 --output v0.4.1_results
   ```

3. **結果比較**
   ```bash
   python tools/compare_extraction_results.py \
     --v0.3.5 v0.3.5_results \
     --v0.4.0 v0.4.0_results \
     --v0.4.1 v0.4.1_results \
     --generate-report
   ```

## 🚨 リスク分析

### 高リスク
1. **過度な調整による副作用**
   - 対策: 段階的なパラメータ調整
   - 監視: 各調整後の個別テスト実行

2. **YOLO/SAM統合の複雑化**
   - 対策: 統合ロジックの単純化
   - 監視: パフォーマンスベンチマーク

### 中リスク
1. **特定画像タイプでの劣化**
   - 対策: 多様なテストデータでの検証
   - 監視: エッジケースの継続的収集

2. **メモリ使用量増加**
   - 対策: メモリ効率的な実装
   - 監視: メモリプロファイリング

## 📋 実装チェックリスト

### Phase 1: 基本修正
- [ ] `enhanced_solid_fill_processor.py`の適応的エッジ閾値実装
- [ ] 色均一性パラメータの調整
- [ ] 基本的な回帰テスト実行

### Phase 2: 統合改善
- [ ] YOLO/SAM情報の統合
- [ ] 重み付き分類システムの実装
- [ ] フォールバック機構の追加

### Phase 3: 最適化
- [ ] ランキングアルゴリズムの調整
- [ ] パフォーマンス最適化
- [ ] 最終的な品質検証

## 🎯 成功基準

### 必須条件（すべて満たす必要あり）
- [ ] 全身抽出成功率 ≥ 90%
- [ ] B評価達成率 ≥ 50%
- [ ] 背景誤検出率 ≤ 5%
- [ ] 処理時間の著しい劣化なし（+50%以内）

### 理想条件（可能な限り達成）
- [ ] A評価率の向上（目標: 30%以上）
- [ ] 処理時間の改善
- [ ] メモリ使用量の最適化
- [ ] コードの可読性向上

---

**関連ドキュメント**:
- [AI協議議事録](./issue-regression-001-discussion.md)
- [PROGRESS_TRACKER.md](../workflows/PROGRESS_TRACKER.md)
- [spec.md](../../spec.md)