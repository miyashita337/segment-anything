# Phase 1: 全身検出アルゴリズム分析レポート

**作成日**: 2025-07-17  
**タスク**: [P1-001] 全身検出アルゴリズム分析  
**目的**: 現在のfullbody_priority手法の詳細分析と改善方針策定

---

## 🔍 現在のfullbody_priority手法分析

### 実装場所
- **主要実装**: `features/extraction/models/yolo_wrapper.py:315`
- **統合箇所**: `features/evaluation/utils/learned_quality_assessment.py`
- **重み設定**: fullbody: 40%, area: 20%, central: 15%, grounded: 15%, confidence: 10%

### 現在のアルゴリズム

#### 1. アスペクト比ベース全身判定（yolo_wrapper.py:282-288）
```python
# 全身キャラクター判定の現在実装
aspect_ratio = bbox[3] / max(bbox[2], 1)  # height / width
if 1.2 <= aspect_ratio <= 2.5:  # 全身キャラクター範囲
    scores['fullbody'] = min((aspect_ratio - 0.5) / 2.0, 1.0)
else:
    scores['fullbody'] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)
```

#### 2. 画像特性による全身検出（learned_quality_assessment.py:170-188）
```python
# has_full_body判定
if aspect_ratio >= 1.5:  # 縦長画像
    has_full_body = True
elif height > width * 1.2:  # 高さが幅の1.2倍以上
    has_full_body = True
else:
    has_full_body = False
```

---

## 📊 現在手法の問題点分析

### 1. 単一指標依存の限界
- **問題**: アスペクト比のみに依存する判定
- **影響**: 複雑ポーズ、部分隠蔽、デフォルメキャラクターで誤判定
- **例**: 座った姿勢（アスペクト比1.0）が全身でも顔判定される

### 2. 閾値の固定化
- **問題**: 1.2-2.5の固定範囲では多様性に対応不可
- **影響**: キャラクタースタイル（SD、ちびキャラ等）で適応不良
- **例**: デフォルメキャラは1.0-1.5の範囲が多い

### 3. 人体構造の非考慮
- **問題**: 顔・胴体・手足の位置関係を無視
- **影響**: 顔だけ/足だけ抽出の防止不可
- **例**: 縦長フレームでも顔のみが大きく写った画像を全身と誤判定

### 4. コンテキスト情報の不足
- **問題**: 画像内容（背景、キャラクター数）を考慮しない
- **影響**: 複数キャラクターでの誤選択
- **例**: 複数キャラクターのうち部分的なキャラクターを選択

---

## 🎯 改善戦略の方向性

### 1. 多指標統合アプローチ
- **アスペクト比**: 基本指標として継続使用
- **人体構造認識**: 顔・胴体・手足の検出と位置関係分析
- **エッジ分布**: 境界線パターンによる完全性評価
- **セマンティック領域**: 肌色、髪色、服装の分布分析

### 2. 動的閾値システム
- **画像サイズ適応**: 解像度に応じた閾値調整
- **スタイル適応**: 漫画/アニメ/リアル系での最適化
- **キャラクター種別**: SD/等身/デフォルメ別パラメータ
- **学習ベース調整**: 成功例から最適閾値を学習

### 3. 階層的判定システム
- **Level 1**: 粗い判定（現在のアスペクト比ベース）
- **Level 2**: 人体構造による詳細判定
- **Level 3**: 機械学習による最終判定
- **フォールバック**: 判定困難時の安全な選択

---

## 🛠️ 実装すべき新機能

### 1. 人体構造認識モジュール
```python
class HumanStructureAnalyzer:
    """人体構造分析クラス"""
    
    def analyze_body_completeness(self, image, mask):
        """身体の完全性を分析"""
        return {
            'has_face': bool,
            'has_torso': bool, 
            'has_arms': bool,
            'has_legs': bool,
            'completeness_score': float,
            'missing_parts': List[str]
        }
    
    def detect_face_region(self, image):
        """顔領域の検出"""
        # OpenCV Face Cascade or dlib implementation
        
    def analyze_limb_distribution(self, mask):
        """手足の分布分析"""
        # エッジ検出による手足位置の推定
```

### 2. 改良版全身判定システム
```python
class EnhancedFullBodyDetector:
    """改良版全身検出システム"""
    
    def evaluate_fullbody_score(self, image, mask_data):
        """多指標による全身スコア評価"""
        scores = {
            'aspect_ratio': self._aspect_ratio_score(mask_data),
            'body_structure': self._body_structure_score(image, mask_data),
            'edge_distribution': self._edge_distribution_score(mask_data),
            'semantic_regions': self._semantic_region_score(image, mask_data)
        }
        
        # 重み付き統合
        weights = {'aspect_ratio': 0.3, 'body_structure': 0.4, 
                   'edge_distribution': 0.2, 'semantic_regions': 0.1}
        
        return sum(scores[k] * weights[k] for k in scores.keys())
```

### 3. 部分抽出検出システム
```python
class PartialExtractionDetector:
    """部分抽出検出システム"""
    
    def detect_incomplete_extraction(self, image, mask):
        """不完全抽出の検出"""
        issues = []
        
        # 顔のみ抽出検出
        if self._is_face_only(image, mask):
            issues.append('face_only')
            
        # 手足切断検出
        if self._has_limb_truncation(mask):
            issues.append('limb_truncated')
            
        # 胴体欠損検出
        if self._missing_torso(image, mask):
            issues.append('torso_missing')
            
        return issues
```

---

## 📋 次の実装ステップ（P1-002, P1-003対応）

### Phase 1.1: 基盤技術実装（今週）
1. **人体構造認識の基礎実装**
   - OpenCV/dlibベースの顔検出統合
   - エッジ検出による手足分析
   - セマンティック領域分析

2. **部分抽出検出システム**
   - 既存マスクの完全性評価
   - 問題パターンの分類
   - 修正提案の生成

3. **テストケース作成**
   - test_small/での検証
   - 多様なキャラクタータイプでのテスト
   - 成功/失敗ケースの定量評価

### Phase 1.2: 統合・最適化（来週）
1. **既存システムとの統合**
   - YOLOWrapperへの新機能統合
   - LearnedQualityAssessmentへの反映
   - extract_character.pyでの利用

2. **パフォーマンス最適化**
   - GPU利用の最適化
   - 処理時間の短縮
   - メモリ使用量の削減

3. **バリデーション**
   - kana06データセットでの検証
   - 成功率の定量評価
   - エラーケースの分析

---

## 🎯 期待される改善効果

### 定量的改善目標
- **全身検出精度**: 現在70%推定 → 目標90%+
- **部分抽出の減少**: 顔のみ/手足切断を50%削減
- **総合成功率**: 32.6% → 45%+（Phase 1全体目標）

### 定性的改善
- 多様なキャラクタースタイルへの対応
- 複雑ポーズでの安定性向上
- ユーザーの手動修正頻度の減少

---

*この分析レポートは P1-002（部分抽出検出システム実装）と P1-003（全身判定基準の改善）の設計基盤となります。*