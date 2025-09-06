# INTG-002 根本原因分析報告書

**実行日時**: 2025-07-26 23:40  
**分析対象**: Phase 1改善後の品質劣化問題（総合スコア28.6%）

## 🔍 根本原因特定結果

### 原因1: Phase 1改善コードのdeprecated化
**問題**: コミット23fa1ea「Project Structure Cleanup」でPhase 1の改善機能が除外

**影響範囲**:
```yaml
移動されたPhase 1改善ファイル:
  - deprecated/temp_implementations/kana08_stable_batch.py  # 61.5%成功率実績
  - features/evaluation/enhanced_sci_engine.py             # SCI計算改善
  - features/extraction/quality_guard_system.py            # 品質保護システム
  - features/extraction/robust_extractor.py                # 安定抽出器
  - features/processing/limb_protection_system.py          # 手足切断防止
```

### 原因2: 古いツールバージョンの使用
**問題**: 現在のパイプラインがv0.0.1の古いツールを使用

**技術詳細**:
```python
現在使用中: features/extraction/commands/extract_character.py（新アーキテクチャ）
- 基本的なYOLO+SAM機能のみ
- 品質評価機能なし
- MediaPipe Pose未統合
- A/B評価システムなし

Phase 1改善版: deprecated/temp_implementations/kana08_stable_batch.py
- Kana08StableExtractor実装
- 品質評価・改善システム統合
- MediaPipe Pose最適化適用
- confidence_threshold=0.07（アニメ特化）
```

## 📊 品質劣化の技術的詳細

### A/B評価率: 75% → 0% (-100%劣化)
**原因**: 品質評価システムの完全欠如
```python
# Phase 1改善版（deprecated）
features/extraction/quality_guard_system.py:
  - A評価保護機能
  - 品質スコア閾値管理
  - 適応的改善処理

# 現在使用中（新アーキテクチャ）
features/extraction/commands/extract_character.py:
  - 品質評価機能なし
  - すべて未評価のまま出力
```

### SCI値: 0.853 → 0.400 (-53.1%劣化)
**原因**: 強化SCI計算エンジンの未使用
```python
# Phase 1改善版（deprecated）
features/evaluation/enhanced_sci_engine.py:
  - EnhancedFaceDetector統合
  - EnhancedPoseDetector統合
  - MediaPipe最適化適用

# 現在使用中（v0.0.1）
基本的なSCI計算のみ:
  - 顔検出率: 低精度
  - ポーズ検出: 未最適化
  - 統合評価なし
```

### 処理成功率: 62% → 100% (+38%改善)
**原因**: 基本的なYOLO+SAM機能は正常動作
- 単純な検出・抽出は機能
- 品質は考慮せず全て「成功」として処理

## 🔧 解決方針

### Phase 1: 緊急復旧（即座実行）
1. **deprecated/temp_implementations/kana08_stable_batch.py**をメインツールに統合
2. **features/evaluation/enhanced_sci_engine.py**を有効化
3. **品質評価システム**の再統合

### Phase 2: システム統合（1-2日）
1. **tools/run_auto_pipeline.py**にPhase 1改善を統合
2. **品質保証ワークフロー**の自動化
3. **劣化防止システム**の実装

## 📋 期待効果

### 復旧後の予想品質指標:
```yaml
A/B評価率: 0% → 75% (Phase 1実績に回復)
SCI値: 0.400 → 0.853 (Phase 1実績に回復)
処理成功率: 100% (現在の水準維持)
総合品質スコア: 28.6% → 85%+ (大幅改善)
```

## 🚨 教訓と再発防止

### 重要な学び
1. **ファイル整理時の機能影響分析不足**
2. **品質テストの実行タイミング問題**
3. **改善機能の依存関係管理不備**

### 再発防止策
1. **必須機能のコア認定**（整理対象外指定）
2. **品質テスト自動化**（整理後即座実行）
3. **機能復旧手順書**の作成

---

**結論**: Phase 1の真の改善は存在したが、ファイル整理により無効化された。  
deprecated機能を復元することで、品質を即座にPhase 1水準まで回復可能。