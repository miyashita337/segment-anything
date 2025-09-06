# 普遍的仕様書（変更禁止事項）

**⚠️ 重要**: この情報は [`docs/technical_specifications.md`](technical_specifications.md) に統合されました。  
**最新情報については**: `docs/technical_specifications.md` を参照してください。

**作成日**: 2025-07-24  
**重要度**: 最高  
**変更制限**: このドキュメントの項目は基本的に変更禁止

## 🔒 変更すべきでない普遍的仕様

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

### 4. 品質評価原則
```yaml
# 評価システム基盤（変更不可）
evaluation_foundation:
  pla_calculation: "IoU（Intersection over Union）"
  sci_calculation: "MediaPipe + OpenCV構造解析" 
  ple_calculation: "時系列進捗効率分析"
  objectivity_requirement: "完全客観的数値評価"
  reproducibility: "100%再現可能な結果"
```

### 5. ファイル構造原則
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

## 🔄 変更可能な部分（実装詳細）

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

### アルゴリズム改善
```yaml
# 改善可能な領域
improvable_areas:
  preprocessing_methods: "前処理手法の改良"
  postprocessing_steps: "後処理ステップの追加"
  quality_calculation_details: "品質計算の詳細改善"
  performance_optimization: "実行速度の最適化"
  error_handling_enhancement: "エラー処理の強化"
```

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
- [ ] 客観的品質評価を使用しているか？

### ✅ 互換性確認
- [ ] 既存のspec.mdとの整合性があるか？
- [ ] PRINCIPLE.mdのセキュリティ原則に準拠しているか？
- [ ] 3指標システム（PLA/SCI/PLE）を活用しているか？

## 🚨 緊急時対応

### 普遍的仕様違反の発見時
1. **即座に実装を停止**
2. **違反内容を documenta-tion/issues/ に記録**
3. **原因分析と修正計画を策定**
4. **修正後に普遍性チェックリストで再確認**

### 仕様変更の検討が必要な場合
1. **仕様変更の必要性を文書化**
2. **影響範囲の詳細分析**
3. **段階的移行計画の策定**
4. **全テストスイートでの動作確認**

---

**重要**: この普遍的仕様書は、プロジェクトの根幹を成す不変の要素です。  
変更は慎重に検討し、必ず全体影響を評価してから実行してください。