# 統合品質チェックシステム実行ガイド

**バージョン**: 1.0  
**最終更新**: 2025-07-26  
**対象**: segment-anything プロジェクト全参加者

## 🎯 概要

統合品質チェックシステムは、現在の10指標（評価4 + マスク3 + 客観3指標）を用いて、キャラクター抽出の品質を包括的に評価・可視化するシステムです。すべてのバッチ処理・機能実装後に必須実行することで、継続的な品質向上を実現します。

## 📊 システム構成

### 🔍 統合品質チェック（主軸）
- **ファイル**: `tools/unified_quality_checker.py`
- **機能**: 抽出結果JSONから10指標の品質評価実行
- **出力**: 統合品質レポート（JSON形式）

### 📈 品質ダッシュボード（可視化）
- **ファイル**: `tools/quality_dashboard.py`
- **機能**: 品質レポートからHTMLダッシュボード生成
- **出力**: インタラクティブな品質可視化

### 🎯 客観指標テスト（開発用）
- **ファイル**: `features/evaluation/objective_metrics.py`
- **機能**: PLA/SCI/PLEの独立テスト・検証
- **出力**: 指標別詳細結果表示

## 🚀 標準実行ワークフロー

### ステップ1: 統合品質チェック実行

```bash
# 基本実行
python3 tools/unified_quality_checker.py --results path/to/extraction_results.json

# 出力パス指定
python3 tools/unified_quality_checker.py --results path/to/extraction_results.json --output custom_report.json

# 静音モード（サマリー非表示）
python3 tools/unified_quality_checker.py --results path/to/extraction_results.json --quiet
```

**実行例（kana08データ）**:
```bash
python3 tools/unified_quality_checker.py --results /mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge/extraction_results_kana08_final.json
```

### ステップ2: 品質ダッシュボード生成

```bash
# 基本実行（自動出力先）
python3 tools/quality_dashboard.py --report path/to/unified_quality_report.json

# 出力ディレクトリ指定
python3 tools/quality_dashboard.py --report path/to/unified_quality_report.json --output dashboard_output/
```

**実行例**:
```bash
python3 tools/quality_dashboard.py --report /mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge/unified_quality_report_kana08_20250726_071111.json
```

### ステップ3: 客観指標テスト（オプション）

```bash
# 全指標テスト
python3 features/evaluation/objective_metrics.py --test all

# 個別指標テスト
python3 features/evaluation/objective_metrics.py --test pla
python3 features/evaluation/objective_metrics.py --test sci  
python3 features/evaluation/objective_metrics.py --test ple
```

## 📋 出力ファイル・結果の解釈

### 統合品質レポート（JSON）

**ファイル形式**: `unified_quality_report_<dataset>_<timestamp>.json`

**主要項目**:
```json
{
  "overall_score": 0.2,           // 総合スコア（0-1）
  "passed_metrics": 2,            // 合格指標数
  "total_metrics": 10,            // 総指標数
  "status": "FAIL",               // 総合判定
  "evaluation_metrics": [...],    // 評価指標詳細
  "mask_metrics": [...],          // マスク品質詳細
  "objective_metrics": [...],     // 客観指標詳細
  "priority_improvements": [...], // 優先改善事項
  "technical_recommendations": [...]  // 技術推奨事項
}
```

### 品質ダッシュボード（HTML）

**ファイル**: `dashboard/dashboard.html` + 関連PNG画像

**表示項目**:
- 📊 総合指標レーダーチャート
- 📈 カテゴリ別合格率
- 📋 指標詳細比較
- 🎯 ステータス分布
- 🚀 改善優先度ランキング

### 客観指標テスト結果

**コンソール出力例**:
```
🎯 客観的3指標システムテスト
==================================================

📊 PLA (Pixel-Level Accuracy) テスト
結果: 0.839 (信頼度: 0.912)
ステータス: passed
詳細: IoU=0.839, Dice=0.912

🎭 SCI (Semantic Completeness Index) テスト  
結果: 0.463 (信頼度: 0.725)
ステータス: failed
詳細: Face=0.850, Limb=0.623, Contour=0.341

📈 PLE (Progressive Learning Efficiency) テスト
結果: 0.000 (信頼度: 0.000)
ステータス: insufficient_data
詳細: 履歴データ不足
```

## 🎯 品質基準・判定指標

### 総合品質スコア基準

| スコア範囲 | 評価レベル | 対応方針 |
|------------|------------|----------|
| 70%以上 | **優秀** | 現状維持・微調整 |
| 50-69% | **良好** | 部分改善実行 |
| 30-49% | **要改善** | 集中改善プロジェクト |
| 30%未満 | **緊急対応** | 全面見直し必要 |

### 個別指標重要度

#### 🔥 高優先度（必達指標）
- **FPS（処理速度）**: ≥ 0.2 → 基本性能要件
- **PLA（ピクセル精度）**: ≥ 0.75 → 抽出精度の基盤

#### ⚡ 中優先度（改善対象）
- **Largest-Character Accuracy**: ≥ 0.80 → 検出成功率
- **A/B評価率**: ≥ 0.70 → 品質満足度
- **SCI（意味完全性）**: ≥ 0.70 → 構造的品質

#### 📊 低優先度（監視対象）
- **マスク品質指標**: カバレッジ、コンパクトネス、フィル率
- **PLE（学習効率）**: 継続的改善効率

## 🚨 品質問題・対応ガイド

### 品質劣化検出時の対応手順

#### 1. 即座停止判定
**条件**: 総合スコア前回比-10%以上低下
```bash
# 前回レポートとの比較確認
diff previous_report.json current_report.json
```

#### 2. 原因調査
**詳細分析**:
- `unified_quality_report.json`の`priority_improvements`確認
- 失敗指標の`notes`・`improvement_suggestions`分析
- ダッシュボードでの視覚的問題特定

#### 3. 改善実装
**優先順位**:
1. **検出範囲拡張**（YOLO問題）
2. **輪郭後処理**（マスク品質問題）
3. **ノイズ除去**（SCI問題）
4. **SAM後処理改良**（全般品質問題）

#### 4. 再検証
```bash
# 改善後の品質チェック再実行
python3 tools/unified_quality_checker.py --results improved_results.json
python3 tools/quality_dashboard.py --report improved_report.json
```

### よくある品質問題・解決策

#### 問題1: 検出成功率低下（Largest-Character Accuracy < 0.8）
**原因**: YOLO閾値設定・モデル適合性
**解決策**: 
- アニメ特化YOLO使用確認
- 閾値調整（0.07 → アダプティブ）
- 前処理強化

#### 問題2: A/B評価率低迷（< 0.3）
**原因**: SAM後処理品質・マスク精度
**解決策**:
- エッジ精密化処理追加
- ノイズ除去アルゴリズム強化
- 境界線スムージング

#### 問題3: SCI低スコア（< 0.5）
**原因**: 輪郭計算エラー・重み配分
**解決策**:
- OpenCV Bool型 → uint8変換
- アニメ特化重み調整（顔60%, ポーズ25%, 輪郭15%）
- MediaPipe姿勢推定強化

## 📈 継続的品質向上

### 週次・月次目標

#### 週次目標
- **総合スコア**: +5%向上
- **最低1指標**: 閾値達成
- **改善実装**: 1つ以上完了

#### 月次目標  
- **総合スコア**: 70%達成
- **高優先度指標**: すべて合格
- **品質安定性**: 変動±5%以内

### 品質履歴管理

**ファイル**: `quality_history.json`
**用途**: PLE計算・長期トレンド分析
**更新**: 統合品質チェック実行時自動

## 🔗 関連ファイル・システム

### コアファイル
- `tools/unified_quality_checker.py` - 統合チェック本体
- `tools/quality_dashboard.py` - ダッシュボード生成
- `features/evaluation/objective_metrics.py` - 客観指標実装
- `features/processing/postprocessing/postprocessing.py` - マスク品質計算

### 設定・データファイル  
- `quality_history.json` - 品質履歴（PLE用）
- `unified_quality_report_*.json` - 品質レポート
- `dashboard/dashboard.html` - 品質ダッシュボード

### 依存関係
- OpenCV - 画像処理・輪郭解析
- MediaPipe - 姿勢推定（SCI計算）
- Matplotlib - グラフ・チャート生成
- NumPy - 数値計算

## ❓ トラブルシューティング

### Q1: "No module named 'mediapipe'" エラー
**A**: MediaPipeは必須依存関係ではありません
```bash
# オプション：MediaPipeインストール
pip install mediapipe

# または：フォールバック実装で継続実行
# → SCI計算で自動的にフォールバック処理実行
```

### Q2: 品質レポートJSONが見つからない
**A**: 統合品質チェック実行後に生成される自動ファイル名を確認
```bash
# ファイル名パターン確認
ls unified_quality_report_*_*.json

# 最新ファイル使用
python3 tools/quality_dashboard.py --report $(ls -t unified_quality_report_*.json | head -1)
```

### Q3: ダッシュボードが空白表示
**A**: PNG画像生成エラーの可能性
```bash
# 実行時エラーログ確認
python3 tools/quality_dashboard.py --report report.json 2>&1 | grep ERROR

# 出力ディレクトリ権限確認
ls -la dashboard/
```

### Q4: 客観指標テストでエラー多発
**A**: テスト画像データ・依存関係確認
```bash
# 依存関係再インストール
pip install -r requirements.txt

# テスト用画像データ確認
python3 features/evaluation/objective_metrics.py --test sci
```

## 🎉 期待される効果

### 品質向上効果
- **継続的監視**: 品質劣化の早期発見
- **データ駆動改善**: 具体的数値基準による改善
- **視覚的理解**: ダッシュボードによる直感的把握

### 開発効率向上  
- **自動化**: 手動評価作業の大幅削減
- **標準化**: 一貫した品質評価基準
- **トレーサビリティ**: 改善履歴・効果の可視化

### プロジェクト価値向上
- **品質保証**: 客観的品質基準の確立
- **継続改善**: PDCA サイクルの自動化
- **ベストプラクティス**: 再現可能な品質向上手法

---

**注意**: このシステムは継続的に改善されます。最新情報は PROGRESS_TRACKER.md を参照してください。