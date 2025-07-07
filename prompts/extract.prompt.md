# Character Extraction Command

## 概要

`/extract_character` コマンドは、Segment Anything Model (SAM) と YOLOv8 を組み合わせて、漫画画像からキャラクターを自動的に抽出するコマンドです。

## 使用方法

### 基本的な使い方

```bash
# 単一画像の処理（基本）
/extract_character path/to/image.jpg

# 出力先を指定
/extract_character path/to/image.jpg -o output/character

# マスクと透明背景版も保存
/extract_character path/to/image.jpg --save-mask --save-transparent

# バッチ処理
/extract_character input_directory/ --batch -o output_directory/

# バッチ処理（全オプション）
/extract_character input_directory/ --batch -o output_directory/ --save-mask --save-transparent

# Phase 4使用例（推奨: 成功率 52% → 70-75%）
/extract_character image.jpg --phase4 --save-mask --save-transparent

# 個別機能の使用例
/extract_character image.jpg --mask-inversion-detection --adaptive-range --quality-prediction

# 漫画処理（Phase 2 + Phase 4）
/extract_character manga_page.jpg --manga-mode --effect-removal --phase4
```

### 基本オプション

- `--enhance-contrast`: 画像のコントラストを強化
- `--filter-text`: テキスト領域をフィルタリング（デフォルト: ON）
- `--save-mask`: マスクファイルも保存（デフォルト: OFF）
- `--save-transparent`: 透明背景版も保存（デフォルト: OFF）
- `--min-yolo-score FLOAT`: YOLO最小スコア閾値（デフォルト: 0.1）
- `--verbose`: 詳細な出力を表示

### 高度なオプション

#### 複雑ポーズ・ダイナミック構図対応
- `--difficult-pose`: 複雑ポーズ処理モードを有効化
- `--low-threshold`: 低閾値設定を使用（YOLO スコア 0.02）
- `--auto-retry`: プログレッシブ設定での自動リトライを有効化
- `--high-quality`: 高品質SAM処理を有効化

#### Phase 2: 漫画前処理オプション
- `--manga-mode`: 漫画特化前処理を有効化
- `--effect-removal`: エフェクト線除去を有効化
- `--panel-split`: マルチコマ分割を有効化

#### Phase 4: 統合システムオプション（🆕最新機能）
- `--phase4`: Phase 4統合システムを有効化（全機能ON）
- `--mask-inversion-detection`: マスク逆転検出・修正を有効化
- `--adaptive-range`: 適応的抽出範囲調整を有効化
- `--quality-prediction`: 品質予測・フィードバックを有効化

## 処理フロー

### 従来処理フロー
1. **画像前処理**: 画像読み込み、リサイズ、正規化
2. **SAM マスク生成**: 全領域のセグメンテーション
3. **キャラクター候補フィルタリング**: 面積、アスペクト比による絞り込み
4. **YOLO スコアリング**: 人物検出スコアによる最適マスク選択
5. **テキストフィルタリング**: テキスト領域の除外（オプション）
6. **マスク精製**: エッジ平滑化、ノイズ除去
7. **キャラクター抽出**: 背景除去、クロップ
8. **結果保存**: 抽出画像、マスク、透明背景版の保存

### Phase 4統合処理フロー（🆕最新）
1. **画像前処理**: 画像読み込み、リサイズ、正規化
2. **YOLO検出**: 人物領域の事前検出
3. **品質予測**: 抽出前の品質予測とリスク要因特定
4. **姿勢分析**: 体部位検出と複雑度分析
5. **適応的範囲調整**: 姿勢に応じた抽出範囲動的調整
6. **SAM処理**: 最適化されたパラメータでマスク生成
7. **マスク逆転検出・修正**: 自動的な品質チェックと修正
8. **フィードバック学習**: 結果を次回処理に反映
9. **結果保存**: 高品質な抽出結果の保存

## 出力ファイル

- `{name}.jpg`: 抽出されたキャラクター画像（黒背景）
- `{name}_mask.png`: セグメンテーションマスク（`--save-mask`オプション時）
- `{name}_transparent.png`: 透明背景のキャラクター画像（`--save-transparent`オプション時）

## 要件

- SAM model checkpoint: `sam_vit_h_4b8939.pth`
- YOLO model: `yolov8n.pt` (自動ダウンロード)
- 推奨GPU メモリ: 6GB以上
- 推奨RAM: 8GB以上

## 性能向上効果

### Phase 4による改善効果
- **成功率向上**: 52.2% → 70-75% (+18-23%改善)
- **マスク逆転問題**: 60-80%削減 (5ケース → 1-2ケース)
- **抽出範囲問題**: 50-62%削減 (8ケース → 3-4ケース)
- **A評価比率**: 43% → 60-65%

### 各機能の効果
- `--mask-inversion-detection`: マスク逆転自動修正（66.7%実行率）
- `--adaptive-range`: 姿勢に応じた範囲調整（100%実行率）
- `--quality-prediction`: 事前品質予測とパラメータ最適化
- `--phase4`: 全機能統合で最大性能

## パフォーマンスメトリクス

コマンド実行時に以下の情報が表示されます：

- 処理時間の内訳
- メモリ使用量（RAM/GPU）
- マスク品質メトリクス
- YOLO検出スコア

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - 画像サイズを小さくする
   - バッチサイズを減らす
   - GPUメモリを確認

2. **キャラクターが検出されない**
   - `--min-yolo-score` を下げる（例: 0.05）
   - `--low-threshold` を使用（YOLO スコア 0.02）
   - `--enhance-contrast` を試す
   - 画像品質を確認

3. **テキストと誤認識**
   - `--filter-text` が有効か確認
   - テキスト密度閾値を調整

4. **マスクが逆転している**
   - `--mask-inversion-detection` を有効化（Phase 4）
   - `--phase4` で自動修正

5. **抽出範囲が不適切**
   - `--adaptive-range` を有効化（Phase 4）
   - `--difficult-pose` を試す
   - `--phase4` で自動調整

6. **複雑な姿勢・動的ポーズ**
   - `--phase4` を使用（推奨）
   - `--difficult-pose` + `--high-quality` の組み合わせ
   - `--auto-retry` で複数設定を自動試行

7. **漫画特有の問題（エフェクト線、マルチコマ）**
   - `--manga-mode` + `--effect-removal` + `--panel-split`
   - `--phase4` と組み合わせで最高性能

### ログ出力の見方

#### 従来処理の場合
```
🔄 開始: SAM Model Loading (RAM: 2048.1MB, GPU: 0.0MB)
✅ 完了: SAM Model Loading (1.23秒, RAM: 4096.2MB, GPU: 2048.1MB)
📊 生成マスク: 1245 → キャラクター候補: 23
🎯 最適マスク選択: YOLO score=0.456, combined score=0.623
📐 マスク品質: coverage=0.234, compactness=0.789
✅ キャラクター抽出完了: 5.67秒
```

#### Phase 4処理の場合
```
🚀 Phase 4統合システム実行中...
🎯 YOLO検出: bbox=(6, 35, 548, 975), confidence=0.475
✅ Phase 4処理成功
   品質スコア: 0.339
   実行調整: ['適応的範囲調整実行', 'マスク逆転修正']
   処理時間: 0.068秒
✅ Phase 4キャラクター抽出完了: 2.86秒
```

## API使用例

```python
from commands.extract_character import extract_character_from_path

# 従来処理
result = extract_character_from_path(
    image_path="manga_page.jpg",
    output_path="output/character",
    enhance_contrast=True,
    min_yolo_score=0.15
)

# Phase 4統合システム（推奨）
result = extract_character_from_path(
    image_path="manga_page.jpg",
    output_path="output/character",
    enable_phase4=True,
    save_mask=True,
    save_transparent=True
)

# 個別機能の使用
result = extract_character_from_path(
    image_path="complex_pose.jpg",
    output_path="output/character",
    enable_mask_inversion_detection=True,
    enable_adaptive_range=True,
    enable_quality_prediction=True,
    difficult_pose=True,
    high_quality=True
)

if result['success']:
    print(f"抽出成功: {result['output_path']}")
    print(f"処理時間: {result['processing_time']:.2f}秒")
    if 'phase4_stats' in result:
        print(f"Phase 4調整: {result['phase4_adjustments']}")
        print(f"品質スコア: {result['mask_quality']['coverage_ratio']:.3f}")
else:
    print(f"抽出失敗: {result['error']}")
```

## 関連コマンド

- `/start`: モデル初期化（自動実行）
- `test_phase4_system.py`: Phase 4システムの動作テスト
- Phase 3インタラクティブ機能: `commands/interactive_extract.py`
- サンプル画像: `assets/masks1.png`

## Phase 4技術詳細

### マスク逆転検出・修正
- 色彩複雑度比分析による自動判定
- エッジ一貫性チェック
- 自動反転修正機能

### 適応的範囲調整
- 姿勢複雑度分析（Simple/Dynamic/Complex/Extreme）
- 体部位検出（頭部、胴体、足部、腕部）
- 動的バウンディングボックス拡張

### 品質予測・フィードバック
- 抽出前のリスク要因特定
- 最適パラメータの自動選択
- 処理結果の学習・改善

### 統合システム
- 最大3回の反復処理
- リアルタイム品質モニタリング
- 自動パラメータ調整