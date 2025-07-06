# Character Extraction Command

## 概要

`/extract_character` コマンドは、Segment Anything Model (SAM) と YOLOv8 を組み合わせて、漫画画像からキャラクターを自動的に抽出するコマンドです。

## 使用方法

### 基本的な使い方

```bash
# 単一画像の処理
/extract_character path/to/image.jpg

# 出力先を指定
/extract_character path/to/image.jpg -o output/character

# バッチ処理
/extract_character input_directory/ --batch -o output_directory/
```

### オプション

- `--enhance-contrast`: 画像のコントラストを強化
- `--filter-text`: テキスト領域をフィルタリング（デフォルト: ON）
- `--save-mask`: マスクファイルも保存（デフォルト: ON）
- `--save-transparent`: 透明背景版も保存（デフォルト: ON）
- `--min-yolo-score FLOAT`: YOLO最小スコア閾値（デフォルト: 0.1）
- `--verbose`: 詳細な出力を表示

## 処理フロー

1. **画像前処理**: 画像読み込み、リサイズ、正規化
2. **SAM マスク生成**: 全領域のセグメンテーション
3. **キャラクター候補フィルタリング**: 面積、アスペクト比による絞り込み
4. **YOLO スコアリング**: 人物検出スコアによる最適マスク選択
5. **テキストフィルタリング**: テキスト領域の除外（オプション）
6. **マスク精製**: エッジ平滑化、ノイズ除去
7. **キャラクター抽出**: 背景除去、クロップ
8. **結果保存**: 抽出画像、マスク、透明背景版の保存

## 出力ファイル

- `{name}.jpg`: 抽出されたキャラクター画像（黒背景）
- `{name}_mask.png`: セグメンテーションマスク
- `{name}_transparent.png`: 透明背景のキャラクター画像

## 要件

- SAM model checkpoint: `sam_vit_h_4b8939.pth`
- YOLO model: `yolov8n.pt` (自動ダウンロード)
- 推奨GPU メモリ: 6GB以上
- 推奨RAM: 8GB以上

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
   - `--enhance-contrast` を試す
   - 画像品質を確認

3. **テキストと誤認識**
   - `--filter-text` が有効か確認
   - テキスト密度閾値を調整

### ログ出力の見方

```
🔄 開始: SAM Model Loading (RAM: 2048.1MB, GPU: 0.0MB)
✅ 完了: SAM Model Loading (1.23秒, RAM: 4096.2MB, GPU: 2048.1MB)
📊 生成マスク: 1245 → キャラクター候補: 23
🎯 最適マスク選択: YOLO score=0.456, combined score=0.623
📐 マスク品質: coverage=0.234, compactness=0.789
✅ キャラクター抽出完了: 5.67秒
```

## API使用例

```python
from commands.extract_character import extract_character_from_path

# 単一画像処理
result = extract_character_from_path(
    image_path="manga_page.jpg",
    output_path="output/character",
    enhance_contrast=True,
    min_yolo_score=0.15
)

if result['success']:
    print(f"抽出成功: {result['output_path']}")
    print(f"処理時間: {result['processing_time']:.2f}秒")
else:
    print(f"抽出失敗: {result['error']}")
```

## 関連コマンド

- `/start`: モデル初期化（自動実行）
- サンプル画像: `assets/masks1.png`