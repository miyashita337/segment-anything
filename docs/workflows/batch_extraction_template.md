# バッチキャラクター抽出ワークフローテンプレート

## 📋 概要

segment-anything を使用したアニメキャラクターのバッチ抽出の標準化されたワークフロー。
このテンプレートは実際のバッチ処理実行で確立された手順をベースに作成されています。

## 🎯 対象

- アニメ・漫画キャラクター画像の大量バッチ処理
- LoRA 学習用データセット作成
- 品質を重視した抽出処理

## 📦 前提条件

### 環境要件

**詳細は [../../spec.md](../../spec.md) を参照してください。**

主要要件:
- Python 3.8+ (推奨: 3.10)
- CUDA対応GPU (推奨: 8GB VRAM以上)
- 必要なモデルファイルとパッケージ

### ディレクトリ構造

```
/mnt/c/AItools/lora/train/[character_name]/
├── org/[dataset_name]/          # 入力画像
└── clipped_boundingbox/
    └── [dataset_name]_[version]/    # 出力ディレクトリ
```

## 🚀 実行手順

### ステップ 1: 環境準備

```bash
# プロジェクトディレクトリに移動
cd /mnt/c/AItools/segment-anything

# 仮想環境有効化（必要に応じて）
source sam-env/bin/activate

# GPU確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### ステップ 2: 入力データ確認

```bash
# 入力ディレクトリの画像数確認
INPUT_DIR="/mnt/c/AItools/lora/train/[character_name]/org/[dataset_name]"
ls -la "$INPUT_DIR" | grep -E '\.(jpg|png)$' | wc -l
```

### ステップ 3: 出力ディレクトリ準備

```bash
# 出力ディレクトリ作成
OUTPUT_DIR="/mnt/c/AItools/lora/train/[character_name]/clipped_boundingbox/[dataset_name]_[version]"
mkdir -p "$OUTPUT_DIR"
```

### ステップ 4: 進捗追跡ファイル作成

```bash
# 進捗追跡ドキュメント作成
PROGRESS_FILE="docs/request/[dataset_name]_extraction_progress_$(date +%Y%m%d).md"
```

進捗ファイルテンプレート:

```markdown
# [Dataset Name] キャラクター抽出進捗

## 基本情報

- **実行日**: [日付]
- **バージョン**: [../../spec.md](../../spec.md) を参照
- **入力**: [入力パス]
- **出力**: [出力パス]
- **品質手法**: balanced

## パラメータ設定

- YOLO 閾値: 0.07
- 出力形式: 元ファイル名保持
- 透明処理: なし

## 実行状況

[リアルタイム更新]
```

### ステップ 5: バッチ抽出実行

#### オプション 1: test_phase2_simple.py 使用（推奨）

```bash
python3 tools/test_phase2_simple.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --score_threshold 0.07
```

#### オプション 2: run_batch_extraction.py 使用

```bash
python3 temp/scripts/migration/run_batch_extraction.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"
```

### ステップ 6: 結果確認

```bash
# 出力ファイル数確認
OUTPUT_COUNT=$(ls -la "$OUTPUT_DIR" | grep -E '\.(jpg|png|webp)$' | wc -l)
INPUT_COUNT=$(ls -la "$INPUT_DIR" | grep -E '\.(jpg|png|webp)$' | wc -l)

echo "入力: $INPUT_COUNT枚"
echo "出力: $OUTPUT_COUNT枚"
echo "成功率: $(echo "scale=1; $OUTPUT_COUNT * 100 / $INPUT_COUNT" | bc)%"
```

### ステップ 7: 品質確認（手動）

抽出結果をサンプリングして品質確認:

```bash
# ランダムサンプル5枚を表示
find "$OUTPUT_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.webp" | shuf -n 5
```

## 📊 パフォーマンス指標

### 参考実績 (実測値)

- **処理画像数**: 26 枚
- **成功率**: 100% (26/26)
- **平均処理時間**: 8-9 秒/画像
- **品質手法**: balanced
- **品質スコア範囲**: 0.482-0.938

### 期待値

- **処理速度**: 5-10 秒/画像
- **成功率**: 95%以上
- **品質スコア**: 0.5 以上

## 🛠 トラブルシューティング

### よくある問題と解決方法

#### 1. インポートエラー

```bash
# 症状: ImportError: cannot import name 'sam_model_registry'
# 解決: Python パス確認
cd /mnt/c/AItools/segment-anything
python -c "import sys; print(sys.path)"
```

#### 2. Unicode エラー

```bash
# 症状: UnicodeDecodeError (cp932)
# 解決: 既に絵文字を削除済み、最新コード使用
git pull origin main
```

#### 3. GPU メモリ不足

```bash
# 解決: バッチサイズ削減
# yolov8n.pt (軽量モデル) を使用
```

#### 4. 処理速度低下

```bash
# 確認事項:
# - GPU利用可能性確認
# - VRAM使用量確認
# - システムリソース確認
nvidia-smi  # GPU確認
```

## 🔧 カスタマイズ

### 品質手法変更

```bash
# balanced (推奨)
--quality_method balanced

# サイズ優先
--quality_method size_priority

# 信頼度優先
--quality_method confidence_priority
```

### 閾値調整

```bash
# 高感度（より多く検出）
--score_threshold 0.05

# 標準
--score_threshold 0.07

# 高精度（厳選）
--score_threshold 0.1
```

## 📝 バージョン管理

### コード変更時の手順

```bash
# 1. 変更をコミット
git add -A
git commit -m "Fix: [変更内容] for batch processing"

# 2. プッシュ
git push origin main

# 3. 処理実行前に最新版取得
git pull origin main
```

## 🎯 品質向上のポイント

### 1. 入力画像の品質

- 解像度: 512px 以上推奨
- 形式: JPG/PNG/WebP
- キャラクターが明確に写っている

### 2. パラメータ調整

- アニメキャラクター: YOLO 閾値 0.07 が最適
- balanced 手法が多くの場合で最良結果

### 3. 後処理

- 手動品質確認を推奨
- A 評価以外は個別確認

## 📚 関連ドキュメント

- [segment-anything/CLAUDE.md](../CLAUDE.md) - プロジェクト全体説明
- [segment-anything/README.md](../README.md) - 基本使用方法
- [進捗追跡テンプレート](./progress_tracking_template.md)

## 🚨 注意事項

- **バックアップ**: 処理前に入力データのバックアップを推奨
- **ディスク容量**: 出力ディスクに十分な容量を確保
- **処理時間**: 大量データは数時間要する場合がある
- **GPU 負荷**: 長時間処理時は GPU 温度監視推奨

---

_最終更新: 2025-07-21_  
_バージョン: [../../spec.md](../../spec.md) を参照_
