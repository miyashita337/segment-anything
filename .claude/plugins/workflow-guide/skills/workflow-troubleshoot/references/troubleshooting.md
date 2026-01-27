# トラブルシューティングガイド

## 🚨 概要

segment-anything v0.4.0バッチ処理で発生する一般的な問題と解決方法。  
今回のkana08処理で解決された問題も含む包括的なガイド。

## 📋 問題分類

### Level 1: 環境・セットアップ問題
### Level 2: 実行時エラー
### Level 3: 品質・結果問題
### Level 4: パフォーマンス問題

---

## 🔧 Level 1: 環境・セットアップ問題

### 1.1 インポートエラー

#### 問題: `ImportError: cannot import name 'sam_model_registry'`
```bash
ImportError: cannot import name 'sam_model_registry' from 'segment_anything'
```

**原因**: Python パス設定の問題

**解決方法**:
```bash
# 1. プロジェクトルートで実行
cd /mnt/c/AItools/segment-anything

# 2. Python パス確認
python -c "import sys; print('\\n'.join(sys.path))"

# 3. 環境変数設定
export PYTHONPATH="/mnt/c/AItools/segment-anything:$PYTHONPATH"

# 4. 再実行
python tools/test_phase2_simple.py --help
```

**予防策**: 常にプロジェクトルートから実行

#### 問題: `ImportError: cannot import name 'Sam' from 'segment_anything.modeling'`
```bash
ImportError: cannot import name 'Sam' from 'segment_anything.modeling'
```

**原因**: 相対インポートの問題

**解決方法**: 既に修正済み
```python
# 修正済み (predictor.py line 10)
from .modeling import Sam  # 相対インポートに変更
```

**確認**: `git pull origin main` で最新コードを取得

### 1.2 Unicode エンコーディングエラー

#### 問題: `UnicodeDecodeError: 'cp932' codec`
```bash
UnicodeDecodeError: 'cp932' codec can't decode byte 0xf0
```

**原因**: Windows環境での絵文字文字

**解決方法**: 既に修正済み
- 全Python ファイルから絵文字を削除
- ASCII互換文字に置換

**確認**: 最新コードを使用
```bash
git pull origin main
```

### 1.3 モジュール未インストール

#### 問題: `ModuleNotFoundError: No module named 'ultralytics'`
```bash
ModuleNotFoundError: No module named 'ultralytics'
```

**解決方法**:
```bash
# 1. 仮想環境確認
which python3
source sam-env/bin/activate

# 2. ultralytics インストール
pip install ultralytics

# 3. 依存関係確認
pip list | grep ultralytics
```

### 1.5 Pythonコマンドエラー

#### 問題: `Command 'python' not found`
```bash
Command 'python' not found, did you mean:
  command 'python3' from package 'python3'
```

**解決方法**: `python3` コマンドを使用
```bash
# 正しい実行方法
python3 tools/test_phase2_simple.py --help
python3 -c "import torch; print(torch.cuda.is_available())"

# エイリアス設定 (オプション)
alias python=python3
```

### 1.4 CUDA利用不可

#### 問題: GPU加速が使用できない
```bash
CUDA available: False
```

**確認手順**:
```bash
# 1. NVIDIA ドライバー確認
nvidia-smi

# 2. CUDA インストール確認
nvcc --version

# 3. PyTorch CUDA 確認
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Ver: {torch.version.cuda}')"
```

**解決方法**:
```bash
# CUDA版PyTorch再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## ⚡ Level 2: 実行時エラー

### 2.1 メモリ不足エラー

#### 問題: `CUDA out of memory`
```bash
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**即時対応**:
```bash
# GPU メモリクリア
python3 -c "import torch; torch.cuda.empty_cache()"
```

**根本解決**:
```bash
# 1. 軽量モデル使用
--model_path yolov8n.pt  # yolov8x.pt → yolov8n.pt

# 2. バッチサイズ削減
--batch_size 1

# 3. 画像サイズ削減
--max_image_size 1024  # 2048 → 1024
```

### 2.2 ファイルパス問題

#### 問題: ファイルが見つからない
```bash
FileNotFoundError: [Errno 2] No such file or directory
```

**確認事項**:
```bash
# 1. パス存在確認
ls -la "/path/to/input/directory"

# 2. 権限確認
ls -ld "/path/to/input/directory"

# 3. ファイル形式確認
find "/path/to/input" -name "*.jpg" -o -name "*.png" -o -name "*.webp" | wc -l
```

**解決方法**:
```bash
# 絶対パス使用
INPUT_DIR="/mnt/c/AItools/lora/train/yado/org/kana08"
OUTPUT_DIR="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_0_4_0"
```

### 2.3 モデルファイル問題

#### 問題: SAMモデルが読み込めない
```bash
FileNotFoundError: sam_vit_h_4b8939.pth not found
```

**解決方法**:
```bash
# 1. モデルファイル確認
ls -la sam_vit_h_4b8939.pth

# 2. ダウンロード（必要に応じて）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 3. 権限確認
chmod 644 sam_vit_h_4b8939.pth
```

### 2.4 引数解析エラー

#### 問題: コマンドライン引数が認識されない
```bash
error: unrecognized arguments: --input_dir
```

**解決方法**: 既に修正済み
```python
# 修正済み: argparse対応
parser.add_argument("--input_dir", required=True, help="Input directory path")
parser.add_argument("--output_dir", required=True, help="Output directory path")
```

---

## 🎯 Level 3: 品質・結果問題

### 3.1 抽出成功率が低い

#### 問題: 成功率 < 80%

**診断手順**:
```bash
# 1. 入力画像品質確認
for img in "$INPUT_DIR"/*.jpg; do
    identify "$img" | grep -E "[0-9]+x[0-9]+"
done

# 2. YOLO検出率確認
python -c "
import cv2
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
results = model('test_image.jpg', conf=0.07)
print(f'Detections: {len(results[0].boxes)}')
"
```

**解決方法**:
```bash
# 1. 閾値調整
--score_threshold 0.05  # 0.07 → 0.05 (高感度)

# 2. 品質手法変更
--quality_method confidence_priority  # balanced → confidence_priority

# 3. 前処理強化
--enhance_contrast true
--filter_text true
```

### 3.2 手足切断問題

#### 問題: キャラクターの手足が切断される

**即時対応**: Phase 3インタラクティブモード
```bash
python commands/quick_interactive.py image.jpg \
  --points 750,1000,pos 800,1200,pos 500,500,neg
```

**根本対策**:
```bash
# 1. マスク拡張パラメータ調整
--mask_expansion_factor 1.2  # デフォルト: 1.0

# 2. 全身優先モード
--quality_method fullbody_priority

# 3. 低閾値モード
--low_threshold true
```

### 3.3 背景混入問題

#### 問題: 不要な背景要素が抽出に含まれる

**解決方法**:
```bash
# 1. マンガモード有効化
--manga_mode true
--effect_removal true

# 2. テキスト除去
--filter_text true

# 3. 厳密モード
--score_threshold 0.1  # 高い閾値で厳選
```

### 3.4 品質スコアが低い

#### 問題: 平均品質スコア < 0.5

**分析**:
```bash
# 品質分布確認
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
scores = [r['quality_score'] for r in data['results'] if 'quality_score' in r]
print(f'Average: {sum(scores)/len(scores):.3f}')
print(f'Min: {min(scores):.3f}, Max: {max(scores):.3f}')
"
```

**改善方法**:
```bash
# 1. 複数手法テスト
for method in balanced size_priority confidence_priority; do
    echo "Testing $method"
    python tools/test_phase2_simple.py \
      --quality_method $method \
      --input_dir small_test_set
done

# 2. パラメータスイープ
for threshold in 0.05 0.07 0.1; do
    echo "Testing threshold $threshold"
    # テスト実行
done
```

---

## ⚡ Level 4: パフォーマンス問題

### 4.1 処理速度が遅い

#### 問題: > 15秒/画像

**診断**:
```bash
# GPU使用率確認
nvidia-smi dmon -s pucvmet -d 1

# システムリソース確認
htop
```

**最適化**:
```bash
# 1. 軽量モデル使用
--model_path yolov8n.pt

# 2. 画像リサイズ
--max_image_size 1024

# 3. バッチ処理無効化
--batch_processing false
```

### 4.2 メモリ使用量が多い

#### 問題: RAM/VRAM使用量過多

**監視**:
```bash
# メモリ使用量監視
watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

**最適化**:
```bash
# 1. ガベージコレクション強化
python -c "
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
"

# 2. 処理画像数制限
--max_batch_size 10

# 3. 中間ファイル削除
--cleanup_intermediate true
```

### 4.3 ディスク容量不足

#### 問題: 出力ディスクが満杯

**確認**:
```bash
# ディスク使用量確認
df -h /mnt/c/AItools/lora/train/

# 大きなファイル確認
find /mnt/c/AItools -size +100M -type f | head -10
```

**対策**:
```bash
# 1. 不要ファイル削除
find . -name "*_mask.png" -delete  # マスクファイル削除
find . -name "*_transparent.png" -delete  # 透明版削除

# 2. 圧縮設定
--output_quality 85  # JPEG品質調整

# 3. 別ドライブ使用
OUTPUT_DIR="/mnt/d/extraction_results"
```

---

## 📊 診断コマンド集

### 環境診断
```bash
#!/bin/bash
echo "=== Environment Diagnosis ==="
echo "Python: $(python --version)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Ultralytics: $(pip show ultralytics | grep Version)"
```

### ファイルシステム診断
```bash
#!/bin/bash
INPUT_DIR="$1"
echo "=== File System Diagnosis ==="
echo "Input directory: $INPUT_DIR"
echo "Exists: $([ -d "$INPUT_DIR" ] && echo 'Yes' || echo 'No')"
echo "Readable: $([ -r "$INPUT_DIR" ] && echo 'Yes' || echo 'No')"
echo "Image count: $(find "$INPUT_DIR" -name "*.jpg" -o -name "*.png" | wc -l)"
echo "Disk space: $(df -h "$INPUT_DIR" | tail -1 | awk '{print $4}')"
```

### パフォーマンス診断
```bash
#!/bin/bash
echo "=== Performance Diagnosis ==="
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "GPU Utilization: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
echo "CPU Count: $(nproc)"
echo "RAM Total: $(free -h | grep Mem | awk '{print $2}')"
echo "RAM Available: $(free -h | grep Mem | awk '{print $7}')"
```

---

## 🚨 緊急時対応

### 処理中断時
```bash
# 1. プロセス確認
ps aux | grep python

# 2. 安全な停止
pkill -TERM -f "test_phase2_simple.py"

# 3. GPU メモリクリア
python -c "import torch; torch.cuda.empty_cache()"

# 4. レジューム準備
# 処理済みファイルを確認してから再開
```

### システム不安定時
```bash
# 1. GPU リセット
sudo nvidia-smi --gpu-reset

# 2. Python プロセス全停止
pkill -f python

# 3. システムリソース確認
free -h && df -h

# 4. 再起動（必要に応じて）
sudo reboot
```

---

## 📝 問題報告テンプレート

### 問題報告フォーマット
```markdown
## 問題報告

### 環境情報
- OS: [OS/バージョン]
- Python: [バージョン]
- CUDA: [バージョン]
- GPU: [モデル/VRAM]

### 発生した問題
- エラーメッセージ: [正確なエラー]
- 発生タイミング: [いつ発生するか]
- 再現性: [常に/時々/一度だけ]

### 実行したコマンド
```bash
[実際に実行したコマンド]
```

### 期待する結果
[期待していた動作]

### 実際の結果
[実際に起こった動作]

### 試行した解決方法
1. [試したこと1]
2. [試したこと2]

### 添付ファイル
- ログファイル: [ある場合]
- スクリーンショット: [ある場合]
```

---

*最終更新: 2025-07-21*  
*対象バージョン: [../../docs/technical/specifications/system_spec.md](../../docs/technical/specifications/system_spec.md) を参照*