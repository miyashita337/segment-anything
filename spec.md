# 環境仕様書 (Environment Specifications)

## ハードウェア要件

### 最小要件
- **CPU**: Intel Core i5以上またはAMD Ryzen 5以上
- **RAM**: 8GB以上
- **GPU**: NVIDIA GPU（CUDA対応）4GB VRAM以上推奨
- **ストレージ**: 10GB以上の空き容量

### 推奨要件
- **CPU**: Intel Core i7以上またはAMD Ryzen 7以上
- **RAM**: 16GB以上
- **GPU**: NVIDIA GPU 8GB VRAM以上（RTX 3070以上推奨）
- **ストレージ**: SSD 20GB以上の空き容量

## ソフトウェア要件

### Python環境
```yaml
python_version: ">=3.8"
recommended_version: "3.10"
```

### 必須パッケージ
- torch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- ultralytics>=8.0.0
- segment-anything>=1.0
- numpy>=1.24.0
- Pillow>=10.0.0

### 開発依存関係
- pytest>=7.4.0
- black>=23.0.0
- flake8>=6.0.0
- mypy>=1.5.0
- isort>=5.12.0

## モデルファイル要件

### SAM (Segment Anything Model)
- **ViT-H**: `sam_vit_h_4b8939.pth` (2.6GB)
- **ViT-L**: `sam_vit_l_0b3195.pth` (2.3GB)
- **ViT-B**: `sam_vit_b_01ec64.pth` (358MB)

### YOLO Models
- **YOLOv8n**: `yolov8n.pt` (6.2MB)
- **YOLOv8x**: `yolov8x.pt` (136MB)

## 対応画像形式

### 入力対応形式
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

### 出力形式
- PNG（マスク画像）
- JPEG（抽出結果）
- JSON（メタデータ）

## 実行環境

### 仮想環境
```bash
python -m venv sam-env
source sam-env/bin/activate  # Linux/macOS
sam-env\Scripts\activate     # Windows
```

### インストール
```bash
pip install -e .[dev]
```

### 環境検証
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
./bin/shell/linter.sh
python -m pytest tests/ -v
```