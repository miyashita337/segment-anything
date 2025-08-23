# システム仕様書

**プロジェクト**: segment-anything  
**バージョン**: v0.9.23  
**最終更新**: 2025-08-23

## 📋 概要

このドキュメントは segment-anything プロジェクトで使用するハードウェア・ソフトウェア・モデルファイルの完全な仕様を定義します。  
実装時およびマージ時には、この仕様との整合性を必ず確認してください。

## 💻 ハードウェア要件

### 必須要件
- **CPU**: x86_64 アーキテクチャ
- **RAM**: 最小 8GB、推奨 16GB以上
- **GPU**: CUDA対応GPU（必須）
- **VRAM**: 最小 4GB、推奨 8GB以上
- **ストレージ**: 最小 20GB の空き容量

### 推奨環境
- **GPU**: NVIDIA RTX 3080 以上または同等品
- **VRAM**: 12GB 以上
- **RAM**: 32GB 以上
- **ストレージ**: SSD 50GB 以上

## 🐍 ソフトウェア要件

### Python環境
```yaml
python_version: ">=3.8,<3.12"
recommended_version: "3.10"
virtual_environment: "必須"
```

### 必須パッケージ

> **詳細**: [`docs/dependency_reference.md`](docs/dependency_reference.md) を参照

```yaml
core_packages:
  torch: ">=1.7.0"
  torchvision: ">=0.8.0"
  ultralytics: "latest"
  opencv-python: ">=4.5.0"
  numpy: ">=1.19.0"
  pillow: ">=8.0.0"

ml_packages:
  segment-anything: "custom"  # Meta実装 + カスタム拡張
  
optional_packages:
  jupyter: "latest"
  matplotlib: ">=3.3.0"
  tqdm: ">=4.60.0"
```

### CUDA環境
```yaml
cuda_version: ">=11.0"
recommended_cuda: "11.8"
pytorch_cuda_support: "必須"
```

### システムパッケージ
```bash
# Ubuntu/Debian
apt-get install:
  - build-essential
  - python3-dev
  - libgl1-mesa-glx
  - libglib2.0-0

# Windows (WSL推奨)
wsl_version: "WSL2"
```

## 🤖 モデルファイル要件

### SAM (Segment Anything Model)
```yaml
model_file: "sam_vit_h_4b8939.pth"
size: "2.6GB"
download_url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
checksum_sha256: "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
location: "プロジェクトルート"
```

### YOLO (Object Detection)
```yaml
primary_model: "yolov8x6_animeface.pt"
fallback_model: "yolov8n.pt"
model_source: "ultralytics"
anime_optimized: true
confidence_threshold: 0.07
```

### 代替モデル
```yaml
sam_alternatives:
  - "sam_vit_l_0b3195.pth"  # Large model
  - "sam_vit_b_01ec64.pth"  # Base model

yolo_alternatives:
  - "yolov8s.pt"  # Small model (低VRAM環境)
  - "yolov8m.pt"  # Medium model
  - "yolov8l.pt"  # Large model
```

## 📁 ディレクトリ構造

> **詳細**: [`docs/project_structure_reference.md`](docs/project_structure_reference.md) を参照

### 標準プロジェクト構造
```
segment-anything/
├── spec.md                     # この仕様書
├── core/                       # Meta Facebook実装
│   └── segment_anything/
├── features/                   # カスタム実装
│   ├── extraction/
│   ├── evaluation/
│   └── common/
├── tools/                      # 実行スクリプト
├── tests/                      # テストスイート
├── docs/                       # ドキュメント
│   └── workflows/
├── sam_vit_h_4b8939.pth       # SAMモデル
└── yolov8x6_animeface.pt      # YOLOモデル
```

### データディレクトリ構造
```
/path/to/data/
├── input/
│   └── [dataset_name]/         # 動的データセット名
├── output/
│   └── [dataset_name]_[version]/
└── backup/                     # バックアップ（推奨）
```

## 🖼️ 対応画像形式

### 入力形式
```yaml
supported_formats:
  - ".jpg"
  - ".jpeg"
  - ".png"
  - ".webp"

priority_order:
  1. "jpg"
  2. "png" 
  3. "webp"

image_requirements:
  min_resolution: "512x512"
  recommended_resolution: "1024x1024"
  max_file_size: "10MB"
  color_space: "RGB"
```

### 出力形式
```yaml
default_output: ".jpg"
quality_setting: 95
transparency_support: ".png"  # 必要時のみ
```

## ⚙️ 実行時パラメータ

### デフォルト設定
```yaml
processing_settings:
  quality_method: "balanced"
  score_threshold: 0.07
  batch_size: 1
  gpu_memory_fraction: 0.8
  max_image_size: 2048

performance_settings:
  timeout_per_image: 300  # 5分
  max_retry_attempts: 3
  memory_cleanup_interval: 10  # 10画像ごと
```

### 品質レベル別設定
```yaml
quality_presets:
  high_quality:
    score_threshold: 0.05
    model: "yolov8x6_animeface.pt"
    sam_model: "sam_vit_h_4b8939.pth"
    
  balanced:
    score_threshold: 0.07
    model: "yolov8x6_animeface.pt"
    sam_model: "sam_vit_h_4b8939.pth"
    
  fast:
    score_threshold: 0.1
    model: "yolov8n.pt"
    sam_model: "sam_vit_b_01ec64.pth"
```

## 🧪 テスト環境要件

### UnitTest環境
```yaml
test_framework: "pytest"
coverage_target: ">=80%"
test_data_size: "<100MB"

required_test_checks:
  - python_version_check
  - cuda_availability_check
  - model_file_existence_check
  - package_version_check
  - gpu_memory_check
```

### CI/CD環境
```yaml
github_actions:
  runner: "ubuntu-latest"
  python_versions: ["3.8", "3.9", "3.10"]
  cuda_support: "optional"  # GPU Runnerが利用可能な場合
```

## 🔧 コマンド仕様

### 標準実行コマンド
```bash
# Python実行（優先順位）
python3 [script.py]  # 第一選択
python [script.py]   # フォールバック

# 環境チェック
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import sys; print(sys.version)"
```

### バッチ処理コマンド
```bash
# 推奨実行方法
python3 tools/test_phase2_simple.py \
  --input_dir "/path/to/input" \
  --output_dir "/path/to/output" \
  --score_threshold 0.07
```

## 📊 パフォーマンス指標

### 期待値
```yaml
processing_speed:
  target: "5-10 seconds/image"
  acceptable: "<=15 seconds/image"
  
success_rate:
  target: ">=95%"
  minimum: ">=90%"
  
quality_score:
  target: ">=0.7"
  minimum: ">=0.5"
  
memory_usage:
  ram_peak: "<=4GB"
  vram_peak: "<=6GB"
```

### ベンチマーク環境
```yaml
reference_hardware:
  gpu: "NVIDIA RTX 3080"
  vram: "10GB"
  ram: "32GB"
  cpu: "Intel i7-10700K"
```

## 🚨 制約事項

### 命名規則
```yaml
禁止事項:
  - 特定データセット名のハードコード
  - テストパス以外での固有名詞使用
  - バージョン情報のハードコード

推奨事項:
  - 変数名: [dataset_name], [character_name], [version]
  - 設定参照: "../../spec.md を参照"
```

### セキュリティ
```yaml
api_keys:
  storage: "環境変数またはGitHub Secrets"
  hardcode: "禁止"
  
file_permissions:
  models: "644"
  scripts: "755"
  configs: "600"
```

## 🔄 更新プロセス

### 自動更新トリガー
- Pull Request マージ時
- 仕様変更検出時
- モデルファイル更新時

### 手動更新タイミング
- 新機能追加時
- パフォーマンス要件変更時
- ハードウェア要件変更時

---

**注意**: この仕様書は実装と同期して維持される必要があります。  
変更時は必ず UnitTest での環境整合性チェックを実行してください。