# segment-anything complete requirements
# 核心技術スタック（変更禁止）

# SAM (Segment Anything Model) - Meta AI
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git

# YOLO Object Detection - Ultralytics
ultralytics>=8.0.0

# 画像処理基盤
opencv-python>=4.5.0
Pillow>=8.0.0

# 科学計算基盤
numpy>=1.19.0
scipy>=1.7.0

# ML/AI フレームワーク
torch>=1.7.0
torchvision>=0.8.0

# 客観評価システム（新規・重要）
mediapipe>=0.10.0

# 画像処理補助
scikit-image>=0.18.0

# データ処理・分析
pandas>=1.3.0

# 可視化
matplotlib>=3.3.0

# 進捗表示・UX
tqdm>=4.60.0

# 開発・テスト環境（[dev]インストール用）
flake8>=4.0.0
black==23.*
mypy>=0.910
isort==5.12.0
pytest>=6.0.0
pytest-cov>=3.0.0

# 通知システム（オプション）
# pushover-complete  # 必要に応じてアンコメント