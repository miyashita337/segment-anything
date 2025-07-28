# 依存関係管理統一リファレンス

**作成日**: 2025-07-28  
**重要度**: 高  
**目的**: プロジェクトの依存関係情報を一元化した統一リファレンス

---

## 📋 このドキュメントについて

このドキュメントは、プロジェクトの依存関係管理に関する**唯一の正式な参照元**です。  
すべての依存関係関連の情報は、この統一リファレンスを参照してください。

**⚠️ 重要**: 他のドキュメントで依存関係を扱う場合は、「詳細: `docs/dependency_reference.md` を参照」と記載してください。

---

## 🔒 核心依存関係（変更禁止）

これらのライブラリはプロジェクトの根幹を成すため、**バージョンの大幅変更や代替ライブラリへの変更は禁止**です。

### 1. SAM (Segment Anything Model)
```yaml
library: segment-anything
source: Meta AI Facebook Research
installation: "git+https://github.com/facebookresearch/segment-anything.git"
role: "メイン分割アルゴリズム"
criticality: "最高"
version_policy: "Meta公式の最新安定版を使用"
```

### 2. YOLO (Ultralytics)
```yaml
library: ultralytics
minimum_version: "8.0.0"
role: "物体検出（キャラクター候補特定）"
criticality: "最高"
version_policy: "8.x系列内で最新バージョン"
breaking_change_risk: "低（API安定）"
```

### 3. OpenCV
```yaml
library: opencv-python
minimum_version: "4.5.0"
role: "基本画像処理・変換"
criticality: "最高"
version_policy: "4.x系列最新（5.x移行は慎重検討）"
```

### 4. PyTorch
```yaml
library: torch, torchvision
minimum_version: "torch>=1.7.0, torchvision>=0.8.0"
role: "ML計算基盤・GPU処理"
criticality: "最高"
cuda_dependency: "必須"
version_policy: "CUDA互換性を最優先"
```

### 5. Python Runtime
```yaml
python_version: ">=3.8,<3.12"
role: "実行環境"
criticality: "最高"
cuda_requirement: "必須（GPU処理のため）"
```

---

## 📊 客観評価依存関係（新規重要）

v1.0.0での客観的評価システム導入に伴う新規依存関係：

### MediaPipe
```yaml
library: mediapipe
minimum_version: "0.10.0"
role: "人体姿勢推定（SCI計算）"
criticality: "高（客観評価に必須）"
purpose: "顔検出、関節点検出、構造完全性評価"
```

### scikit-image
```yaml
library: scikit-image
minimum_version: "0.18.0"
role: "高度画像分析（輪郭品質評価）"
criticality: "中（品質向上に寄与）"
```

---

## 📦 完整依存関係リスト

### コア依存関係
```txt
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
```

### 開発環境依存関係
```txt
# コード品質管理
flake8>=4.0.0
black==23.*           # 特定バージョン固定（フォーマット一貫性）
mypy>=0.910
isort==5.12.0         # 特定バージョン固定（フォーマット一貫性）

# テスト環境
pytest>=6.0.0
pytest-cov>=3.0.0

# 通知システム（オプション）
# pushover-complete   # 必要に応じてアンコメント
```

**注意**: `black`, `isort`は特定バージョン固定。他バージョンでフォーマット結果が変わるため。

---

## 🔄 依存関係の更新方針

### 定期更新（推奨）
```bash
# 月1回の定期更新確認
pip list --outdated

# 安全な更新（パッチバージョンのみ）
pip install --upgrade torch torchvision ultralytics opencv-python

# 慎重更新（メジャーバージョン変更前にテスト）
pip install --upgrade mediapipe scikit-image
```

### 更新時のテスト手順
```bash
# 1. 更新前のベンチマーク取得
python tools/benchmark_current_system.py --save baseline_before_update.json

# 2. 依存関係更新
pip install --upgrade [target_package]

# 3. 基本動作確認
python test_phase2_simple.py

# 4. 客観評価確認
python tools/objective_quality_evaluation.py --batch test_small/ --compare baseline_before_update.json

# 5. 問題があれば即座にロールバック
pip install [target_package]==previous_version
```

---

## 🚨 トラブルシューティング

### よくある問題と解決法

#### 1. CUDA関連エラー
```bash
# 症状: torch.cuda.is_available() が False
# 原因: PyTorchのCUDAバージョンとシステムCUDAの不整合

# 確認方法
nvidia-smi  # システムCUDAバージョン確認
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDAバージョン

# 解決方法：適切なCUDA版PyTorchを再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8の場合
```

#### 2. MediaPipe初期化エラー
```bash
# 症状: ModuleNotFoundError: No module named 'mediapipe.solutions'
# 原因: MediaPipeのバージョン不整合

# 解決方法
pip uninstall mediapipe
pip install mediapipe>=0.10.0
```

#### 3. OpenCV表示エラー（Linux環境）
```bash
# 症状: cv2.imshow() でエラー
# 原因: GUI関連ライブラリ不足

# Ubuntu/Debian解決方法
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# WSL環境では表示機能を使わない
export DISPLAY=""  # 表示機能無効化
```

### メモリ不足エラー
```bash
# GPU VRAM不足時の対処
export YOLO_MODEL=yolov8n.pt  # デフォルトはyolov8x.pt

# RAM不足時の対処
# バッチサイズを削減（コード内で調整）
```

---

## 📦 環境構築の標準手順

### 新規環境セットアップ
```bash
# 1. Python仮想環境作成
python3 -m venv sam-env
source sam-env/bin/activate  # Linux
# sam-env\Scripts\activate  # Windows

# 2. 基本依存関係インストール
pip install --upgrade pip

# 3. 開発環境セットアップ（推奨）
pip install -e .[dev]

# または基本インストール
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx ultralytics easyocr

# 4. 動作確認
python -c "import torch, ultralytics, cv2, mediapipe; print('All core libraries imported successfully')"
python tools/testing/test_phase2_simple.py
```

### CUDA環境確認
```bash
# CUDA利用可能性確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# GPU メモリ確認
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
else:
    print('CUDA not available')
"
```

---

## 🔍 依存関係監査

### セキュリティチェック
```bash
# 脆弱性チェック（推奨：月1回）
pip audit

# 使用許可されていないライセンスの確認
pip-licenses --summary
```

### 使用状況分析
```bash
# 実際に使用されている依存関係の確認
pipdeptree

# 未使用依存関係の特定
pip-check
```

---

## 📋 依存関係管理チェックリスト

### ✅ 月次チェック項目
```yaml
monthly_checks:
  - [ ] pip list --outdated で更新確認
  - [ ] pip audit でセキュリティチェック
  - [ ] 更新後の動作確認テスト実行
  - [ ] requirements.txt の更新（必要に応じて）

quarterly_checks:
  - [ ] 核心依存関係の代替技術調査
  - [ ] 依存関係グラフの整理
  - [ ] 不要依存関係の削除検討
  - [ ] Docker環境での動作確認
```

### ⚠️ 注意事項
- **核心依存関係の変更は慎重に**: SAM、YOLO、OpenCV、PyTorchの変更は全体影響が大きい
- **CUDA互換性を最優先**: GPU処理が前提のプロジェクトのため
- **バージョン固定の理由を理解**: `black`、`isort`は一貫性のため特定バージョン固定
- **テスト必須**: 依存関係更新後は必ず品質評価テストを実行

---

## 🚨 緊急時対応

### 依存関係障害時の対応手順
```yaml
dependency_emergency:
  
  immediate_actions:
    - "問題のある依存関係を特定"
    - "前バージョンへの即座ロールバック"
    - "基本動作確認テスト実行"
  
  investigation:
    - "エラーログの詳細分析"
    - "依存関係の競合状況確認" 
    - "回避方法・代替手段の検討"
    
  resolution:
    - "修正版の依存関係更新"
    - "全体テストスイートの実行"
    - "本番環境への慎重な適用"
```

### 緊急用ロールバックコマンド
```bash
# PyTorch関連のロールバック例
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu118

# YOLO関連のロールバック例
pip install ultralytics==8.0.120

# MediaPipe関連のロールバック例
pip install mediapipe==0.10.0
```

---

**重要**: 依存関係管理は品質とセキュリティに直結します。  
特に核心依存関係の変更は、十分なテストと影響評価を経てから実行してください。

**更新履歴**:
- 2025-07-28: 統一リファレンス作成（`library_dependencies_guide.md` + `dependency_management.md` 統合）