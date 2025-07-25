# 依存関係管理ガイド

**作成日**: 2025-07-24  
**対象**: segment-anything プロジェクトの依存ライブラリ管理

## 🎯 依存関係の分類

### 🔒 核心依存関係（変更禁止）

これらのライブラリはプロジェクトの根幹を成すため、**バージョンの大幅変更や代替ライブラリへの変更は禁止**です。

#### 1. SAM (Segment Anything Model)
```yaml
library: segment-anything
source: Meta AI Facebook Research
installation: "git+https://github.com/facebookresearch/segment-anything.git"
role: "メイン分割アルゴリズム"
criticality: "最高"
version_policy: "Meta公式の最新安定版を使用"
```

#### 2. YOLO (Ultralytics)
```yaml
library: ultralytics
minimum_version: "8.0.0"
role: "物体検出（キャラクター候補特定）"
criticality: "最高"
version_policy: "8.x系列内で最新バージョン"
breaking_change_risk: "低（API安定）"
```

#### 3. OpenCV
```yaml
library: opencv-python
minimum_version: "4.5.0"
role: "基本画像処理・変換"
criticality: "最高"
version_policy: "4.x系列最新（5.x移行は慎重検討）"
```

#### 4. PyTorch
```yaml
library: torch, torchvision
minimum_version: "torch>=1.7.0, torchvision>=0.8.0"
role: "ML計算基盤・GPU処理"
criticality: "最高"
cuda_dependency: "必須"
version_policy: "CUDA互換性を最優先"
```

### 📊 客観評価依存関係（新規重要）

v1.0.0での客観的評価システム導入に伴う新規依存関係：

#### MediaPipe
```yaml
library: mediapipe
minimum_version: "0.10.0"
role: "人体姿勢推定（SCI計算）"
criticality: "高（客観評価に必須）"
purpose: "顔検出、関節点検出、構造完全性評価"
```

#### scikit-image
```yaml
library: scikit-image
minimum_version: "0.18.0"
role: "高度画像分析（輪郭品質評価）"
criticality: "中（品質向上に寄与）"
```

### 🛠️ 開発環境依存関係

#### コード品質管理
```yaml
quality_tools:
  flake8: ">=4.0.0"  # Pythonスタイルチェック
  black: "==23.*"    # コード整形（特定バージョン固定）
  mypy: ">=0.910"    # 型チェック
  isort: "==5.12.0"  # import整理（特定バージョン固定）

# 注意: black, isortは特定バージョン固定
# 理由: 他バージョンでフォーマット結果が変わるため
```

#### テスト環境
```yaml
test_tools:
  pytest: ">=6.0.0"      # テスト実行
  pytest-cov: ">=3.0.0"  # カバレッジ測定
```

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

## 🚨 依存関係のトラブルシューティング

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

## 📦 環境構築の標準手順

### 新規環境セットアップ
```bash
# 1. Python仮想環境作成
python3 -m venv sam-env
source sam-env/bin/activate  # Linux
# sam-env\Scripts\activate  # Windows

# 2. 基本依存関係インストール
pip install --upgrade pip
pip install -r requirements_complete.txt

# 3. 開発環境セットアップ（開発者のみ）
pip install -e .[dev]

# 4. 動作確認
python -c "import torch, ultralytics, cv2, mediapipe; print('All core libraries imported successfully')"
python test_phase2_simple.py
```

### Docker環境（将来的推奨）
```dockerfile
# Dockerfile（参考実装）
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    git

COPY requirements_complete.txt .
RUN pip3 install -r requirements_complete.txt

WORKDIR /workspace
COPY . .
```

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

## 📋 依存関係管理チェックリスト

### ✅ 月次チェック項目
```yaml
monthly_checks:
  - [ ] pip list --outdated で更新確認
  - [ ] pip audit でセキュリティチェック
  - [ ] 更新後の動作確認テスト実行
  - [ ] requirements_complete.txt の更新

quarterly_checks:
  - [ ] 核心依存関係の代替技術調査
  - [ ] 依存関係グラフの整理
  - [ ] 不要依存関係の削除検討
  - [ ] Docker環境での動作確認
```

### 🚨 緊急時対応
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

---

**重要**: 依存関係管理は品質とセキュリティに直結します。  
特に核心依存関係の変更は、十分なテストと影響評価を経てから実行してください。