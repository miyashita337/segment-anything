# 環境確認ガイド（segment-anything専用）

**最終更新**: 2025-08-02  
**目的**: セッション継続時の環境管理不備を防止し、Google Sheetsアクセス等の問題を根本解決

## 📋 概要

segment-anythingプロジェクトでの作業において、仮想環境（sam-env）の未有効化により発生する問題を防止するための標準手順を定義します。

## 🚨 発生する問題

### 環境未切り替え時の典型的問題
- ❌ Google Sheetsアクセス失敗（`ModuleNotFoundError: No module named 'google'`）
- ❌ PyTorchやUltralyticsなどの依存関係不足
- ❌ CUDA利用不可による処理速度低下
- ❌ ワークフロー逸脱とタスク遅延

## ✅ 必須チェックリスト

### segment-anything作業開始前（必須実行）

```bash
# 📋 作業前チェックリスト
□ プロジェクトディレクトリ確認: pwd = /mnt/c/AItools/segment-anything
□ 仮想環境確認: $VIRTUAL_ENV contains "sam-env"  
□ 必須パッケージ確認: pip list | grep -E "(torch|google-auth|ultralytics)"
□ GPU確認: python -c "import torch; print(torch.cuda.is_available())"
□ Google Sheetsアクセステスト: python3 tools/progress_tracker/cli.py status
```

## 🔧 自動化された環境確認

### 推奨実行方法
```bash
# 自動環境確認・切り替えスクリプト実行
source bin/shell/check_env.sh
```

### スクリプトの機能
- ✅ プロジェクトディレクトリ自動確認
- ✅ sam-env環境の自動切り替え（Windows/Linux両対応）  
- ✅ 必須パッケージの存在確認
- ✅ CUDA利用可能性確認
- ✅ 問題発生時の具体的解決策提示

## 📊 環境確認項目詳細

### 1. プロジェクトディレクトリ確認
```bash
# 期待値
pwd
# → /mnt/c/AItools/segment-anything
```

### 2. 仮想環境確認
```bash
# 期待値
echo $VIRTUAL_ENV
# → /mnt/c/AItools/segment-anything/sam-env (またはsam-envが含まれるパス)

# 現在のPython確認
which python
# → sam-env内のPythonパスを表示
```

### 3. 必須パッケージ確認

#### 必須パッケージリスト
- **PyTorch**: CUDA対応版（YOLO, SAM処理用）
- **google-auth関連**: Google Sheetsアクセス用
  - google-auth
  - google-auth-oauthlib
  - google-auth-httplib2
  - google-api-python-client
- **ultralytics**: YOLO v8実行用
- **opencv-python**: 画像処理用

#### 確認コマンド
```bash
# パッケージ存在確認
pip list | grep -E "(torch|google|ultralytics|opencv)"

# 機能確認
python -c "
import torch
import google.auth
import ultralytics
import cv2
print('✅ 全必須パッケージ確認OK')
"
```

### 4. CUDA確認
```bash
# CUDA利用可能性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU情報
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## 🛠 トラブルシューティング

### 問題1: sam-env環境が見つからない
```bash
# 症状
ls sam-env/
# → "No such file or directory"

# 解決策
python -m venv sam-env
source sam-env/Scripts/activate  # Windows
# source sam-env/bin/activate    # Linux
pip install -e .
```

### 問題2: Google認証パッケージ不足
```bash
# 症状
ModuleNotFoundError: No module named 'google'

# 解決策
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 問題3: PyTorch CUDA版未インストール  
```bash
# 症状
torch.cuda.is_available() → False

# 解決策
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 問題4: Google Sheetsアクセスエラー
```bash
# 症状
認証エラーまたは接続エラー

# 解決策
# 1. 認証ファイル確認
ls config/google_sheets_auth.json

# 2. テスト実行
python3 tools/progress_tracker/cli.py status
```

## 🔄 再発防止メカニズム

### 1. 定期的な環境確認
```bash
# 作業セッション開始時（必須）
source bin/shell/check_env.sh

# 長時間作業時（推奨：1時間ごと）
echo "現在の環境: $VIRTUAL_ENV"
```

### 2. 環境切り替え履歴の記録
```bash
# 環境切り替えログ
echo "$(date): 環境切り替え $VIRTUAL_ENV" >> logs/env_switch.log
```

### 3. 自動化の推進
- 作業開始時の環境確認を習慣化
- スクリプト実行の標準化
- 問題発生時の自動復旧

## 📈 効果測定

### 導入前後の比較指標
- **Google Sheetsアクセス成功率**: 目標 100%
- **セッション開始時の環境準備時間**: 目標 1分以内
- **環境起因のエラー発生回数**: 目標 0回/日

### 継続監視項目
- 環境確認スクリプトの実行頻度
- 依存関係関連エラーの発生頻度
- ワークフロー準拠率

## 🎯 運用ルール

### 必須実行タイミング
1. **新しいセッション開始時**（最重要）
2. **P1-B003等のタスク開始前**
3. **Google Sheetsアクセス前**
4. **バッチ処理実行前**

### 推奨実行タイミング
- 長時間作業の中断・再開時
- エラー発生時の復旧作業前
- 新機能実装開始時

---

**重要**: このガイドの手順を確実に実行することで、セッション継続時の環境管理問題を根本的に解決し、ワークフロー準拠の安定した開発環境を実現できます。