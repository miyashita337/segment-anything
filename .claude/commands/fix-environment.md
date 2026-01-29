# 環境問題トラブルシューティング

sam-env仮想環境で発生する一般的な環境問題の診断と解決を行います。

## 対象エラー

### 1. sympy循環インポートエラー

**エラーメッセージ**:
```
ImportError: cannot import name 'Add' from partially initialized module 'sympy.core.add'
(most likely due to a circular import)
```

**原因**: PyTorch/torchvision と sympy のバージョン間に互換性問題

**解決方法**:
```bash
source sam-env/bin/activate
pip cache purge
pip install --force-reinstall sympy
```

**代替方法（上記が失敗した場合）**:
```bash
source sam-env/bin/activate
pip install sympy==1.12
```

### 2. torchvision インポートエラー

**エラーメッセージ**:
```
from torchvision.transforms.functional import resize, to_pil_image
ModuleNotFoundError: No module named 'torchvision'
```

**解決方法**:
```bash
source sam-env/bin/activate
pip install torchvision --upgrade
```

### 3. CUDA関連エラー

**エラーメッセージ**:
```
RuntimeError: CUDA out of memory
```

**解決方法**:
1. GPUメモリ使用状況確認: `nvidia-smi`
2. 他のGPUプロセスを終了
3. バッチサイズを縮小

## 診断コマンド

```bash
# 環境確認
source sam-env/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import sympy; print(f'sympy: {sympy.__version__}')"

# GPU確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 実行手順

1. エラーメッセージを確認
2. 上記の対象エラーと照合
3. 該当する解決方法を実行
4. 診断コマンドで修正確認
