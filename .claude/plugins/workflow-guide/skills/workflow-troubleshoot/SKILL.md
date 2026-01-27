---
description: |
  segment-anything プロジェクトのワークフロー問題を診断・解決するスキル。

  トリガーフレーズ:
  - 「ワークフローでエラーが出た」
  - 「抽出処理で失敗した」
  - 「品質評価が低い」
  - 「バッチ処理がうまくいかない」
  - 「SAMの処理でエラー」
---

# Workflow Troubleshoot Skill

segment-anything プロジェクトのワークフロー問題を診断・解決するためのガイダンスを提供します。

## Quick Diagnosis

問題の種類を特定してください：

### 1. 環境・セットアップ問題
- ImportError / ModuleNotFoundError
- CUDA利用不可
- モデルファイル不足

**対処**: 環境確認コマンドを実行
```bash
# GPU確認
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# パス確認
cd /mnt/c/AItools/segment-anything
python3 -c "import sys; print('\\n'.join(sys.path))"
```

### 2. 実行時エラー
- CUDA out of memory
- FileNotFoundError
- 処理中断

**対処**: リソース確認
```bash
nvidia-smi  # GPU状態
free -h     # メモリ状態
df -h       # ディスク容量
```

### 3. 品質・結果問題
- 成功率が低い（< 80%）
- 手足切断
- 背景混入

**対処**: パラメータ調整が必要。詳細は `./references/quality_evaluation.md` を参照

### 4. バッチ処理問題
- 処理が途中で止まる
- 出力ファイルがない
- 処理速度が遅い

**対処**: バッチ処理ガイドを参照。詳細は `./references/batch_extraction.md` を参照

---

## Common Issues

### ImportError: cannot import name 'sam_model_registry'

**原因**: Python パス設定の問題

**解決**:
```bash
cd /mnt/c/AItools/segment-anything
export PYTHONPATH="/mnt/c/AItools/segment-anything:$PYTHONPATH"
```

### UnicodeDecodeError: 'cp932' codec

**原因**: Windows環境での文字コード問題

**解決**: 最新コードを取得
```bash
git pull origin main
```

### CUDA out of memory

**原因**: GPU メモリ不足

**解決**:
```bash
# 1. GPU メモリクリア
python3 -c "import torch; torch.cuda.empty_cache()"

# 2. 軽量モデル使用
--model_path yolov8n.pt  # yolov8x.pt → yolov8n.pt
```

### FileNotFoundError: sam_vit_h_4b8939.pth not found

**原因**: SAMモデルファイル不足

**解決**:
```bash
# モデルダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 成功率が低い（< 80%）

**原因**: パラメータ設定が最適でない

**解決**:
```bash
# 閾値調整
--score_threshold 0.05  # より高感度

# 品質手法変更
--quality_method confidence_priority
```

---

## 詳細リファレンス

問題の種類に応じて、以下の詳細ドキュメントを参照してください：

| 問題カテゴリ | 参照ファイル |
|-------------|-------------|
| バッチ処理全般 | `./references/batch_extraction.md` |
| エラー対処 | `./references/troubleshooting.md` |
| 品質評価・改善 | `./references/quality_evaluation.md` |

---

## 環境診断コマンド

問題切り分けのための診断コマンド：

```bash
#!/bin/bash
echo "=== Environment Diagnosis ==="
echo "Python: $(python3 --version)"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Working Dir: $(pwd)"
```

---

## 推奨コマンド

### キャラクター抽出（推奨）
```bash
python3 features/extraction/commands/extract_character.py \
  "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  --batch \
  --verbose \
  --strict-validation
```

### インタラクティブ抽出（緊急時）
```bash
python3 features/extraction/commands/quick_interactive.py image.jpg
```

---

## サポート

このスキルで解決できない場合：
1. `./references/` の詳細ドキュメントを確認
2. `docs/workflows/` の完全版ドキュメントを参照
3. GitHub Issues で報告
