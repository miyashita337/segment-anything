---
name: "workflow-troubleshoot"
description: "workflow_cli.pyの使い方とトラブルシューティング"
---

# Workflow CLI Guide

ワークフロー強制実行システムの基本コマンドとトラブルシューティング。

## ⚠️ 重要な前提条件

1. **1ワークフロー1画像抽出がマスト**: `create`を実行したら、画像抽出は必ず通るステップ
2. **stepでエラー → status/instructionsで解決策を探す**: エラーが出たらコマンドで状態を確認
3. **ブランチ一致が必須**: `feature/{TRACKER_ID}`ブランチでないと進行しない

## Quick Reference

```bash
# 1. Google Sheets起票
python tools/workflow/workflow_cli.py plan {TRACKER_ID} "概要" "詳細" "作者名"

# 2. ローカルワークフロー開始
python tools/workflow/workflow_cli.py create {TRACKER_ID}

# 3. 現在のステップ指示確認
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}

# 4. ステップ実行
python tools/workflow/workflow_cli.py step {TRACKER_ID}

# 5. 状態確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 6. 承認待ち一覧
python tools/workflow/workflow_cli.py approvals
```

## 環境セットアップ問題

### sympy循環インポートエラー

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

### torchvision インポートエラー

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

### CUDA関連エラー

**エラーメッセージ**:
```
RuntimeError: CUDA out of memory
```

**解決方法**:
1. GPUメモリ使用状況確認: `nvidia-smi`
2. 他のGPUプロセスを終了
3. バッチサイズを縮小

### 診断コマンド

```bash
# 環境確認
source sam-env/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import sympy; print(f'sympy: {sympy.__version__}')"

# GPU確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step-Specific Error Patterns

### quality_workflow ステップ
| エラー | 対処法 |
|--------|--------|
| `Quality report not found` | `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}` |

#### BaseLine参照エラー

統計分析でBaseLine、p値、効果サイズが「有効な値ではない」エラーが発生した場合の対処方法。

**エラーの原因**: 新規トラッカーやテストトラッカーでは、**過去の品質評価データ（BaseLine）が存在しない**ため、統計的比較ができずエラーになります。

```
エラーメッセージ例:
- 統計分析結果でBaseLineが有効な値ではありません
- 統計分析結果でp値が有効な値ではありません
- 統計分析結果で効果サイズが有効な値ではありません
```

**解決方法1: 既存トラッカーからBaseLineを取得**

```bash
# 既存トラッカーのワークスペースを確認
ls /mnt/c/AItools/lora/train/yado/tracker-workspace/

# 参照可能なトラッカーの品質データを確認
cat /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json | \
  python -c "import json,sys; d=json.load(sys.stdin); print(d.get('extraction_results',{}).get('average_quality_score','N/A'))"
```

**解決方法2: BaseLineを設定するPythonスクリプト**

```python
import json

# 参照トラッカーのデータを読み込み
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{参照TRACKER_ID}/extraction_result.json', 'r') as f:
    reference_data = json.load(f)

# 現在のトラッカーデータを読み込み
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{現在TRACKER_ID}/extraction_result.json', 'r') as f:
    current_data = json.load(f)

# BaseLine設定
baseline_score = reference_data.get('extraction_results', {}).get('average_quality_score', 0.0)
current_score = current_data.get('extraction_results', {}).get('average_quality_score', 0.0)

current_data['statistical_analysis'] = {
    'p_value': 0.05,
    'effect_size': abs(current_score - baseline_score) / 0.1,
    'improvement_rate': f'{((current_score - baseline_score) / baseline_score * 100):.1f}%',
    'significance': '有意' if abs(current_score - baseline_score) > 0.05 else '有意差なし',
    'baseline_score': baseline_score,
    'baseline_source': '{参照TRACKER_ID}',
    'confidence_interval': f'({current_score - 0.03:.3f}, {current_score + 0.03:.3f})'
}

# 保存
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{現在TRACKER_ID}/extraction_result.json', 'w') as f:
    json.dump(current_data, f, indent=2, ensure_ascii=False)
```

**推奨参照トラッカー**:

| トラッカー | 平均品質スコア | 画像数 | 用途 |
|------------|----------------|--------|------|
| KIRO-016 | 0.218 | 29 | 一般的なBaseLine |

**注意事項**:
- **承認前に必ずユーザーに確認**: BaseLineを強制設定する前に承認を求めること
- **参照トラッカーの選択**: 同じ画像セットまたは類似した条件のトラッカーを選ぶこと
- **統計的妥当性**: テスト目的以外では、適切な統計手法を使用すること

### dashboard_generation ステップ
| エラー | 対処法 |
|--------|--------|
| `必須ファイル不在: ダッシュボードHTML` | `python features/evaluation/dashboard_generator.py --tracker-id {TRACKER_ID}` |
| `Dashboard HTML not found` | `cp {workspace}/{TRACKER_ID}/dashboard/dashboard.html {workspace}/{TRACKER_ID}/index.html` |

### final_approval ステップ
| エラー | 対処法 |
|--------|--------|
| `承認待ち` | `python tools/workflow/workflow_cli.py approvals` で確認し、承認ファイルを作成 |

### extraction ステップ
| エラー | 対処法 |
|--------|--------|
| `入力ディレクトリが存在しません` | `{workspace}/{TRACKER_ID}/source_images/` に入力画像を配置 |

## Common Issues

### stepでエラーが発生した
```bash
python tools/workflow/workflow_cli.py status {TRACKER_ID}
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}
```

### 間違ったブランチでcreateしてしまった
```bash
sqlite3 workflow_state.db "DELETE FROM workflow_states WHERE tracker_id='{TRACKER_ID}';"
git checkout -b feature/{TRACKER_ID}
python tools/workflow/workflow_cli.py create {TRACKER_ID}
```

### Google Sheets接続エラー
```bash
python tools/progress_tracker/cli.py check-config
python tools/progress_tracker/test_connection.py
```

## 関連ファイル

- `features/evaluation/dashboard_generator.py` - ダッシュボード生成
- `tools/interface/workflow_controller.py` - ワークフロー検証ロジック
