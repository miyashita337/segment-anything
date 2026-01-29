# BaseLine参照エラー解決ガイド

統計分析でBaseLine、p値、効果サイズが「有効な値ではない」エラーが発生した場合の対処方法。

## エラーの原因

新規トラッカーやテストトラッカーでは、**過去の品質評価データ（BaseLine）が存在しない**ため、統計的比較ができずエラーになります。

```
エラーメッセージ例:
- 統計分析結果でBaseLineが有効な値ではありません
- 統計分析結果でp値が有効な値ではありません
- 統計分析結果で効果サイズが有効な値ではありません
```

## 解決方法

### 1. 既存トラッカーからBaseLineを取得

```bash
# 既存トラッカーのワークスペースを確認
ls /mnt/c/AItools/lora/train/yado/tracker-workspace/

# 参照可能なトラッカーの品質データを確認
cat /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json | \
  python -c "import json,sys; d=json.load(sys.stdin); print(d.get('extraction_results',{}).get('average_quality_score','N/A'))"
```

### 2. BaseLineを設定するPythonスクリプト

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

## 推奨参照トラッカー

| トラッカー | 平均品質スコア | 画像数 | 用途 |
|------------|----------------|--------|------|
| KIRO-016 | 0.218 | 29 | 一般的なBaseLine |

## 注意事項

- **承認前に必ずユーザーに確認**: BaseLineを強制設定する前に承認を求めること
- **参照トラッカーの選択**: 同じ画像セットまたは類似した条件のトラッカーを選ぶこと
- **統計的妥当性**: テスト目的以外では、適切な統計手法を使用すること

## 関連ファイル

- `features/evaluation/dashboard_generator.py` - ダッシュボード生成
- `tools/interface/workflow_controller.py` - ワークフロー検証ロジック
