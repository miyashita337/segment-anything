# MERGE-001 抽出プログラム統合 - 移行ガイド

## 概要
P1-016-020で実装された改善機能を統合するため、以下の変更を実施しました：

## 変更内容

### 非推奨化プログラム
- `tools/core/sam_yolo_character_segment.py` → `deprecated/legacy_programs/sam_yolo_character_segment.py`

### 新メインプログラム
- `features/extraction/commands/extract_character.py` （統合版）

## 機能統合内容

### P1-018: バッチサイズ制御
```bash
# 旧
--max_files 10

# 新
--max-files 10
```

### P1-019: プロセス安定性
```bash
# 新機能
--resume  # チェックポイントから再開
```

### P1-020: SAM最適化
```bash
# 新機能
--sam-optimization-profile p1_020_optimized  # 93%高速化
```

## 新しい使用方法

### 基本コマンド
```bash
# 単一画像抽出
python3 -m features.extraction.commands.extract_character input.jpg -o output.png

# バッチ処理（P1-018-020統合版）
python3 -m features.extraction.commands.extract_character input_dir/ -o output_dir/ \
  --batch \
  --max-files 10 \
  --resume \
  --sam-optimization-profile p1_020_optimized \
  --verbose
```

### 品質ワークフロー統合
`tools/scripts/run_quality_workflow.sh` は自動的に新プログラムを使用します。

## パフォーマンス改善
- 処理時間: 93%削減（6分 → 24.7秒）
- 成功率: 30% → 80%
- SAMボトルネック: 85.5% → 14.7%

## 移行完了日
2025-07-31（MERGE-001実装完了）