# 一時的スクリプト管理

このディレクトリには、一時的に作成されたスクリプトを整理・保管します。

## ディレクトリ構造

```
temp/
├── README.md                # このファイル
├── scripts/                 # 一時的なPythonスクリプト
│   ├── analysis/           # 分析用スクリプト
│   ├── testing/            # テスト用スクリプト
│   └── migration/          # 移行・変換用スクリプト
└── logs/                   # 実行ログ
```

## スクリプト一覧

### analysis/ (1ファイル)
- analyze_extraction_failures.py - 抽出失敗分析用

### testing/ (12ファイル)
- test_anime_yolo_batch.py - アニメYOLOバッチテスト
- test_anime_yolo_model.py - アニメYOLOモデルテスト
- test_basic_functionality.py - 基本機能テスト
- test_box_expansion_demo.py - ボックス拡張デモ
- test_fullbody_extraction.py - 全身抽出テスト
- test_fullbody_with_expansion.py - 拡張付き全身テスト
- test_line_pattern_debug.py - ラインパターンデバッグ
- test_phase_a_box_expansion.py - Phase Aボックス拡張
- test_phase_a_box_expansion_visualization.py - Phase A可視化
- test_real_image_screentone.py - 実画像スクリーントーン
- test_screentone_debug.py - スクリーントーンデバッグ
- test_yolo_threshold_005.py - YOLO閾値0.005テスト
- yolo_threshold_comparison_test.py - YOLO閾値比較テスト

### migration/ (6ファイル)
- run_batch_extraction.py - バッチ抽出実行
- run_batch_v034.py - v0.34バッチ実行
- run_batch_v035.py - v0.35バッチ実行
- run_kana07_batch.py - kana07バッチ実行
- run_small_test_v034.py - v0.34小規模テスト

## 管理ルール

1. **新規作成時**: 必ずこのREADMEを更新
2. **保存期間**: 最終使用から30日
3. **移行先**: 
   - 恒久的に必要 → `tools/` または適切なモジュール
   - テスト用 → `tests/` に正式なUnitTestとして実装
4. **削除基準**: 30日間未使用、または機能が正式実装済み

## 更新履歴

- 2025-07-20: ディレクトリ作成、初期構造定義
- 2025-07-20 02:48: 18個の一時スクリプトを整理・移動完了