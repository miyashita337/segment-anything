# Tools Directory Structure Plan

## 新しいディレクトリ構造

```
tools/
├── core/                    # 継続使用される中核ツール
│   ├── google_sheets_updater.py
│   ├── quality_dashboard.py
│   ├── run_auto_pipeline.py
│   ├── run_objective_evaluation.py
│   ├── unified_quality_checker.py
│   └── sam_yolo_character_segment.py
│
├── batch/                   # バッチ処理系スクリプト
│   ├── kana08_enhanced_stable_batch.py
│   ├── kana08_stable_batch_restored.py
│   └── batch_task_ticketing.py
│
├── testing/                 # テスト・評価系
│   ├── test_difficult_pose.py
│   ├── test_phase2_simple.py
│   ├── test_phase3_cli.py
│   ├── test_priority_integration.py
│   ├── test_resume_functionality.py
│   ├── validate_evaluation_data.py
│   └── evaluation/          # 評価レポート生成
│       ├── evaluate_batch_images.py
│       ├── evaluate_batch_images_v2.py
│       ├── generate_evaluation_report.py
│       ├── generate_evaluation_report_v2.py
│       └── week3_final_benchmark.py
│
├── scripts/                 # 一時的・特定目的スクリプト
│   ├── p1_a001_release.py
│   ├── calculate_p1_a001_metrics.py
│   ├── ph2_002_dashboard_generator.py
│   ├── fix_chart_font.py
│   ├── update_sheet_headers.py
│   └── status_update_hook.py
│
├── utils/                   # ユーティリティ・共通機能
│   ├── init_models.py
│   ├── cleanup_repository.py
│   ├── audit_path_compliance.py
│   └── file_protection_checklist.py
│
├── legacy/                  # レガシー・重複機能（移行対象）
│   ├── read_google_sheets.py
│   ├── read_google_sheets_demo.py
│   ├── read_sheets_with_api.py
│   └── unified_quality_checker_legacy.py
│
└── progress_tracker/        # 既存のprogress_trackerモジュール（そのまま）
    ├── cli.py
    ├── config.py
    ├── connection_monitor.py
    ├── data_models.py
    ├── migration_tool.py
    ├── progress_manager.py
    ├── sheets_client.py
    ├── test_connection.py
    ├── update_dates.py
    └── workflow_integration.py
```

## 移行計画

### 即座に移動するファイル
1. **core/** - 6ファイル
2. **batch/** - 3ファイル
3. **testing/** - 6ファイル + evaluation/5ファイル
4. **scripts/** - 6ファイル
5. **utils/** - 4ファイル
6. **legacy/** - 4ファイル

### 合計
- 移動対象: 34ファイル
- progress_tracker/: 10ファイル（そのまま）
- 総計: 44ファイル（整理後も同数）

## 実行手順
1. ディレクトリ作成
2. ファイル移動
3. import文の調整（必要に応じて）
4. 動作確認