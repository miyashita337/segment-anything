# 包括的ワークフローテストシステム

## 📋 概要

本文書では、Level 1-4の包括的ワークフローテストシステムの仕様、使用方法、保守手順について説明します。

## 🎯 システム目的

### 主な目標
- **仕様違反の防止**: 入力パス検証等の仕様違反を事前検出
- **品質保証**: SAM+YOLO抽出から統計分析までの品質確保
- **自動化促進**: CI/CD統合による継続的テスト実行
- **デバッグ支援**: 問題発生時の迅速な原因特定

### 解決する問題
1. **入力パス未検証問題**: 存在しないディレクトリでの処理継続
2. **品質評価の一貫性**: 抽出品質の統一的評価基準
3. **統計分析の信頼性**: Cohen's d計算、p値検定の正確性
4. **承認プロセスの透明性**: Pushover通知、承認フローの自動化

## 🏗️ システム構成

### Level 1: 基本ワークフローテスト
**ファイル**: `tests/workflow/test_basic_workflow.py`

#### テスト範囲
- 入力パス検証（26テストケース）
- トラッカーID検証
- ワークスペース作成
- 基本エラーハンドリング

#### 主要テストケース
```python
def test_input_path_validation_nonexistent_path()
def test_tracker_id_format_validation()
def test_workspace_creation_with_permissions()
def test_claude_md_compliance()
```

#### 品質保証ポイント
- CLAUDE.md準拠の厳格な入力検証
- 統一エラーメッセージ形式
- 代替案提案の完全禁止

### Level 2: 品質ワークフローテスト
**ファイル**: `tests/workflow/test_quality_workflow.py`

#### テスト範囲
- SAM+YOLO抽出処理（23テストケース）
- 品質評価システム
- ダッシュボード生成
- 画像処理品質チェック

#### 主要テストケース
```python
def test_sam_yolo_extraction_balanced_method()
def test_quality_evaluation_comprehensive()
def test_dashboard_generation_with_statistics()
def test_grade_distribution_analysis()
```

#### 品質保証ポイント
- 5つの品質評価手法の検証
- A-F評価の統計的分析
- Base64画像埋め込み機能

### Level 3: 統計分析ワークフローテスト
**ファイル**: `tests/workflow/test_statistical_workflow.py`

#### テスト範囲
- Cohen's d効果サイズ計算（36テストケース）
- Welch's t検定実装
- Google Sheets API統合
- 信頼区間計算

#### 主要テストケース
```python
def test_cohens_d_calculation_accuracy()
def test_welch_t_test_implementation()
def test_google_sheets_integration()
def test_confidence_interval_calculation()
```

#### 品質保証ポイント
- 統計的妥当性の数学的検証
- Google Sheets API のモック統合
- 複数トラッカー間の比較分析

### Level 4: 承認ワークフローテスト
**ファイル**: `tests/workflow/test_approval_workflow.py`

#### テスト範囲
- Pushover通知システム（42テストケース）
- 承認プロセス管理
- 進捗追跡機能
- 統合ワークフロー

#### 主要テストケース
```python
def test_pushover_extraction_complete_notification()
def test_approval_request_lifecycle()
def test_progress_task_dependencies()
def test_integrated_approval_workflow()
```

#### 品質保証ポイント
- 通知システムの信頼性
- 承認プロセスの透明性
- 進捗管理の一貫性

## 🔧 モックシステム

### Mock SAM+YOLO抽出器
**ファイル**: `tests/mocks/mock_sam_yolo.py`

#### 特徴
- 現実的な成功率設定（balanced: 85%, confidence: 78%）
- A-F評価の統計的分布
- 可変品質スコア生成

#### 設定例
```python
METHOD_SUCCESS_RATES = {
    "balanced": 0.85,
    "confidence_priority": 0.78,
    "size_priority": 0.82,
    "fullbody_priority": 0.75,
    "central_priority": 0.80
}
```

### Mock Google Sheets クライアント
**ファイル**: `tests/mocks/mock_google_sheets.py`

#### 機能
- トラッカーデータ管理
- 統計分析結果の保存
- Cohen's d計算の統合

#### API例
```python
client.update_tracker_data(tracker_id, data)
cohens_d = client.calculate_cohens_d(current, baseline)
```

### Mock Pushover システム
**ファイル**: `tests/mocks/mock_pushover.py`

#### 通知タイプ
- 抽出完了通知
- 承認依頼通知
- 品質アラート通知
- カスタム通知

#### 使用例
```python
client.send_extraction_complete_notification(
    tracker_id="TEST-001",
    total_images=50,
    successful_extractions=45,
    success_rate=90.0
)
```

### Mock 承認システム
**ファイル**: `tests/mocks/mock_approval_system.py`

#### 承認ステージ
- `EXTRACTION_START`: 抽出開始
- `QUALITY_CHECK`: 品質チェック
- `STATISTICAL_ANALYSIS`: 統計分析
- `FINAL_APPROVAL`: 最終承認
- `DEPLOYMENT`: デプロイ

#### 承認フロー
```python
request = system.create_approval_request(
    tracker_id, stage, title, details
)
system.approve_request(request_id, approver, comment)
```

### Mock 進捗管理
**ファイル**: `tests/mocks/mock_progress_manager.py`

#### タスク管理機能
- タスクライフサイクル管理
- 依存関係処理
- 進捗率計算
- 統計分析

## 🚀 実行方法

### 基本実行
```bash
# 全レベル実行
./bin/shell/run_workflow_tests.sh

# 特定レベル実行
./bin/shell/run_workflow_tests.sh --level level_1
./bin/shell/run_workflow_tests.sh --level level_2
./bin/shell/run_workflow_tests.sh --level level_3
./bin/shell/run_workflow_tests.sh --level level_4

# 簡潔モード
./bin/shell/run_workflow_tests.sh --quiet

# 失敗時停止
./bin/shell/run_workflow_tests.sh --stop-on-failure

# 結果ファイル出力
./bin/shell/run_workflow_tests.sh --output results.json
```

### Python直接実行
```bash
# テストランナー直接実行
python3 tests/test_runner.py --level all --verbose

# 個別テストファイル実行
pytest tests/workflow/test_basic_workflow.py -v
pytest tests/workflow/test_quality_workflow.py -v
pytest tests/workflow/test_statistical_workflow.py -v
pytest tests/workflow/test_approval_workflow.py -v
```

### CI/CD統合
```yaml
# .github/workflows/workflow-tests.yml
name: Workflow Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Workflow Tests
        run: ./bin/shell/run_workflow_tests.sh --output ci-results.json
```

## 📊 テスト結果の解釈

### 成功基準
- **全体成功率**: 95%以上
- **レベル成功率**: 100%（全レベル通過）
- **個別テスト成功率**: 各レベル95%以上

### 結果ファイル構造
```json
{
  "overall_result": "PASSED",
  "execution_info": {
    "start_time": "2025-08-27T10:00:00",
    "end_time": "2025-08-27T10:05:30",
    "total_duration_seconds": 330.5,
    "levels_executed": 4
  },
  "aggregate_stats": {
    "total_tests": 127,
    "total_passed": 124,
    "total_failed": 2,
    "total_skipped": 1,
    "total_errors": 0,
    "success_rate": 97.6
  },
  "level_stats": {
    "levels_passed": 4,
    "levels_failed": 0,
    "levels_errors": 0,
    "level_success_rate": 100.0
  }
}
```

## 🔍 トラブルシューティング

### よくある問題と解決策

#### 1. インポートエラー
```bash
ModuleNotFoundError: No module named 'tests.mocks'
```

**解決策**:
```bash
export PYTHONPATH="/mnt/c/AItools/segment-anything:${PYTHONPATH:-}"
```

#### 2. パーミッションエラー
```bash
Permission denied: ./bin/shell/run_workflow_tests.sh
```

**解決策**:
```bash
chmod +x bin/shell/run_workflow_tests.sh
```

#### 3. テストファイル不足
```bash
❌ テストファイル不足: tests/workflow/test_basic_workflow.py
```

**解決策**:
- 必要なテストファイルが存在することを確認
- git pull で最新版を取得

#### 4. Python パッケージ不足
```bash
⚠️ パッケージ不足: pytest
```

**解決策**:
```bash
pip install pytest pytest-mock
```

### デバッグモード実行
```bash
# デバッグ情報付きで実行
./bin/shell/run_workflow_tests.sh --verbose --level level_1

# 特定テストのみ実行
pytest tests/workflow/test_basic_workflow.py::test_input_path_validation_nonexistent_path -v -s
```

## 📈 品質指標

### テスト品質メトリクス
- **コードカバレッジ**: 90%以上
- **テストケース数**: 127件（Level 1: 26, Level 2: 23, Level 3: 36, Level 4: 42）
- **実行時間**: 300秒以下
- **成功率**: 95%以上

### テストケース分布
```
Level 1 (基本): 26テストケース (20.5%)
├── 入力パス検証: 10ケース
├── トラッカーID検証: 8ケース
├── ワークスペース作成: 5ケース
└── エラーハンドリング: 3ケース

Level 2 (品質): 23テストケース (18.1%)
├── SAM+YOLO抽出: 8ケース
├── 品質評価: 7ケース
├── ダッシュボード生成: 5ケース
└── 画像処理: 3ケース

Level 3 (統計): 36テストケース (28.3%)
├── Cohen's d計算: 12ケース
├── 統計検定: 10ケース
├── Google Sheets統合: 8ケース
└── 信頼区間: 6ケース

Level 4 (承認): 42テストケース (33.1%)
├── Pushover通知: 15ケース
├── 承認プロセス: 12ケース
├── 進捗管理: 10ケース
└── 統合ワークフロー: 5ケース
```

## 🔄 継続的改善

### 定期メンテナンス
- **月次**: テストケース見直し、成功率分析
- **四半期**: モックシステム更新、新機能対応
- **半年**: 全体アーキテクチャ見直し

### 機能拡張計画
1. **リアルタイム監視**: テスト実行状況のリアルタイム表示
2. **パフォーマンステスト**: 大規模データでの性能検証
3. **セキュリティテスト**: 入力検証の脆弱性チェック
4. **並列実行**: テストレベル並列実行による高速化

### 品質改善サイクル
1. **問題検出**: テスト失敗の分析
2. **根本原因分析**: 仕様違反・バグの特定
3. **修正実装**: 問題修正・テストケース追加
4. **回帰テスト**: 修正による副作用確認
5. **継続監視**: 同様問題の再発防止

## 📚 参考資料

### 関連ドキュメント
- [入力パス検証チェックリスト](../checklists/input_path_validation_checklist.md)
- [統合トラッカーテンプレート](../templates/unified_tracker_template.md)
- [技術仕様書](../technical_specifications.md)

### 実装ファイル
- **バリデーション**: `tools/utils/input_path_validator.py`
- **ワークフロー検証**: `tools/utils/workflow_validator.py`
- **テストランナー**: `tests/test_runner.py`
- **実行スクリプト**: `bin/shell/run_workflow_tests.sh`

### 外部リソース
- [pytest公式ドキュメント](https://docs.pytest.org/)
- [モックオブジェクト設計パターン](https://docs.python.org/3/library/unittest.mock.html)
- [CI/CD テスト戦略](https://docs.github.com/en/actions/automating-builds-and-tests)

## 📞 サポート

### 問題報告
問題や提案がある場合は、以下の情報を含めてチーケットを作成してください：

1. **実行環境**: OS、Python バージョン、実行コマンド
2. **エラー詳細**: エラーメッセージ、スタックトレース
3. **期待結果**: 本来期待していた動作
4. **再現手順**: 問題を再現する具体的な手順

### よくある質問

**Q: テスト実行に時間がかかりすぎる**
A: `--level` オプションで特定レベルのみ実行するか、`--quiet` オプションで出力を簡潔化してください。

**Q: 一部のテストが環境依存で失敗する**
A: モックシステムを使用しているため環境依存は最小限ですが、Python バージョンや必須パッケージを確認してください。

**Q: 新しいテストケースを追加したい**
A: 該当するレベルのテストファイルに新しい `test_` 関数を追加し、適切なモックシステムを使用してください。

---

**最終更新**: 2025-08-27  
**文書バージョン**: v1.0  
**対応システムバージョン**: Level 1-4 統合テストシステム