# 日次ユーザータスクガイド

**作成日**: 2025-07-24  
**対象**: segment-anything プロジェクトのメイン開発者・ユーザー

## 📅 毎日実行すべきタスク

### 🌅 朝のタスク（プロジェクト開始時）

#### 1. システム状態確認（5分）
```bash
# GPU・環境チェック
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB' if torch.cuda.is_available() else 'CUDA not available')"

# 仮想環境確認
which python && pip list | grep -E "(torch|ultralytics|opencv)"

# プロジェクト状態確認
cd /mnt/c/AItools/segment-anything
git status --porcelain
```

#### 2. 前日の進捗レポート確認（3分）
```bash
# 日次進捗レポートの確認
python tools/daily_progress_tracker.py --date yesterday --summary

# アラート確認
python tools/check_alerts.py --since yesterday
```

#### 3. 今日の作業計画確認（2分）
```bash
# マイルストーン進捗確認
python tools/milestone_tracker.py --current-status

# 今日のタスク表示
python tools/show_daily_tasks.py --date today
```

### 🔧 開発作業中のタスク

#### メイン開発サイクル（随時実行）
```bash
# 1. コード変更後の品質チェック
./linter.sh  # flake8, black, mypy, isort

# 2. 小規模テスト実行
python test_phase2_simple.py  # 基本動作確認

# 3. 変更の動作確認
python tools/test_current_changes.py --quick
```

#### バッチ処理実行時
```bash
# 1. バッチ処理の実行（大規模データセット処理時）
python extract_kana03.py --quality_method balanced --input test_small/

# 2. リアルタイム品質監視
python tools/monitor_batch_quality.py --follow

# 3. 処理完了後の客観評価
python tools/objective_quality_evaluation.py --batch results_batch/ --generate-report
```

### 🌆 夕方のタスク（作業終了時）

#### 1. 今日の成果確認（10分）
```bash
# 客観的評価レポート生成
python tools/daily_progress_tracker.py --date today --full-analysis

# 出力例確認項目:
# - PLA平均値が目標（0.75）以上か？
# - SCI平均値が目標（0.70）以上か？  
# - PLE値が改善傾向（0.05以上）か？
# - アラートが発生していないか？
```

#### 2. 作業ログの保存（5分）
```bash
# Git変更の確認・コミット
git add .
git status
git commit -m "Daily work: [今日の主な作業内容]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 重要: 画像ファイルがcommitされていないことを必ず確認
git log --name-only -1 | grep -E "\.(jpg|png|webp)$" && echo "⚠️ 画像ファイル発見 - 確認要" || echo "✅ 画像ファイルなし"
```

#### 3. 明日の準備（3分）
```bash
# 明日のタスク計画生成
python tools/plan_tomorrow_tasks.py --based-on-today

# 継続監視システムの状態確認
python tools/monitoring_health_check.py --schedule-tomorrow
```

## 📊 週次タスク（毎週金曜日）

### 🔍 週次分析・レビュー（30分）

#### 1. 週間進捗分析
```bash
# 週間トレンドレポート生成
python tools/generate_weekly_report.py --week-start $(date -d "last monday" +%Y-%m-%d)

# マイルストーン進捗評価
python tools/milestone_tracker.py --weekly-review
```

#### 2. システム性能評価
```bash
# 週間性能統計
python tools/performance_analysis.py --period week

# 品質トレンド分析
python tools/quality_trend_analysis.py --period week --generate-charts
```

#### 3. 次週計画策定
```bash
# 次週の目標設定
python tools/set_weekly_targets.py --based-on-current-progress

# 改善点の特定
python tools/identify_improvement_areas.py --period week
```

## 🎯 月次タスク（毎月最終金曜日）

### 📈 月次総合評価（60分）

#### 1. 月間成果分析
```bash
# 月間総合レポート
python tools/generate_monthly_report.py --month $(date +%Y-%m)

# マイルストーン達成度評価
python tools/milestone_achievement_analysis.py --month $(date +%Y-%m)
```

#### 2. システム健全性チェック
```bash
# 依存関係の更新確認
pip list --outdated

# セキュリティチェック
python tools/security_audit.py --comprehensive

# ディスク使用量確認
python tools/storage_usage_analysis.py --cleanup-suggestions
```

## 🚨 緊急時・アラート発生時のタスク

### アラート種別別対応

#### 1. 性能退行アラート
```bash
# 直近の変更を確認
git log --oneline -10

# 前回成功時との比較
python tools/compare_with_last_success.py --detailed

# ロールバック検討
python tools/suggest_rollback_options.py
```

#### 2. 品質低下アラート
```bash
# 詳細品質分析
python tools/detailed_quality_analysis.py --problematic-images

# 失敗パターン分析
python tools/failure_pattern_analysis.py --recent

# 改善提案生成
python tools/generate_improvement_suggestions.py --based-on-failures
```

#### 3. システム障害アラート
```bash
# システム診断
python tools/system_diagnostics.py --comprehensive

# ログ分析
python tools/analyze_error_logs.py --since "1 hour ago"

# 復旧手順実行
python tools/system_recovery.py --guided
```

## 📋 チェックリスト形式の日次確認

### ✅ 毎日必須チェック項目

```yaml
朝の確認:
  - [ ] CUDA環境正常動作
  - [ ] 仮想環境適切に有効化
  - [ ] 前日レポート内容確認
  - [ ] 今日の目標明確化

開発中の確認:
  - [ ] コード品質チェック（linter.sh）通過
  - [ ] 小規模テスト実行・成功
  - [ ] 画像ファイル非コミット確認

夕方の確認:
  - [ ] PLA目標値（0.75）以上達成
  - [ ] SCI目標値（0.70）以上達成
  - [ ] PLE改善傾向（0.05以上）確認
  - [ ] アラート未発生確認
  - [ ] Git変更適切にコミット
```

## 🔄 自動化可能なタスク

### 現在手動だが自動化推奨のタスク

```python
# 自動化スクリプト例（実装推奨）
def automate_daily_tasks():
    """日次タスクの自動実行"""
    
    # 1. 朝の環境チェック自動実行
    if datetime.now().hour == 9:  # 朝9時
        run_system_health_check()
        send_daily_status_notification()
    
    # 2. 夕方のレポート自動生成
    if datetime.now().hour == 18:  # 夕方6時
        generate_daily_progress_report()
        check_for_alerts()
        
    # 3. 週次レポート自動実行
    if datetime.now().weekday() == 4:  # 金曜日
        generate_weekly_analysis()
```

## 💡 効率化のためのヒント

### よく使うコマンドのエイリアス設定
```bash
# ~/.bashrc または ~/.zshrc に追加
alias sam-env='cd /mnt/c/AItools/segment-anything && source sam-env/bin/activate'
alias sam-test='python test_phase2_simple.py'
alias sam-lint='./linter.sh'
alias sam-status='python tools/daily_progress_tracker.py --date today --summary'
alias sam-quality='python tools/objective_quality_evaluation.py --batch results_batch/ --quick'
```

### 作業効率向上Tips
1. **並列実行**: 品質評価は時間がかかるため、バックグラウンド実行を活用
2. **プリセット活用**: よく使う設定はconfig/に保存して再利用
3. **通知活用**: 長時間処理は完了通知設定で効率化
4. **履歴活用**: 過去の成功パターンを参考に作業手順を最適化

---

**重要**: これらのタスクを毎日実行することで、進捗の可視化・品質の継続改善・問題の早期発見が実現されます。