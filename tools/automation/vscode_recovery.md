# VSCode ハングアップ復旧手順

## 現在の状況
- kana05バッチ処理は順調進行中（22/39完了, 53.8%）
- SAM処理がCPU 110%で稼働中（正常）
- VS Codeがハングアップ状態

## 即座復旧手順

### 1. VS Code強制終了と再起動
```bash
# VS Codeプロセス強制終了
pkill -f "code"
killall code 2>/dev/null

# 30秒待機
sleep 30

# VS Code再起動
code /mnt/c/AItools/segment-anything
```

### 2. リソース最適化設定
VS Code設定で以下を適用：

```json
{
    "files.watcherExclude": {
        "**/node_modules/**": true,
        "**/.git/**": true,
        "**/results_batch/**": true,
        "**/tracker-workspace/**": true,
        "**/*.log": true
    },
    "search.exclude": {
        "**/tracker-workspace/**": true,
        "**/results_batch/**": true,
        "**/*.log": true
    },
    "files.exclude": {
        "**/tracker-workspace/*/extraction/batch_temp_*/**": true
    }
}
```

### 3. バッチ処理への影響回避
- バッチ処理は継続中（PID 72236, 73703）
- VS Code再起動はバッチ処理に影響しません
- Claude Codeプロセス（PID 73093）も独立動作

## 進捗監視コマンド（VS Code復旧後）
```bash
# 進捗確認
tail -f kana05_production.log

# プロセス確認
ps aux | grep -E "(simple_batch_runner|sam_yolo)"

# シンプル監視起動
python3 tools/automation/simple_progress_monitor.py PH3-007-PRODUCTION --interval 60
```

## 予防策
1. **ファイル監視除外**: 大量の一時ファイルを監視対象から除外
2. **定期的な再起動**: 重い処理中は1-2時間ごとにVS Code再起動
3. **分離作業**: CPU集約的な処理中は別ターミナルで監視

## 注意事項
- ❌ バッチ処理プロセスを停止しない
- ✅ VS Code再起動は安全
- ✅ Claude Codeは独立動作で継続可能