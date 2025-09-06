# Pushover バッチ処理進捗通知システム

## 概要

長時間のバッチ処理中でも安心して待機できるよう、Pushoverを使用した進捗通知システムを実装しました。VSCodeがハングアップしても、スマートフォンで進捗と推定完了時刻を確認できます。

## 機能

### 通知タイミング
- **処理開始時**: 全体の画像数と推定処理時間
- **マイルストーン達成時**: 25%, 50%, 75%到達時（優先度高）
- **定期通知**: 30分ごとの進捗更新（カスタマイズ可能）
- **エラー発生時**: 重大エラーの即座通知
- **処理完了時**: 最終結果と統計情報

### 通知内容
```
📊 バッチ処理進捗 [22/39]
━━━━━━━━━━━━━━━━━━━━
✅ 進捗: 56.4% (22/39枚)
⏱️ 経過: 1時間23分
🎯 推定完了: 17:09
📈 成功率: 95.5% (21/22)
💨 処理速度: 267秒/枚
❌ 失敗: 1枚
━━━━━━━━━━━━━━━━━━━━
```

## セットアップ

### 1. Pushover設定ファイルの準備
```bash
# 設定ファイルをコピー
cp config/pushover.json.example config/pushover.json

# 編集してAPIトークンとユーザーキーを設定
nano config/pushover.json
```

```json
{
  "token": "your_application_token_here",
  "user": "your_user_key_here",
  "device": "",
  "title": "Character Extraction"
}
```

### 2. 通知テスト
```bash
# 通知システムのテスト
python3 -c "
from features.common.notification.global_pushover import test_notifications
test_notifications()
"
```

## 使用方法

### 基本的な使用（デフォルト30分間隔）
```bash
python3 tools/automation/simple_batch_runner.py PH3-007-PRODUCTION \
  --input-dir /path/to/images \
  --pushover-interval 30
```

### バックグラウンド実行（推奨）
```bash
# Pushover通知は自動的に有効化されます
./tools/automation/background_batch_runner.sh PH3-007-PRODUCTION \
  --input-dir /path/to/images \
  --batch-size 1 \
  --timeout 900
```

### 通知間隔のカスタマイズ
```bash
# 10分ごとに通知
--pushover-interval 10

# 60分ごとに通知
--pushover-interval 60
```

### 通知を無効化
```bash
# Pushover通知を完全に無効化
--no-pushover
```

## 高度な設定

### 通知の優先度
- **通常進捗**: 優先度0（通常）
- **マイルストーン**: 優先度1（高）
- **エラー**: 優先度2（緊急、繰り返し通知）

### カスタム通知音
- 開始時: `cosmic`
- 成功時: `magic`
- エラー時: `siren`
- 警告時: `falling`

## トラブルシューティング

### 通知が届かない場合
1. Pushover設定ファイルの確認
   ```bash
   cat config/pushover.json
   ```

2. APIトークンとユーザーキーの検証
   ```bash
   python3 test_pushover_notification.py
   ```

3. ネットワーク接続の確認
   ```bash
   curl -s --form-string "token=APP_TOKEN" \
     --form-string "user=USER_KEY" \
     --form-string "message=test" \
     https://api.pushover.net/1/messages.json
   ```

### エラー通知のみ受信したい場合
```python
# simple_batch_runner.py のカスタマイズ例
if self.use_pushover and error_occurred:
    notify_error("エラー発生", str(error))
```

## 利点

1. **安心感**: 画面が固まっても進捗が分かる
2. **時間管理**: 推定完了時刻で他の作業を計画可能
3. **リモート監視**: 外出先でも処理状況を確認
4. **エラー即応**: 問題発生時に即座に通知

## 注意事項

- Pushover APIには月間メッセージ数の制限があります（通常7,500メッセージ/月）
- 大量の画像処理で頻繁な通知を設定すると制限に達する可能性があります
- 通知間隔は処理規模に応じて適切に設定してください

## 実行例

### kana05（39枚）の処理例
```bash
# 推定時間: 約3.3時間（300秒/枚 × 39枚）
# 通知回数: 開始1回 + マイルストーン3回 + 定期6回 + 完了1回 = 約11回

./tools/automation/background_batch_runner.sh PH3-007-PRODUCTION \
  --input-dir /mnt/c/AItools/lora/train/yado/org/kana05 \
  --batch-size 1 \
  --timeout 900
```

通知により以下が可能になります：
- 17:09完了予定 → 17:00頃に戻ればよいと判断
- 進捗56.4% → 順調に進行中と確認
- 成功率95.5% → 品質も問題なし

これにより、強制終了の必要がなくなり、安定した長時間処理が可能になります。