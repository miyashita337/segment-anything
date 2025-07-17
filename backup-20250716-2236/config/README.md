# 設定ファイルについて

このディレクトリには、Character Extraction システムで使用する設定ファイルを配置します。

## Pushover通知設定

バッチ処理完了時にスマートフォンに通知を送るためのPushover設定です。

### 手順

1. **Pushoverアカウント作成**
   - https://pushover.net/ でアカウント作成
   - モバイルアプリをダウンロード

2. **アプリケーション作成**
   - Pushover ダッシュボードで新しいアプリケーションを作成
   - Application Token を取得

3. **設定ファイル作成**
   ```bash
   cd config
   cp pushover.json.example pushover.json
   ```

4. **設定ファイル編集**
   ```json
   {
     "token": "your_application_token_here",    # ← Application Token
     "user": "your_user_key_here",             # ← User Key  
     "device": "",                             # ← デバイス名（空白可）
     "title": "Character Extraction"           # ← 通知タイトル
   }
   ```

   - `token`: Pushoverで作成したアプリケーションのトークン
   - `user`: PushoverのUser Key（ダッシュボードTOPに表示）
   - `device`: 特定デバイスのみに送信したい場合（空白で全デバイス）
   - `title`: 通知のタイトル

5. **動作確認**
   ```bash
   python3 utils/notification.py
   ```

### 通知内容

バッチ処理完了時に以下の情報が通知されます：

- 処理結果（成功数/総数）
- 成功率
- 失敗数
- 処理時間
- 1画像あたりの平均処理時間

### セキュリティ

- `pushover.json` は `.gitignore` に含まれており、Gitで追跡されません
- API TokenやUser Keyが含まれるため、ファイルの取り扱いに注意してください

### トラブルシューティング

- **通知が来ない**: API TokenとUser Keyを確認
- **設定ファイルエラー**: JSON形式が正しいか確認
- **権限エラー**: Pushoverアプリケーションが有効か確認