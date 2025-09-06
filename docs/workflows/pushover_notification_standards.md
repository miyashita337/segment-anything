# Pushover通知システム標準仕様

## 🚨 重要原則

**Pushover通知関連の全ての実装は `features/common/notification/pushover_image_sender.py` を使用すること**

- ❌ **新規Pushoverスクリプト作成禁止**: 事あるごとに新たなPushover実装を作らない
- ✅ **既存モジュール拡張**: 機能追加は `pushover_image_sender.py` を更新する
- ✅ **統一API使用**: 全てのプロジェクトは統一されたPushover APIを使用

## 📁 統一Pushover実装

### メインモジュール
```
features/common/notification/pushover_image_sender.py
```

### 主要機能
1. **画像付き通知**: `send_extraction_complete_with_images()`
2. **単一画像通知**: `send_pushover_with_image()`
3. **テキスト通知**: `global_pushover.py` へのフォールバック
4. **設定管理**: 複数設定ファイル形式対応

### 使用方法

#### 抽出完了通知（全画像添付）
```python
from features.common.notification.pushover_image_sender import send_extraction_complete_with_images

send_extraction_complete_with_images(
    title="QI-005抽出完了",
    extraction_dir="/path/to/extraction/",
    successful=20,
    total=25,
    failed=5,
    duration=3600.0
)
```

#### 単一画像通知
```python
from features.common.notification.pushover_image_sender import send_pushover_with_image

send_pushover_with_image(
    title="処理完了",
    message="画像処理が完了しました",
    image_path="/path/to/image.jpg"
)
```

## 📊 統一仕様

### 通知送信パターン
1. **詳細統計付き**: 最初の画像に処理結果統計を添付
2. **シンプル通知**: 2枚目以降は画像名のみ
3. **API制限対策**: 1秒間隔での順次送信
4. **エラーハンドリング**: 失敗時のフォールバック実装

### 設定ファイル対応
- segment-anything形式: `api_token`, `user_key`
- manga-character-extractor-api形式: `pushover.api_token`, `pushover.user_key`
- 旧形式: `token`, `user`

### 通知優先度・音声設定
- 成功率80%以上: `magic`音、通常優先度
- 成功率50-79%: `pushover`音、通常優先度
- 成功率50%未満: `pushover`音、高優先度
- 2枚目以降: 音なし

## 🔧 実装統合状況

### 統合済みファイル（20/31）
- `features/extraction/commands/extract_character.py` ✅
- その他18ファイル ✅

### 未統合ファイル（11/31）
- 個別対応が必要なファイルまたは非推奨ファイル

## 📋 新機能追加ガイドライン

### Pushover機能拡張時の手順
1. **pushover_image_sender.py を編集**: 新機能を既存モジュールに追加
2. **互換性確保**: 既存の呼び出し元が動作し続けることを保証
3. **テスト実行**: `python features/common/notification/pushover_image_sender.py`
4. **ドキュメント更新**: このファイルの更新

### 禁止事項
- ❌ 新たなPushover通知スクリプトの作成
- ❌ 直接API呼び出しの実装
- ❌ 設定ファイル読み込みの重複実装
- ❌ 独自の通知形式の実装

### 推奨事項
- ✅ `pushover_image_sender.py` の関数拡張
- ✅ 統一された設定管理の使用
- ✅ エラーハンドリングの再利用
- ✅ API制限対策の活用

## 🧪 テスト方法

### 単体テスト
```bash
python features/common/notification/pushover_image_sender.py
```

### 統合テスト
```bash
# 実際の抽出処理でのテスト
python features/extraction/commands/extract_character.py --batch
```

## 📈 バージョン履歴

- **v0.9.2**: 初回統一実装、全画像添付機能
- **v0.9.3** (予定): 追加機能拡張、パフォーマンス改善

## ⚠️ 重要事項

**このドキュメントは全Pushover実装の単一真実源（Single Source of Truth）です。**

新しいPushover機能が必要な場合は：
1. このドキュメントを参照
2. `pushover_image_sender.py` を拡張
3. このドキュメントを更新

**絶対に新たなPushoverスクリプトを作成しないでください。**