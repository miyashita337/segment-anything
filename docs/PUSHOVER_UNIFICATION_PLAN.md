# Pushover通知システム統一化計画

## 🚨 現状の問題

### 重複実装の乱立
1. **global_pushover.py** - 標準共通モジュール（推奨）
2. **notification.py** - 別の共通モジュール
3. **extraction_notifier.py** - 抽出専用モジュール  
4. **17ファイル** - 直接API実装

### 影響
- 同じ機能が複数箇所に実装されている
- 修正時に全箇所を修正する必要がある
- 通知が届かない原因の特定が困難
- メンテナンス性の著しい低下

## ✅ 統一化方針

### Phase 1: 標準モジュールの確立
```python
# 全てのファイルでこれを使用
from features.common.notification.global_pushover import (
    notify_success,      # 成功通知
    notify_error,        # エラー通知
    notify_warning,      # 警告通知
    notify_process_complete,  # 処理完了通知
    notify_long_process_start,  # 長時間処理開始
    notify_critical_error  # 重大エラー
)
```

### Phase 2: 既存実装の移行

#### 優先度高（即座に移行）
- `features/extraction/commands/extract_character.py` ✅ 完了 
- `integrated_quality_pipeline.py`

#### 優先度中（段階的移行）
- バッチ処理系ファイル
- ワークフロースクリプト

#### 優先度低（将来的移行）
- テストファイル
- deprecated配下のファイル

### Phase 3: 旧実装の削除
1. `extraction_notifier.py` → global_pushover.pyに統合
2. `notification.py` → 削除またはglobal_pushover.pyへのエイリアス化
3. 直接API実装 → 全て共通モジュール使用に変更

## 📊 移行チェックリスト

### 直接API実装ファイル（要修正）
- [ ] background_extraction_integrate_3_6_03_full.py
- [ ] background_extraction_integrate_3_6_04.py
- [ ] complete_extraction_integrate_3_6_04.py
- [ ] features/common/notification/notification.py
- [ ] pushover_compare_integrate_3_6.py
- [ ] qc_batch_extraction.py
- [ ] send_all_images_3_6_04.py
- [ ] send_final_notification_3_6_04.py
- [ ] send_pushover_images.py
- [ ] test_pushover.py
- [ ] tools/audit_path_compliance.py
- [ ] tools/core/integrated_quality_pipeline.py
- [ ] tools/scripts/pushover_individual_sender_integrate_3_6_03.py
- [ ] tools/scripts/pushover_sender_integrate_3_6_01.py
- [ ] tools/scripts/pushover_sender_integrate_3_6_02.py
- [ ] tools/scripts/pushover_sender_integrate_3_6_03.py

### 独自モジュール使用ファイル（要変更）
- [ ] features/extraction/commands/extract_character.py
- [ ] その他extraction_notifierを使用するファイル

## 🛡️ 再発防止策

### 1. コーディング規約
```python
# ❌ 禁止
import requests
response = requests.post("https://api.pushover.net/1/messages.json", ...)

# ✅ 推奨
from features.common.notification.global_pushover import notify_success
notify_success("処理完了", "詳細メッセージ")
```

### 2. レビューポイント
- 新規ファイルでPushover通知が必要な場合は必ずglobal_pushover.pyを使用
- 直接API実装は絶対に禁止
- PRレビュー時に通知実装の重複をチェック

### 3. テストコード
```python
# tests/unit/test_pushover_unification.py
def test_no_direct_api_implementation():
    """直接API実装が存在しないことを確認"""
    files = find_all_python_files()
    for file in files:
        content = read_file(file)
        assert "api.pushover.net" not in content or file == "global_pushover.py"
```

## 📈 期待効果

1. **保守性向上**: 修正箇所が1箇所に集約
2. **信頼性向上**: 通知が確実に届く
3. **開発効率向上**: 新規実装時の迷いがなくなる
4. **テスト容易性**: モック化が簡単
5. **設定管理**: pushover.json一箇所で管理

## 🚀 実行計画

### 即座実行（今回）
1. features/extraction/commands/extract_character.py ✅ 完了
3. ドキュメント作成 ✅ 完了

### 次回実行
1. 全ファイルの一括移行スクリプト作成
2. 旧実装の削除
3. ユニットテスト追加

---
作成日: 2025-08-08
理由: QI-002/003/004でPushover通知が届かない問題の根本原因が実装の乱立と判明