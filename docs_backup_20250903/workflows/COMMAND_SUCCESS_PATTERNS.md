# 成功コマンドパターン集

## 🎯 **必ず使用すべき実績あるコマンド**

### **キャラクター抽出（実績100%成功）**

#### ✅ Windows環境（推奨・実績あり）
```bash
# QC-SUCCESS-RESTORE, P1-B004で実証済み
MEMORY_LIMIT_DISABLED=true sam-env/Scripts/python.exe features/extraction/commands/extract_character.py \
  "C:/AItools/lora/train/yado/org/kana08/" \
  -o "C:/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction/" \
  --batch --verbose --max-files 5
```

#### ❌ WSL/Linux環境（避けるべき）
```bash
# 依存関係・パス問題で失敗率高
sam-env/bin/python3 features/extraction/commands/extract_character.py
```

### **Pushover通知（自動実行される箇所）**

#### ✅ 統合パイプライン内で自動実行
```bash
# tools/core/integrated_quality_pipeline.py
# - send_extraction_results_to_pushover() 
# - 抽出完了時に最大10枚自動送信
# - config/pushover.json必須
```

#### ✅ 品質チェック完了時
```bash
# tools/core/unified_quality_checker.py  
# - send_completion_notification()
# - 品質レポート+成功画像グリッド送信
```

### **完全ワークフロー（実績あり）**

```bash
# 1. Windows環境での抽出実行
MEMORY_LIMIT_DISABLED=true sam-env/Scripts/python.exe features/extraction/commands/extract_character.py \
  "C:/AItools/lora/train/yado/org/kana08/" \
  -o "C:/AItools/lora/train/yado/tracker-workspace/QI-002/extraction/" \
  --batch --verbose

# 2. 品質チェック3コマンド（Pushover自動実行）
python3 tools/core/unified_quality_checker.py \
  --results "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-002/extraction_result.json"

# 3. ダッシュボード生成
python3 tools/quality_dashboard.py \
  --report "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-002/quality/unified_quality_report.json" \
  --output "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-002/dashboard/"
```

## 🚫 **絶対に避けるべきパターン**

### ❌ 新しいコマンドの即興作成
- 既存成功コマンドがあるのに新規作成
- WSL環境での無理な実行
- 依存関係確認の省略

### ❌ 設定ファイル不備の放置
- config/pushover.json未確認
- パス設定の環境間不整合
- エラーログ無視

## ✅ **必須チェック事項**

### 実行前チェックリスト
- [ ] 既存成功コマンドの存在確認（CLAUDE.md参照）
- [ ] config/pushover.json存在・内容確認
- [ ] Windows/WSL環境の適切な選択
- [ ] MEMORY_LIMIT_DISABLED=true設定
- [ ] sam-env/Scripts/python.exe（Windows推奨）

### 実行後確認
- [ ] 抽出画像の実際の生成確認
- [ ] Pushover通知の受信確認  
- [ ] ダッシュボード生成確認
- [ ] エラーログの詳細確認