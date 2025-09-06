# レガシープログラム・アーカイブリスト

**最終更新**: 2025-07-30  
**目的**: 混乱していたバッチプログラム群の整理と非推奨化管理

---

## 🏷️ プログラム分類と推奨状況

### ✅ **現在推奨（アクティブ）**

| プログラム | 用途 | 優先度 |
|-----------|------|--------|
| `features/extraction/commands/extract_character.py` | 日常的な抽出・新アーキテクチャ | **最優先** |
| `features/extraction/commands/quick_interactive.py` | 緊急時救済・100%成功率 | **条件付き** |
| `tools/core/run_auto_pipeline.py` | 大規模バッチ処理 | **特殊用途** |

### ⚠️ **データセット特化（保守モード）**

| プログラム | 用途 | 状態 |
|-----------|------|------|
| `tools/batch/kana08_enhanced_stable_batch.py` | kana08データセット専用 | **保守のみ** |
| `tools/batch/kana08_stable_batch_restored.py` | kana08復元版 | **非推奨** |

### ❌ **非推奨・アーカイブ対象**

#### 高優先度非推奨（即座削除・移動検討）
```bash
# 以下は deprecated/ ディレクトリに移動推奨
tools/batch/kana08_stable_batch_restored.py      # 復元版（enhanced版で置換済み）
temp/scripts/migration/run_batch_extraction.py   # 一時移行スクリプト
deprecated/tools_archive/                        # 既にアーカイブ済み
```

#### 中優先度非推奨（段階的移行）
```bash
# レガシー実装群
tools/automation/simple_batch_runner.py          # シンプル版（統合済み）
tools/automation/batched_extraction_runner.py   # 旧バッチシステム
temp/scripts/testing/yolo_threshold_comparison_test.py  # テスト用
```

---

## 🔄 移行計画・整理プロセス

### Phase 1: 即座実行（高優先度）
1. **重複ファイル削除**
   ```bash
   # 明らかに不要なファイル
   rm tools/batch/kana08_stable_batch_restored.py
   mv temp/scripts/migration/run_batch_extraction.py deprecated/
   ```

2. **CLAUDE.md の更新完了**
   - ✅ プログラム分類・使用指針の明確化完了
   - ✅ 推奨実行パターンの統一完了

### Phase 2: 段階的移行（中優先度）
3. **レガシーツール非推奨化**
   ```bash
   # deprecated/ に移動
   mkdir -p deprecated/legacy_batch_tools/
   mv tools/automation/simple_batch_runner.py deprecated/legacy_batch_tools/
   mv tools/automation/batched_extraction_runner.py deprecated/legacy_batch_tools/
   ```

4. **ドキュメント統合完了**
   - ✅ 重複ドキュメント統合完了
   - ✅ 設定変数一元管理完了

### Phase 3: 完全クリーンアップ（低優先度）
5. **temp/ ディレクトリ整理**
   - 実験用スクリプトの評価・アーカイブ
   - 不要ファイルの削除

---

## 📋 整理実施記録

### 2025-07-30 実施事項
- ✅ **プログラム分類完了**: CLAUDE.md に推奨使用パターン明記
- ✅ **設定統合完了**: `config/workspace_config.py` + `tools/config_manager.py`
- ✅ **ドキュメント統合**: 重複3ファイル → 統一1ファイル
- ✅ **自動化システム**: `tools/audit_path_compliance.py` + cron設定

### 削除・移動待ちファイル
```bash
# 即座削除推奨
tools/batch/kana08_stable_batch_restored.py    # enhanced版で置換済み

# deprecated/ 移動推奨  
temp/scripts/migration/run_batch_extraction.py  # 一時移行用
tools/automation/simple_batch_runner.py         # 機能統合済み
```

---

## 🎯 期待効果

### Before（整理前）
- ❌ 6つの似たようなバッチプログラムが混在
- ❌ どれを使うべきかユーザーが迷う状態
- ❌ 機能重複による保守負荷
- ❌ ドキュメントも重複・散在

### After（整理後）
- ✅ **明確な使用指針**: 用途別に4つのカテゴリで整理
- ✅ **推奨プログラム明確化**: `extract_character.py` を標準に
- ✅ **設定一元管理**: すべての設定が1箇所で完結
- ✅ **自動監視**: 日次監査で品質維持

---

## 📚 参考・関連ドキュメント

- **統一設定**: `docs/CONFIG_VARIABLES_REFERENCE.md`
- **統合ディレクトリ**: `docs/workflows/OUTPUT_DIRECTORY_UNIFIED.md`
- **プログラム使用指針**: `CLAUDE.md` のキャラクター抽出実行セクション
- **自動化システム**: `tools/audit_path_compliance.py`

---

**注意**: このアーカイブリストは整理プロセスの記録です。実際の削除・移動は段階的に実施し、重要機能への影響を慎重に確認してください。