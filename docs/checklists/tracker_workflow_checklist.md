# トラッカーワークフロー完了チェックリスト

**作成日**: 2025-07-28  
**目的**: 4回目仕様違反の恒久対策として、すべてのトラッカータスクで必須確認事項を標準化

---

## 🚨 **重要な背景**

### 過去の問題パターン
- **1-3回目**: ワークスペース出力なし、品質チェック未実行、ダッシュボード未生成
- **4回目**: P1-005で同じ問題を繰り返し
- **根本原因**: プリフライト確認の欠如、仕様書参照の省略

---

## 📋 **Phase 0: タスク開始前必須確認**

### ✅ 仕様書確認（必須）
- [ ] `spec/OUTPUT_PATH_STANDARDS.md` 全文確認
- [ ] `docs/workflows/output_directory_config.md` 確認  
- [ ] `features/common/output_path_manager.py` API確認
- [ ] CLAUDE.md のトラッカーワークフロー要件確認

### ✅ 既存事例調査（必須）
- [ ] `/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/` 内の完了済みトラッカー確認
- [ ] P1-A001, P1-A002, P1-A003, P1-A004 の構造パターン調査
- [ ] 成功事例の実装方法参考

### ✅ ワークスペース要件確認（必須）
- [ ] トラッカーID決定（例: P1-005, PH2-001）
- [ ] `/workspace/{TRACKER_ID}/` 構造設計
- [ ] 必要ディレクトリ: extraction/, quality/, dashboard/, tests/
- [ ] OutputPathManager使用計画

---

## 📋 **Phase 1: 実装計画**

### ✅ アーキテクチャ設計
- [ ] 機能実装の詳細設計
- [ ] ワークスペース統合計画
- [ ] 品質ワークフロー統合計画

### ✅ 実装前準備
- [ ] ワークスペースディレクトリ作成: `mkdir -p /mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/{TRACKER_ID}/{extraction,quality,dashboard,tests}`
- [ ] OutputPathManager インスタンス化
- [ ] テスト計画策定

---

## 📋 **Phase 2: 実装実行**

### ✅ コア機能実装
- [ ] 主要機能の実装完了
- [ ] 単体テスト作成・実行
- [ ] 統合テスト作成・実行

### ✅ ワークスペース統合
- [ ] OutputPathManager を使用した出力先設定
- [ ] ワークスペース構造への適合確認
- [ ] ファイル出力の動作確認

---

## 📋 **Phase 3: 品質ワークフロー実行**

### ✅ 品質ワークフロー準備
- [ ] テスト画像の準備確認
- [ ] `tools/scripts/run_quality_workflow.sh` の動作確認
- [ ] 実行権限・パス確認

### ✅ 品質ワークフロー実行（必須）
- [ ] **実行コマンド**: `./tools/scripts/run_quality_workflow.sh {TRACKER_ID}`
- [ ] 抽出パイプライン実行完了
- [ ] 品質チェック3コマンド完了
- [ ] ダッシュボード生成完了

### ✅ 出力確認
- [ ] `workspace/{TRACKER_ID}/extraction/` にファイル存在
- [ ] `workspace/{TRACKER_ID}/quality/unified_quality_report.json` 存在
- [ ] `workspace/{TRACKER_ID}/dashboard/dashboard.html` 存在・アクセス可能
- [ ] `workspace/{TRACKER_ID}/tests/` にテスト結果存在

---

## 📋 **Phase 4: 完了検証**

### ✅ 自動検証実行（必須）
- [ ] **検証コマンド**: `python3 tools/utils/validate_tracker_completion.py {TRACKER_ID}`
- [ ] 総合ステータス: `COMPLETE` 確認
- [ ] 全チェック項目合格確認

### ✅ 手動確認
- [ ] ダッシュボードブラウザアクセス: `file:///workspace/{TRACKER_ID}/dashboard/dashboard.html`
- [ ] 品質レポートの内容確認
- [ ] 抽出結果の妥当性確認

### ✅ ドキュメント化
- [ ] 実装完了レポート作成
- [ ] 改善効果の記録
- [ ] 学習事項の記録

---

## 📋 **Phase 5: 最終確認・リリース**

### ✅ システム統合確認
- [ ] 既存システムとの互換性確認
- [ ] 依存関係の問題なし確認
- [ ] パフォーマンス影響確認

### ✅ 最終チェック
- [ ] **すべての必須要件満たす確認**:
  - ワークスペース完全構築
  - 品質ワークフロー実行済み
  - ダッシュボード生成済み
  - 自動検証合格
- [ ] Google Sheets更新（可能であれば）
- [ ] 完了通知送信

---

## ❌ **絶対回避すべきNGパターン**

### 🚫 実装のみで完了報告
```
❌ 機能実装完了 → 「タスク完了」報告
✅ 機能実装 → 品質ワークフロー → 検証 → 完了報告
```

### 🚫 ワークスペース要件無視
```
❌ プロジェクトルートへの直接出力
✅ /workspace/{TRACKER_ID}/ への標準出力
```

### 🚫 品質ワークフロー省略
```
❌ 「機能動作するから大丈夫」
✅ run_quality_workflow.sh 必須実行
```

### 🚫 仕様書確認省略
```
❌ 「前回と同じやり方で」
✅ 毎回仕様書確認から開始
```

---

## 🔧 **緊急時の対処法**

### ワークスペース未作成
```bash
mkdir -p /mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/{TRACKER_ID}/{extraction,quality,dashboard,tests}
```

### 品質ワークフロー失敗
```bash
# パス確認
ls tools/scripts/run_quality_workflow.sh
# 権限確認
chmod +x tools/scripts/run_quality_workflow.sh
# 再実行
./tools/scripts/run_quality_workflow.sh {TRACKER_ID}
```

### 検証失敗時
```bash
# 詳細確認
python3 tools/utils/validate_tracker_completion.py {TRACKER_ID}
# 不足要素の個別対応
```

---

## 📊 **成功指標**

### 短期目標
- [ ] このチェックリスト使用でのタスク100%完了
- [ ] 仕様違反ゼロ
- [ ] 品質ワークフロー実行率100%

### 中期目標
- [ ] 全トラッカータスクでの標準適用
- [ ] 自動化ツールの継続改善
- [ ] プロセス効率化

---

**注意**: このチェックリストは4回の仕様違反を受けて作成された実用的な文書です。**必ず全項目を確認してからタスク完了報告**を行ってください。