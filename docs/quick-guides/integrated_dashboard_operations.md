# 統合ダッシュボード運用クイックガイド

**最終更新**: 2025-09-15
**目的**: トラッカーワークフロー完了時の統合ダッシュボード必須手順

---

## 🚨 **必須3ステップ (1分で完了)**

### **Step 1: index.html作成**
```bash
# 統合サーバー認識用ファイル作成
cp /path/to/tracker-workspace/TRACKER-ID/dashboard/dashboard.html \
   /path/to/tracker-workspace/TRACKER-ID/index.html
```

### **Step 2: サーバー再スキャン**
```bash
# ダッシュボード再スキャン実行
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/refresh
```
**期待結果**: `{"status": "success", "message": "XX dashboards rescanned"}`

### **Step 3: 認識確認**
```bash
# トラッカー認識確認
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/tracker/TRACKER-ID
```
**成功**: HTML統合ダッシュボードが返る
**失敗**: `❌ ダッシュボードが見つかりません`

---

## 🔧 **失敗時対処法**

**問題**: Step 3でエラーが返る場合
1. **ワークスペース構造確認**: `ls -la /path/to/tracker-workspace/TRACKER-ID/`
2. **index.html存在確認**: Step 1を再実行
3. **サーバー状況確認**: `ps aux | grep integrated_dashboard_server`

---

## 📋 **チェックリスト統合要件**

**ステップ9B**: 🔴 **完了報告前curl必須実行**
すべての完了報告前に上記Step 3実行し、要件満たされていることを検証すること

**重要**: このチェックをスキップした場合、品質ワークフロー未完了とみなす