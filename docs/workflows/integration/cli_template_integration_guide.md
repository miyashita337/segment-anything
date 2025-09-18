# CLI・テンプレート統合ガイド

## 🎯 **統合の目的**

従来の手動チェックリストと新しい機械的強制実行システムを統合し、
**人間の判断** + **機械的保証** のハイブリッドワークフローを実現する。

---

## 🏗️ **統合アーキテクチャ**

### **3層構造の設計**

```
┌─────────────────────────────────────────┐
│ Layer 1: 人間判断レイヤー                │
│ - 統合テンプレート (unified_tracker_template.md) │
│ - 計画・承認・例外処理                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Layer 2: 機械的強制レイヤー               │
│ - CLI ツール (workflow_cli.py)           │
│ - 状態管理・検証・承認システム            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Layer 3: 詳細手順レイヤー                │
│ - 13ステップチェックリスト                │
│ - 具体的コマンド・検証手順                │
└─────────────────────────────────────────┘
```

---

## 🔄 **統合ワークフロー**

### **Phase 0: 初期化**
```bash
# 1. 人間判断: トラッカー要件確認
# 2. 機械的実行: ワークフロー作成
python tools/workflow/workflow_cli.py create TRACKER-001

# 3. 状態確認
python tools/workflow/workflow_cli.py status TRACKER-001
```

### **Phase 1: 計画フェーズ**
```bash
# 1. 人間判断: 統合テンプレート使用
#    - SOW作成
#    - 計画書作成
#    - リスク分析

# 2. 機械的検証: ブランチ確認
python tools/workflow/workflow_cli.py step TRACKER-001
# → branch_verification ステップ実行

# 3. 詳細手順: チェックリスト確認
#    - Phase 0.5 ブランチ検証完了確認
#    - sam-env 環境確認
```

### **Phase 2: 実装フェーズ**
```bash
# 1. 人間判断: 実装方針決定
#    - 技術アーキテクチャ
#    - 実装手順

# 2. 機械的強制: 承認システム
python tools/workflow/workflow_cli.py step TRACKER-001
# → sow_creation で承認要求発生

# 3. 承認処理: ファイルベース承認
# 承認ファイル作成後、進行再開
```

---

## 📋 **具体的統合手順**

### **1. トラッカー開始時**

**人間の作業:**
```markdown
# unified_tracker_template.md を使用
- [ ] トラッカーID決定: TRACKER-001
- [ ] 要件分析完了
- [ ] SOW作成完了
```

**機械的実行:**
```bash
# ワークフロー初期化
python tools/workflow/workflow_cli.py create TRACKER-001

# 現在状態確認
python tools/workflow/workflow_cli.py status TRACKER-001
```

### **2. 各ステップ実行時**

**機械的指示取得:**
```bash
# 現在のステップ指示を取得
python tools/workflow/workflow_cli.py instructions TRACKER-001
```

**人間の判断:**
```markdown
# 取得した指示に基づいて作業実行
# 例: Git Branch Verification
- [ ] 現在ブランチ確認: git branch --show-current
- [ ] feature/TRACKER-001 ブランチ作成
```

**機械的進行:**
```bash
# ステップ完了試行
python tools/workflow/workflow_cli.py step TRACKER-001
```

### **3. 承認が必要な場合**

**承認要求表示:**
```
🚨 CRITICAL: APPROVAL REQUIRED - ALL PROGRESS BLOCKED
📋 Approval ID: TRACKER-001_sow_creation_1736176088
🎯 Step: Statement of Work Creation
```

**人間の承認作業:**
```bash
# 承認ファイル作成
echo '{
  "approved": true,
  "approved_by": "ユーザー名",
  "approved_at": "2025-01-06T14:21:28.825734",
  "comments": "SOW内容を確認し承認します",
  "evidence": "要件・スコープ・成果物が明確"
}' > .workflow_approvals/TRACKER-001_sow_creation_1736176088_approved.json
```

**進行再開:**
```bash
# 承認後、自動的に次ステップへ進行
python tools/workflow/workflow_cli.py step TRACKER-001
```

---

## 🎛️ **CLI コマンド活用法**

### **日常的な状態確認**
```bash
# 現在の作業状況確認
python tools/workflow/workflow_cli.py status TRACKER-001

# 承認待ち一覧確認
python tools/workflow/workflow_cli.py approvals

# 次に何をすべきか確認
python tools/workflow/workflow_cli.py instructions TRACKER-001
```

### **進行管理**
```bash
# ステップ実行
python tools/workflow/workflow_cli.py step TRACKER-001

# バックグラウンドプロセス確認
python tools/workflow/workflow_cli.py process TRACKER-001
```

---

## 📊 **統合の利点**

### **人間の負担軽減**
- ✅ 何をすべきかが明確に表示される
- ✅ 手順スキップが物理的に不可能
- ✅ 承認が必要なタイミングが自動判定

### **品質保証の向上**
- ✅ 機械的検証による確実性
- ✅ 外部状態管理による改ざん防止
- ✅ 段階的承認による安全性

### **作業効率の向上**
- ✅ 現在状況の即座確認
- ✅ 次アクションの明確指示
- ✅ 日本語表示による理解しやすさ

---

## 🔧 **実装例: 統合ワークフロー**

### **完全統合スクリプト例**
```bash
#!/bin/bash
# integrated_workflow.sh - 統合ワークフロー実行スクリプト

TRACKER_ID="$1"

echo "🚀 統合ワークフロー開始: $TRACKER_ID"

# 1. ワークフロー作成
echo "📋 ワークフロー初期化中..."
python tools/workflow/workflow_cli.py create "$TRACKER_ID"

# 2. 状態確認ループ
while true; do
    echo "📊 現在の状態確認..."
    python tools/workflow/workflow_cli.py status "$TRACKER_ID"
    
    echo "📝 次のステップ指示:"
    python tools/workflow/workflow_cli.py instructions "$TRACKER_ID"
    
    echo "🔄 ステップ実行を試行しますか? (y/n)"
    read -r response
    
    if [[ "$response" == "y" ]]; then
        python tools/workflow/workflow_cli.py step "$TRACKER_ID"
    else
        echo "⏸️ 作業を一時停止します"
        break
    fi
    
    # 承認待ちチェック
    echo "⏳ 承認待ち確認..."
    python tools/workflow/workflow_cli.py approvals
    
    echo "続行しますか? (y/n)"
    read -r continue_response
    [[ "$continue_response" != "y" ]] && break
done

echo "✅ 統合ワークフロー完了"
```

---

## 📚 **関連ドキュメント更新**

### **統合テンプレート更新**
```markdown
# unified_tracker_template.md に追加

## 🤖 CLI統合セクション

### ワークフロー初期化
```bash
python tools/workflow/workflow_cli.py create {TRACKER_ID}
```

### 各フェーズでの確認
```bash
# 現在状況確認
python tools/workflow/workflow_cli.py status {TRACKER_ID}

# 次ステップ指示取得
python tools/workflow/workflow_cli.py instructions {TRACKER_ID}
```
```

### **チェックリスト更新**
```markdown
# tracker_workflow_checklist.md に追加

## 🤖 機械的検証項目

### Phase 0.5: ブランチ検証
- [ ] CLI実行: `python tools/workflow/workflow_cli.py step TRACKER-001`
- [ ] 検証結果: ✅ branch_verification 完了

### 各ステップ
- [ ] 指示確認: `python tools/workflow/workflow_cli.py instructions TRACKER-001`
- [ ] ステップ実行: `python tools/workflow/workflow_cli.py step TRACKER-001`
- [ ] 状態確認: `python tools/workflow/workflow_cli.py status TRACKER-001`
```

---

この統合により、**人間の創造性・判断力** と **機械の確実性・一貫性** を組み合わせた
最適なワークフローが実現されます！ 🎉