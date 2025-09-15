# ハイブリッド検証システム使用ガイド

## 概要

ハイブリッド検証システムは、厳格な品質保証と柔軟性の両方を実現するため、以下の2段階検証を組み合わせます：

1. **第1段階（必須）**: `validate_workflow_compliance.sh` による厳格検証
2. **第2段階（補完）**: `semantic_workflow_compliance.sh` によるセマンティック検証

## 🎯 実現される価値

| 機能 | validate_workflow | semantic_workflow | hybrid_system |
|---|---|---|---|
| **品質保証** | ✅ 高い | ⚠️ やや低い | ✅ 高い |
| **エラー防止** | ✅ 確実 | ⚠️ 不確実 | ✅ 確実 |
| **柔軟性** | ❌ 低い | ✅ 高い | ✅ 高い |
| **詳細分析** | ❌ 基本的 | ✅ 詳細 | ✅ 詳細 |

## 🔧 使用方法

### 1. 環境変数による制御

```bash
# 厳格検証のみ（本番推奨）
export VALIDATION_MODE=strict

# セマンティック検証のみ（開発・実験用）
export VALIDATION_MODE=semantic

# ハイブリッド検証（デフォルト・推奨）
export VALIDATION_MODE=hybrid
# または環境変数未設定
```

### 2. Hook設定

```json
{
  "PreToolUse": {
    "command": "bash tools/hooks/hybrid_workflow_compliance.sh",
    "match": {
      "tool": "Bash",
      "args.command": "*extract_character.py*|*progress_tracker/cli.py*|*run_quality_workflow.sh*|*python*tracker*"
    },
    "description": "INTG-087: ハイブリッド検証システム"
  }
}
```

### 3. 検証フロー

#### strictモード
```
入力 → 厳格検証 → 成功/失敗
```

#### semanticモード  
```
入力 → セマンティック検証 → 成功/失敗
```

#### hybridモード（推奨）
```
入力 → 厳格検証 → 失敗時即停止
      ↓ 成功時
      セマンティック検証 → 警告のみ（処理継続）
      ↓
      総合判定（詳細情報付き）
```

## 📊 検証結果の形式

### strictモード結果
```json
{
  "allow": true,
  "message": "ワークフロー検証成功"
}
```

### hybridモード結果
```json
{
  "allow": true,
  "mode": "hybrid",
  "validation_results": {
    "strict": "passed",
    "semantic": "passed"
  },
  "message": "ハイブリッド検証成功（厳格検証合格）",
  "details": [
    "strict_validation: passed",
    "semantic_validation: passed"
  ],
  "recommendations": [
    "💡 セマンティック分析: 追加の洞察を提供"
  ]
}
```

## 🚀 推奨運用方針

### 本番環境
```bash
export VALIDATION_MODE=strict
```
- 最高の品質保証
- 高速処理
- 確実性重視

### 開発環境
```bash
export VALIDATION_MODE=hybrid
```
- 品質保証 + 詳細分析
- 柔軟性と安全性の両立
- 段階的改善情報の提供

### 実験環境
```bash
export VALIDATION_MODE=semantic
```
- 新機能の検証
- ドキュメント変更への適応テスト
- AI支援ワークフローの評価

## ⚠️ 重要な注意点

### 品質保証の優先順位
1. **厳格検証の失敗 = 即座にブロック**（品質優先）
2. **セマンティック検証の失敗 = 警告のみ**（情報提供）

### エラー処理
- 厳格検証エラー: `exit 1`（処理停止）
- セマンティック検証エラー: 警告ログ（処理継続）
- システムエラー: クリーンアップ後終了

### ログファイル
- 個別ログ: `{workspace}/.workflow/logs/hybrid_compliance.log`
- デフォルト: `/tmp/hybrid_workflow_compliance.log`

## 🔍 トラブルシューティング

### よくある問題

#### 1. 厳格検証の失敗
```bash
# エラー例
❌ 厳格検証失敗 - 実行をブロック

# 対処方法
1. validate_workflow_compliance.shの要件を確認
2. ブランチ、入力パス、フェーズ状態をチェック
3. チェックリストファイルの状態確認
```

#### 2. セマンティック検証の警告
```bash
# 警告例
⚠️ セマンティック検証で警告検出（ブロックしない）

# 対処方法
1. 警告内容の確認（recommendations参照）
2. 必要に応じて手動確認
3. 継続可能（品質は厳格検証で保証済み）
```

#### 3. 権限エラー
```bash
# スクリプト実行権限の確認
chmod +x tools/hooks/hybrid_workflow_compliance.sh
```

## 📈 今後の拡張計画

### Phase 1: 基本機能（✅完了）
- 3モード対応（strict/semantic/hybrid）
- 2段階検証システム
- 詳細ログ機能

### Phase 2: 機能拡張（予定）
- モード別設定ファイル
- 検証結果のメトリクス収集
- 自動モード切り替え

### Phase 3: AI統合（構想）
- 検証結果の学習機能
- 適応的品質基準調整
- インテリジェントな推奨事項

## 📋 関連ファイル

- **メインスクリプト**: `tools/hooks/hybrid_workflow_compliance.sh`
- **厳格検証**: `tools/hooks/validate_workflow_compliance.sh`
- **セマンティック検証**: `tools/hooks/semantic_workflow_compliance.sh`
- **Hook設定**: `.claude/hooks.json`
- **設定ファイル**: `config/execution_rules.yaml`

---

**ハイブリッド検証システムにより、品質を保ちながら柔軟性も実現できます。**