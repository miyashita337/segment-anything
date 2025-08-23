# Claude for GitHub Actions 復旧ガイド

**作成日**: 2025-08-23  
**目的**: claude-for-github.ymlの安全な段階的復旧  
**前提条件**: ANTHROPIC_API_KEY設定済み

---

## 📋 復旧プロセス

### Phase 1: 手動テスト ✅ 完了
- **状況**: workflow_dispatchのみ有効化済み
- **テスト方法**: GitHub Actions → Manual trigger で実行
- **確認項目**: 
  - ✅ アクション実行成功
  - ✅ API認証成功  
  - ✅ エラーログなし

### Phase 2: Issue Commentテスト（次ステップ）
**実行前提**: Phase 1 手動テスト成功

1. **テンプレート適用**:
   ```bash
   cp .github/workflows/claude-for-github-phase2.yml.template .github/workflows/claude-for-github.yml
   ```

2. **テスト方法**: IssueにCommentを追加
3. **監視項目**:
   - API使用量の増加パターン
   - Response時間
   - エラー率

### Phase 3: フル復旧（最終ステップ）  
**実行前提**: Phase 2 Issue Comment テスト成功

1. **テンプレート適用**:
   ```bash
   cp .github/workflows/claude-for-github-phase3.yml.template .github/workflows/claude-for-github.yml
   ```

2. **全イベント有効化**:
   - Issues (opened, edited)
   - Issue Comments (created, edited)
   - Pull Requests (opened, edited, synchronize)
   - PR Review Comments (created, edited)

---

## 🚨 緊急時対処

### 予期しない大量実行発生時
```bash
# 即座に無効化
git checkout HEAD~1 -- .github/workflows/claude-for-github.yml
git commit -m "emergency: disable claude-for-github due to excessive API usage"
git push
```

### API制限到達時
```yaml
# 一時的にworkflow_dispatchのみに変更
on:
  workflow_dispatch: # 手動実行のみに制限
```

---

## 📊 監視ポイント

### API使用量
- **正常範囲**: 1-5 requests/hour
- **注意レベル**: 10+ requests/hour  
- **危険レベル**: 50+ requests/hour

### エラーパターン
```
- Authentication failed → ANTHROPIC_API_KEY確認
- Rate limit exceeded → 一時無効化必要
- Action not found → anthropics/claude-code-action@v1利用不可
```

### 成功指標
- ✅ GitHub Issues/PRへの適切な応答
- ✅ エラー率 < 5%
- ✅ API使用量が予測範囲内
- ✅ 24時間安定稼働

---

## 🔄 ロールバック手順

### Phase 2 → Phase 1
```bash
# 手動実行のみに戻す
on:
  workflow_dispatch: # Manual only
# issue_comment を削除
```

### 完全無効化
```bash
# 全体をコメントアウト
sed 's/^/# /' .github/workflows/claude-for-github.yml > temp && mv temp .github/workflows/claude-for-github.yml
```

---

**注意**: 各Phaseでの十分なテスト・監視を怠らず、問題発生時は即座に前のPhaseまたは無効化に戻すこと