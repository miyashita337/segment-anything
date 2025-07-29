# 出力パス設定標準化ガイドライン

**最終更新**: 2025-07-27  
**目的**: PH2-002問題の再発防止と統一的な出力パス管理  
**適用範囲**: 全新規機能および既存コードの改修

---

## 🚨 重要な背景

### PH2-002問題の教訓
PH2-002ダッシュボード生成時に発生した問題：
- **原因**: 仕様書記載のトラッカーID管理を無視し、相対パス `"dashboard_output"` を使用
- **影響**: 出力ファイルが間違った場所に配置され、管理困難になった
- **根本**: 既存仕様書の参照不足と、確立されたワークスペース構造の無視

### 仕様書準拠の重要性
```
正しい仕様: /workspace/{tracker_id}/dashboard/
間違った実装: ./dashboard_output/
```

---

## 📋 必須ルール

### 1. 🚫 禁止事項（絶対NG）

```python
# ❌ 相対パス（プロジェクトルート基準）
output_dir = Path("dashboard_output")
output_dir = Path("results")
output_dir = Path("temp")

# ❌ カレントディレクトリ基準
output_dir = Path("./output")
output_dir = Path("../results")

# ❌ ハードコードされた固定ディレクトリ名
self.output_dir = Path("dashboard_output")
results_dir = "results_batch"

# ❌ トラッカーID無視
output_path = "/some/fixed/path/output.html"
```

### 2. ✅ 推奨実装パターン

#### パターンA: OutputPathManager使用（最推奨）
```python
from features.common.output_path_manager import (
    OutputPathManager, 
    OutputCategory,
    ensure_compliant_output
)

# 基本使用
manager = OutputPathManager("PH2-002")
dashboard_dir = manager.ensure_output_dir(OutputCategory.DASHBOARD)
report_path = manager.get_output_path(
    OutputCategory.DASHBOARD, 
    filename="comprehensive_report.html"
)

# 簡易版
output_file = ensure_compliant_output(
    tracker_id="PH2-002",
    category=OutputCategory.DASHBOARD,
    filename="dashboard.html"
)
```

#### パターンB: 仕様準拠の直接実装（許容）
```python
# 仕様書準拠の最低限実装
WORKSPACE_BASE = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace")

def get_tracker_output_path(tracker_id: str, category: str, filename: str = None) -> Path:
    """仕様準拠の出力パス生成"""
    path = WORKSPACE_BASE / tracker_id / category
    if filename:
        path = path / filename
    return path

# 使用例
tracker_id = "PH2-002"  # 変数化必須
output_dir = get_tracker_output_path(tracker_id, "dashboard")
output_dir.mkdir(parents=True, exist_ok=True)
```

---

## 🗂 標準ディレクトリ構造

### ワークスペース基本構造
```
/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/
├── {tracker_id}/                 # 例: PH2-002, baseline
│   ├── dashboard/               # HTMLダッシュボード、可視化
│   ├── extraction/              # 抽出結果画像
│   ├── quality/                 # 品質レポート（JSON）
│   ├── tests/                   # テスト結果
│   └── temp/                    # 一時ファイル
├── baseline/                    # ベースライン結果
├── backup/                      # バックアップデータ
└── comparisons/                 # 比較分析結果
```

### OutputCategory対応表
| カテゴリ | ディレクトリ | 用途 |
|---------|------------|------|
| `DASHBOARD` | dashboard/ | HTMLダッシュボード、チャート |
| `EXTRACTION` | extraction/ | 抽出された画像ファイル |
| `QUALITY_REPORT` | quality/ | JSON品質レポート |
| `TEST_RESULT` | tests/ | 単体・統合テスト結果 |
| `TEMP` | temp/ | 一時ファイル（自動削除対象） |

---

## 🔧 実装ガイド

### 新機能開発時
1. **要件分析段階**:
   - トラッカーIDの決定（PH3-001等）
   - 出力カテゴリの選択
   - ファイル命名規則の決定

2. **設計段階**:
   ```python
   # クラス初期化時
   class MyDashboardGenerator:
       def __init__(self, tracker_id: str):
           self.tracker_id = tracker_id
           self.path_manager = OutputPathManager(tracker_id)
           self.output_dir = self.path_manager.ensure_output_dir(
               OutputCategory.DASHBOARD
           )
   ```

3. **実装段階**:
   ```python
   # ファイル出力時
   def save_report(self, data: Dict, filename: str):
       output_path = self.path_manager.get_output_path(
           OutputCategory.DASHBOARD,
           filename=filename
       )
       with open(output_path, 'w') as f:
           json.dump(data, f, indent=2)
   ```

### 既存コード改修時
1. **現状調査**: `tools/audit_path_compliance.py` で問題箇所特定
2. **影響範囲確認**: 出力ファイルを参照している他のコードの調査
3. **段階的移行**: 高優先度問題から順次対応

---

## 📊 品質保証

### 実装前チェック
- [ ] `docs/guidelines/SPECIFICATION_COMPLIANCE_CHECKLIST.md` の確認
- [ ] 既存の類似実装パターンの調査
- [ ] トラッカーID変数化の確認

### 実装後検証
```python
# 準拠性チェック
manager = OutputPathManager("your-tracker-id")
compliance = manager.validate_compliance()
assert compliance["compliant"], f"Issues: {compliance['issues']}"

# パス確認
expected_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace")
assert output_path.is_relative_to(expected_base), "Path not in workspace"
```

### 自動テスト統合
```python
def test_output_path_compliance():
    """出力パス準拠性テスト"""
    manager = OutputPathManager("TEST")
    
    # 正しいベースパス
    dashboard_path = manager.get_output_path(OutputCategory.DASHBOARD)
    assert "workspace/TEST/dashboard" in str(dashboard_path)
    
    # ディレクトリ作成確認
    test_dir = manager.ensure_output_dir(OutputCategory.TEMP)
    assert test_dir.exists()
```

---

## 🔍 監視・メンテナンス

### 定期監査
```bash
# 週次実行推奨
python3 tools/audit_path_compliance.py
```

### CI/CD統合
```yaml
# .github/workflows/quality-check.yml 例
- name: Path Compliance Check
  run: |
    python3 tools/audit_path_compliance.py
    if [ $? -gt 0 ]; then
      echo "❌ Path compliance issues detected"
      exit 1
    fi
```

### ログ監視
```python
# 本番環境での監視
import logging
logger = logging.getLogger(__name__)

def validate_output_location(output_path: Path):
    """出力先検証"""
    workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace")
    if not output_path.is_relative_to(workspace_base):
        logger.error(f"⚠️ Non-compliant output path: {output_path}")
        # アラート送信等
```

---

## 🚀 移行計画

### フェーズ1: 新規開発（即座適用）
- 全新機能でOutputPathManager必須使用
- コードレビューでの準拠性チェック強化

### フェーズ2: 高優先度修正（1週間以内）
```bash
# 高優先度176件の対応
python3 tools/audit_path_compliance.py | grep "🚨 High"
```

### フェーズ3: 全体最適化（1ヶ月以内）
- 既存全コードの段階的移行
- 自動化ツールの整備
- ドキュメント体系化

---

## 📚 参考資料

### 必須参照ドキュメント
- `docs/workflows/output_directory_config.md` - 公式仕様
- `docs/guidelines/SPECIFICATION_COMPLIANCE_CHECKLIST.md` - チェックリスト
- `features/common/output_path_manager.py` - 実装リファレンス

### 実装例
- `workspace/PH2-001/` - 正しい構造例
- `workspace/PH2-002/` - 修正後の構造例

### トラブルシューティング
- **権限エラー**: `sudo chown -R $USER workspace/`
- **パス長制限**: Windows環境での260文字制限に注意
- **並列処理**: 複数プロセスからの同時ディレクトリ作成競合

---

## ✅ 成功指標

### 短期目標（1週間）
- [ ] 新規開発での100%準拠
- [ ] 高優先度問題（176件）の50%以上解決
- [ ] CI/CDチェック統合

### 中期目標（1ヶ月）
- [ ] 全高優先度問題の解決
- [ ] 既存コードの80%以上改修
- [ ] 監査ツールの定期実行確立

### 長期目標（3ヶ月）
- [ ] プロジェクト全体の100%準拠
- [ ] 自動修正ツールの開発
- [ ] 他プロジェクトへの標準展開

---

**注意**: このガイドラインは実際の問題（PH2-002）に基づいて作成された実用的な文書です。継続的な改善と実際の使用体験に基づく更新を行ってください。