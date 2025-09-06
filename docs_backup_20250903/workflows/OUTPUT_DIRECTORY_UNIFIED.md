# 出力ディレクトリ統一管理ガイド

**最終更新**: 2025-08-08  
**バージョン**: 3.0（ダッシュボード統合版）  
**旧バージョンからの統合**: `OUTPUT_PATH_STANDARDS.md` + `output_directory_config.md`

---

## 🎯 目的と統合の背景

このドキュメントは以下を統合した統一リファレンスです：
- `/spec/OUTPUT_PATH_STANDARDS.md` - パス標準化ガイドライン
- `/docs/workflows/output_directory_config.md` - ディレクトリ設定

### 統合により解決した問題
1. **重複ドキュメントの混乱**: 3つの類似ファイルを1つに統合
2. **設定の散在**: 環境変数とPython設定の一元管理実現
3. **手動更新の負荷**: 自動生成システムによる保守性向上

---

## 🗂 標準ディレクトリ構造

### ワークスペース基本構造
```
${TRACKER_WORKSPACE_BASE}/workspace/
├── {tracker_id}/                 # 例: P1-011, PH2-002, baseline
│   ├── dashboard/               # HTMLダッシュボード、可視化（詳細: dashboard_management_guide.md）
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
| `DASHBOARD` | dashboard/ | HTMLダッシュボード、チャート（**自動的に http://100.123.241.106:8088/tracker/ に統合表示**） |
| `EXTRACTION` | extraction/ | 抽出された画像ファイル |
| `QUALITY_REPORT` | quality/ | JSON品質レポート |
| `TEST_RESULT` | tests/ | 単体・統合テスト結果 |
| `TEMP` | temp/ | 一時ファイル（自動削除対象） |

### 🌐 ダッシュボード統合システム（2025-08-08追加）

**重要**: 全トラッカーのダッシュボードは以下で統一管理されます：

- **統合URL**: http://100.123.241.106:8088/tracker/{TRACKER_ID}
- **Basic認証**: admin / dashboard2025!
- **自動認識**: `dashboard/` ディレクトリに生成されたHTMLファイルは自動的に8088サーバーで認識
- **詳細**: [`dashboard_management_unified.md`](./dashboard_management_unified.md) を参照

---

## ⚙️ 統一設定管理システム

### 🔧 設定変数の一元管理

すべての設定変数は `config/workspace_config.py` で管理され、以下で参照可能：

```bash
# 設定変数一覧表示
python3 tools/config_manager.py --show-vars

# 設定妥当性検証
python3 tools/config_manager.py --validate

# 環境変数スクリプト生成
python3 tools/config_manager.py --generate-env bash > bin/shell/set_env_vars.sh
```

**📋 利用可能変数**: 詳細は `docs/CONFIG_VARIABLES_REFERENCE.md` を参照

### 🐍 Python実装パターン

#### パターンA: OutputPathManager使用（最推奨）
```python
from features.common.output_path_manager import (
    OutputPathManager, 
    OutputCategory,
    ensure_compliant_output
)

# 基本使用
manager = OutputPathManager("P1-011")
dashboard_dir = manager.ensure_output_dir(OutputCategory.DASHBOARD)
report_path = manager.get_output_path(
    OutputCategory.DASHBOARD, 
    filename="comprehensive_report.html"
)
```

#### パターンB: 設定統合システム使用
```python
# config/workspace_config.py の WorkspaceConfig を使用
from config.workspace_config import WorkspaceConfig

def get_tracker_output_path(tracker_id: str, category: str, filename: str = None) -> Path:
    """仕様準拠の出力パス生成"""
    workspace_root = WorkspaceConfig.get_tracker_workspace(tracker_id)
    path = workspace_root / category
    if filename:
        path = path / filename
    return path
```

---

## 🚫 禁止事項（重要）

### ❌ 絶対に使用禁止
```python
# 相対パス（プロジェクトルート基準）
output_dir = Path("dashboard_output")
output_dir = Path("results")

# カレントディレクトリ基準
output_dir = Path("./output")

# ハードコードされた固定パス
self.output_dir = Path("dashboard_output")
output_path = "/some/fixed/path/output.html"
```

### ✅ 正しい実装
```python
# 統一設定システム使用
from config.workspace_config import WorkspaceConfig

tracker_id = "P1-011"  # 変数化必須
output_dir = WorkspaceConfig.get_tracker_workspace(tracker_id) / "dashboard"
output_dir.mkdir(parents=True, exist_ok=True)
```

---

## 🔍 監視・自動化システム

### 日次自動監査（予定）
```bash
# 20:00に自動実行・Pushover通知
python3 tools/audit_path_compliance.py
```

### 手動検証コマンド
```bash
# 設定全体の妥当性検証
python3 tools/config_manager.py --validate

# パス準拠性チェック
manager = OutputPathManager("TEST")
compliance = manager.validate_compliance()
assert compliance["compliant"], f"Issues: {compliance['issues']}"
```

---

## 📚 移行・統合ガイド

### 既存コード移行手順
1. **現状調査**: `tools/config_manager.py --validate` で問題特定
2. **段階的移行**: ハードコードパス → 設定変数使用
3. **検証**: 移行後の動作確認

### 統合完了事項
- ✅ 重複ドキュメント統合（3→1ファイル）
- ✅ 設定変数一元管理システム構築
- ✅ 自動生成ツール実装
- ✅ 環境変数スクリプト自動生成

### 実装予定事項
- 🔄 日次監査システム（cron + Pushover）
- 🔄 CI/CD統合
- 🔄 Markdownファイル変数自動展開

---

## 🔗 関連リソース

### 必須参照ドキュメント
- **設定変数**: `docs/CONFIG_VARIABLES_REFERENCE.md`
- **実装リファレンス**: `features/common/output_path_manager.py`
- **統合管理ツール**: `tools/config_manager.py`

### 自動化スクリプト
- **環境変数設定**: `bin/shell/set_env_vars.sh`（自動生成）
- **設定検証**: `python3 tools/config_manager.py --validate`

### Google Sheets連携
- **移行計画**: フェーズ1-3の詳細管理はGoogle Sheets進捗トラッカーで実施
- **進捗確認**: `python3 tools/progress_tracker/cli.py status`

---

**注意**: このドキュメントは統合版です。旧バージョンのドキュメントは参照しないでください。最新情報は自動生成システムで管理されています。