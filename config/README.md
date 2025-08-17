# Workspace Config 一元管理システム

## 概要

全ワークスペースパス設定を一元管理し、設定変更時の影響範囲を最小化するconfig システムです。

## 主要ファイル

- `workspace_config.py` - メイン設定管理クラス
- `__init__.py` - パッケージ初期化

## 使用方法

### Python から使用

```python
from config.workspace_config import WorkspaceConfig

# ワークスペースベースパス取得
base_path = WorkspaceConfig.get_workspace_base()
# 結果: "/mnt/c/AItools/lora/train/yado/tracker-workspace"

# ワークスペースルートパス取得（/workspace サフィックス付き）
root_path = WorkspaceConfig.get_workspace_root()
# 結果: Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace")

# 特定トラッカーのワークスペースパス
tracker_path = WorkspaceConfig.get_tracker_workspace("P1-010")
# 結果: Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-010")
```

### シェルスクリプトから使用

```bash
# 環境変数取得
WORKSPACE_CONFIG_OUTPUT=$(python3 -c "
from config.workspace_config import WorkspaceConfig
env_vars = WorkspaceConfig.export_environment_variables()
for key, value in env_vars.items():
    print(f'{key}=\"{value}\"')
")

# 環境変数設定
eval "$WORKSPACE_CONFIG_OUTPUT"

# 使用
echo "Base: $TRACKER_WORKSPACE_BASE"
echo "Root: $TRACKER_WORKSPACE_ROOT"
```

## 環境変数での設定変更

```bash
# カスタムパス設定
export TRACKER_WORKSPACE_BASE="/custom/path/to/workspace"

# 設定確認
python3 config/workspace_config.py
```

## 統合済みファイル

以下のファイルがconfig システムに統合済み：

1. `features/common/output_path_manager.py`
2. `tools/utils/validate_tracker_completion.py`
3. `tools/scripts/run_quality_workflow.sh`

## パス変更方法

### 方法1: 環境変数（推奨）

```bash
export TRACKER_WORKSPACE_BASE="/new/workspace/path"
```

### 方法２: デフォルト値変更

`workspace_config.py` の `DEFAULT_WORKSPACE_BASE` を変更：

```python
DEFAULT_WORKSPACE_BASE = "/new/default/path"
```

## 設定確認

```bash
# 現在の設定確認
python3 config/workspace_config.py

# 妥当性検証
python3 -c "
from config.workspace_config import WorkspaceConfig
print('Valid:', WorkspaceConfig.validate_workspace_path())
"
```

## 既存ファイルへの影響

このconfig システム導入により、ハードコードされたパスがすべて動的取得に変更され、パス変更時の修正箇所が1か所（config ファイルまたは環境変数）に集約されました。

---

## Pushover通知設定（既存）

バッチ処理完了時にスマートフォンに通知を送るためのPushover設定です。

### 手順

1. **Pushoverアカウント作成**
   - https://pushover.net/ でアカウント作成
   - モバイルアプリをダウンロード

2. **アプリケーション作成**
   - Pushover ダッシュボードで新しいアプリケーションを作成
   - Application Token を取得

3. **設定ファイル作成**
   ```bash
   cd config
   cp pushover.json.example pushover.json
   ```

4. **設定ファイル編集**
   ```json
   {
     "token": "your_application_token_here",    # ← Application Token
     "user": "your_user_key_here",             # ← User Key  
     "device": "",                             # ← デバイス名（空白可）
     "title": "Character Extraction"           # ← 通知タイトル
   }
   ```