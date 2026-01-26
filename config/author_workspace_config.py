#!/usr/bin/env python3
"""
QUAL-036: 作者別ワークスペース設定システム
作者検出結果を他スクリプトと共有するための共通設定モジュール

Usage:
    # 設定
    from config.author_workspace_config import set_current_author, get_current_workspace
    set_current_author("yado", "/mnt/c/AItools/lora/train/yado/org/kana04/")
    
    # 取得
    workspace = get_current_workspace()
    author = get_current_author()
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 設定ファイルパス
CONFIG_FILE = Path(__file__).parent / "current_author_config.json"

# グローバル作者情報（メモリキャッシュ）
_current_author_info: Optional[Dict[str, Any]] = None


def set_current_author(author_name: str, input_path: str, tracker_id: Optional[str] = None) -> None:
    """
    現在の作者情報を設定

    Args:
        author_name: 作者名（yado, kiri, zundamon）
        input_path: 入力パス
        tracker_id: 現在のトラッカーID（オプション）
    """
    global _current_author_info

    author_info = {
        "author_name": author_name,
        "input_path": input_path,
        "tracker_id": tracker_id,
        "workspace_path": f"/mnt/c/AItools/lora/train/{author_name}/tracker-workspace",
        "timestamp": datetime.now().isoformat(),
        "set_by": "QUAL-036",
    }

    # メモリキャッシュ更新
    _current_author_info = author_info

    # 環境変数に設定（他プロセス連携）
    os.environ["CURRENT_AUTHOR"] = author_name
    os.environ["CURRENT_WORKSPACE"] = author_info["workspace_path"]
    os.environ["CURRENT_INPUT_PATH"] = input_path

    if tracker_id:
        os.environ["CURRENT_TRACKER_ID"] = tracker_id

    # 設定ファイルに永続化
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(author_info, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 作者設定保存完了: {author_name} → {CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"⚠️ 設定ファイル保存失敗: {e}")


def get_current_author() -> Optional[str]:
    """現在の作者名を取得"""
    author_info = _get_author_info()
    return author_info.get("author_name") if author_info else None


def get_current_workspace() -> Optional[str]:
    """現在のワークスペースパスを取得"""
    author_info = _get_author_info()
    return author_info.get("workspace_path") if author_info else None


def get_current_input_path() -> Optional[str]:
    """現在の入力パスを取得"""
    author_info = _get_author_info()
    return author_info.get("input_path") if author_info else None


def get_current_tracker_id() -> Optional[str]:
    """現在のトラッカーIDを取得"""
    author_info = _get_author_info()
    return author_info.get("tracker_id") if author_info else None


def get_all_author_info() -> Optional[Dict[str, Any]]:
    """全作者情報を取得"""
    return _get_author_info()


def _get_author_info() -> Optional[Dict[str, Any]]:
    """作者情報取得（キャッシュ・環境変数・ファイルの順で試行）"""
    global _current_author_info

    # 1. メモリキャッシュから取得
    if _current_author_info:
        return _current_author_info

    # 2. 環境変数から取得
    if "CURRENT_AUTHOR" in os.environ:
        author_info = {
            "author_name": os.environ.get("CURRENT_AUTHOR"),
            "workspace_path": os.environ.get("CURRENT_WORKSPACE"),
            "input_path": os.environ.get("CURRENT_INPUT_PATH"),
            "tracker_id": os.environ.get("CURRENT_TRACKER_ID"),
            "source": "environment_variables",
        }
        _current_author_info = author_info
        return author_info

    # 3. 設定ファイルから取得
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                author_info = json.load(f)
                _current_author_info = author_info
                return author_info
        except Exception as e:
            logger.warning(f"⚠️ 設定ファイル読み込み失敗: {e}")

    return None


def clear_current_author() -> None:
    """作者設定をクリア"""
    global _current_author_info

    # メモリキャッシュクリア
    _current_author_info = None

    # 環境変数クリア
    for env_var in [
        "CURRENT_AUTHOR",
        "CURRENT_WORKSPACE",
        "CURRENT_INPUT_PATH",
        "CURRENT_TRACKER_ID",
    ]:
        if env_var in os.environ:
            del os.environ[env_var]

    # 設定ファイル削除
    if CONFIG_FILE.exists():
        try:
            CONFIG_FILE.unlink()
            logger.info("✅ 作者設定クリア完了")
        except Exception as e:
            logger.warning(f"⚠️ 設定ファイル削除失敗: {e}")


def export_for_shell() -> str:
    """シェルスクリプト用export文を生成"""
    author_info = _get_author_info()
    if not author_info:
        return "# No author configuration found"

    exports = []
    if author_info.get("author_name"):
        exports.append(f'export CURRENT_AUTHOR="{author_info["author_name"]}"')
    if author_info.get("workspace_path"):
        exports.append(f'export CURRENT_WORKSPACE="{author_info["workspace_path"]}"')
    if author_info.get("input_path"):
        exports.append(f'export CURRENT_INPUT_PATH="{author_info["input_path"]}"')
    if author_info.get("tracker_id"):
        exports.append(f'export CURRENT_TRACKER_ID="{author_info["tracker_id"]}"')

    return "\n".join(exports)


def get_workspace_for_tracker(tracker_id: str) -> str:
    """
    トラッカー用ワークスペースパスを生成

    Args:
        tracker_id: トラッカーID

    Returns:
        完全なワークスペースパス
    """
    base_workspace = get_current_workspace()
    if not base_workspace:
        # フォールバック
        base_workspace = "/mnt/c/AItools/lora/train/yado/tracker-workspace"

    return f"{base_workspace}/{tracker_id}"


# 初期化時に環境変数・ファイルから自動読み込み
def _auto_initialize():
    """自動初期化（インポート時実行）"""
    try:
        author_info = _get_author_info()
        if author_info:
            logger.debug(f"📋 作者設定自動読み込み: {author_info.get('author_name')}")
    except Exception as e:
        logger.debug(f"⚠️ 自動初期化エラー（無視可能）: {e}")


# インポート時に自動初期化実行
_auto_initialize()

if __name__ == "__main__":
    # テスト・デモ実行
    logging.basicConfig(level=logging.INFO)

    print("🧪 QUAL-036 作者別ワークスペース設定システム テスト")
    print("=" * 60)

    # テスト1: 作者設定
    print("\n1. 作者設定テスト")
    set_current_author("yado", "/mnt/c/AItools/lora/train/yado/org/kana04/", "QUAL-036")

    # テスト2: 情報取得
    print(f"   作者名: {get_current_author()}")
    print(f"   ワークスペース: {get_current_workspace()}")
    print(f"   入力パス: {get_current_input_path()}")
    print(f"   トラッカーID: {get_current_tracker_id()}")

    # テスト3: Shell export
    print(f"\n2. Shell Export:")
    print(export_for_shell())

    # テスト4: トラッカー用パス
    print(f"\n3. トラッカー用パス:")
    print(f"   QUAL-036: {get_workspace_for_tracker('QUAL-036')}")

    print("\n✅ テスト完了")
