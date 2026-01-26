#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pathshim.py - Cross-platform path resolution system (Layer 1)

安全な実行時パス変換システム
- WSL/Windows/Linux間のパス自動変換
- 既存207ファイルに触れず安全に動作
- 無限再帰リスクなし
"""

import os
import re
from pathlib import Path
from typing import Union

# WSL形式パスの正規表現 (/mnt/c/... 形式)
_MNT_C = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")


def resolve(p: Union[str, Path]) -> Path:
    """
    WSL-style、Windows-style、相対パスを現在のOSに適したPathに変換

    Args:
        p: 変換対象パス（文字列またはPath）

    Returns:
        Path: 現在のOSで有効なPathオブジェクト

    Examples:
        # WSL環境で実行時
        resolve("/mnt/c/AItools/data/test.jpg") -> Path("/mnt/c/AItools/data/test.jpg")

        # Windows環境で実行時
        resolve("/mnt/c/AItools/data/test.jpg") -> Path("C:/AItools/data/test.jpg")

        # CI Linux環境で実行時（Layer 0のシンボリックリンクが有効）
        resolve("/mnt/c/AItools/data/test.jpg") -> Path("/mnt/c/AItools/data/test.jpg")
    """
    p = Path(p).expanduser()

    # ケース1: /mnt/c/foo/bar (WSL ハードコード形式)
    m = _MNT_C.match(str(p))
    if m:
        drive, tail = m.group(1).upper(), m.group(2) or "/"

        # Windows環境の場合、C:/foo/bar形式に変換
        if os.name == "nt":
            return Path(f"{drive}:{tail}")

        # Linux/WSL環境の場合、そのまま返す
        # (Layer 0のシンボリックリンクまたは実際のWSLマウントが有効)
        return p

    # ケース2: C:\foo\bar (生Windows形式)
    if os.name != "nt" and len(str(p)) > 2 and str(p)[1] == ":":
        # Linux環境でWindows絶対パスを受けた場合、WSL形式に変換
        drive = str(p)[0].lower()
        tail = str(p)[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{tail}")

    # ケース3: 相対パス、Unix絶対パス（変換不要）
    return p


def resolve_str(p: Union[str, Path]) -> str:
    """
    resolve()の文字列版

    Args:
        p: 変換対象パス

    Returns:
        str: 変換後パス文字列
    """
    return str(resolve(p))


def is_wsl_path(p: Union[str, Path]) -> bool:
    """
    WSL形式パス（/mnt/c/...）かどうか判定

    Args:
        p: 判定対象パス

    Returns:
        bool: WSL形式の場合True
    """
    return bool(_MNT_C.match(str(p)))


def is_windows_path(p: Union[str, Path]) -> bool:
    """
    Windows形式パス（C:\...）かどうか判定

    Args:
        p: 判定対象パス

    Returns:
        bool: Windows形式の場合True
    """
    p_str = str(p)
    return len(p_str) > 2 and p_str[1] == ":"


def get_environment_type() -> str:
    """
    現在の実行環境タイプを取得

    Returns:
        str: "windows", "wsl", "linux", "ci"
    """
    if os.getenv("CI_ENVIRONMENT") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
        return "ci"
    elif os.name == "nt":
        return "windows"
    elif os.path.exists("/mnt/c") and os.path.exists("/proc/version"):
        # /proc/versionでWSLかどうか確認
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    return "wsl"
        except Exception:
            pass

    return "linux"


def debug_path_info(p: Union[str, Path]) -> dict:
    """
    デバッグ用：パス変換情報を詳細表示

    Args:
        p: 対象パス

    Returns:
        dict: 変換情報辞書
    """
    original = str(p)
    resolved = resolve(p)

    return {
        "original": original,
        "resolved": str(resolved),
        "environment": get_environment_type(),
        "is_wsl_path": is_wsl_path(p),
        "is_windows_path": is_windows_path(p),
        "exists": resolved.exists(),
        "is_absolute": resolved.is_absolute(),
    }


# 使用例とテスト関数
if __name__ == "__main__":
    # テスト用パス（O3-Search 3層戦略検証用）
    test_paths = [
        "/mnt/c/AItools/lora/train/yado/org/kana08/",
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/",
        "C:\\AItools\\lora\\train\\yado\\org\\kana08\\",
        "./test_small/kana05_0001.jpg",
        "/tmp/test.jpg",
    ]

    print(f"🌍 Environment: {get_environment_type()}")
    print("=" * 60)

    for path in test_paths:
        info = debug_path_info(path)
        print(f"Original: {info['original']}")
        print(f"Resolved: {info['resolved']}")
        print(f"WSL: {info['is_wsl_path']}, Windows: {info['is_windows_path']}")
        print(f"Exists: {info['exists']}, Absolute: {info['is_absolute']}")
        print("-" * 40)
