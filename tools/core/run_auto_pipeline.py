#!/usr/bin/env python3
"""
完全自動パイプライン実行スクリプト
CLAUDE.md準拠の最終自動化プログラム
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from core.automation.auto_pipeline import get_auto_pipeline


def main():
    """メイン実行"""
    print("🚀 完全自動パイプライン開始")
    print("CLAUDE.md準拠のクリーンアーキテクチャ構築完了")
    print("=" * 60)

    # パイプライン取得・実行
    pipeline = get_auto_pipeline()
    pipeline.run_full_pipeline()

    print("🎉 完全自動パイプライン完了")


if __name__ == "__main__":
    main()
