#!/usr/bin/env python3
"""
トラッカーID置換実行スクリプト（LIVE RUN）
DRY RUN検証済みの安全な置換実行
"""

import json
import sys
from datetime import datetime
from tracker_replacement_engine import TrackerReplacementEngine


def main():
    """LIVE RUN実行"""
    print("=" * 60)
    print("フェーズ3.1: トラッカーID置換 - LIVE RUN実行")
    print("=" * 60)
    print("⚠️  注意: 実際にファイルとディレクトリを変更します")
    print("バックアップが自動作成されます...")
    print("=" * 60)

    try:
        # 置換エンジン初期化
        engine = TrackerReplacementEngine()

        # LIVE RUN実行
        live_results = engine.execute_replacement(dry_run=False)

        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"tools/analysis/replacement_live_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(live_results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 LIVE RUN結果保存: {results_file}")

        # 成功判定
        if live_results["verification"]["success"]:
            print("🎉 トラッカーID標準化完了!")
            print(f"   バックアップ場所: {live_results['backup_path']}")
            return True
        else:
            print("⚠️  置換実行完了、但し検証で警告あり")
            print("   詳細確認が必要です")
            return False

    except Exception as e:
        print(f"❌ LIVE RUN実行エラー: {e}")
        print("バックアップから復元してください")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
