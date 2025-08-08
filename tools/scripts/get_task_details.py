#!/usr/bin/env python3
"""
指定されたトラッカーIDの詳細情報を取得するスクリプト
"""

import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.config import get_default_config

def get_task_details(tracker_id: str):
    """指定されたトラッカーIDの詳細情報を取得"""
    try:
        # 設定取得
        config = get_default_config()
        
        # 進捗管理システム初期化
        manager = ProgressManager(config)
        
        # タスク詳細取得
        task = manager.client.get_task(tracker_id)
        
        if task is None:
            print(f"❌ タスク {tracker_id} が見つかりません")
            return
        
        print(f"📋 タスク詳細: {tracker_id}")
        print("=" * 60)
        print(f"📍 ステータス: {task.status.value}")
        print(f"📅 登録日付: {task.created_date}")
        print(f"🔄 更新日付: {task.updated_date}")
        print(f"📝 概要: {task.description}")
        print("\n🔧 コンポーネント状況:")
        print(f"  動作確認: {task.operation_check.value}")
        print(f"  テストUNIT: {task.unit_test.value}")
        print(f"  品質評価: {task.quality_evaluation.value}")
        print(f"  統合実行スクリプト: {task.integration_script.value}")
        print(f"  ダッシュボード生成: {task.dashboard_generation.value}")
        print(f"  抽出パイプライン: {task.extraction_pipeline.value}")
        
        if hasattr(task, 'metrics') and task.metrics:
            print("\n📊 品質メトリクス:")
            if task.metrics.lca is not None:
                print(f"  LCA: {task.metrics.lca}")
            if task.metrics.ab_evaluation_rate is not None:
                print(f"  A/B評価率: {task.metrics.ab_evaluation_rate}%")
            if task.metrics.fps is not None:
                print(f"  FPS: {task.metrics.fps}")
            if task.metrics.c_plus_rate is not None:
                print(f"  C以上評価率: {task.metrics.c_plus_rate}%")
            if task.metrics.avg_coverage_rate is not None:
                print(f"  平均カバレッジ率: {task.metrics.avg_coverage_rate}%")
            if task.metrics.avg_compactness is not None:
                print(f"  平均コンパクトネス: {task.metrics.avg_compactness}")
            if task.metrics.avg_fill_rate is not None:
                print(f"  平均フィル率: {task.metrics.avg_fill_rate}%")
            if task.metrics.sci is not None:
                print(f"  SCI: {task.metrics.sci}")
            if task.metrics.pla is not None:
                print(f"  PLA: {task.metrics.pla}")
            if task.metrics.ple is not None:
                print(f"  PLE: {task.metrics.ple}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        logging.error(f"タスク詳細取得エラー: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python get_task_details.py <TRACKER_ID>")
        print("例: python get_task_details.py P1-016")
        sys.exit(1)
    
    tracker_id = sys.argv[1]
    get_task_details(tracker_id)