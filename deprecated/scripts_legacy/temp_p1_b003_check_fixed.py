#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P1-B003タスク詳細確認スクリプト（修正版）
Google Sheetsから直接P1-B003の情報を取得
"""

import os
import sys
import json
from pathlib import Path

# プロジェクトパスを設定
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tools.progress_tracker.sheets_client import GoogleSheetsClient
    from tools.progress_tracker.data_models import TaskStatus
    from tools.progress_tracker.config import get_default_config
    
    def get_p1_b003_details():
        """P1-B003の詳細情報を取得"""
        
        # 環境変数設定
        os.environ["PROGRESS_TRACKER_SHEET_NAME"] = "シート1"
        
        try:
            # 設定取得
            config = get_default_config()
            
            # Google Sheetsクライアント初期化
            client = GoogleSheetsClient(config)
            print("Google Sheets接続成功")
            
            # 全タスク取得
            tasks = client.get_all_tasks()
            print(f"総タスク数: {len(tasks)}")
            
            # P1-B003を検索
            p1_b003_task = None
            for task in tasks:
                if task.tracker_id == "P1-B003":
                    p1_b003_task = task
                    break
            
            if p1_b003_task:
                print("\nP1-B003 タスク詳細:")
                print(f"   トラッカーID: {p1_b003_task.tracker_id}")
                print(f"   優先度: {p1_b003_task.priority}")
                print(f"   ステータス: {p1_b003_task.status}")
                print(f"   登録日: {p1_b003_task.registration_date}")
                print(f"   更新日: {p1_b003_task.update_date}")
                print(f"   説明: {p1_b003_task.description}")
                
                # 詳細情報をJSON形式で出力
                task_details = {
                    "tracker_id": p1_b003_task.tracker_id,
                    "priority": p1_b003_task.priority,
                    "status": p1_b003_task.status,
                    "registration_date": str(p1_b003_task.registration_date),
                    "update_date": str(p1_b003_task.update_date),
                    "description": p1_b003_task.description
                }
                
                with open("p1_b003_details.json", "w", encoding="utf-8") as f:
                    json.dump(task_details, f, ensure_ascii=False, indent=2)
                
                print("詳細情報を p1_b003_details.json に保存しました")
                
                return task_details
            else:
                print("P1-B003タスクが見つかりません")
                
                # 利用可能なタスクID一覧を表示
                print("\n利用可能なタスクID:")
                for task in tasks[:10]:  # 最初の10件を表示
                    print(f"   - {task.tracker_id}: {task.description[:50] if task.description else 'No description'}...")
                
                return None
                
        except Exception as e:
            print(f"エラー: {e}")
            return None
    
    if __name__ == "__main__":
        get_p1_b003_details()
        
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールがインストールされていない可能性があります")
    sys.exit(1)