#!/usr/bin/env python3
"""
Google Sheets 137件全トラッカーID完全置換スクリプト
旧IDから新IDへの完全置換実行
"""

import json
import sys
import os
import time
from typing import Dict, List

# 必要なパスを追加
sys.path.append('/mnt/c/AItools/segment-anything')

try:
    from tools.progress_tracker.sheets_client import GoogleSheetsClient
    from tools.progress_tracker.config import get_default_config
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("必要なモジュールが見つかりません")
    sys.exit(1)

class GoogleSheetsCompleteUpdater:
    def __init__(self):
        self.mapping_file = "/tmp/tracker_function_mapping.json"
        self.replacement_map = {}
        self.config = get_default_config()
        self.client = GoogleSheetsClient(self.config)
        self.stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "errors": []
        }
    
    def load_mapping(self) -> bool:
        """マッピングデータの読み込み"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 置換マップ作成
            for category, trackers in data["mapping"].items():
                for tracker in trackers:
                    old_id = tracker["original_id"]
                    new_id = tracker["new_id"]
                    self.replacement_map[old_id] = new_id
            
            print(f"✅ マッピング読み込み完了: {len(self.replacement_map)}件")
            return True
            
        except Exception as e:
            print(f"❌ マッピング読み込み失敗: {e}")
            return False
    
    def get_all_tracker_data(self) -> List[tuple]:
        """全トラッカーデータの取得"""
        try:
            # A列の全データを取得
            all_values = self.client.get_sheet_values("A:A")
            
            tracker_data = []
            if all_values:
                for i, row in enumerate(all_values, 1):
                    if row and len(row) > 0:
                        tracker_id = row[0]
                        if tracker_id and tracker_id in self.replacement_map:
                            tracker_data.append((i, tracker_id, self.replacement_map[tracker_id]))
            
            print(f"📊 更新対象発見: {len(tracker_data)}件")
            return tracker_data
            
        except Exception as e:
            print(f"❌ データ取得失敗: {e}")
            return []
    
    def update_single_tracker(self, row: int, old_id: str, new_id: str) -> bool:
        """単一トラッカーの更新"""
        try:
            # A列の特定行を更新
            range_name = f"A{row}"
            self.client.update_sheet_values(range_name, [[new_id]])
            
            print(f"   ✅ 行{row:3}: {old_id:15} → {new_id}")
            return True
            
        except Exception as e:
            error_msg = f"行{row}: {old_id} → {new_id} 更新失敗 - {e}"
            self.stats["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
            return False
    
    def run_complete_update(self):
        """Google Sheets完全更新の実行"""
        print("🚀 Google Sheets 137件全トラッカーID置換開始")
        print("=" * 60)
        
        # マッピング読み込み
        if not self.load_mapping():
            return False
        
        # 全データ取得
        tracker_data = self.get_all_tracker_data()
        if not tracker_data:
            print("❌ 更新対象が見つかりませんでした")
            return False
        
        # 更新実行
        print(f"📝 {len(tracker_data)}件の更新を開始...")
        self.stats["total_updates"] = len(tracker_data)
        
        for row, old_id, new_id in tracker_data:
            if self.update_single_tracker(row, old_id, new_id):
                self.stats["successful_updates"] += 1
            else:
                self.stats["failed_updates"] += 1
            
            # API制限回避のため小さな待機
            time.sleep(0.1)
        
        # 結果レポート
        self.print_summary()
        return True
    
    def print_summary(self):
        """処理結果サマリー"""
        print("\n" + "=" * 60)
        print("📊 Google Sheets更新完了レポート")
        print("=" * 60)
        print(f"📋 総更新対象: {self.stats['total_updates']}件")
        print(f"✅ 成功: {self.stats['successful_updates']}件")
        print(f"❌ 失敗: {self.stats['failed_updates']}件")
        
        if self.stats["errors"]:
            print(f"\n⚠️ エラー詳細:")
            for error in self.stats["errors"][:10]:  # 最初の10件表示
                print(f"   - {error}")
            if len(self.stats["errors"]) > 10:
                print(f"   ... 他{len(self.stats['errors'])-10}件")
        else:
            print("✅ エラー: なし")
        
        # 成功率計算
        if self.stats["total_updates"] > 0:
            success_rate = (self.stats["successful_updates"] / self.stats["total_updates"]) * 100
            print(f"\n📈 成功率: {success_rate:.1f}%")
    
    def verify_updates(self):
        """更新結果の検証"""
        print("\n🔍 更新結果検証中...")
        
        try:
            # 全データ再取得
            all_values = self.client.get_sheet_values("A:A")
            
            found_old_ids = []
            found_new_ids = []
            
            if all_values:
                for i, row in enumerate(all_values, 1):
                    if row and len(row) > 0:
                        tracker_id = row[0]
                        if tracker_id in self.replacement_map.keys():
                            found_old_ids.append((i, tracker_id))
                        elif tracker_id in self.replacement_map.values():
                            found_new_ids.append((i, tracker_id))
            
            print(f"❌ 残存旧ID: {len(found_old_ids)}件")
            for row, old_id in found_old_ids[:5]:
                print(f"   行{row}: {old_id}")
            
            print(f"✅ 新ID確認: {len(found_new_ids)}件")
            for row, new_id in found_new_ids[:5]:
                print(f"   行{row}: {new_id}")
            
            if len(found_old_ids) == 0:
                print("🎉 Google Sheets更新100%完了！")
            else:
                print(f"⚠️ {len(found_old_ids)}件の旧IDが残存しています")
                
        except Exception as e:
            print(f"❌ 検証エラー: {e}")

if __name__ == "__main__":
    updater = GoogleSheetsCompleteUpdater()
    
    # 実行モード選択
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        updater.load_mapping()
        updater.verify_updates()
    else:
        success = updater.run_complete_update()
        if success:
            updater.verify_updates()