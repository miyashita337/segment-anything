#!/usr/bin/env python3
"""
ベースライン特定システム

Pattern C（明示的指定） → Pattern B（更新日付ベース）で
ローリングベースラインを特定する。
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config


class BaselineDetector:
    """ベースライン特定システム"""
    
    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)
        
        # Google Sheetsクライアント初期化
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)
    
    def find_baseline_from_metadata(self, tracker_id: str) -> Optional[str]:
        """
        Pattern C: メタデータから明示的ベースライン取得
        
        Args:
            tracker_id: 対象トラッカーID
            
        Returns:
            Optional[str]: ベースライントラッカーID（未発見時はNone）
        """
        tracker_dir = self.workspace_base / tracker_id
        metadata_path = tracker_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"🔍 {tracker_id}: metadata.json未発見")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            baseline = metadata.get("baseline_tracker")
            if baseline:
                print(f"✅ {tracker_id}: 明示的ベースライン発見 → {baseline}")
                return baseline
            else:
                print(f"🔍 {tracker_id}: metadata.jsonにbaseline_tracker未定義")
                return None
                
        except Exception as e:
            print(f"❌ {tracker_id}: metadata.json読み込み失敗 → {e}")
            return None
    
    def get_all_trackers_by_update_date(self) -> List[Dict[str, str]]:
        """
        Pattern B: Google Sheets更新日付ベースでトラッカー取得
        
        Returns:
            List[Dict]: 更新日付順のトラッカー情報リスト
        """
        try:
            # Google Sheetsから全データ取得
            all_values = self.sheets_client.get_sheet_values('A:T')  # A-T列まで取得
            
            if not all_values or len(all_values) < 2:
                print("❌ Google Sheetsデータ取得失敗")
                return []
            
            headers = all_values[0]
            print(f"📋 Google Sheetsヘッダー: {headers[:5]}...")  # 最初の5列のみ表示
            
            # 更新日付列を特定（E列が更新日付）
            update_date_col = None
            for i, header in enumerate(headers):
                if '更新日' in header or 'update' in header.lower():
                    update_date_col = i
                    break
            
            if update_date_col is None:
                print("❌ 更新日付列が見つかりません")
                # E列（インデックス4）をデフォルトとして使用
                update_date_col = 4
                print(f"⚠️ デフォルトでE列（{update_date_col}）を更新日付として使用")
            
            trackers = []
            
            for i, row in enumerate(all_values[1:], 2):  # ヘッダー行をスキップ
                if len(row) > 0:
                    tracker_id = row[0] if len(row) > 0 else ""
                    status = row[2] if len(row) > 2 else ""  # C列がステータス
                    update_date = row[update_date_col] if len(row) > update_date_col else ""
                    
                    # トラッカーIDが有効で、完了状態（/releaseまたは着手中）のもののみ
                    if tracker_id and (status == "/release" or status == "着手中"):
                        trackers.append({
                            "tracker_id": tracker_id,
                            "status": status,
                            "update_date": update_date,
                            "row_num": i
                        })
            
            # 更新日付でソート（古い順）
            def parse_date(date_str: str) -> datetime:
                """日付文字列をパース"""
                if not date_str:
                    return datetime.min
                
                # 複数の日付形式に対応
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                
                print(f"⚠️ 日付パース失敗: {date_str}")
                return datetime.min
            
            trackers.sort(key=lambda x: parse_date(x['update_date']))
            
            print(f"📊 更新日付順トラッカー取得: {len(trackers)}件")
            for tracker in trackers[:5]:  # 最初の5件を表示
                print(f"   {tracker['tracker_id']}: {tracker['update_date']} ({tracker['status']})")
            
            return trackers
            
        except Exception as e:
            print(f"❌ Google Sheets取得エラー: {e}")
            return []
    
    def find_baseline_from_update_date(self, tracker_id: str) -> Optional[str]:
        """
        Pattern B: 更新日付ベースでベースライン特定
        
        Args:
            tracker_id: 対象トラッカーID
            
        Returns:
            Optional[str]: ベースライントラッカーID
        """
        trackers = self.get_all_trackers_by_update_date()
        
        if not trackers:
            print(f"❌ {tracker_id}: トラッカーリスト取得失敗")
            return None
        
        # 対象トラッカーのインデックス特定
        current_idx = None
        for i, tracker in enumerate(trackers):
            if tracker['tracker_id'] == tracker_id:
                current_idx = i
                break
        
        if current_idx is None:
            print(f"❌ {tracker_id}: Google Sheetsに未発見")
            return None
        
        if current_idx == 0:
            print(f"🎯 {tracker_id}: 初回トラッカー（ベースラインなし）")
            return None
        
        baseline_tracker = trackers[current_idx - 1]['tracker_id']
        print(f"✅ {tracker_id}: 更新日付ベースベースライン → {baseline_tracker}")
        
        return baseline_tracker
    
    def determine_baseline_tracker(self, tracker_id: str) -> Optional[str]:
        """
        統合ベースライン特定システム
        Pattern C → Pattern B の順で試行
        
        Args:
            tracker_id: 対象トラッカーID
            
        Returns:
            Optional[str]: ベースライントラッカーID
        """
        print(f"🎯 {tracker_id}のベースライン特定開始...")
        
        # Pattern C: メタデータからの明示的取得
        baseline = self.find_baseline_from_metadata(tracker_id)
        if baseline:
            return baseline
        
        # Pattern B: 更新日付ベースでの特定
        baseline = self.find_baseline_from_update_date(tracker_id)
        if baseline:
            return baseline
        
        print(f"❌ {tracker_id}: ベースライン特定失敗")
        return None
    
    def validate_baseline_data(self, baseline_tracker: str) -> bool:
        """
        ベースライントラッカーのデータ存在確認
        
        Args:
            baseline_tracker: ベースライントラッカーID
            
        Returns:
            bool: データ存在の可否
        """
        tracker_dir = self.workspace_base / baseline_tracker
        
        # extraction_result.json存在確認
        json_path = tracker_dir / "extraction_result.json"
        if json_path.exists():
            print(f"✅ {baseline_tracker}: extraction_result.json存在")
            return True
        
        # 抽出ディレクトリ存在確認
        extraction_dir = tracker_dir / "extraction"
        if extraction_dir.exists() and any(extraction_dir.iterdir()):
            print(f"⚠️ {baseline_tracker}: extraction_result.json不在、但し抽出データ存在")
            return True
        
        print(f"❌ {baseline_tracker}: データ未発見")
        return False


def main():
    """テスト実行"""
    detector = BaselineDetector()
    
    # QCC-022のベースライン特定
    baseline = detector.determine_baseline_tracker('QCC-022')
    
    if baseline:
        print(f"\n🎯 QCC-022のベースライン: {baseline}")
        
        # データ存在確認
        if detector.validate_baseline_data(baseline):
            print(f"✅ {baseline}: データ検証完了")
        else:
            print(f"❌ {baseline}: データ不足")
    else:
        print(f"\n❌ QCC-022: ベースライン特定失敗")
    
    return baseline


if __name__ == "__main__":
    main()