#!/usr/bin/env python3
"""
Google Sheetsチケット更新日付自動登録システム

過去3ヶ月のgit commitログからチケット番号を検索し、
Google SheetsのE列（更新日付）に最終更新日を自動設定する。

使用法:
    PROGRESS_TRACKER_SHEET_NAME="シート1" python3 update_tracker_dates_from_git.py
"""

import re
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config


class GitCommitDateExtractor:
    """Git commitログからチケット更新日付を抽出するクラス"""
    
    def __init__(self):
        """初期化"""
        self.target_tickets = [
            'QCC-022', 'QI-004', 'QI-003', 'QI-002', 'QCC-FIX-001',
            'QCC-021', 'QCA-001', 'QCC-011', 'QI-006', 'INTEGRATE-3-6',
            'QI-001', 'P1-022', 'QC-SUCCESS-RESTORE', 'PH2-002', 'P1-B004',
            'P1-023', 'P1-B003', 'P1-021', 'P1-B001', 'PH2-007'
        ]
        
        # チケットID検索用正規表現パターン
        # より柔軟なパターンで、区切り文字（空白、括弧、コロン等）を考慮
        self.ticket_pattern = re.compile(r'\b(' + '|'.join(re.escape(ticket) for ticket in self.target_tickets) + r')\b', re.IGNORECASE)
    
    def get_commits_since_months(self, months: int = 3) -> List[Tuple[str, str, str]]:
        """
        過去N ヶ月のcommitログを取得
        
        Args:
            months: 取得対象月数
            
        Returns:
            List[(commit_hash, iso_date, commit_message)]
        """
        try:
            # git logコマンド実行
            cmd = [
                'git', 'log',
                f'--since={months} months ago',
                '--format=%H|%ci|%s'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('|', 2)
                    if len(parts) == 3:
                        commit_hash, commit_date, commit_message = parts
                        commits.append((commit_hash, commit_date, commit_message))
            
            print(f"📊 過去{months}ヶ月のcommit取得完了: {len(commits)}件")
            return commits
            
        except subprocess.CalledProcessError as e:
            print(f"❌ git logコマンド実行エラー: {e}")
            return []
        except Exception as e:
            print(f"❌ commit取得エラー: {e}")
            return []
    
    def extract_ticket_dates(self, commits: List[Tuple[str, str, str]]) -> Dict[str, Tuple[str, str, str]]:
        """
        commitログからチケットIDと最終更新日付をマッピング
        
        Args:
            commits: [(commit_hash, iso_date, commit_message)]
            
        Returns:
            Dict[ticket_id: (latest_date, commit_hash, commit_message)]
        """
        ticket_commits = {}
        
        print(f"🔍 {len(commits)}件のcommitからチケットID検索中...")
        
        for commit_hash, commit_date, commit_message in commits:
            # チケットIDをcommitメッセージから検索
            matches = self.ticket_pattern.findall(commit_message)
            
            for match in matches:
                ticket_id = match.upper()  # 大文字統一
                
                # 既存のエントリと比較して最新日付を保持
                if ticket_id not in ticket_commits:
                    ticket_commits[ticket_id] = (commit_date, commit_hash, commit_message)
                    print(f"   ✅ {ticket_id}: {commit_date[:10]} - {commit_message[:50]}...")
                else:
                    # 日付比較（ISO形式なので文字列比較で十分）
                    if commit_date > ticket_commits[ticket_id][0]:
                        ticket_commits[ticket_id] = (commit_date, commit_hash, commit_message)
                        print(f"   🔄 {ticket_id}: 更新 {commit_date[:10]} - {commit_message[:50]}...")
        
        print(f"\n📋 チケット-日付マッピング完了: {len(ticket_commits)}件")
        
        # 見つからないチケットを報告
        found_tickets = set(ticket_commits.keys())
        missing_tickets = set(self.target_tickets) - found_tickets
        
        if missing_tickets:
            print(f"⚠️  commitログに見つからないチケット ({len(missing_tickets)}件):")
            for ticket in sorted(missing_tickets):
                print(f"   - {ticket}")
        
        return ticket_commits


class GoogleSheetsDateUpdater:
    """Google Sheetsの更新日付列を更新するクラス"""
    
    def __init__(self):
        """初期化"""
        self.config = get_default_config()
        self.client = GoogleSheetsClient(self.config)
        self.update_date_column = 'E'  # 更新日付列
    
    def get_current_sheet_data(self) -> Dict[str, int]:
        """
        現在のシートデータを取得してチケットID-行番号マッピングを作成
        
        Returns:
            Dict[ticket_id: row_number]
        """
        try:
            # A列（チケットID）とE列（更新日付）を取得
            values = self.client.get_sheet_values('A:E')
            
            if not values:
                print("❌ シートデータ取得失敗")
                return {}
            
            ticket_row_map = {}
            
            for i, row in enumerate(values[1:], 2):  # ヘッダーをスキップして2行目から
                if row and len(row) > 0:
                    ticket_id = row[0].strip()
                    if ticket_id:
                        ticket_row_map[ticket_id] = i
            
            print(f"📊 シートデータ取得完了: {len(ticket_row_map)}行")
            return ticket_row_map
            
        except Exception as e:
            print(f"❌ シートデータ取得エラー: {e}")
            return {}
    
    def format_git_date(self, git_date: str) -> str:
        """
        git日付をGoogle Sheets用形式に変換
        
        Args:
            git_date: "2025-08-14 01:17:52 +0900" 形式
            
        Returns:
            "2025-08-14 01:17:52" 形式
        """
        try:
            # タイムゾーン情報を除去
            date_part = git_date.rsplit(' ', 1)[0]
            return date_part
        except Exception:
            return git_date
    
    def update_tracker_dates(self, ticket_commits: Dict[str, Tuple[str, str, str]]) -> Dict[str, any]:
        """
        チケット更新日付をGoogle Sheetsに一括更新
        
        Args:
            ticket_commits: {ticket_id: (date, hash, message)}
            
        Returns:
            更新結果レポート
        """
        # 現在のシートデータ取得
        ticket_row_map = self.get_current_sheet_data()
        
        if not ticket_row_map:
            return {'success': False, 'error': 'シートデータ取得失敗'}
        
        # 更新対象チケットを特定
        updates = []
        skipped_not_found = []
        skipped_no_row = []
        
        for ticket_id, (commit_date, commit_hash, commit_message) in ticket_commits.items():
            if ticket_id not in ticket_row_map:
                skipped_no_row.append(ticket_id)
                continue
            
            row_number = ticket_row_map[ticket_id]
            formatted_date = self.format_git_date(commit_date)
            
            updates.append({
                'ticket_id': ticket_id,
                'row': row_number,
                'date': formatted_date,
                'commit_hash': commit_hash[:8],
                'commit_message': commit_message[:100]
            })
        
        # 見つからないチケット
        target_tickets = [
            'QCC-022', 'QI-004', 'QI-003', 'QI-002', 'QCC-FIX-001',
            'QCC-021', 'QCA-001', 'QCC-011', 'QI-006', 'INTEGRATE-3-6',
            'QI-001', 'P1-022', 'QC-SUCCESS-RESTORE', 'PH2-002', 'P1-B004',
            'P1-023', 'P1-B003', 'P1-021', 'P1-B001', 'PH2-007'
        ]
        
        found_tickets = set(ticket_commits.keys())
        skipped_not_found = list(set(target_tickets) - found_tickets)
        
        # バッチ更新実行
        update_success = 0
        update_errors = []
        
        print(f"\n🔄 Google Sheets更新開始: {len(updates)}件")
        
        for update in updates:
            try:
                cell_range = f"{self.update_date_column}{update['row']}"
                self.client.update_sheet_values(cell_range, [[update['date']]])
                
                print(f"   ✅ {update['ticket_id']}: {update['date']} (#{update['commit_hash']})")
                update_success += 1
                
            except Exception as e:
                error_msg = f"{update['ticket_id']}: {str(e)}"
                update_errors.append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # 結果レポート
        result = {
            'success': True,
            'total_updates': len(updates),
            'successful_updates': update_success,
            'failed_updates': len(update_errors),
            'skipped_not_found': skipped_not_found,
            'skipped_no_row': skipped_no_row,
            'update_errors': update_errors,
            'update_details': updates
        }
        
        return result


def main():
    """メイン処理"""
    print("🚀 Google Sheetsチケット更新日付自動登録システム開始")
    print("=" * 60)
    
    # Git commitログ解析
    extractor = GitCommitDateExtractor()
    commits = extractor.get_commits_since_months(3)
    
    if not commits:
        print("❌ commitログ取得失敗")
        return
    
    # チケット-日付マッピング生成
    ticket_commits = extractor.extract_ticket_dates(commits)
    
    if not ticket_commits:
        print("❌ チケットIDが見つかりませんでした")
        return
    
    # Google Sheets更新
    updater = GoogleSheetsDateUpdater()
    result = updater.update_tracker_dates(ticket_commits)
    
    # 結果レポート
    print("\n" + "=" * 60)
    print("📊 更新結果レポート")
    print("=" * 60)
    
    if result['success']:
        print(f"✅ 総更新対象: {result['total_updates']}件")
        print(f"✅ 更新成功: {result['successful_updates']}件")
        print(f"❌ 更新失敗: {result['failed_updates']}件")
        print(f"⏭️  commitログなし: {len(result['skipped_not_found'])}件")
        print(f"⏭️  シート行なし: {len(result['skipped_no_row'])}件")
        
        if result['skipped_not_found']:
            print(f"\n⚠️  commitログに見つからないチケット:")
            for ticket in sorted(result['skipped_not_found']):
                print(f"   - {ticket}")
        
        if result['skipped_no_row']:
            print(f"\n⚠️  シートに行が見つからないチケット:")
            for ticket in sorted(result['skipped_no_row']):
                print(f"   - {ticket}")
        
        if result['update_errors']:
            print(f"\n❌ 更新エラー:")
            for error in result['update_errors']:
                print(f"   - {error}")
        
        print(f"\n🎯 Google Sheets更新完了")
        
    else:
        print(f"❌ 更新処理失敗: {result.get('error', '不明なエラー')}")


if __name__ == "__main__":
    main()